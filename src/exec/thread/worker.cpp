/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include "common/global/global.hpp"
#include "exec/exec.hpp"
#include "exec/thread/worker.hpp"

#define CCL_WORKER_CHECK_STOP_ITERS     (16384)
#define CCL_WORKER_CHECK_UPDATE_ITERS   (16384)
#define CCL_WORKER_CHECK_AFFINITY_ITERS (16384)
#define CCL_WORKER_PROCESS_ALL_ITERS    (4096)

static void* ccl_worker_func(void* args);

ccl_worker::ccl_worker(size_t idx, std::unique_ptr<ccl_sched_queue> queue)
        : ccl_base_thread(idx, ccl_worker_func),
          should_lock(false),
          is_locked(false),
          strict_sched_queue(std::unique_ptr<ccl_strict_sched_queue>(new ccl_strict_sched_queue())),
          sched_queue(std::move(queue)) {}

void ccl_worker::add(ccl_sched* sched) {
    LOG_DEBUG("add sched ", sched, ", type ", ccl_coll_type_to_str(sched->coll_param.ctype));

    CCL_ASSERT(sched);
    CCL_ASSERT(!sched->bin);
    CCL_ASSERT(sched->get_in_bin_status() != ccl_sched_in_bin_added);

    if (sched->strict_order) {
        /* to keep valid non-completed req until safe releasing */
        sched->req->increase_counter(1);
        strict_sched_queue->add(sched);
    }
    else {
        sched_queue->add(sched);
    }
}

ccl::status ccl_worker::do_work(size_t& processed_count) {
    do_work_counter++;

    auto ret = process_strict_sched_queue();
    if (ret != ccl::status::success)
        return ret;

    ret = process_sched_queue(processed_count,
                              (do_work_counter % CCL_WORKER_PROCESS_ALL_ITERS) ? false : true);
    if (ret != ccl::status::success)
        return ret;

#ifdef ENABLE_DEBUG
    if (processed_count == 0 && (do_work_counter % CCL_WORKER_PROCESS_ALL_ITERS * 1024) == 0) {
        //sched_queue->dump(std::cout);
    }
#endif

    return ccl::status::success;
}

ccl::status ccl_worker::process_strict_sched_queue() {
    auto& queue = strict_sched_queue->peek();
    if (queue.empty())
        return ccl::status::success;

    size_t erased_scheds = 0;

    /* try to finish previous postponed operations */
    for (auto sched_it = queue.begin(); sched_it != queue.end(); sched_it++) {
        ccl_sched* sched = *sched_it;

        if (sched->get_in_bin_status() == ccl_sched_in_bin_erased) {
            CCL_ASSERT(!sched->bin);
            erased_scheds++;

            /* only single sched in active strict queue can be erased since previous call */
            CCL_ASSERT(erased_scheds == 1);

            /* now it is safe to release this sched */
            sched->req->complete();
            continue;
        }

        if (sched->get_in_bin_status() == ccl_sched_in_bin_none) {
            CCL_ASSERT(!sched->bin, "unexpected bin ", sched->bin);
            /* here we add sched from strict_queue to regular queue for real execution */
            LOG_DEBUG("add sched ", sched, " from strict_queue to exec_queue, req ", sched->req);
            sched_queue->add(sched);
        }

        CCL_ASSERT(sched->get_in_bin_status() == ccl_sched_in_bin_added,
                   "sched ",
                   sched,
                   " unexpected in_bin_status ",
                   sched->get_in_bin_status());

        sched->do_progress();

        if (!sched->is_strict_order_satisfied()) {
            /*
                we can't state that current operation is started with strict order
                remove all previous operations from queue, as they were successfully started with strict order
                and return to strict starting for current operation on the next call
            */
            std::vector<ccl_sched*>(sched_it, queue.end()).swap(queue);
            return ccl::status::success;
        }
        else {
            /* now it is safe to release this sched */
            sched->req->complete();
        }
    }

    queue.clear();

    return ccl::status::success;
}

ccl::status ccl_worker::process_sched_queue(size_t& completed_sched_count, bool process_all) {
    completed_sched_count = 0;
    if (process_all) {
        auto bins = sched_queue->peek_all();

        if (bins.empty())
            return ccl::status::success;

        size_t completed_sched_count_local = 0;
        for (auto& bin : bins) {
            process_sched_bin(bin, completed_sched_count_local);
            completed_sched_count += completed_sched_count_local;
        }

        if (completed_sched_count)
            LOG_DEBUG("process_all, completed_sched_count ", completed_sched_count);

        return ccl::status::success;
    }
    else {
        ccl_sched_bin* bin = sched_queue->peek();
        if (!bin)
            return ccl::status::success;
        return process_sched_bin(bin, completed_sched_count);
    }
}

ccl::status ccl_worker::process_sched_bin(ccl_sched_bin* bin, size_t& completed_sched_count) {
    CCL_ASSERT(bin);
    completed_sched_count = 0;

    size_t bin_size = bin->size();
    CCL_ASSERT(bin_size > 0);

    LOG_TRACE("bin ", bin, ", sched_count ", bin_size);

    /* ensure communication progress */

    for (size_t sched_idx = 0; sched_idx < 1 /*bin_size*/; sched_idx++) {
        ccl_sched* sched = bin->get(sched_idx);
        ccl_comm* comm = sched->coll_param.comm;
        atl_status_t atl_status = comm->atl->atl_ep_poll(bin->get_atl_ep());
        CCL_THROW_IF_NOT(atl_status == ATL_STATUS_SUCCESS, "bad status ", atl_status);
    }

    //    if (ccl::global_data::get().is_ft_enabled) {
    //        if (atl_status != ATL_STATUS_SUCCESS)
    //            return ccl::status::blocked_due_to_resize;
    //    }
    //    else {
    //        CCL_THROW_IF_NOT(atl_status == ATL_STATUS_SUCCESS, "bad status ", atl_status);
    //    }

    // iterate through the scheds stored in the bin
    for (size_t sched_idx = 0; sched_idx < bin_size;) {
        ccl_sched* sched = bin->get(sched_idx);
        CCL_ASSERT(sched && bin == sched->bin);

        sched->do_progress();

        if (sched->start_idx == sched->entries.size()) {
            // the last entry in the schedule has been completed, clean up the schedule and complete its request
            LOG_DEBUG("complete and dequeue: sched ",
                      sched,
                      ", coll ",
                      ccl_coll_type_to_str(sched->coll_param.ctype),
                      ", req ",
                      sched->req,
                      ", entry_count ",
                      sched->entries.size());

            // remove completed schedule from the bin
            sched_queue->erase(bin, sched_idx);
            CCL_ASSERT(!sched->bin);
            bin_size--;
            LOG_DEBUG("completing request ", sched->req);
            sched->complete();
            ++completed_sched_count;
        }
        else {
            // this schedule is not completed yet, switch to the next sched in bin scheds list
            // progression of unfinished schedules will be continued in the next call of @ref ccl_bin_progress
            ++sched_idx;
        }
    }

    return ccl::status::success;
}

void ccl_worker::clear_queue() {
    strict_sched_queue->clear();
    sched_queue->clear();
}

static inline bool ccl_worker_check_conditions(ccl_worker* worker,
                                               size_t iter_count,
                                               int do_work_status) {
    bool should_stop = false;

    if ((iter_count % CCL_WORKER_CHECK_STOP_ITERS) == 0) {
        if (worker->should_stop.load(std::memory_order_acquire))
            should_stop = true;
    }

    if (ccl::global_data::get().is_ft_enabled &&
        unlikely(do_work_status == ccl::status::blocked_due_to_resize ||
                 iter_count % CCL_WORKER_CHECK_UPDATE_ITERS == 0)) {
        if (worker->should_lock.load(std::memory_order_acquire)) {
            worker->clear_queue();
            worker->is_locked = true;
            while (worker->should_lock.load(std::memory_order_relaxed)) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
            worker->is_locked = false;
        }
    }

    if ((iter_count % CCL_WORKER_CHECK_AFFINITY_ITERS) == 0) {
        int start_affinity = worker->get_start_affinity();
        int affinity = worker->get_affinity();
        if (start_affinity != affinity) {
            LOG_ERROR("worker ",
                      worker->get_idx(),
                      " unexpectedly changed affinity from ",
                      start_affinity,
                      " to ",
                      affinity);
        }
    }

    return should_stop;
}

static void* ccl_worker_func(void* args) {
    auto worker = static_cast<ccl_worker*>(args);

    auto worker_idx = worker->get_idx();

    LOG_INFO("worker_idx ", worker_idx);

    size_t iter_count = 0;
    size_t processed_count = 0;
    size_t max_spin_count = ccl::global_data::env().spin_count;
    size_t spin_count = max_spin_count;
    ccl::status ret;

    ccl::global_data::get().is_worker_thread = true;

    worker->started = true;

    do {
        try {
            ret = worker->do_work(processed_count);

            if (ccl_worker_check_conditions(worker, iter_count, ret))
                break;
        }
        catch (ccl::exception& ccl_e) {
            CCL_FATAL("worker ", worker_idx, " caught internal exception: ", ccl_e.what());
        }
        catch (std::exception& e) {
            CCL_FATAL("worker ", worker_idx, " caught exception: ", e.what());
        }
        catch (...) {
            CCL_FATAL("worker ", worker_idx, " caught general exception");
        }

        iter_count++;

        if (processed_count == 0) {
            spin_count--;
            if (!spin_count) {
                ccl_yield(ccl::global_data::env().yield_type);
                spin_count = 1;
            }
        }
        else {
            spin_count = max_spin_count;
        }
    } while (true);

    worker->started = false;

    return nullptr;
}
