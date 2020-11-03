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
#include <numeric>

#include "exec/exec.hpp"
#include "exec/thread/service_worker.hpp"
#include "exec/thread/worker.hpp"
#include "common/env/env.hpp"
#include "sched/extra_sched.hpp"

size_t ccl_executor::get_worker_idx_by_sched_id(ccl_sched* sched) {
    return sched->sched_id % workers.size();
}

size_t ccl_executor::get_worker_idx_round_robin(ccl_sched* sched) {
    ++rr_worker_idx %= workers.size();
    return rr_worker_idx;
}

size_t ccl_executor::calculate_atl_ep_count(size_t worker_count) {
    size_t ep_count = worker_count;

    if (ccl::global_data::env().priority_mode != ccl_priority_none) {
        ep_count *= CCL_PRIORITY_BUCKET_COUNT;
    }

    return ep_count;
}

atl_attr_t ccl_executor::generate_atl_attr(const ccl::env_data& env) {

    atl_attr_t attr;

    attr.ep_count = calculate_atl_ep_count(env.worker_count);
    attr.enable_shm = env.enable_shm;
    /*
        TODO:
        executor may be destroyed before cached rma-based schedule made memory deregistration
        need to refactor global objects dependencies
        don't use ring_rma till that
    */
    attr.enable_rma = 0; // env.enable_rma;
    attr.sync_coll = env.sync_coll;
    attr.extra_ep = env.extra_ep;

    return attr;
}

std::unique_ptr<ccl_sched_queue> ccl_executor::create_sched_queue(size_t idx,
                                                                  size_t ep_per_worker) {
    std::vector<size_t> ep_vec(ep_per_worker);
    std::iota(std::begin(ep_vec), std::end(ep_vec), idx * ep_per_worker);
    std::unique_ptr<ccl_sched_queue> sched_queue{ new ccl_sched_queue(ep_vec) };
    return sched_queue;
}

ccl_executor::ccl_executor(const char* main_addr) {

    auto& env = ccl::global_data::env();

    get_worker_idx_fn =
        (env.enable_fusion || env.enable_unordered_coll)
            ? &ccl_executor::get_worker_idx_by_sched_id
            : &ccl_executor::get_worker_idx_round_robin;

    /* generate ATL attr for all future communicators */
    atl_wrapper::attr = generate_atl_attr(env);

    set_local_coord();
    CCL_THROW_IF_NOT(env.env_2_worker_affinity(get_local_proc_idx(), get_local_proc_count()));
    start_workers();
}

void ccl_executor::start_workers() {

    auto& env = ccl::global_data::env();

    auto worker_count = env.worker_count;
    auto ep_count = calculate_atl_ep_count(worker_count);

    if (env.worker_offload) {
        CCL_THROW_IF_NOT(
            env.worker_affinity.size() >= get_local_proc_count() * worker_count,
            "unexpected worker affinity length ",
            env.worker_affinity.size(),
            ", should be ",
            get_local_proc_count() * worker_count);
    }

    size_t ep_per_worker = ep_count / worker_count;
    for (size_t idx = 0; idx < worker_count; idx++) {
        if (env.enable_fusion && idx == 0) {
            LOG_DEBUG("create service worker");
            workers.emplace_back(new ccl_service_worker(idx,
                                                        create_sched_queue(idx, ep_per_worker),
                                                        *ccl::global_data::get().fusion_manager));
        }
        else {
            workers.emplace_back(new ccl_worker(idx, create_sched_queue(idx, ep_per_worker)));
        }

        if (env.worker_offload) {
            size_t affinity =
                env.worker_affinity[get_local_proc_idx() * worker_count + idx];

            CCL_THROW_IF_NOT(workers.back()->start(affinity) == ccl_status_success,
                             "failed to start worker # ",
                             idx);

            LOG_DEBUG("started worker: local_proc_idx ",
                      get_local_proc_idx(),
                      ", worker_idx ",
                      idx,
                      ", affinity ",
                      affinity);
        }
    }
}

ccl_executor::~ccl_executor() {
    // TODO: Rework to support listener
    //    if (listener) {
    //        listener->stop();
    //        LOG_DEBUG("stopped listener");
    //
    //        lock_workers();
    //        unlock_workers();
    //    }
    //    listener.reset();

    for (size_t idx = 0; idx < workers.size(); idx++) {
        if (ccl::global_data::env().worker_offload) {
            if (workers[idx]->stop() != ccl_status_success) {
                LOG_ERROR("failed to stop worker # ", idx);
            }
            else
                LOG_DEBUG("stopped worker # ", idx);

            workers[idx].reset();
        }
    }
}

void ccl_executor::lock_workers() {
    size_t idx;
    for (idx = 0; idx < workers.size(); idx++) {
        workers[idx]->should_lock = true;
    }

    idx = 0;
    while (idx < workers.size()) {
        if (workers[idx]->is_locked.load(std::memory_order_relaxed)) {
            idx++;
        }
        else {
            ccl_yield(ccl::global_data::env().yield_type);
        }
    }
}

void ccl_executor::unlock_workers() {
    size_t idx;
    for (idx = 0; idx < workers.size(); idx++) {
        workers[idx]->should_lock = false;
    }
    idx = 0;
    while (idx < workers.size()) {
        if (!workers[idx]->is_locked.load(std::memory_order_relaxed)) {
            idx++;
        }
    }
}

void ccl_executor::update_workers() {
    size_t ep_count = calculate_atl_ep_count(workers.size());
    size_t ep_per_worker = ep_count / workers.size();

    LOG_INFO("atl ep_count ", ep_count);

    for (size_t idx = 0; idx < workers.size(); idx++) {
        workers[idx]->reset_queue(create_sched_queue(idx, ep_per_worker));
    }
}

// TODO: Rework to support listener
//ccl_status_t ccl_executor::create_listener(ccl_resize_fn_t resize_func) {
//    if (listener) {
//        LOG_ERROR("attempt to create listener twice");
//        return ccl_status_runtime_error;
//    }
//
//    if (resize_func != NULL)
//        ccl::global_data::get().atl->atl_set_resize_function((atl_resize_fn_t)resize_func);
//
//    /* pin listener thread together with first worker thread */
//    auto worker_affinity = ccl::global_data::env().worker_affinity;
//    size_t affinity_idx = get_local_proc_idx() * ccl::global_data::env().worker_count;
//    CCL_THROW_IF_NOT(worker_affinity.size() > affinity_idx);
//    size_t affinity = worker_affinity[affinity_idx];
//
//    listener = std::unique_ptr<ccl_listener>(new ccl_listener());
//    listener->start(affinity);
//
//    LOG_DEBUG("started listener");
//
//    return ccl_status_success;
//}

void ccl_executor::start(ccl_extra_sched* extra_sched) {
    CCL_ASSERT(extra_sched->internal_type == ccl_sched_internal_unordered_coll,
               "should be unordered_coll for now");

    extra_sched->set_counter(1);
    workers[0]->add(extra_sched);
}

void ccl_executor::start(ccl_master_sched* sched) {
    size_t worker_idx;
    for (size_t idx = 0; idx < sched->partial_scheds.size(); idx++) {
        worker_idx = (this->*get_worker_idx_fn)(sched->partial_scheds[idx].get());
        workers[worker_idx]->add(sched->partial_scheds[idx].get());
    }
}

void ccl_executor::wait(const ccl_request* req) {
    /* set urgent state for fusion manager */
    req->urgent = true;

    /* wait completion */
    while (!req->is_completed()) {
        do_work();
    }

    req->urgent = false;
}

bool ccl_executor::test(const ccl_request* req) {
    if (!req->is_completed()) {
        req->urgent = true;
        do_work();
        return false;
    }

    req->urgent = false;
    return true;
}

void ccl_executor::do_work() {
    size_t processed_count;
    if (ccl::global_data::env().worker_offload) {
        ccl_yield(ccl::global_data::env().yield_type);
    }
    else {
        for (auto& worker : workers) {
            worker->do_work(processed_count);
        }
    }
}

size_t ccl_executor::get_worker_count() const {
    return workers.size();
}
void ccl_executor::set_local_coord() {

    // TODO: works only for hydra
    const char* mpi_local_ranks_env = "MPI_LOCALNRANKS";
    const char* mpi_local_id_env = "MPI_LOCALRANKID";

    char* local_count = getenv(mpi_local_ranks_env);
    if (local_count) {
        char* local_id = getenv(mpi_local_id_env);
        if (local_id) {
            local_proc_count = std::atoi(local_count);
            local_proc_idx = std::atoi(local_id);
            return;
        }
    }

    local_proc_count = 1;
    local_proc_idx = 0;

    LOG_INFO("WARNING: ",
             mpi_local_ranks_env,
             " or ",
             mpi_local_id_env,
             " not found. Use default: ",
             local_proc_count,
             " , ",
             local_proc_idx);
}
