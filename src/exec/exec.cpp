/*
 Copyright 2016-2019 Intel Corporation
 
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

#include "exec/exec.hpp"
#include "exec/thread/service_worker.hpp"
#include "exec/thread/worker.hpp"
#include "unordered_coll/unordered_coll.hpp"
#include "sched/extra_sched.hpp"

size_t ccl_executor::get_atl_comm_count(size_t worker_count)
{
    size_t comm_count = worker_count;

    if (env_data.priority_mode != ccl_priority_none)
    {
        comm_count *= CCL_PRIORITY_BUCKET_COUNT;
    }

    return comm_count;
}

std::unique_ptr<ccl_sched_queue> ccl_executor::create_sched_queue(size_t idx, size_t comm_per_worker)
{
    std::vector<atl_comm_t*> comm_vec(atl_comms + idx * comm_per_worker,
                                      atl_comms + (idx + 1) * comm_per_worker);
    std::unique_ptr<ccl_sched_queue> sched_queue{new ccl_sched_queue(comm_vec)};
    return sched_queue;
}

ccl_executor::ccl_executor()
{
    auto worker_count = env_data.worker_count;
    workers.reserve(worker_count);
    auto comm_count = get_atl_comm_count(worker_count);

    atl_attr.comm_count = comm_count;
    atl_attr.enable_shm = env_data.enable_shm;
    atl_attr.enable_rma = env_data.enable_rma;

    LOG_INFO("init ATL, requested comm_count ", atl_attr.comm_count);

    atl_status_t atl_status = atl_init(ccl_atl_transport_to_str(env_data.atl_transport),
                                       nullptr, nullptr, &atl_proc_coord,
                                       &atl_attr, &atl_comms, &atl_desc);
    CCL_THROW_IF_NOT(atl_status == atl_status_success && atl_desc && atl_comms,
                     "ATL init failed, res ", atl_status, ", desc ", atl_desc, ", comm ",atl_comms);

    global_data.is_ft_enabled = is_ft_enabled(atl_desc);

    LOG_INFO("global_proc_idx ", atl_proc_coord.global_idx,
             ", global_proc_count ", atl_proc_coord.global_count,
             ", local_proc_idx ", atl_proc_coord.local_idx,
             ", local_proc_count ", atl_proc_coord.local_count,
             ", worker_count ", worker_count);

    if (get_global_proc_idx() == 0)
    {
        LOG_INFO("\nATL parameters:",
                 "\n  comm_count:             ", atl_attr.comm_count,
                 "\n  enable_shm:             ", atl_attr.enable_shm,
                 "\n  is_tagged_coll_enabled: ", atl_attr.is_tagged_coll_enabled,
                 "\n  tag_bits:               ", atl_attr.tag_bits,
                 "\n  max_tag:                ", atl_attr.max_tag,
                 "\n  enable_rma:             ", atl_attr.enable_rma,
                 "\n  max_order_waw_size:     ", atl_attr.max_order_waw_size);
    }
}

void ccl_executor::start_workers()
{
    auto worker_count = env_data.worker_count;
    auto comm_count = get_atl_comm_count(worker_count);

    if (env_data.worker_offload)
    {
        CCL_THROW_IF_NOT(env_data.worker_affinity.size() >= get_local_proc_count() * worker_count,
                         "unexpected worker affinity length ", env_data.worker_affinity.size(),
                         ", should be ", get_local_proc_count() * worker_count);
    }

    size_t comm_per_worker = comm_count / worker_count;
    for (size_t idx = 0; idx < worker_count; idx++)
    {
        if (env_data.enable_fusion && idx == 0)
        {
            LOG_DEBUG("create service worker");
            workers.emplace_back(new ccl_service_worker(this, idx, create_sched_queue(idx, comm_per_worker),
                                                        *global_data.fusion_manager));
        }
        else
        {
            workers.emplace_back(new ccl_worker(this, idx, create_sched_queue(idx, comm_per_worker)));
        }

        if (env_data.worker_offload)
        {
            size_t affinity = env_data.worker_affinity[get_local_proc_idx() * worker_count + idx];
            workers.back()->start();
            workers.back()->pin(affinity);
            LOG_INFO("started worker: global_proc_idx ", get_global_proc_idx(),
                     ", local_proc_idx ", get_local_proc_idx(),
                     ", worker_idx ", idx,
                     ", affinity ", affinity);
        }
    }
}

ccl_executor::~ccl_executor()
{
    if (listener)
    {
        listener->stop();
        LOG_DEBUG("stopped listener");

        lock_workers();
        unlock_workers();
    }

    for (size_t idx = 0; idx < workers.size(); idx++)
    {
        if (env_data.worker_offload)
        {
            workers[idx]->stop();
            LOG_DEBUG("stopped worker # ", idx);
        }
    }

    if (get_global_proc_idx() == 0)
        LOG_INFO("finalizing ATL");

    auto result = atl_finalize(atl_desc, atl_comms);

    if (get_global_proc_idx() == 0)
        LOG_INFO("finalized ATL");

    if (result != atl_status_success)
    {
        LOG_ERROR("ATL finalize failed, error ", result);
    }
}

void ccl_executor::lock_workers()
{
    size_t idx;
    for (idx = 0; idx < workers.size(); idx++)
    {
        workers[idx]->should_lock = true;
    }

    idx = 0;
    while (idx < workers.size())
    {
        if (workers[idx]->is_locked.load(std::memory_order_relaxed))
        {
            idx++;
        }
        else
        {
            ccl_yield(env_data.yield_type);
        }
    }
}

void ccl_executor::unlock_workers()
{
    size_t idx;
    for (idx = 0; idx < workers.size(); idx++)
    {
        workers[idx]->should_lock = false;
    }
    idx = 0;
    while (idx < workers.size())
    {
        if (!workers[idx]->is_locked.load(std::memory_order_relaxed))
        {
            idx++;
        }
    }
}

void ccl_executor::update_workers()
{
    size_t comm_count = get_atl_comm_count(workers.size());
    size_t comm_per_worker = comm_count / workers.size();

    LOG_INFO("atl comm count ", comm_count);

    for (size_t idx = 0; idx < workers.size(); idx++)
    {
        workers[idx]->reset_queue(create_sched_queue(idx, comm_per_worker));
    }
}

ccl_status_t ccl_executor::create_listener(ccl_resize_fn_t resize_func)
{
    if (listener)
    {
        LOG_ERROR("attempt to twice create listener");
        return ccl_status_runtime_error;
    }

    if (resize_func != NULL)
        atl_set_resize_function(global_data.executor->atl_desc, (atl_resize_fn_t) resize_func);

    listener = std::unique_ptr<ccl_listener>(new ccl_listener(&global_data));
    listener->start();

    /* pin listener thread together with first worker thread */
    size_t affinity = env_data.worker_affinity[get_local_proc_idx() * env_data.worker_count];
    listener->pin(affinity);
    LOG_DEBUG("started listener");

    return ccl_status_success;
}

void ccl_executor::start(ccl_extra_sched* extra_sched)
{
    CCL_ASSERT(extra_sched->internal_type == ccl_sched_internal_unordered_coll,
               "should be unordered_coll for now");

    extra_sched->set_counter(1);
    workers[0]->add(extra_sched);
}

void ccl_executor::start(ccl_master_sched* sched)
{
    size_t worker_idx;
    for (size_t idx = 0; idx < sched->partial_scheds.size(); idx++)
    {
        if (env_data.atl_transport == ccl_atl_mpi && sched->coll_param.ctype == ccl_coll_sparse_allreduce)
        {
            /* to workaround issue with multi-threaded MPI_Iprobe support in IMPI */
            worker_idx = 0;
        }
        else
        {
            worker_idx = sched->partial_scheds[idx]->sched_id % workers.size();
        }

        workers[worker_idx]->add(sched->partial_scheds[idx].get());
    }
}

void ccl_executor::wait(const ccl_request* req)
{
    /* set urgent state for fusion manager */
    req->urgent = true;

    /* wait completion */
    while (!req->is_completed())
    {
        do_work();
    }

    req->urgent = false;
}

bool ccl_executor::test(const ccl_request* req)
{
    if (!req->is_completed())
    {
        req->urgent = true;
        do_work();
        return false;
    }

    req->urgent = false;
    return true;
}

void ccl_executor::do_work()
{
    size_t processed_count;
    if (env_data.worker_offload)
    {
        ccl_yield(env_data.yield_type);
    }
    else
    {
        for (auto& worker : workers)
        {
            worker->do_work(processed_count);
        }
    }
}

size_t ccl_executor::get_worker_count() const
{
    return workers.size();
}
