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
#include "coll_util.hpp"

#include "sched/entry/coll/coll_entry_helper.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/entry/ze/ze_event_signal_entry.hpp"
#include "sched/entry/ze/ze_event_wait_entry.hpp"

namespace ccl {

void add_wait_events(ccl_sched* sched, const std::vector<ze_event_handle_t>& wait_events) {
    if (wait_events.size() > 0) {
        entry_factory::create<ze_event_wait_entry>(sched, wait_events);
        sched->add_barrier();
    }
}

void add_signal_event(ccl_sched* sched, ze_event_handle_t signal_event) {
    if (signal_event) {
        entry_factory::create<ze_event_signal_entry>(sched, signal_event);
        sched->add_barrier();
    }
}

ze_event_handle_t add_signal_event(ccl_sched* sched) {
    auto signal_event = sched->get_memory().event_manager->create();
    add_signal_event(sched, signal_event);
    return signal_event;
}

void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      ze_event_pool_handle_t ipc_pool,
                      size_t ipc_event_idx) {
    if (ipc_pool && global_data::env().enable_ze_barrier) {
        entry_factory::create<ze_barrier_entry>(sched, comm, ipc_pool, ipc_event_idx);
    }
    else {
        ccl_coll_entry_param barrier_param{};
        barrier_param.ctype = ccl_coll_barrier;
        barrier_param.comm = comm;

        /* TODO: optimize p2p based barrier */
        //barrier_param.hint_algo.barrier = ccl_coll_barrier_ring;

        coll_entry_helper::add_coll_entry<ccl_coll_barrier>(sched, barrier_param);
    }
    sched->add_barrier();
}

ze_event_handle_t add_comm_barrier(ccl_sched* sched,
                                   ccl_comm* comm,
                                   const std::vector<ze_event_handle_t>& wait_events,
                                   ze_event_pool_handle_t ipc_pool,
                                   size_t ipc_event_idx) {
    auto signal_event = sched->get_memory().event_manager->create();
    if (sched->get_memory().use_single_list) {
        add_wait_events(sched, wait_events);
        add_comm_barrier(sched, comm, ipc_pool, ipc_event_idx);
        add_signal_event(sched, signal_event);
    }
    else {
        add_comm_barrier(sched, comm, ipc_pool, ipc_event_idx);
        add_signal_event(sched, signal_event);
    }
    return signal_event;
}

void add_handle_exchange(ccl_sched* sched,
                         ccl_comm* comm,
                         const std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                         int skip_rank,
                         ze_event_pool_handle_t pool,
                         size_t event_idx) {
    if (sched->coll_attr.to_cache) {
        sched->set_entry_exec_mode(ccl_sched_entry_exec_once);
        entry_factory::create<ze_handle_exchange_entry>(sched, comm, in_buffers, skip_rank);
        sched->add_barrier();
        sched->set_entry_exec_mode(ccl_sched_entry_exec_regular);

        // TODO: no need barrier for the first iteration where ze_handle_exchange_entry exists
        add_comm_barrier(sched, comm, pool, event_idx);
    }
    else {
        entry_factory::create<ze_handle_exchange_entry>(sched, comm, in_buffers, skip_rank);
        sched->add_barrier();
    }
}

} // namespace ccl
