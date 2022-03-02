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
#pragma once

#include "common/global/global.hpp"
#include "sched/entry/coll/coll_entry_param.hpp"
#include "sched/entry/copy/copy_helper.hpp"
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "sched/entry/ze/ze_handle_exchange_entry.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

namespace ccl {

void add_coll_entry(ccl_sched* sched, const ccl_coll_entry_param& param);

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)

void add_wait_events(ccl_sched* sched, const std::vector<ze_event_handle_t>& wait_events);
void add_signal_event(ccl_sched* sched, ze_event_handle_t signal_event);
ze_event_handle_t add_signal_event(ccl_sched* sched);

void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      ze_event_pool_handle_t ipc_pool = {},
                      size_t ipc_event_idx = 0);

ze_event_handle_t add_comm_barrier(ccl_sched* sched,
                                   ccl_comm* comm,
                                   const std::vector<ze_event_handle_t>& wait_events,
                                   ze_event_pool_handle_t ipc_pool = {},
                                   size_t ipc_event_idx = 0);

void add_handle_exchange(ccl_sched* sched,
                         ccl_comm* comm,
                         const std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                         int skip_rank = ccl_comm::invalid_rank,
                         ze_event_pool_handle_t pool = nullptr,
                         size_t event_idx = 0);

void add_coll(ccl_sched* sched,
              const ccl_coll_entry_param& param,
              std::vector<ze_event_handle_t>& wait_events);

void add_scaleout(ccl_sched* sched,
                  const ccl_coll_entry_param& coll_param,
                  const bool is_single_node,
                  std::vector<ze_event_handle_t>& wait_events,
                  const copy_attr& h2d_copy_attr = copy_attr(copy_direction::h2d),
                  ccl_comm* global_comm = nullptr,
                  ccl_buffer global_recv = {},
                  int global_root = 0);

#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

} // namespace ccl
