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
#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "thread_group_scheduler.hpp"
#include "common/comm/l0/context/device_storage.hpp"

namespace native {
thread_group_context::~thread_group_context() {}

bool thread_group_context::sync_barrier(const ccl::device_indices_t& device_indices_t,
                                        ccl::context_comm_addr& comm_addr,
                                        device_storage& devices) {
    //check on group creation final condition
    device_group_ctx_ptr group_ctx{};
    if (false == thread_device_group_ctx.insert({ comm_addr.thread_idx, group_ctx }).second) {
        LOG_ERROR("cannot register devices group ctx for thread idx: ", comm_addr.thread_idx);
        abort();
    }

    LOG_DEBUG("Thread ", comm_addr.to_string(), " reached thread group communicator barrier");

    if (thread_device_group_ctx.size() != comm_addr.thread_count) {
        // not all threads are registered yet - wait for all
        LOG_DEBUG("Thread ", comm_addr.to_string(), " waits on barrier");
        return false; //slave thread
    }

    //Current thread finalize communicator creation
    LOG_INFO("Thread ", comm_addr.to_string(), " starts hardware topologies creation");
    LOG_DEBUG("Final thread ", comm_addr.to_string(), " ready to communicate");

    return true; //master thread
}

void thread_group_context::aggregate_device_indices(size_t thread_id,
                                                    const ccl::device_indices_t& new_indices) {}

const ccl::process_device_indices_t& thread_group_context::get_thread_group_device_indices() const {
    return per_thread_indices;
}

const ccl::device_indices_t& thread_group_context::get_device_group_indices(
    size_t thread_id) const {
    auto it = per_thread_indices.find(thread_id);
    if (it == per_thread_indices.end()) {
        LOG_ERROR("Cannot find device group for thread: ", thread_id, ". Empty indices");
        static const ccl::device_indices_t empty;
        return empty;
    }
    return it->second;
}

thread_group_context::device_group_ctx_ptr thread_group_context::get_device_group_ctx(
    size_t thread_id) {
    auto it = thread_device_group_ctx.find(thread_id);
    if (it == thread_device_group_ctx.end()) {
        LOG_ERROR("Cannot find device group for thread: ", thread_id, ". Empty context");
        return {};
    }
    return it->second;
}
} // namespace native
