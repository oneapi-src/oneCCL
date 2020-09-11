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
#include <unistd.h>

#include "sched/gpu_sched.hpp"

ccl_gpu_sched::ccl_gpu_sched(native::specific_indexed_device_storage& devices,
                             size_t group_size,
                             const ccl_coll_param& coll_param /* = ccl_coll_param()*/)
        : ccl_sched(coll_param, nullptr),
          ccl_request(),
          expected_group_size(group_size),
          group_comm_devices(devices) {
    //preallocation
    entry_fences.reserve(expected_group_size);
    //WHY deque??? entries.reserve(expected_group_size);
}

void ccl_gpu_sched::complete() {}

ze_fence_handle_t* entry_request = nullptr;
void ccl_gpu_sched::set_fence(ze_fence_handle_t entry_fence) //TODO temporary
{
    assert(entry_fence);
    entry_fences.push_back(entry_fence);
}

bool ccl_gpu_sched::wait(size_t nanosec) {
    bool ready = true;
    for (auto fence : entry_fences) {
        ze_result_t ret = zeFenceHostSynchronize(fence, nanosec);
        if (ret != ZE_RESULT_SUCCESS) {
            if (ret == ZE_RESULT_NOT_READY or ret == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
                ready = false;
                break;
            }
            throw std::runtime_error(std::string("zeFenceHostSynchronize failed: ") +
                                     native::to_string(ret));
        }
    }
    return ready;
}
