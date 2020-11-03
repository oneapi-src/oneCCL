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
#include "sched/queue/queue.hpp"
#include "sched/sched.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "common/comm/l0/gpu_device_types.hpp"

class ccl_gpu_sched;

class alignas(CACHELINE_SIZE) ccl_gpu_sched : public ccl_sched, public ccl_request {
public:
    static constexpr const char* class_name() {
        return "gpu_worker_sched";
    }

    ccl_gpu_sched(native::specific_indexed_device_storage& devices,
                  size_t expected_group_size,
                  const ccl_coll_param& coll_param = ccl_coll_param());

    ccl_gpu_sched(const ccl_sched& other) = delete;
    ccl_gpu_sched& operator=(const ccl_gpu_sched& other) = delete;

    ~ccl_gpu_sched() = default;

    void complete() override;
    bool wait(size_t nanosec);
    //TODO temporary
    void set_fence(ze_fence_handle_t entry_fence);

    template <class device_t>
    native::indexed_device_container<device_t>& get_devices() {
        return std::get<device_t::type_idx()>(group_comm_devices);
    }

    size_t get_group_size() const {
        return expected_group_size;
    }

private:
    size_t expected_group_size;
    std::vector<ze_fence_handle_t> entry_fences;
    native::specific_indexed_device_storage& group_comm_devices;
};
