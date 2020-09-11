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
#include "sched/gpu_sched.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "common/comm/l0/gpu_device_types.hpp"

class alignas(CACHELINE_SIZE) ccl_gpu_concurrent_sched : public ccl_sched_base, public ccl_request {
public:
    static constexpr const char* class_name() {
        return "gpu_concurrent_sched";
    }

    using ccl_gpu_concurrent_sched_ptr = std::unique_ptr<ccl_gpu_concurrent_sched>;
    static ccl_gpu_concurrent_sched_ptr create(size_t thread_count,
                                               const ccl_coll_param& param = ccl_coll_param());

    ccl_gpu_concurrent_sched(size_t expected_threads_count,
                             const ccl_coll_param& coll_param = ccl_coll_param());
    ccl_gpu_concurrent_sched(const ccl_gpu_concurrent_sched& src) = delete;
    ~ccl_gpu_concurrent_sched();

    std::shared_ptr<ccl_gpu_sched> create_gpu_sched(
        size_t thread_id,
        native::specific_indexed_device_storage& thread_devices,
        size_t expected_group_size,
        const ccl_coll_param& coll_param = ccl_coll_param());

    std::shared_ptr<ccl_gpu_sched> get_gpu_sched(size_t thread_id);

private:
    std::vector<std::shared_ptr<ccl_gpu_sched>> partial_scheds;
};
