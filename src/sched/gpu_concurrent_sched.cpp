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
#include <functional>
#include "sched/gpu_concurrent_sched.hpp"

ccl_gpu_concurrent_sched::ccl_gpu_concurrent_sched(
    size_t expected_threads_count,
    const ccl_coll_param& coll_param /* = ccl_coll_param()*/)
        : ccl_sched_base(coll_param),
          ccl_request(),
          partial_scheds(expected_threads_count) {
#ifdef ENABLE_DEBUG
    set_dump_callback([this](std::ostream& out) {
        out << "exeecuted: " << this << std::endl; //dump(out);
    });
#endif
}

ccl_gpu_concurrent_sched::~ccl_gpu_concurrent_sched() {}

std::shared_ptr<ccl_gpu_sched> ccl_gpu_concurrent_sched::get_gpu_sched(size_t thread_id) {
    if (thread_id >= partial_scheds.size()) {
        LOG_ERROR(
            "Requested thread id: ", thread_id, " is more than expected: ", partial_scheds.size());
    }
    return partial_scheds[thread_id];
}

std::shared_ptr<ccl_gpu_sched> ccl_gpu_concurrent_sched::create_gpu_sched(
    size_t thread_id,
    native::specific_indexed_device_storage& thread_devices,
    size_t expected_group_size,
    const ccl_coll_param& coll_param /* = ccl_coll_param()*/) {
    if (thread_id >= partial_scheds.size()) {
        LOG_ERROR(
            "Requested thread id: ", thread_id, " is more than expected: ", partial_scheds.size());
    }

    //it is safe to create thread ccl_gpu_sched in specific slot (determined by thread_id) in preallocaed partial scheds vector
    auto sched = std::make_shared<ccl_gpu_sched>(thread_devices, expected_group_size, coll_param);
    partial_scheds[thread_id] = sched;
    return sched;
}

ccl_gpu_concurrent_sched::ccl_gpu_concurrent_sched_ptr ccl_gpu_concurrent_sched::create(
    size_t thread_count,
    const ccl_coll_param& param /* = ccl_coll_param()*/) {
    /*TODO use cache*/
    return ccl_gpu_concurrent_sched_ptr(new ccl_gpu_concurrent_sched(thread_count, param));
}
