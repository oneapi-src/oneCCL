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
#include "sched/entry/reduce_local_entry.hpp"

#include "common/comm/l0/modules/kernel_utils.hpp"
#include "common/datatype/datatype.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/gpu/ze_primitives.hpp"
#include "sched/entry/gpu/ze_cache.hpp"
#include "sched/queue/queue.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

void reduce_local_entry::init() {
    if (ze_base_entry::is_initialized) {
        return;
    }

    LOG_DEBUG("initialization");

    ze_base_entry::init(init_mode::compute);

    ccl::global_data::get().ze_cache->get(context, device, "kernels.spv", &module);

    kernel_name =
        "reduce_local_inplace_kernel_" + to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    ccl::global_data::get().ze_cache->get(worker_idx, module, kernel_name, &kernel);
    LOG_DEBUG("get kernel: name: ", kernel_name);

    ze_group_size_t group_size;
    get_suggested_group_size(kernel, in_cnt, &group_size);
    LOG_DEBUG("suggested group size: ", to_string(group_size));

    get_suggested_group_count(group_size, in_cnt, &group_count);
    LOG_DEBUG("suggested group count: ", to_string(group_count));

    ZE_CALL(zeKernelSetGroupSize,
            (kernel, group_size.groupSizeX, group_size.groupSizeY, group_size.groupSizeZ));

    size_t bytes = in_cnt * dtype.size();
    in_buf_ptr = in_buf.get_ptr(bytes);
    inout_buf_ptr = inout_buf.get_ptr(bytes);
    ze_kernel_args_t kernel_args = { { sizeof(in_cnt), &in_cnt },
                                     { sizeof(in_buf_ptr), &in_buf_ptr },
                                     { sizeof(inout_buf_ptr), &inout_buf_ptr } };

    LOG_DEBUG("kernel ", kernel, " args:\n", to_string(kernel_args));
    set_kernel_args(kernel, kernel_args);

    ZE_CALL(zeCommandListAppendLaunchKernel,
            (ze_base_entry::comp_primitives.list,
             kernel,
             &group_count,
             ze_base_entry::entry_event,
             0,
             nullptr));
    ZE_CALL(zeCommandListClose, (ze_base_entry::comp_primitives.list));

    LOG_DEBUG("initialization complete");
}

void reduce_local_entry::update() {
    CCL_THROW_IF_NOT(use_device);

    ze_base_entry::update();
    if (status == ccl_sched_entry_status_complete && !sched->coll_attr.to_cache) {
        finalize();
    }
}

void reduce_local_entry::check_use_device() {
    use_device = false;
    ccl_stream* stream = (ccl_stream*)sched->coll_param.stream;
    if (fn || !stream)
        return;

    size_t bytes = in_cnt * dtype.size();
    sycl::queue* q = stream->get_native_stream(sched->queue->get_idx());
    CCL_THROW_IF_NOT(q, "null sycl queue");
    auto in_ptr_type = sycl::get_pointer_type(in_buf.get_ptr(bytes), q->get_context());
    auto inout_ptr_type = sycl::get_pointer_type(inout_buf.get_ptr(bytes), q->get_context());

    LOG_DEBUG("in_ptr_type: ",
              ccl::utils::usm_type_to_str(in_ptr_type),
              ", inout_ptr_type: ",
              ccl::utils::usm_type_to_str(inout_ptr_type),
              ", native_stream: ",
              stream->to_string(),
              ", in_count: ",
              in_cnt)

    if ((in_ptr_type == sycl::usm::alloc::device) && (inout_ptr_type == sycl::usm::alloc::device)) {
        use_device = true;
    }
}

void reduce_local_entry::start_on_device() {
    init();

    ze_base_entry::start();
    status = ccl_sched_entry_status_started;
}

void reduce_local_entry::finalize() {
    if (!ze_base_entry::is_initialized) {
        return;
    }

    LOG_DEBUG("finalization");

    // kernel cache
    ccl::global_data::get().ze_cache->push(worker_idx, module, kernel_name, kernel);

    ze_base_entry::finalize();

    LOG_DEBUG("finalization complete");
}
