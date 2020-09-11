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
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
#include "common/comm/single_device_communicator/single_device_base.hpp"
#include "common/comm/comm.hpp"

namespace native {
struct device_group_context;
}

class single_device_communicator
        : public typed_single_device_base_communicator<single_device_communicator,
                                                       ccl::gpu_communicator_traits> {
public:
    using base_t = typed_single_device_base_communicator<single_device_communicator,
                                                         ccl::gpu_communicator_traits>;

    single_device_communicator(ccl::unified_device_type&& device,
                               size_t thread_idx,
                               size_t proces_idx,
                               const ccl::device_comm_split_attr& attr);
#ifdef MULTI_GPU_SUPPORT
    void visit(ccl::gpu_comm_attr& comm_attr) override;
#endif
    ccl::request_t barrier(ccl::stream::impl_value_t& op_stream,
                           const ccl::barrier_attr& attr,
                           const ccl::vector_class<ccl::event>& deps) override;

    DEVICE_COMM_IMPL_DECLARATION
    DEVICE_COMM_IMPL_CLASS_DECLARATION
    DEVICE_COMM_IMPL_SPARSE_DECLARATION
    DEVICE_COMM_IMPL_SPARSE_CLASS_DECLARATION

    void set_ccl_comm(std::shared_ptr<ccl_comm> imp);

private:
    std::shared_ptr<ccl_comm> comm_impl;
};

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
