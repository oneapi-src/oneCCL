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
#include "common/comm/usm_visitor/usm_visitors.hpp"

namespace native {
struct device_group_context;
}

class single_device_communicator
        : public typed_single_device_base_communicator<single_device_communicator,
                                                       ccl::gpu_communicator_traits>,
          /*USM-converter visitors*/
          public allgather_usm_visitor<single_device_communicator>,
          public allreduce_usm_visitor<single_device_communicator>,
          public alltoall_usm_visitor<single_device_communicator>,
          public alltoallv_usm_visitor<single_device_communicator>,
          public broadcast_usm_visitor<single_device_communicator>,
          public reduce_usm_visitor<single_device_communicator>,
          public reduce_scatter_usm_visitor<single_device_communicator>,
          public sparse_allreduce_usm_visitor<single_device_communicator> {
public:
    using base_t = typed_single_device_base_communicator<single_device_communicator,
                                                         ccl::gpu_communicator_traits>;
    //TODO -S- Use event_impl for all communicator impl!
    using coll_request_t = ccl::event;
    using allgather_usm_visitor_base_t = allgather_usm_visitor<single_device_communicator>;
    using allreduce_usm_visitor_base_t = allreduce_usm_visitor<single_device_communicator>;
    using alltoall_usm_visitor_base_t = alltoall_usm_visitor<single_device_communicator>;
    using alltoallv_usm_visitor_base_t = alltoallv_usm_visitor<single_device_communicator>;
    using broadcast_usm_visitor_base_t = broadcast_usm_visitor<single_device_communicator>;
    using reduce_usm_visitor_base_t = reduce_usm_visitor<single_device_communicator>;
    using reduce_scatter_usm_visitor_base_t =
        reduce_scatter_usm_visitor<single_device_communicator>;
    using sparse_allreduce_usm_visitor_base_t =
        sparse_allreduce_usm_visitor<single_device_communicator>;

    single_device_communicator(ccl::unified_device_type&& device,
                               ccl::unified_device_context_type&& context,
                               size_t thread_idx,
                               size_t proces_idx,
                               const ccl::comm_split_attr& attr);

    ~single_device_communicator();

    std::shared_ptr<ccl::communicator_interface> split(const ccl::comm_split_attr& attr) override;

#ifdef MULTI_GPU_SUPPORT
    void visit(ccl::gpu_comm_attr& comm_attr) override;
#endif
    coll_request_t barrier(const ccl::stream::impl_value_t& op_stream,
                           const ccl::barrier_attr& attr,
                           const ccl::vector_class<ccl::event>& deps) override;

    DEVICE_COMM_IMPL_DECLARATION
    DEVICE_COMM_IMPL_CLASS_DECLARATION
    DEVICE_COMM_IMPL_SPARSE_DECLARATION
    DEVICE_COMM_IMPL_SPARSE_CLASS_DECLARATION

    void set_ccl_comm(std::shared_ptr<ccl_comm> imp);

    //TODO use visit() to set `context`
    void set_context(const ccl::unified_device_context_type::ccl_native_t& context);
    void set_context(const ccl::context& context);
private:
    std::shared_ptr<ccl_comm> comm_impl;
};

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
