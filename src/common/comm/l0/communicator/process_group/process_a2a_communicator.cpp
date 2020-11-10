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
#include "oneapi/ccl.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "common/comm/l0/communicator/process_group/process_a2a_communicator_impl.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"

using namespace ccl;

process_a2a_communicator::process_a2a_communicator(ccl::unified_device_type&& device,
                                                   ccl::unified_context_type&& ctx,
                                                   size_t thread_idx,
                                                   size_t process_idx,
                                                   const ccl::comm_split_attr& attr)
        : base_t(std::move(device), std::move(ctx), thread_idx, process_idx, /*comm_attr, */ attr) {
}

void process_a2a_communicator::visit(ccl::gpu_comm_attr& comm_attr) {
    ctx = comm_attr.get_process_context();

    //get rank & size
    auto topology = ctx->get_process_topology<base_t::topology_class()>(process_id, thread_id);
    this->initialize_comm_addr(get_device_path(), topology);

    this->set_comm_group_id(comm_attr.get_unique_id());
}

ccl::event process_a2a_communicator::barrier(const ccl::stream::impl_value_t& stream,
                                             const ccl::barrier_attr& attr,
                                             const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented yet");
}

/* allgatherv */
ccl::event process_a2a_communicator::allgatherv_impl(const void* send_buf,
                                                     size_t send_count,
                                                     void* recv_buf,
                                                     const ccl::vector_class<size_t>& recv_counts,
                                                     ccl::datatype dtype,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::allgatherv_attr& attr,
                                                     const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
ccl::event process_a2a_communicator::allgatherv_impl(const void* send_buf,
                                                     size_t send_count,
                                                     const ccl::vector_class<void*>& recv_bufs,
                                                     const ccl::vector_class<size_t>& recv_counts,
                                                     ccl::datatype dtype,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::allgatherv_attr& attr,

                                                     const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* allreduce */
ccl::event process_a2a_communicator::allreduce_impl(const void* send_buf,
                                                    void* recv_buf,
                                                    size_t count,
                                                    ccl::datatype dtype,
                                                    ccl::reduction reduction,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::allreduce_attr& attr,
                                                    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoall */
ccl::event process_a2a_communicator::alltoall_impl(const void* send_buf,
                                                   void* recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::alltoall_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
ccl::event process_a2a_communicator::alltoall_impl(const ccl::vector_class<void*>& send_buf,
                                                   const ccl::vector_class<void*>& recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::alltoall_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
ccl::event process_a2a_communicator::alltoallv_impl(const void* send_buf,
                                                    const ccl::vector_class<size_t>& send_counts,
                                                    void* recv_buf,
                                                    const ccl::vector_class<size_t>& recv_counts,
                                                    ccl::datatype dtype,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::alltoallv_attr& attr,
                                                    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
ccl::event process_a2a_communicator::alltoallv_impl(const ccl::vector_class<void*>& send_buf,
                                                    const ccl::vector_class<size_t>& send_counts,
                                                    ccl::vector_class<void*> recv_buf,
                                                    const ccl::vector_class<size_t>& recv_counts,
                                                    ccl::datatype dtype,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::alltoallv_attr& attr,

                                                    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
ccl::event process_a2a_communicator::broadcast_impl(void* buf,
                                                    size_t count,
                                                    ccl::datatype dtype,
                                                    int root,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::broadcast_attr& attr,
                                                    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* reduce */
ccl::event process_a2a_communicator::reduce_impl(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl::reduction reduction,
                                                 int root,
                                                 const ccl::stream::impl_value_t& stream,
                                                 const ccl::reduce_attr& attr,
                                                 const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* reduce_scatter */
ccl::event process_a2a_communicator::reduce_scatter_impl(
    const void* send_buf,
    void* recv_buf,
    size_t recv_count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* sparse_allreduce */
ccl::event process_a2a_communicator::sparse_allreduce_impl(
    const void* send_ind_buf,
    size_t send_ind_count,
    const void* send_val_buf,
    size_t send_val_count,
    void* recv_ind_buf,
    size_t recv_ind_count,
    void* recv_val_buf,
    size_t recv_val_count,
    ccl::datatype index_dtype,
    ccl::datatype value_dtype,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

COMM_INTERFACE_COLL_INSTANTIATION(process_a2a_communicator);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(process_a2a_communicator);
#endif /* CCL_ENABLE_SYCL */
