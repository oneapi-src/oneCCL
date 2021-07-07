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
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
#include "common/comm/single_device_communicator/single_device_communicator_impl.hpp"
#ifdef MULTI_GPU_SUPPORT
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"
#endif //MULTI_GPU_SUPPORT
using namespace ccl;

single_device_communicator::single_device_communicator(ccl::unified_device_type&& device,
                                                       ccl::unified_context_type&& context,
                                                       size_t thread_idx,
                                                       size_t process_idx,
                                                       const ccl::comm_split_attr& attr)
        : base_t(std::move(device),
                 std::move(context),
                 thread_idx,
                 process_idx /*, comm_attr*/,
                 attr) {}

single_device_communicator::~single_device_communicator() {}

std::shared_ptr<ccl::communicator_interface> single_device_communicator::split(
    const ccl::comm_split_attr& attr) {
    // TODO
    throw ccl::exception(std::string(__FUNCTION__) + " - 'is not implemented");
    return {};
}

void single_device_communicator::set_ccl_comm(std::shared_ptr<ccl_comm> impl) {
    CCL_ASSERT(!comm_impl, "comm_impl must be nullptr before first udage");
    comm_impl = impl;

    comm_rank = comm_impl->rank();
    comm_size = comm_impl->size();
}

//TODO use visit() to set `context`
void single_device_communicator::set_context(
    const ccl::unified_context_type::ccl_native_t& in_context) {
    context = in_context;
}
void single_device_communicator::set_context(const ccl::context& in_context) {
    context = in_context.get_native();
}

#ifdef MULTI_GPU_SUPPORT
void single_device_communicator::visit(ccl::gpu_comm_attr& comm_attr) {
    auto process_ctx = comm_attr.get_process_context();
    auto thread_ctx = process_ctx->get_thread_context(process_id);
    auto device_ctx = thread_ctx->get_device_group_ctx(thread_id);

    //get rank & size

    /*  this->initialize_comm_addr(get_device_path(),
                               ctx->get_group_topology<base_t::topology_class()>());
*/
    this->set_comm_group_id(comm_attr.get_unique_id());
}
#endif
ccl::event single_device_communicator::barrier(const ccl::stream::impl_value_t& op_stream,
                                               const ccl::barrier_attr& attr,
                                               const ccl::vector_class<ccl::event>& deps) {
    // TODO what exactly we need to do with 'attr' here?

    ccl_barrier_impl(comm_impl.get(), op_stream.get(), deps);

    // TODO what exactly we need to return here? ccl_barrier_impl() is void func
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(nullptr));
}

/* allgatherv */
ccl::event single_device_communicator::allgatherv_base_impl(
    const void* send_buf,
    size_t send_count,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    const ccl::stream::impl_value_t& stream,
    const ccl_coll_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(std::unique_ptr<ccl::event_impl>(
        new ccl::host_event_impl(ccl_allgatherv_impl(send_buf,
                                                     send_count,
                                                     recv_buf,
                                                     recv_counts.data(),
                                                     dtype,
                                                     attr,
                                                     comm_impl.get(),
                                                     stream.get(),
                                                     deps))));
}

ccl::event single_device_communicator::allgatherv_impl(const void* send_buf,
                                                       size_t send_count,
                                                       void* recv_buf,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       ccl::datatype dtype,
                                                       const ccl::stream::impl_value_t& stream,
                                                       const ccl::allgatherv_attr& attr,
                                                       const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    return allgatherv_base_impl(
        send_buf, send_count, recv_buf, recv_counts, dtype, stream, internal_attr, deps);
}

ccl::event single_device_communicator::allgatherv_impl(const void* send_buf,
                                                       size_t send_count,
                                                       const ccl::vector_class<void*>& recv_bufs,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       ccl::datatype dtype,
                                                       const ccl::stream::impl_value_t& stream,
                                                       const ccl::allgatherv_attr& attr,
                                                       const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.vector_buf = 1;
    return allgatherv_base_impl(send_buf,
                                send_count,
                                (void*)(recv_bufs.data()),
                                recv_counts,
                                dtype,
                                stream,
                                internal_attr,
                                deps);
}

/* allreduce */
ccl::event single_device_communicator::allreduce_impl(const void* send_buf,
                                                      void* recv_buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      ccl::reduction reduction,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::allreduce_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(ccl_allreduce_impl(
        send_buf, recv_buf, count, dtype, reduction, attr, comm_impl.get(), stream.get(), deps))));
}

/* alltoall */
ccl::event single_device_communicator::alltoall_impl(const void* send_buf,
                                                     void* recv_buf,
                                                     size_t count,
                                                     ccl::datatype dtype,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::alltoall_attr& attr,
                                                     const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(ccl_alltoall_impl(
        send_buf, recv_buf, count, dtype, attr, comm_impl.get(), stream.get(), deps))));
}

ccl::event single_device_communicator::alltoall_impl(const ccl::vector_class<void*>& send_buf,
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
ccl::event single_device_communicator::alltoallv_impl(const void* send_buf,
                                                      const ccl::vector_class<size_t>& send_counts,
                                                      void* recv_buf,
                                                      const ccl::vector_class<size_t>& recv_counts,
                                                      ccl::datatype dtype,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::alltoallv_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(std::unique_ptr<ccl::event_impl>(
        new ccl::host_event_impl(ccl_alltoallv_impl(send_buf,
                                                    send_counts.data(),
                                                    recv_buf,
                                                    recv_counts.data(),
                                                    dtype,
                                                    attr,
                                                    comm_impl.get(),
                                                    stream.get(),
                                                    deps))));
}
ccl::event single_device_communicator::alltoallv_impl(const ccl::vector_class<void*>& send_buf,
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
ccl::event single_device_communicator::broadcast_impl(void* buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      int root,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::broadcast_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(
        ccl_broadcast_impl(buf, count, dtype, root, attr, comm_impl.get(), stream.get(), deps))));
}

/* reduce */
ccl::event single_device_communicator::reduce_impl(const void* send_buf,
                                                   void* recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   ccl::reduction reduction,
                                                   int root,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::reduce_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(
        std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(ccl_reduce_impl(send_buf,
                                                                                  recv_buf,
                                                                                  count,
                                                                                  dtype,
                                                                                  reduction,
                                                                                  root,
                                                                                  attr,
                                                                                  comm_impl.get(),
                                                                                  stream.get(),
                                                                                  deps))));
}

/* reduce_scatter */
ccl::event single_device_communicator::reduce_scatter_impl(
    const void* send_buf,
    void* recv_buf,
    size_t recv_count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return ccl::event(std::unique_ptr<ccl::event_impl>(
        new ccl::host_event_impl(ccl_reduce_scatter_impl(send_buf,
                                                         recv_buf,
                                                         recv_count,
                                                         dtype,
                                                         reduction,
                                                         attr,
                                                         comm_impl.get(),
                                                         stream.get(),
                                                         deps))));
}

/* sparse_allreduce */
ccl::event single_device_communicator::sparse_allreduce_impl(
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
    return ccl::event(std::unique_ptr<ccl::event_impl>(
        new ccl::host_event_impl(ccl_sparse_allreduce_impl(send_ind_buf,
                                                           send_ind_count,
                                                           send_val_buf,
                                                           send_val_count,
                                                           recv_ind_buf,
                                                           recv_ind_count,
                                                           recv_val_buf,
                                                           recv_val_count,
                                                           index_dtype,
                                                           value_dtype,
                                                           reduction,
                                                           attr,
                                                           comm_impl.get(),
                                                           stream.get(),
                                                           deps))));
}

COMM_INTERFACE_COLL_INSTANTIATION(single_device_communicator);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(single_device_communicator);
#endif /* CCL_ENABLE_SYCL */

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
