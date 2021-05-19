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
#include "common/comm/single_device_communicator/single_device_communicator.hpp"
#include "common/comm/single_device_communicator/single_device_base_impl.hpp"

#include "oneapi/ccl/native_device_api/interop_utils.hpp"
#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"
#include "common/event/impls/scoped_event.hpp"

#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"

/* allgatherv */

template <class buffer_type>
ccl::event single_device_communicator::allgatherv_base_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl_coll_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return allgatherv_base_impl(send_buf,
                                send_count,
                                recv_buf,
                                recv_counts,
                                ccl::native_type_info<buffer_type>::dtype,
                                stream,
                                attr,
                                deps);
}

template <class buffer_type>
ccl::event single_device_communicator::allgatherv_impl(const buffer_type* send_buf,
                                                       size_t send_count,
                                                       buffer_type* recv_buf,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       const ccl::stream::impl_value_t& stream,
                                                       const ccl::allgatherv_attr& attr,
                                                       const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    return allgatherv_base_impl(
        send_buf, send_count, recv_buf, recv_counts, stream, internal_attr, deps);
}

template <class buffer_type>
ccl::event single_device_communicator::allgatherv_impl(const buffer_type* send_buf,
                                                       size_t send_count,
                                                       ccl::vector_class<buffer_type*>& recv_bufs,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       const ccl::stream::impl_value_t& stream,
                                                       const ccl::allgatherv_attr& attr,
                                                       const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.vector_buf = 1;
    return allgatherv_base_impl(send_buf,
                                send_count,
                                (buffer_type*)(recv_bufs.data()),
                                recv_counts,
                                stream,
                                internal_attr,
                                deps);
}

template <class buffer_type>
ccl::event single_device_communicator::allgatherv_impl(const buffer_type& send_buf,
                                                       size_t send_count,
                                                       buffer_type& recv_buf,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       const ccl::stream::impl_value_t& stream,
                                                       const ccl::allgatherv_attr& attr,
                                                       const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(&send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(&recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::dtype,
                                           attr,
                                           comm_impl.get(),
                                           stream.get(),
                                           deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
template <class buffer_type>
ccl::event single_device_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* allreduce */
template <class buffer_type>
ccl::event single_device_communicator::allreduce_impl(const buffer_type* send_buf,
                                                      buffer_type* recv_buf,
                                                      size_t count,
                                                      ccl::reduction reduction,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::allreduce_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return allreduce_impl(send_buf,
                          recv_buf,
                          count,
                          ccl::native_type_info<buffer_type>::dtype,
                          reduction,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event single_device_communicator::allreduce_impl(const buffer_type& send_buf,
                                                      buffer_type& recv_buf,
                                                      size_t count,
                                                      ccl::reduction reduction,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::allreduce_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(&send_buf),
                                          reinterpret_cast<void*>(&recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::dtype,
                                          reduction,
                                          attr,
                                          comm_impl.get(),
                                          stream.get(),
                                          deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* alltoall */
template <class buffer_type>
ccl::event single_device_communicator::alltoall_impl(const buffer_type* send_buf,
                                                     buffer_type* recv_buf,
                                                     size_t count,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::alltoall_attr& attr,
                                                     const ccl::vector_class<ccl::event>& deps) {
    return alltoall_impl(
        send_buf, recv_buf, count, ccl::native_type_info<buffer_type>::dtype, stream, attr, deps);
}

template <class buffer_type>
ccl::event single_device_communicator::alltoall_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<buffer_type*>& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::event single_device_communicator::alltoall_impl(const buffer_type& send_buf,
                                                     buffer_type& recv_buf,
                                                     size_t count,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::alltoall_attr& attr,
                                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(&send_buf),
                                         reinterpret_cast<void*>(&recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::dtype,
                                         attr,
                                         comm_impl.get(),
                                         stream.get(),
                                         deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event single_device_communicator::alltoall_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
template <class buffer_type>
ccl::event single_device_communicator::alltoallv_impl(const buffer_type* send_buf,
                                                      const ccl::vector_class<size_t>& send_counts,
                                                      buffer_type* recv_buf,
                                                      const ccl::vector_class<size_t>& recv_counts,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::alltoallv_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return alltoallv_impl(send_buf,
                          send_counts,
                          recv_buf,
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          stream,
                          attr,
                          deps);
}

template <class buffer_type>
ccl::event single_device_communicator::alltoallv_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::event single_device_communicator::alltoallv_impl(const buffer_type& send_buf,
                                                      const ccl::vector_class<size_t>& send_counts,
                                                      buffer_type& recv_buf,
                                                      const ccl::vector_class<size_t>& recv_counts,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::alltoallv_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(&send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(&recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<buffer_type>::dtype,
                                          attr,
                                          comm_impl.get(),
                                          stream.get(),
                                          deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event single_device_communicator::alltoallv_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
template <class buffer_type>
ccl::event single_device_communicator::broadcast_impl(buffer_type* buf,
                                                      size_t count,
                                                      int root,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::broadcast_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return broadcast_impl(
        buf, count, ccl::native_type_info<buffer_type>::dtype, root, stream, attr, deps);
}

template <class buffer_type>
ccl::event single_device_communicator::broadcast_impl(buffer_type& buf,
                                                      size_t count,
                                                      int root,
                                                      const ccl::stream::impl_value_t& stream,
                                                      const ccl::broadcast_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(&buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::dtype,
                                          root,
                                          attr,
                                          comm_impl.get(),
                                          stream.get(),
                                          deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce */
template <class buffer_type>
ccl::event single_device_communicator::reduce_impl(const buffer_type* send_buf,
                                                   buffer_type* recv_buf,
                                                   size_t count,
                                                   ccl::reduction reduction,
                                                   int root,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::reduce_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    return reduce_impl(send_buf,
                       recv_buf,
                       count,
                       ccl::native_type_info<buffer_type>::dtype,
                       reduction,
                       root,
                       stream,
                       attr,
                       deps);
}

template <class buffer_type>
ccl::event single_device_communicator::reduce_impl(const buffer_type& send_buf,
                                                   buffer_type& recv_buf,
                                                   size_t count,
                                                   ccl::reduction reduction,
                                                   int root,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::reduce_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(&send_buf),
                                       reinterpret_cast<void*>(&recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::dtype,
                                       reduction,
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       stream_ptr,
                                       deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce_scatter */
template <class buffer_type>
ccl::event single_device_communicator::reduce_scatter_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return reduce_scatter_impl(send_buf,
                               recv_buf,
                               recv_count,
                               ccl::native_type_info<buffer_type>::dtype,
                               reduction,
                               stream,
                               attr,
                               deps);
}

template <class buffer_type>
ccl::event single_device_communicator::reduce_scatter_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();
    ccl_request* req = ccl_reduce_scatter_impl(reinterpret_cast<const void*>(&send_buf),
                                               reinterpret_cast<void*>(&recv_buf),
                                               recv_count,
                                               ccl::native_type_info<buffer_type>::dtype,
                                               reduction,
                                               attr,
                                               comm_impl.get(),
                                               stream_ptr,
                                               deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* sparse_allreduce */
template <class index_buffer_type, class value_buffer_type>
ccl::event single_device_communicator::sparse_allreduce_impl(
    const index_buffer_type* send_ind_buf,
    size_t send_ind_count,
    const value_buffer_type* send_val_buf,
    size_t send_val_count,
    index_buffer_type* recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_type* recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return sparse_allreduce_impl(send_ind_buf,
                                 send_ind_count,
                                 send_val_buf,
                                 send_val_count,
                                 recv_ind_buf,
                                 recv_ind_count,
                                 recv_val_buf,
                                 recv_val_count,
                                 ccl::native_type_info<index_buffer_type>::dtype,
                                 ccl::native_type_info<value_buffer_type>::dtype,
                                 reduction,
                                 stream,
                                 attr,
                                 deps);
}

template <class index_buffer_container_type, class value_buffer_container_type>
ccl::event single_device_communicator::sparse_allreduce_impl(
    const index_buffer_container_type& send_ind_buf,
    size_t send_ind_count,
    const value_buffer_container_type& send_val_buf,
    size_t send_val_count,
    index_buffer_container_type& recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_container_type& recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req =
        ccl_sparse_allreduce_impl(reinterpret_cast<const void*>(&send_ind_buf),
                                  send_ind_count,
                                  reinterpret_cast<const void*>(&send_val_buf),
                                  send_val_count,
                                  reinterpret_cast<void*>(&recv_ind_buf),
                                  recv_ind_count,
                                  reinterpret_cast<void*>(&recv_val_buf),
                                  recv_val_count,
                                  ccl::native_type_info<index_buffer_container_type>::dtype,
                                  ccl::native_type_info<value_buffer_container_type>::dtype,
                                  reduction,
                                  attr,
                                  comm_impl.get(),
                                  stream_ptr,
                                  deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
