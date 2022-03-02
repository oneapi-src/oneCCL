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

#include "comm/comm.hpp"

#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"

#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"

/* allgatherv */
template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(const buffer_type* send_buf,
                                     size_t send_count,
                                     buffer_type* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::dtype,
                                           attr,
                                           this,
                                           get_stream_ptr(stream),
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(const buffer_type* send_buf,
                                     size_t send_count,
                                     ccl::vector_class<buffer_type*>& recv_bufs,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           (void*)(recv_bufs.data()),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::dtype,
                                           internal_attr,
                                           this,
                                           get_stream_ptr(stream),
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(const buffer_type& send_buf,
                                     size_t send_count,
                                     buffer_type& recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(&send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(&recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::dtype,
                                           internal_attr,
                                           this,
                                           get_stream_ptr(stream),
                                           deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_bufs,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(&send_buf),
                                           send_count,
                                           (void*)(recv_bufs.data()),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::dtype,
                                           internal_attr,
                                           this,
                                           get_stream_ptr(stream),
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* allreduce */
template <class buffer_type>
ccl::event ccl_comm::allreduce_impl(const buffer_type* send_buf,
                                    buffer_type* recv_buf,
                                    size_t count,
                                    ccl::reduction reduction,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allreduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(send_buf),
                                          reinterpret_cast<void*>(recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::dtype,
                                          reduction,
                                          attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::allreduce_impl(const buffer_type& send_buf,
                                    buffer_type& recv_buf,
                                    size_t count,
                                    ccl::reduction reduction,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allreduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(&send_buf),
                                          reinterpret_cast<void*>(&recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::dtype,
                                          reduction,
                                          internal_attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* alltoall */
template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(const buffer_type* send_buf,
                                   buffer_type* recv_buf,
                                   size_t count,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(send_buf),
                                         reinterpret_cast<void*>(recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::dtype,
                                         attr,
                                         this,
                                         get_stream_ptr(stream),
                                         deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(const ccl::vector_class<buffer_type*>& send_buf,
                                   const ccl::vector_class<buffer_type*>& recv_buf,
                                   size_t count,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(const buffer_type& send_buf,
                                   buffer_type& recv_buf,
                                   size_t count,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(&send_buf),
                                         reinterpret_cast<void*>(&recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::dtype,
                                         internal_attr,
                                         this,
                                         get_stream_ptr(stream),
                                         deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(
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
ccl::event ccl_comm::alltoallv_impl(const buffer_type* send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    buffer_type* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<buffer_type>::dtype,
                                          attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::alltoallv_impl(const ccl::vector_class<buffer_type*>& send_buf,
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
ccl::event ccl_comm::alltoallv_impl(const buffer_type& send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    buffer_type& recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(&send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(&recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<buffer_type>::dtype,
                                          internal_attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::alltoallv_impl(
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
ccl::event ccl_comm::broadcast_impl(buffer_type* buf,
                                    size_t count,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::dtype,
                                          root,
                                          attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::broadcast_impl(buffer_type& buf,
                                    size_t count,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(&buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::dtype,
                                          root,
                                          internal_attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce */
template <class buffer_type>
ccl::event ccl_comm::reduce_impl(const buffer_type* send_buf,
                                 buffer_type* recv_buf,
                                 size_t count,
                                 ccl::reduction reduction,
                                 int root,
                                 const ccl::stream::impl_value_t& stream,
                                 const ccl::reduce_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::dtype,
                                       reduction,
                                       root,
                                       attr,
                                       this,
                                       get_stream_ptr(stream),
                                       deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::reduce_impl(const buffer_type& send_buf,
                                 buffer_type& recv_buf,
                                 size_t count,
                                 ccl::reduction reduction,
                                 int root,
                                 const ccl::stream::impl_value_t& stream,
                                 const ccl::reduce_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(&send_buf),
                                       reinterpret_cast<void*>(&recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::dtype,
                                       reduction,
                                       root,
                                       internal_attr,
                                       this,
                                       get_stream_ptr(stream),
                                       deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce_scatter */
template <class buffer_type>
ccl::event ccl_comm::reduce_scatter_impl(const buffer_type* send_buf,
                                         buffer_type* recv_buf,
                                         size_t recv_count,
                                         ccl::reduction reduction,
                                         const ccl::stream::impl_value_t& stream,
                                         const ccl::reduce_scatter_attr& attr,
                                         const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_reduce_scatter_impl(reinterpret_cast<const void*>(send_buf),
                                               reinterpret_cast<void*>(recv_buf),
                                               recv_count,
                                               ccl::native_type_info<buffer_type>::dtype,
                                               reduction,
                                               attr,
                                               this,
                                               get_stream_ptr(stream),
                                               deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

template <class buffer_type>
ccl::event ccl_comm::reduce_scatter_impl(const buffer_type& send_buf,
                                         buffer_type& recv_buf,
                                         size_t recv_count,
                                         ccl::reduction reduction,
                                         const ccl::stream::impl_value_t& stream,
                                         const ccl::reduce_scatter_attr& attr,
                                         const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    ccl_request* req = ccl_reduce_scatter_impl(reinterpret_cast<const void*>(&send_buf),
                                               reinterpret_cast<void*>(&recv_buf),
                                               recv_count,
                                               ccl::native_type_info<buffer_type>::dtype,
                                               reduction,
                                               internal_attr,
                                               this,
                                               get_stream_ptr(stream),
                                               deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}
