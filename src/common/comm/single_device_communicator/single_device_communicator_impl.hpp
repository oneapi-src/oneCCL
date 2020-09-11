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

#include "common/request/request.hpp"
#include "common/request/host_request.hpp"
#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"

/* allgatherv */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    // c-api require null stream for host-stream for backward compatibility
    /* const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;
*/
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                           attr,
                                           comm_impl.get(),
                                           stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
template <class buffer_type>
ccl::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::coll_request_t single_device_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(&send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(&recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                           attr,
                                           comm_impl.get(),
                                           stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
template <class buffer_type>
ccl::request_t single_device_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* allreduce */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::allreduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    using namespace native;

    static constexpr ccl::device_group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::ccl_error(std::string(
            "Single device communicator for group_id: " + ::to_string(group_id) +
            ", class_id: " + ::to_string(class_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    size_t comm_rank = rank();
    LOG_DEBUG("communicator for device idx: ", get_device_path(), ", rank idx: ", comm_rank);

    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(send_buf),
                                          reinterpret_cast<void*>(recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          reduction,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::coll_request_t single_device_communicator::allreduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(&send_buf),
                                          reinterpret_cast<void*>(&recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          reduction,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoall */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::alltoall_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(send_buf),
                                         reinterpret_cast<void*>(recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                         attr,
                                         comm_impl.get(),
                                         stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
template <class buffer_type>
ccl::request_t single_device_communicator::alltoall_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<buffer_type*>& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::coll_request_t single_device_communicator::alltoall_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(&send_buf),
                                         reinterpret_cast<void*>(&recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                         attr,
                                         comm_impl.get(),
                                         stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
template <class buffer_type>
ccl::request_t single_device_communicator::alltoall_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::alltoallv_impl(
    const buffer_type* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
template <class buffer_type>
ccl::coll_request_t single_device_communicator::alltoallv_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::coll_request_t single_device_communicator::alltoallv_impl(
    const buffer_type& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    static_assert(
        !std::is_same<buffer_type,
                      ccl::vector_class<
                          ccl::reference_wrapper_class<cl::sycl::buffer<unsigned long, 1>>>>::value,
        "???");
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(&send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(&recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
template <class buffer_type>
ccl::coll_request_t single_device_communicator::alltoallv_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::broadcast_impl(
    buffer_type* buf,
    size_t count,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          root,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::coll_request_t single_device_communicator::broadcast_impl(
    buffer_type& buf,
    size_t count,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(&buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          root,
                                          attr,
                                          comm_impl.get(),
                                          stream.get());
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::reduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                       reduction,
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::coll_request_t single_device_communicator::reduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(&send_buf),
                                       reinterpret_cast<void*>(&recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                       reduction,
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
/* reduce_scatter */
template <class buffer_type>
ccl::coll_request_t single_device_communicator::reduce_scatter_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
template <class buffer_type>
ccl::coll_request_t single_device_communicator::reduce_scatter_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* sparse_allreduce */
template <class index_buffer_type, class value_buffer_type>
ccl::coll_request_t single_device_communicator::sparse_allreduce_impl(
    const index_buffer_type* send_ind_buf,
    size_t send_ind_count,
    const value_buffer_type* send_val_buf,
    size_t send_val_count,
    index_buffer_type* recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_type* recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req =
        ccl_sparse_allreduce_impl((const void*)send_ind_buf,
                                  send_ind_count,
                                  (const void*)send_val_buf,
                                  send_val_count,
                                  (void*)recv_ind_buf,
                                  recv_ind_count,
                                  (void*)recv_val_buf,
                                  recv_val_count,
                                  ccl::native_type_info<index_buffer_type>::ccl_datatype_value,
                                  ccl::native_type_info<value_buffer_type>::ccl_datatype_value,
                                  reduction,
                                  attr,
                                  comm_impl.get(),
                                  stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class index_buffer_container_type, class value_buffer_container_type>
ccl::coll_request_t single_device_communicator::sparse_allreduce_impl(
    const index_buffer_container_type& send_ind_buf,
    size_t send_ind_count,
    const value_buffer_container_type& send_val_buf,
    size_t send_val_count,
    index_buffer_container_type& recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_container_type& recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    const ccl_stream* stream_ptr = stream.get();

    ccl_request* req = ccl_sparse_allreduce_impl(
        reinterpret_cast<const void*>(&send_ind_buf),
        send_ind_count,
        reinterpret_cast<const void*>(&send_val_buf),
        send_val_count,
        reinterpret_cast<void*>(&recv_ind_buf),
        recv_ind_count,
        reinterpret_cast<void*>(&recv_val_buf),
        recv_val_count,
        ccl::native_type_info<index_buffer_container_type>::ccl_datatype_value,
        ccl::native_type_info<value_buffer_container_type>::ccl_datatype_value,
        reduction,
        attr,
        comm_impl.get(),
        stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
