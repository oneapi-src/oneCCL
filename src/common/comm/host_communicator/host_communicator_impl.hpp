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

#include "common/global/global.hpp"

#include "common/comm//host_communicator/host_communicator.hpp"
#include "common/request/host_request.hpp"
#include "coll/coll.hpp"
#include "common/stream/stream.hpp"

/* allgathev */
template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::allgatherv_impl(const buffer_type* send_buf,
                                                                     size_t send_count,
                                                                     buffer_type* recv_buf,
                                                                     const size_t* recv_counts,
                                                                     const ccl::coll_attr* attr,
                                                                     ccl::stream::impl_t& stream) {
    // c-api require null stream for host-stream for backward compatibility
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(recv_buf),
                                           recv_counts,
                                           ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                           attr,
                                           comm_impl.get(),
                                           stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::allgatherv_impl(const buffer_type& send_buf,
                                                                     size_t send_count,
                                                                     buffer_type& recv_buf,
                                                                     const size_t* recv_counts,
                                                                     const ccl::coll_attr* attr,
                                                                     ccl::stream::impl_t& stream) {
    // c-api require null stream for host-stream for backward compatibility
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(&send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(&recv_buf),
                                           recv_counts,
                                           ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                           attr,
                                           comm_impl.get(),
                                           stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* allreduce */
template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::allreduce_impl(const buffer_type* send_buf,
                                                                    buffer_type* recv_buf,
                                                                    size_t count,
                                                                    ccl::reduction reduction,
                                                                    const ccl::coll_attr* attr,
                                                                    ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(send_buf),
                                          reinterpret_cast<void*>(recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          static_cast<ccl_reduction_t>(reduction),
                                          attr,
                                          comm_impl.get(),
                                          stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::allreduce_impl(const buffer_type& send_buf,
                                                                    buffer_type& recv_buf,
                                                                    size_t count,
                                                                    ccl::reduction reduction,
                                                                    const ccl::coll_attr* attr,
                                                                    ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(&send_buf),
                                          reinterpret_cast<void*>(&recv_buf),
                                          count,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          static_cast<ccl_reduction_t>(reduction),
                                          attr,
                                          comm_impl.get(),
                                          stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoall */
template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::alltoall_impl(const buffer_type* send_buf,
                                                                   buffer_type* recv_buf,
                                                                   size_t count,
                                                                   const ccl::coll_attr* attr,
                                                                   ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(send_buf),
                                         reinterpret_cast<void*>(recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                         attr,
                                         comm_impl.get(),
                                         stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::alltoall_impl(const buffer_type& send_buf,
                                                                   buffer_type& recv_buf,
                                                                   size_t count,
                                                                   const ccl::coll_attr* attr,
                                                                   ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(&send_buf),
                                         reinterpret_cast<void*>(&recv_buf),
                                         count,
                                         ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                         attr,
                                         comm_impl.get(),
                                         stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoallv */
template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::alltoallv_impl(const buffer_type* send_buf,
                                                                    const size_t* send_counts,
                                                                    buffer_type* recv_buf,
                                                                    const size_t* recv_counts,
                                                                    const ccl::coll_attr* attr,
                                                                    ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(send_buf),
                                          send_counts,
                                          reinterpret_cast<void*>(recv_buf),
                                          recv_counts,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          attr,
                                          comm_impl.get(),
                                          stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::alltoallv_impl(const buffer_type& send_buf,
                                                                    const size_t* send_counts,
                                                                    buffer_type& recv_buf,
                                                                    const size_t* recv_counts,
                                                                    const ccl::coll_attr* attr,
                                                                    ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(&send_buf),
                                          send_counts,
                                          reinterpret_cast<void*>(&recv_buf),
                                          recv_counts,
                                          ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                          attr,
                                          comm_impl.get(),
                                          stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* bcast */
template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::bcast_impl(buffer_type* buf,
                                                                size_t count,
                                                                size_t root,
                                                                const ccl::coll_attr* attr,
                                                                ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_bcast_impl(reinterpret_cast<void*>(buf),
                                      count,
                                      ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                      root,
                                      attr,
                                      comm_impl.get(),
                                      stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::bcast_impl(buffer_type& buf,
                                                                size_t count,
                                                                size_t root,
                                                                const ccl::coll_attr* attr,
                                                                ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_bcast_impl(reinterpret_cast<void*>(&buf),
                                      count,
                                      ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                      root,
                                      attr,
                                      comm_impl.get(),
                                      stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce */
template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::reduce_impl(const buffer_type* send_buf,
                                                                 buffer_type* recv_buf,
                                                                 size_t count,
                                                                 ccl::reduction reduction,
                                                                 size_t root,
                                                                 const ccl::coll_attr* attr,
                                                                 ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                       static_cast<ccl_reduction_t>(reduction),
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class buffer_type>
ccl::communicator::coll_request_t host_communicator::reduce_impl(const buffer_type& send_buf,
                                                                 buffer_type& recv_buf,
                                                                 size_t count,
                                                                 ccl::reduction reduction,
                                                                 size_t root,
                                                                 const ccl::coll_attr* attr,
                                                                 ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(&send_buf),
                                       reinterpret_cast<void*>(&recv_buf),
                                       count,
                                       ccl::native_type_info<buffer_type>::ccl_datatype_value,
                                       static_cast<ccl_reduction_t>(reduction),
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class index_buffer_type, class value_buffer_type>
ccl::communicator::coll_request_t host_communicator::sparse_allreduce_impl(
    const index_buffer_type* send_ind_buf,
    size_t send_ind_count,
    const value_buffer_type* send_val_buf,
    size_t send_val_count,
    index_buffer_type* recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_type* recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

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
                                  static_cast<ccl_reduction_t>(reduction),
                                  attr,
                                  comm_impl.get(),
                                  stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class index_buffer_container_type, class value_buffer_container_type>
ccl::communicator::coll_request_t host_communicator::sparse_allreduce_impl(
    const index_buffer_container_type& send_ind_buf,
    size_t send_ind_count,
    const value_buffer_container_type& send_val_buf,
    size_t send_val_count,
    index_buffer_container_type& recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_container_type& recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host))
            ? stream.get()
            : nullptr;

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
        static_cast<ccl_reduction_t>(reduction),
        attr,
        comm_impl.get(),
        stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
