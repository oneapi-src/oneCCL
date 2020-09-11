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

#include "common/comm/host_communicator/host_communicator.hpp"
#include "common/comm/host_communicator/host_communicator_defines.hpp"
#include "common/request/request.hpp"
#include "common/request/host_request.hpp"
#include "coll/coll.hpp"

namespace ccl {

/* allgatherv */
template <class BufferType>
ccl::request_t host_communicator::allgatherv_impl(const BufferType* send_buf,
                                                  size_t send_count,
                                                  BufferType* recv_buf,
                                                  const vector_class<size_t>& recv_counts,
                                                  const allgatherv_attr& attr) {
    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           reinterpret_cast<void*>(recv_buf),
                                           recv_counts.data(),
                                           ccl::native_type_info<BufferType>::ccl_datatype_value,
                                           attr,
                                           comm_impl.get(),
                                           nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class BufferType>
ccl::request_t host_communicator::allgatherv_impl(const BufferType* send_buf,
                                                  size_t send_count,
                                                  const vector_class<BufferType*>& recv_bufs,
                                                  const vector_class<size_t>& recv_counts,
                                                  const allgatherv_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* allreduce */
template <class BufferType, typename T>
ccl::request_t host_communicator::allreduce_impl(const BufferType* send_buf,
                                                 BufferType* recv_buf,
                                                 size_t count,
                                                 ccl::reduction reduction,
                                                 const allreduce_attr& attr) {
    ccl_request* req = ccl_allreduce_impl(reinterpret_cast<const void*>(send_buf),
                                          reinterpret_cast<void*>(recv_buf),
                                          count,
                                          ccl::native_type_info<BufferType>::ccl_datatype_value,
                                          reduction,
                                          attr,
                                          comm_impl.get(),
                                          nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoall */
template <class BufferType, typename T>
ccl::request_t host_communicator::alltoall_impl(const BufferType* send_buf,
                                                BufferType* recv_buf,
                                                size_t count,
                                                const alltoall_attr& attr) {
    ccl_request* req = ccl_alltoall_impl(reinterpret_cast<const void*>(send_buf),
                                         reinterpret_cast<void*>(recv_buf),
                                         count,
                                         ccl::native_type_info<BufferType>::ccl_datatype_value,
                                         attr,
                                         comm_impl.get(),
                                         nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class BufferType, typename T>
ccl::request_t host_communicator::alltoall_impl(const vector_class<BufferType*>& send_buf,
                                                const vector_class<BufferType*>& recv_buf,
                                                size_t count,
                                                const alltoall_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoallv */
template <class BufferType, typename T>
ccl::request_t host_communicator::alltoallv_impl(const BufferType* send_buf,
                                                 const vector_class<size_t>& send_counts,
                                                 BufferType* recv_buf,
                                                 const vector_class<size_t>& recv_counts,
                                                 const alltoallv_attr& attr) {
    ccl_request* req = ccl_alltoallv_impl(reinterpret_cast<const void*>(send_buf),
                                          send_counts.data(),
                                          reinterpret_cast<void*>(recv_buf),
                                          recv_counts.data(),
                                          ccl::native_type_info<BufferType>::ccl_datatype_value,
                                          attr,
                                          comm_impl.get(),
                                          nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

template <class BufferType, typename T>
ccl::request_t host_communicator::alltoallv_impl(const vector_class<BufferType*>& send_bufs,
                                                 const vector_class<size_t>& send_counts,
                                                 const vector_class<BufferType*>& recv_bufs,
                                                 const vector_class<size_t>& recv_counts,
                                                 const alltoallv_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* bcast */
template <class BufferType, typename T>
ccl::request_t host_communicator::broadcast_impl(BufferType* buf,
                                                 size_t count,
                                                 size_t root,
                                                 const broadcast_attr& attr) {
    ccl_request* req = ccl_broadcast_impl(reinterpret_cast<void*>(buf),
                                          count,
                                          ccl::native_type_info<BufferType>::ccl_datatype_value,
                                          root,
                                          attr,
                                          comm_impl.get(),
                                          nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce */
template <class BufferType, typename T>
ccl::request_t host_communicator::reduce_impl(const BufferType* send_buf,
                                              BufferType* recv_buf,
                                              size_t count,
                                              ccl::reduction reduction,
                                              size_t root,
                                              const reduce_attr& attr) {
    ccl_request* req = ccl_reduce_impl(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       count,
                                       ccl::native_type_info<BufferType>::ccl_datatype_value,
                                       reduction,
                                       root,
                                       attr,
                                       comm_impl.get(),
                                       nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce_scatter */
template <class BufferType, typename T>
ccl::request_t host_communicator::reduce_scatter_impl(const BufferType* send_buf,
                                                      BufferType* recv_buf,
                                                      size_t recv_count,
                                                      ccl::reduction reduction,
                                                      const reduce_scatter_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* sparse_allreduce */
template <class index_BufferType, class value_BufferType, typename T>
ccl::request_t host_communicator::sparse_allreduce_impl(const index_BufferType* send_ind_buf,
                                                        size_t send_ind_count,
                                                        const value_BufferType* send_val_buf,
                                                        size_t send_val_count,
                                                        index_BufferType* recv_ind_buf,
                                                        size_t recv_ind_count,
                                                        value_BufferType* recv_val_buf,
                                                        size_t recv_val_count,
                                                        ccl::reduction reduction,
                                                        const sparse_allreduce_attr& attr) {
    ccl_request* req =
        ccl_sparse_allreduce_impl((const void*)send_ind_buf,
                                  send_ind_count,
                                  (const void*)send_val_buf,
                                  send_val_count,
                                  (void*)recv_ind_buf,
                                  recv_ind_count,
                                  (void*)recv_val_buf,
                                  recv_val_count,
                                  ccl::native_type_info<index_BufferType>::ccl_datatype_value,
                                  ccl::native_type_info<value_BufferType>::ccl_datatype_value,
                                  reduction,
                                  attr,
                                  comm_impl.get(),
                                  nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

} // namespace ccl
