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

#include "common/comm/comm.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_request.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

namespace ccl {

class kvs_interface;

class host_communicator {
public:
    size_t rank() const;
    size_t size() const;

    ccl::unique_ptr_class<host_communicator> split(const comm_split_attr& attr);

    /* allgatherv */
    ccl::request_t allgatherv_impl(const void* send_buf,
                                   size_t send_count,
                                   void* recv_buf,
                                   const vector_class<size_t>& recv_counts,
                                   ccl::datatype dtype,
                                   const allgatherv_attr& attr = default_allgatherv_attr);

    ccl::request_t allgatherv_impl(const void* send_buf,
                                   size_t send_count,
                                   const vector_class<void*>& recv_bufs,
                                   const vector_class<size_t>& recv_counts,
                                   ccl::datatype dtype,
                                   const allgatherv_attr& attr = default_allgatherv_attr);

    template <class BufferType>
    ccl::request_t allgatherv_impl(const BufferType* send_buf,
                                   size_t send_count,
                                   BufferType* recv_buf,
                                   const vector_class<size_t>& recv_counts,
                                   const allgatherv_attr& attr = default_allgatherv_attr);

    template <class BufferType>
    ccl::request_t allgatherv_impl(const BufferType* send_buf,
                                   size_t send_count,
                                   const vector_class<BufferType*>& recv_bufs,
                                   const vector_class<size_t>& recv_counts,
                                   const allgatherv_attr& attr = default_allgatherv_attr);

    /* allreduce */
    ccl::request_t allreduce_impl(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction rtype,
                                  const allreduce_attr& attr = default_allreduce_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t allreduce_impl(const BufferType* send_buf,
                                  BufferType* recv_buf,
                                  size_t count,
                                  ccl::reduction rtype,
                                  const allreduce_attr& attr = default_allreduce_attr);

    /* alltoall */
    ccl::request_t alltoall_impl(const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 const alltoall_attr& attr = default_alltoall_attr);

    ccl::request_t alltoall_impl(const vector_class<void*>& send_buf,
                                 const vector_class<void*>& recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 const alltoall_attr& attr = default_alltoall_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t alltoall_impl(const BufferType* send_buf,
                                 BufferType* recv_buf,
                                 size_t count,
                                 const alltoall_attr& attr = default_alltoall_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t alltoall_impl(const vector_class<BufferType*>& send_buf,
                                 const vector_class<BufferType*>& recv_buf,
                                 size_t count,
                                 const alltoall_attr& attr = default_alltoall_attr);

    /* alltoallv */
    ccl::request_t alltoallv_impl(const void* send_buf,
                                  const vector_class<size_t>& send_counts,
                                  void* recv_buf,
                                  const vector_class<size_t>& recv_counts,
                                  ccl::datatype dtype,
                                  const alltoallv_attr& attr = default_alltoallv_attr);

    ccl::request_t alltoallv_impl(const vector_class<void*>& send_bufs,
                                  const vector_class<size_t>& send_counts,
                                  const vector_class<void*>& recv_bufs,
                                  const vector_class<size_t>& recv_counts,
                                  ccl::datatype dtype,
                                  const alltoallv_attr& attr = default_alltoallv_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t alltoallv_impl(const BufferType* send_buf,
                                  const vector_class<size_t>& send_counts,
                                  BufferType* recv_buf,
                                  const vector_class<size_t>& recv_counts,
                                  const alltoallv_attr& attr = default_alltoallv_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t alltoallv_impl(const vector_class<BufferType*>& send_bufs,
                                  const vector_class<size_t>& send_counts,
                                  const vector_class<BufferType*>& recv_bufs,
                                  const vector_class<size_t>& recv_counts,
                                  const alltoallv_attr& attr = default_alltoallv_attr);

    /* barrier */
    ccl::request_t barrier_impl(const barrier_attr& attr = default_barrier_attr);

    /* bcast */
    ccl::request_t broadcast_impl(void* buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  size_t root,
                                  const broadcast_attr& attr = default_broadcast_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t broadcast_impl(BufferType* buf,
                                  size_t count,
                                  size_t root,
                                  const broadcast_attr& attr = default_broadcast_attr);

    /* reduce */
    ccl::request_t reduce_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl::datatype dtype,
                               ccl::reduction rtype,
                               size_t root,
                               const reduce_attr& attr = default_reduce_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t reduce_impl(const BufferType* send_buf,
                               BufferType* recv_buf,
                               size_t count,
                               ccl::reduction rtype,
                               size_t root,
                               const reduce_attr& attr = default_reduce_attr);

    /* reduce_scatter */
    ccl::request_t reduce_scatter_impl(
        const void* send_buf,
        void* recv_buf,
        size_t recv_count,
        ccl::datatype dtype,
        ccl::reduction rtype,
        const reduce_scatter_attr& attr = default_reduce_scatter_attr);

    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>()>::type>
    ccl::request_t reduce_scatter_impl(
        const BufferType* send_buf,
        BufferType* recv_buf,
        size_t recv_count,
        ccl::reduction rtype,
        const reduce_scatter_attr& attr = default_reduce_scatter_attr);

    /* sparse_allreduce */
    ccl::request_t sparse_allreduce_impl(
        const void* send_ind_buf,
        size_t send_ind_count,
        const void* send_val_buf,
        size_t send_val_count,
        void* recv_ind_buf,
        size_t recv_ind_count,
        void* recv_val_buf,
        size_t recv_val_count,
        ccl::datatype ind_dtype,
        ccl::datatype val_dtype,
        ccl::reduction rtype,
        const sparse_allreduce_attr& attr = default_sparse_allreduce_attr);

    template <
        class index_BufferType,
        class value_BufferType,
        class = typename std::enable_if<ccl::is_native_type_supported<value_BufferType>()>::type>
    ccl::request_t sparse_allreduce_impl(
        const index_BufferType* send_ind_buf,
        size_t send_ind_count,
        const value_BufferType* send_val_buf,
        size_t send_val_count,
        index_BufferType* recv_ind_buf,
        size_t recv_ind_count,
        value_BufferType* recv_val_buf,
        size_t recv_val_count,
        ccl::reduction rtype,
        const sparse_allreduce_attr& attr = default_sparse_allreduce_attr);

    host_communicator();
    host_communicator(size_t size, shared_ptr_class<kvs_interface> kvs);
    host_communicator(size_t size, size_t rank, shared_ptr_class<kvs_interface> kvs);
    host_communicator(std::shared_ptr<ccl_comm> impl);
    host_communicator(host_communicator& src) = delete;
    host_communicator(host_communicator&& src) = default;
    host_communicator& operator=(host_communicator& src) = delete;
    host_communicator& operator=(host_communicator&& src) = default;
    ~host_communicator() = default;

private:
    friend struct group_context;
    std::shared_ptr<ccl_comm> comm_impl;
    ccl::comm_split_attr comm_attr;
    size_t comm_rank;
    size_t comm_size;
}; // class host_communicator

} // namespace ccl
