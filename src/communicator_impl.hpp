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

#include "common/log/log.hpp"
#include "oneapi/ccl/ccl_request.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"
#include "oneapi/ccl/ccl_communicator.hpp"
#include "common/comm/host_communicator/host_communicator_impl.hpp"

namespace ccl {

CCL_API communicator::communicator(communicator&& other) : base_t(std::move(other)) {}

CCL_API communicator::communicator(impl_value_t&& impl) : base_t(std::move(impl)) {}

CCL_API communicator::~communicator() noexcept {}

CCL_API communicator& communicator::operator=(communicator&& other) {
    if (other.get_impl() != this->get_impl()) {
        other.get_impl().swap(this->get_impl());
        other.get_impl().reset();
    }
    return *this;
}

CCL_API size_t communicator::rank() const {
    return get_impl()->rank();
}

CCL_API size_t communicator::size() const {
    return get_impl()->size();
}

CCL_API communicator communicator::split(const comm_split_attr& attr) {
    auto impl = get_impl()->split(attr);
    return communicator(std::move(impl));
}

/* TODO temporary function for UT compilation: would be part of ccl::environment in final*/
/**
 * Creates a new host communicator with externally provided size, rank and kvs.
 * Implementation is platform specific and non portable.
 * @return host communicator
 */
communicator communicator::create_communicator() {
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    LOG_TRACE("Create host communicator");

    return communicator(std::unique_ptr<communicator::impl_t>(new communicator::impl_t()));
}

/* TODO temporary function for UT compilation: would be part of ccl::environment in final*/
/**
 * Creates a new host communicator with user supplied size and kvs.
 * Rank will be assigned automatically.
 * @param size user-supplied total number of ranks
 * @param kvs key-value store for ranks wire-up
 * @return host communicator
 */
communicator communicator::create_communicator(const size_t size,
                                               shared_ptr_class<kvs_interface> kvs) {
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    LOG_TRACE("Create host communicator");

    return communicator(std::unique_ptr<communicator::impl_t>(new communicator::impl_t(size, kvs)));
}

/* TODO temporary function for UT compilation: would be part of ccl::environment in final*/
/**
 * Creates a new host communicator with user supplied size, rank and kvs.
 * @param size user-supplied total number of ranks
 * @param rank user-supplied rank
 * @param kvs key-value store for ranks wire-up
 * @return host communicator
 */
communicator communicator::create_communicator(const size_t size,
                                               const size_t rank,
                                               shared_ptr_class<kvs_interface> kvs) {
    LOG_TRACE("Create host communicator");

    return communicator(
        std::unique_ptr<communicator::impl_t>(new communicator::impl_t(size, rank, kvs)));
}

/* allgatherv */
communicator::request_t CCL_API communicator::allgatherv(const void* send_buf,
                                                         size_t send_count,
                                                         void* recv_buf,
                                                         const vector_class<size_t>& recv_counts,
                                                         datatype dtype,
                                                         const allgatherv_attr& attr) {
    return get_impl()->allgatherv_impl(send_buf, send_count, recv_buf, recv_counts, dtype, attr);
}

communicator::request_t CCL_API communicator::allgatherv(const void* send_buf,
                                                         size_t send_count,
                                                         const vector_class<void*>& recv_bufs,
                                                         const vector_class<size_t>& recv_counts,
                                                         datatype dtype,
                                                         const allgatherv_attr& attr) {
    return get_impl()->allgatherv_impl(send_buf, send_count, recv_bufs, recv_counts, dtype, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::allgatherv(const BufferType* send_buf,
                                                         size_t send_count,
                                                         BufferType* recv_buf,
                                                         const vector_class<size_t>& recv_counts,
                                                         const allgatherv_attr& attr) {
    return get_impl()->allgatherv_impl(send_buf, send_count, recv_buf, recv_counts, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::allgatherv(const BufferType* send_buf,
                                                         size_t send_count,
                                                         const vector_class<BufferType*>& recv_bufs,
                                                         const vector_class<size_t>& recv_counts,
                                                         const allgatherv_attr& attr) {
    return get_impl()->allgatherv_impl(send_buf, send_count, recv_bufs, recv_counts, attr);
}

/* allreduce */
communicator::request_t CCL_API communicator::allreduce(const void* send_buf,
                                                        void* recv_buf,
                                                        size_t count,
                                                        datatype dtype,
                                                        reduction rtype,
                                                        const allreduce_attr& attr) {
    return get_impl()->allreduce_impl(send_buf, recv_buf, count, dtype, rtype, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::allreduce(const BufferType* send_buf,
                                                        BufferType* recv_buf,
                                                        size_t count,
                                                        reduction rtype,
                                                        const allreduce_attr& attr) {
    return get_impl()->allreduce_impl(send_buf, recv_buf, count, rtype, attr);
}

/* alltoall */
communicator::request_t CCL_API communicator::alltoall(const void* send_buf,
                                                       void* recv_buf,
                                                       size_t count,
                                                       datatype dtype,
                                                       const alltoall_attr& attr) {
    return get_impl()->alltoall_impl(send_buf, recv_buf, count, dtype, attr);
}

communicator::request_t CCL_API communicator::alltoall(const vector_class<void*>& send_buf,
                                                       const vector_class<void*>& recv_buf,
                                                       size_t count,
                                                       datatype dtype,
                                                       const alltoall_attr& attr) {
    return get_impl()->alltoall_impl(send_buf, recv_buf, count, dtype, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::alltoall(const BufferType* send_buf,
                                                       BufferType* recv_buf,
                                                       size_t count,
                                                       const alltoall_attr& attr) {
    return get_impl()->alltoall_impl(send_buf, recv_buf, count, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::alltoall(const vector_class<BufferType*>& send_buf,
                                                       const vector_class<BufferType*>& recv_buf,
                                                       size_t count,
                                                       const alltoall_attr& attr) {
    return get_impl()->alltoall_impl(send_buf, recv_buf, count, attr);
}

/* alltoallv */
communicator::request_t CCL_API communicator::alltoallv(const void* send_buf,
                                                        const vector_class<size_t>& send_counts,
                                                        void* recv_buf,
                                                        const vector_class<size_t>& recv_counts,
                                                        datatype dtype,
                                                        const alltoallv_attr& attr) {
    return get_impl()->alltoallv_impl(send_buf, send_counts, recv_buf, recv_counts, dtype, attr);
}

communicator::request_t CCL_API communicator::alltoallv(const vector_class<void*>& send_bufs,
                                                        const vector_class<size_t>& send_counts,
                                                        const vector_class<void*>& recv_bufs,
                                                        const vector_class<size_t>& recv_counts,
                                                        datatype dtype,
                                                        const alltoallv_attr& attr) {
    return get_impl()->alltoallv_impl(send_bufs, send_counts, recv_bufs, recv_counts, dtype, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::alltoallv(const BufferType* send_buf,
                                                        const vector_class<size_t>& send_counts,
                                                        BufferType* recv_buf,
                                                        const vector_class<size_t>& recv_counts,
                                                        const alltoallv_attr& attr) {
    return get_impl()->alltoallv_impl(send_buf, send_counts, recv_buf, recv_counts, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::alltoallv(const vector_class<BufferType*>& send_bufs,
                                                        const vector_class<size_t>& send_counts,
                                                        const vector_class<BufferType*>& recv_bufs,
                                                        const vector_class<size_t>& recv_counts,
                                                        const alltoallv_attr& attr) {
    return get_impl()->alltoallv_impl(send_bufs, send_counts, recv_bufs, recv_counts, attr);
}

/* barrier */
communicator::request_t CCL_API communicator::barrier(const barrier_attr& attr) {
    return get_impl()->barrier_impl(attr);
}

/* bcast */
communicator::request_t CCL_API communicator::broadcast(void* buf,
                                                        size_t count,
                                                        datatype dtype,
                                                        size_t root,
                                                        const broadcast_attr& attr) {
    return get_impl()->broadcast_impl(buf, count, dtype, root, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API
communicator::broadcast(BufferType* buf, size_t count, size_t root, const broadcast_attr& attr) {
    return get_impl()->broadcast_impl(buf, count, root, attr);
}

/* reduce */
communicator::request_t CCL_API communicator::reduce(const void* send_buf,
                                                     void* recv_buf,
                                                     size_t count,
                                                     datatype dtype,
                                                     reduction rtype,
                                                     size_t root,
                                                     const reduce_attr& attr) {
    return get_impl()->reduce_impl(send_buf, recv_buf, count, dtype, rtype, root, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::reduce(const BufferType* send_buf,
                                                     BufferType* recv_buf,
                                                     size_t count,
                                                     reduction rtype,
                                                     size_t root,
                                                     const reduce_attr& attr) {
    return get_impl()->reduce_impl(send_buf, recv_buf, count, rtype, root, attr);
}

/* reduce_scatter */
communicator::request_t CCL_API communicator::reduce_scatter(const void* send_buf,
                                                             void* recv_buf,
                                                             size_t recv_count,
                                                             datatype dtype,
                                                             reduction rtype,
                                                             const reduce_scatter_attr& attr) {
    return get_impl()->reduce_scatter_impl(send_buf, recv_buf, recv_count, dtype, rtype, attr);
}

template <class BufferType, typename T>
communicator::request_t CCL_API communicator::reduce_scatter(const BufferType* send_buf,
                                                             BufferType* recv_buf,
                                                             size_t recv_count,
                                                             reduction rtype,
                                                             const reduce_scatter_attr& attr) {
    return get_impl()->reduce_scatter_impl(send_buf, recv_buf, recv_count, rtype, attr);
}

/* sparse_allreduce */
communicator::request_t CCL_API communicator::sparse_allreduce(const void* send_ind_buf,
                                                               size_t send_ind_count,
                                                               const void* send_val_buf,
                                                               size_t send_val_count,
                                                               void* recv_ind_buf,
                                                               size_t recv_ind_count,
                                                               void* recv_val_buf,
                                                               size_t recv_val_count,
                                                               datatype ind_dtype,
                                                               datatype val_dtype,
                                                               reduction rtype,
                                                               const sparse_allreduce_attr& attr) {
    return get_impl()->sparse_allreduce_impl(send_ind_buf,
                                             send_ind_count,
                                             send_val_buf,
                                             send_val_count,
                                             recv_ind_buf,
                                             recv_ind_count,
                                             recv_val_buf,
                                             recv_val_count,
                                             ind_dtype,
                                             val_dtype,
                                             rtype,
                                             attr);
}

template <class index_BufferType, class value_BufferType, typename T>
communicator::request_t CCL_API communicator::sparse_allreduce(const index_BufferType* send_ind_buf,
                                                               size_t send_ind_count,
                                                               const value_BufferType* send_val_buf,
                                                               size_t send_val_count,
                                                               index_BufferType* recv_ind_buf,
                                                               size_t recv_ind_count,
                                                               value_BufferType* recv_val_buf,
                                                               size_t recv_val_count,
                                                               reduction rtype,
                                                               const sparse_allreduce_attr& attr) {
    return get_impl()->sparse_allreduce_impl(send_ind_buf,
                                             send_ind_count,
                                             send_val_buf,
                                             send_val_count,
                                             recv_ind_buf,
                                             recv_ind_count,
                                             recv_val_buf,
                                             recv_val_count,
                                             rtype,
                                             attr);
}

API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(communicator, char);
API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(communicator, int);
API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(communicator, int64_t);
API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(communicator, uint64_t);
API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(communicator, float);
API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(communicator, double);

API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, char);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, int);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, ccl::bfp16);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, float);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, double);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, int64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, char, uint64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, char);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, int);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, ccl::bfp16);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, float);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, double);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, int64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int, uint64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, char);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, int);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, ccl::bfp16);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, float);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, double);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, int64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, int64_t, uint64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, char);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, int);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, ccl::bfp16);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, float);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, double);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, int64_t);
API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(communicator, uint64_t, uint64_t);

} // namespace ccl
