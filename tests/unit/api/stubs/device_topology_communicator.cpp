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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "typed_base_communicator_impl.hpp"
#include "common/comm/l0/communicator/device_group/device_ring_communicator.hpp"
#include "common/comm/l0/communicator/device_group/device_a2a_communicator.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"

using namespace ccl;

device_group_a2a_communicator::device_group_a2a_communicator(
    ccl::unified_device_type&& device,
    size_t thread_idx,
    size_t process_idx,
    const ccl::device_comm_split_attr& attr)
        : base_t(std::move(device), thread_idx, process_idx, /*comm_attr, */ attr) {}

void device_group_a2a_communicator::visit(ccl::gpu_comm_attr& comm_attr) {}

ccl::request_t device_group_a2a_communicator::barrier(ccl::stream::impl_value_t& stream,
                                                      const ccl::barrier_attr& attr,
                                                      const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allgatherv */
ccl::coll_request_t device_group_a2a_communicator::allgatherv_impl(
    const void* send_buf,
    size_t send_count,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
ccl::coll_request_t device_group_a2a_communicator::allgatherv_impl(
    const void* send_buf,
    size_t send_count,
    const ccl::vector_class<void*>& recv_bufs,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allreduce */
ccl::coll_request_t device_group_a2a_communicator::allreduce_impl(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* alltoall */
ccl::coll_request_t device_group_a2a_communicator::alltoall_impl(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
ccl::coll_request_t device_group_a2a_communicator::alltoall_impl(
    const ccl::vector_class<void*>& send_buf,
    const ccl::vector_class<void*>& recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* alltoallv */
ccl::coll_request_t device_group_a2a_communicator::alltoallv_impl(
    const void* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
ccl::coll_request_t device_group_a2a_communicator::alltoallv_impl(
    const ccl::vector_class<void*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    ccl::vector_class<void*> recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

/* bcast */
ccl::coll_request_t device_group_a2a_communicator::broadcast_impl(
    void* buf,
    size_t count,
    ccl::datatype dtype,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* reduce */
ccl::coll_request_t device_group_a2a_communicator::reduce_impl(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* reduce_scatter */
ccl::request_t device_group_a2a_communicator::reduce_scatter_impl(
    const void* send_buf,
    void* recv_buf,
    size_t recv_count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* sparse_allreduce */
ccl::coll_request_t device_group_a2a_communicator::sparse_allreduce_impl(
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
    ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allgatherv */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::request_t device_group_a2a_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allreduce */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::allreduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::allreduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* alltoall */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::alltoall_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::request_t device_group_a2a_communicator::alltoall_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<buffer_type*>& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::alltoall_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::request_t device_group_a2a_communicator::alltoall_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

/* alltoallv */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::alltoallv_impl(
    const buffer_type* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::alltoallv_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::alltoallv_impl(
    const buffer_type& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::alltoallv_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

/* bcast */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::broadcast_impl(
    buffer_type* buf,
    size_t count,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::broadcast_impl(
    buffer_type& buf,
    size_t count,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* reduce */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::reduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::reduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
/* reduce_scatter */
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::reduce_scatter_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_a2a_communicator::reduce_scatter_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* sparse_allreduce */
template <class index_buffer_type, class value_buffer_type>
ccl::coll_request_t device_group_a2a_communicator::sparse_allreduce_impl(
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
    return {};
}

template <class index_buffer_container_type, class value_buffer_container_type>
ccl::coll_request_t device_group_a2a_communicator::sparse_allreduce_impl(
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
    return {};
}

DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_a2a_communicator, char);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_a2a_communicator, int);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_a2a_communicator, int64_t);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_a2a_communicator, uint64_t);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_a2a_communicator, float);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_a2a_communicator, double);

#ifdef CCL_ENABLE_SYCL
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_a2a_communicator,
                                                cl::sycl::buffer<char COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_a2a_communicator,
                                                cl::sycl::buffer<int COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_a2a_communicator,
                                                cl::sycl::buffer<int64_t COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_a2a_communicator,
                                                cl::sycl::buffer<uint64_t COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_a2a_communicator,
                                                cl::sycl::buffer<float COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_a2a_communicator,
                                                cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              char,
                                                              uint64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int,
                                                              uint64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              int64_t,
                                                              uint64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_a2a_communicator,
                                                              uint64_t,
                                                              uint64_t);

#ifdef CCL_ENABLE_SYCL
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_a2a_communicator,
    cl::sycl::buffer<int COMMA 1>,
    cl::sycl::buffer<float COMMA 1>);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_a2a_communicator,
    cl::sycl::buffer<int COMMA 1>,
    cl::sycl::buffer<ccl::bfp16 COMMA 1>);

DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_a2a_communicator,
    cl::sycl::buffer<int64_t COMMA 1>,
    cl::sycl::buffer<float COMMA 1>);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_a2a_communicator,
    cl::sycl::buffer<int64_t COMMA 1>,
    cl::sycl::buffer<ccl::bfp16 COMMA 1>);
#endif //CCL_ENABLE_SYCL

////////////////////////////////////////////////////////////////
device_group_ring_communicator::device_group_ring_communicator(
    ccl::unified_device_type&& device,
    size_t thread_idx,
    size_t process_idx,
    const ccl::device_comm_split_attr& attr)
        : base_t(std::move(device), thread_idx, process_idx, /*comm_attr, */ attr) {}

void device_group_ring_communicator::visit(ccl::gpu_comm_attr& comm_attr) {}

ccl::request_t device_group_ring_communicator::barrier(ccl::stream::impl_value_t& stream,
                                                       const ccl::barrier_attr& attr,
                                                       const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allgatherv */
ccl::coll_request_t device_group_ring_communicator::allgatherv_impl(
    const void* send_buf,
    size_t send_count,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
ccl::coll_request_t device_group_ring_communicator::allgatherv_impl(
    const void* send_buf,
    size_t send_count,
    const ccl::vector_class<void*>& recv_bufs,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allreduce */
ccl::coll_request_t device_group_ring_communicator::allreduce_impl(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* alltoall */
ccl::coll_request_t device_group_ring_communicator::alltoall_impl(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
ccl::coll_request_t device_group_ring_communicator::alltoall_impl(
    const ccl::vector_class<void*>& send_buf,
    const ccl::vector_class<void*>& recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* alltoallv */
ccl::coll_request_t device_group_ring_communicator::alltoallv_impl(
    const void* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
ccl::coll_request_t device_group_ring_communicator::alltoallv_impl(
    const ccl::vector_class<void*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    ccl::vector_class<void*> recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::datatype dtype,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

/* bcast */
ccl::coll_request_t device_group_ring_communicator::broadcast_impl(
    void* buf,
    size_t count,
    ccl::datatype dtype,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* reduce */
ccl::coll_request_t device_group_ring_communicator::reduce_impl(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* reduce_scatter */
ccl::request_t device_group_ring_communicator::reduce_scatter_impl(
    const void* send_buf,
    void* recv_buf,
    size_t recv_count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* sparse_allreduce */
ccl::coll_request_t device_group_ring_communicator::sparse_allreduce_impl(
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
    ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allgatherv */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::request_t device_group_ring_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* allreduce */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::allreduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::allreduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* alltoall */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::alltoall_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::request_t device_group_ring_communicator::alltoall_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<buffer_type*>& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,

    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::alltoall_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::request_t device_group_ring_communicator::alltoall_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

/* alltoallv */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::alltoallv_impl(
    const buffer_type* send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::alltoallv_impl(
    const ccl::vector_class<buffer_type*>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<buffer_type*>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::alltoallv_impl(
    const buffer_type& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    buffer_type& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::alltoallv_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,

    const ccl::vector_class<ccl::event>& dep) {
    return {};
}

/* bcast */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::broadcast_impl(
    buffer_type* buf,
    size_t count,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::broadcast_impl(
    buffer_type& buf,
    size_t count,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::broadcast_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* reduce */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::reduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::reduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
/* reduce_scatter */
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::reduce_scatter_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}
template <class buffer_type>
ccl::coll_request_t device_group_ring_communicator::reduce_scatter_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t recv_count,
    ccl::reduction reduction,
    ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    return {};
}

/* sparse_allreduce */
template <class index_buffer_type, class value_buffer_type>
ccl::coll_request_t device_group_ring_communicator::sparse_allreduce_impl(
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
    return {};
}

template <class index_buffer_container_type, class value_buffer_container_type>
ccl::coll_request_t device_group_ring_communicator::sparse_allreduce_impl(
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
    return {};
}

DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_ring_communicator, char);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_ring_communicator, int);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_ring_communicator, int64_t);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_ring_communicator, uint64_t);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_ring_communicator, float);
DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(device_group_ring_communicator, double);

#ifdef CCL_ENABLE_SYCL
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_ring_communicator,
                                                cl::sycl::buffer<char COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_ring_communicator,
                                                cl::sycl::buffer<int COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_ring_communicator,
                                                cl::sycl::buffer<int64_t COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_ring_communicator,
                                                cl::sycl::buffer<uint64_t COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_ring_communicator,
                                                cl::sycl::buffer<float COMMA 1>);
DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(device_group_ring_communicator,
                                                cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              char,
                                                              uint64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int,
                                                              uint64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              int64_t,
                                                              uint64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              char);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              int);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              ccl::bfp16);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              float);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              double);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              int64_t);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(device_group_ring_communicator,
                                                              uint64_t,
                                                              uint64_t);

#ifdef CCL_ENABLE_SYCL
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_ring_communicator,
    cl::sycl::buffer<int COMMA 1>,
    cl::sycl::buffer<float COMMA 1>);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_ring_communicator,
    cl::sycl::buffer<int COMMA 1>,
    cl::sycl::buffer<ccl::bfp16 COMMA 1>);

DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_ring_communicator,
    cl::sycl::buffer<int64_t COMMA 1>,
    cl::sycl::buffer<float COMMA 1>);
DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(
    device_group_ring_communicator,
    cl::sycl::buffer<int64_t COMMA 1>,
    cl::sycl::buffer<ccl::bfp16 COMMA 1>);
#endif //CCL_ENABLE_SYCL
