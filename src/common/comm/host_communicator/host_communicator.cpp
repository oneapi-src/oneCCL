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
#include "common/global/global.hpp"
#include "common/comm/host_communicator/host_communicator_impl.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "comm_split_attr_creation_impl.hpp"

#include "common/request/request.hpp"
#include "common/request/host_request.hpp"
#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"
#include "coll/ccl_allgather_op_attr.hpp"

namespace ccl {

host_communicator::host_communicator()
        : comm_attr(ccl::create_comm_split_attr()),
          comm_rank(0),
          comm_size(1) {}

host_communicator::host_communicator(size_t size, shared_ptr_class<kvs_interface> kvs)
        : comm_attr(ccl::create_comm_split_attr()),
          comm_rank(0),
          comm_size(size) {
    if (size <= 0) {
        throw ccl_error("Incorrect size value when creating a host communicator");
    }
}

host_communicator::host_communicator(size_t size, size_t rank, shared_ptr_class<kvs_interface> kvs)
        : comm_attr(ccl::create_comm_split_attr()),
          comm_rank(rank),
          comm_size(size) {
    if (rank > size || size <= 0) {
        throw ccl_error("Incorrect rank or size value when creating a host communicator");
    }

    ccl::global_data& data = ccl::global_data::get();
    comm_impl = std::shared_ptr<ccl_comm>(new ccl_comm(rank, size, data.comm_ids->acquire()));
}

host_communicator::host_communicator(std::shared_ptr<ccl_comm> impl)
        : comm_impl(impl),
          comm_attr(
              ccl::
                  create_comm_split_attr()), // TODO should be ccl::environment::instance().create_comm_split_attr() call in final?
          comm_rank(impl->rank()),
          comm_size(impl->size()) {}

size_t host_communicator::rank() const {
    return comm_rank;
}

size_t host_communicator::size() const {
    return comm_size;
}

ccl::unique_ptr_class<host_communicator> host_communicator::split(const comm_split_attr& attr) {
    ccl::global_data& data = ccl::global_data::get();
    auto new_comm = ccl_comm::create_with_color(
        attr.get<ccl::comm_split_attr_id::color>(), data.comm_ids.get(), comm_impl.get());

    comm_attr = attr;

    return unique_ptr_class<host_communicator>(
        new host_communicator(std::shared_ptr<ccl_comm>(new_comm)));
}

/* allgatherv */
ccl::request_t host_communicator::allgatherv_impl(const void* send_buf,
                                                  size_t send_count,
                                                  void* recv_buf,
                                                  const vector_class<size_t>& recv_counts,
                                                  datatype dtype,
                                                  const allgatherv_attr& attr) {
    ccl_request* req = ccl_allgatherv_impl(
        send_buf, send_count, recv_buf, recv_counts.data(), dtype, attr, comm_impl.get(), nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

ccl::request_t host_communicator::allgatherv_impl(const void* send_buf,
                                                  size_t send_count,
                                                  const vector_class<void*>& recv_bufs,
                                                  const vector_class<size_t>& recv_counts,
                                                  datatype dtype,
                                                  const allgatherv_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* allreduce */
ccl::request_t host_communicator::allreduce_impl(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 datatype dtype,
                                                 reduction rtype,
                                                 const allreduce_attr& attr) {
    ccl_request* req =
        ccl_allreduce_impl(send_buf, recv_buf, count, dtype, rtype, attr, comm_impl.get(), nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoall */
ccl::request_t host_communicator::alltoall_impl(const void* send_buf,
                                                void* recv_buf,
                                                size_t count,
                                                datatype dtype,
                                                const alltoall_attr& attr) {
    ccl_request* req =
        ccl_alltoall_impl(send_buf, recv_buf, count, dtype, attr, comm_impl.get(), nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

ccl::request_t host_communicator::alltoall_impl(const vector_class<void*>& send_buf,
                                                const vector_class<void*>& recv_buf,
                                                size_t count,
                                                datatype dtype,
                                                const alltoall_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* alltoallv */
ccl::request_t host_communicator::alltoallv_impl(const void* send_buf,
                                                 const vector_class<size_t>& send_counts,
                                                 void* recv_buf,
                                                 const vector_class<size_t>& recv_counts,
                                                 datatype dtype,
                                                 const alltoallv_attr& attr) {
    ccl_request* req = ccl_alltoallv_impl(send_buf,
                                          send_counts.data(),
                                          recv_buf,
                                          recv_counts.data(),
                                          dtype,
                                          attr,
                                          comm_impl.get(),
                                          nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

ccl::request_t host_communicator::alltoallv_impl(const vector_class<void*>& send_bufs,
                                                 const vector_class<size_t>& send_counts,
                                                 const vector_class<void*>& recv_bufs,
                                                 const vector_class<size_t>& recv_counts,
                                                 datatype dtype,
                                                 const alltoallv_attr& attr) {
    // TODO not implemented
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* barrier */
ccl::request_t host_communicator::barrier_impl(const barrier_attr& attr) {
    // TODO what exactly we need to do with 'attr' here?

    ccl_barrier_impl(comm_impl.get(), nullptr);

    // TODO what exactly we need to return here? ccl_barrier_impl() is void func
    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* bcast */
ccl::request_t host_communicator::broadcast_impl(void* buf,
                                                 size_t count,
                                                 datatype dtype,
                                                 size_t root,
                                                 const broadcast_attr& attr) {
    ccl_request* req = ccl_broadcast_impl(buf, count, dtype, root, attr, comm_impl.get(), nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce */
ccl::request_t host_communicator::reduce_impl(const void* send_buf,
                                              void* recv_buf,
                                              size_t count,
                                              datatype dtype,
                                              reduction rtype,
                                              size_t root,
                                              const reduce_attr& attr) {
    ccl_request* req = ccl_reduce_impl(
        send_buf, recv_buf, count, dtype, rtype, root, attr, comm_impl.get(), nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce_scatter */
ccl::request_t host_communicator::reduce_scatter_impl(const void* send_buf,
                                                      void* recv_buf,
                                                      size_t recv_count,
                                                      datatype dtype,
                                                      reduction rtype,
                                                      const reduce_scatter_attr& attr) {
    // TODO not fully implemented (need to implement reduce_scatter in parallelizer.cpp)
    throw ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    ccl_request* req = ccl_reduce_scatter_impl(
        send_buf, recv_buf, recv_count, dtype, rtype, attr, comm_impl.get(), nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* sparse_allreduce */
ccl::request_t host_communicator::sparse_allreduce_impl(const void* send_ind_buf,
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
    ccl_request* req = ccl_sparse_allreduce_impl(send_ind_buf,
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
                                                 attr,
                                                 comm_impl.get(),
                                                 nullptr);

    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(host_communicator, char);
HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(host_communicator, int);
HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(host_communicator, int64_t);
HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(host_communicator, uint64_t);
HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(host_communicator, float);
HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(host_communicator, double);

HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, char);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, int);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, ccl::bfp16);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, float);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, double);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, int64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, uint64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, char);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, int);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, ccl::bfp16);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, float);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, double);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, int64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, uint64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, char);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, int);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, ccl::bfp16);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, float);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, double);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, int64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, uint64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, char);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, int);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, ccl::bfp16);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, float);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, double);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, int64_t);
HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, uint64_t);

} // namespace ccl
