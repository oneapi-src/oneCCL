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

/**
 * Generating types for collective operations
 * of the host implementation class (host_communicator_impl)
 */
#define HOST_COMM_IMPL_COLL_EXPLICIT_INSTANTIATIONS(comm_class, type) \
\
    template ccl::request_t comm_class::allgatherv_impl(const type* send_buf, \
                                                        size_t send_count, \
                                                        type* recv_buf, \
                                                        const vector_class<size_t>& recv_counts, \
                                                        const allgatherv_attr& attr); \
\
    template ccl::request_t comm_class::allgatherv_impl(const type* send_buf, \
                                                        size_t send_count, \
                                                        const vector_class<type*>& recv_bufs, \
                                                        const vector_class<size_t>& recv_counts, \
                                                        const allgatherv_attr& attr); \
\
    template ccl::request_t comm_class::allreduce_impl(const type* send_buf, \
                                                       type* recv_buf, \
                                                       size_t count, \
                                                       ccl::reduction reduction, \
                                                       const allreduce_attr& attr); \
\
    template ccl::request_t comm_class::alltoall_impl( \
        const type* send_buf, type* recv_buf, size_t count, const alltoall_attr& attr); \
\
    template ccl::request_t comm_class::alltoall_impl(const vector_class<type*>& send_buf, \
                                                      const vector_class<type*>& recv_buf, \
                                                      size_t count, \
                                                      const alltoall_attr& attr); \
\
    template ccl::request_t comm_class::alltoallv_impl(const type* send_buf, \
                                                       const vector_class<size_t>& send_counts, \
                                                       type* recv_buf, \
                                                       const vector_class<size_t>& recv_counts, \
                                                       const alltoallv_attr& attr); \
\
    template ccl::request_t comm_class::alltoallv_impl(const vector_class<type*>& send_bufs, \
                                                       const vector_class<size_t>& send_counts, \
                                                       const vector_class<type*>& recv_bufs, \
                                                       const vector_class<size_t>& recv_counts, \
                                                       const alltoallv_attr& attr); \
\
    template ccl::request_t comm_class::broadcast_impl( \
        type* buf, size_t count, size_t root, const broadcast_attr& attr); \
\
    template ccl::request_t comm_class::reduce_impl(const type* send_buf, \
                                                    type* recv_buf, \
                                                    size_t count, \
                                                    ccl::reduction reduction, \
                                                    size_t root, \
                                                    const reduce_attr& attr); \
\
    template ccl::request_t comm_class::reduce_scatter_impl(const type* send_buf, \
                                                            type* recv_buf, \
                                                            size_t recv_count, \
                                                            ccl::reduction reduction, \
                                                            const reduce_scatter_attr& attr);

#define HOST_COMM_IMPL_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(comm_class, index_type, value_type) \
\
    template ccl::request_t comm_class::sparse_allreduce_impl(const index_type* send_ind_buf, \
                                                              size_t send_ind_count, \
                                                              const value_type* send_val_buf, \
                                                              size_t send_val_count, \
                                                              index_type* recv_ind_buf, \
                                                              size_t recv_ind_count, \
                                                              value_type* recv_val_buf, \
                                                              size_t recv_val_count, \
                                                              ccl::reduction reduction, \
                                                              const sparse_allreduce_attr& attr);

/**
 * Generating API types for collective operations
 * of the host communicator class (communicator)
 */
#define API_HOST_COMM_COLL_EXPLICIT_INSTANTIATION(comm_class, type) \
\
    template ccl::request_t CCL_API comm_class::allgatherv( \
        const type* send_buf, \
        size_t send_count, \
        type* recv_buf, \
        const vector_class<size_t>& recv_counts, \
        const allgatherv_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::allgatherv( \
        const type* send_buf, \
        size_t send_count, \
        const vector_class<type*>& recv_bufs, \
        const vector_class<size_t>& recv_counts, \
        const allgatherv_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::allreduce(const type* send_buf, \
                                                          type* recv_buf, \
                                                          size_t count, \
                                                          ccl::reduction reduction, \
                                                          const allreduce_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::alltoall( \
        const type* send_buf, type* recv_buf, size_t count, const alltoall_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::alltoall(const vector_class<type*>& send_buf, \
                                                         const vector_class<type*>& recv_buf, \
                                                         size_t count, \
                                                         const alltoall_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::alltoallv(const type* send_buf, \
                                                          const vector_class<size_t>& send_counts, \
                                                          type* recv_buf, \
                                                          const vector_class<size_t>& recv_counts, \
                                                          const alltoallv_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::alltoallv(const vector_class<type*>& send_bufs, \
                                                          const vector_class<size_t>& send_counts, \
                                                          const vector_class<type*>& recv_bufs, \
                                                          const vector_class<size_t>& recv_counts, \
                                                          const alltoallv_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::broadcast( \
        type* buf, size_t count, size_t root, const broadcast_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::reduce(const type* send_buf, \
                                                       type* recv_buf, \
                                                       size_t count, \
                                                       ccl::reduction reduction, \
                                                       size_t root, \
                                                       const reduce_attr& attr); \
\
    template ccl::request_t CCL_API comm_class::reduce_scatter(const type* send_buf, \
                                                               type* recv_buf, \
                                                               size_t recv_count, \
                                                               ccl::reduction reduction, \
                                                               const reduce_scatter_attr& attr);

#define API_HOST_COMM_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(comm_class, index_type, value_type) \
\
    template ccl::request_t CCL_API comm_class::sparse_allreduce( \
        const index_type* send_ind_buf, \
        size_t send_ind_count, \
        const value_type* send_val_buf, \
        size_t send_val_count, \
        index_type* recv_ind_buf, \
        size_t recv_ind_count, \
        value_type* recv_val_buf, \
        size_t recv_val_count, \
        ccl::reduction reduction, \
        const sparse_allreduce_attr& attr);
