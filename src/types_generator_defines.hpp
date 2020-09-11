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
 * API types generators
 */
//TODO
#if 0
#define API_COLL_EXPLICIT_CLASS_INSTANTIATION(type) \
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::allgatherv( \
        const type& send_buf, \
        size_t send_count, \
        type& recv_buf, \
        const size_t* recv_counts, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::allreduce( \
        const type& send_buf, \
        type& recv_buf, \
        size_t count, \
        ccl::reduction reduction, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::alltoall( \
        const type& send_buf, \
        type& recv_buf, \
        size_t count, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::alltoallv( \
        const type& send_buf, \
        const size_t* send_counts, \
        type& recv_buf, \
        const size_t* recv_counts, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::broadcast( \
        type& buf, \
        size_t count, \
        size_t root, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::reduce( \
        const type& send_buf, \
        type& recv_buf, \
        size_t count, \
        ccl::reduction reduction, \
        size_t root, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream);

#define API_COLL_EXPLICIT_INSTANTIATION(type) \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::allgatherv( \
        const type* send_buf, \
        size_t send_count, \
        type* recv_buf, \
        const size_t* recv_counts, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::allreduce( \
        const type* send_buf, \
        type* recv_buf, \
        size_t count, \
        ccl::reduction reduction, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::alltoall( \
        const type* send_buf, \
        type* recv_buf, \
        size_t count, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::alltoallv( \
        const type* send_buf, \
        const size_t* send_counts, \
        type* recv_buf, \
        const size_t* recv_counts, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::broadcast( \
        type* buf, \
        size_t count, \
        size_t root, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream); \
\
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::reduce( \
        const type* send_buf, \
        type* recv_buf, \
        size_t count, \
        ccl::reduction reduction, \
        size_t root, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream);

#define API_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(index_type, value_type) \
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::sparse_allreduce( \
        const index_type* send_ind_buf, \
        size_t send_ind_count, \
        const value_type* send_val_buf, \
        size_t send_val_count, \
        index_type* recv_ind_buf, \
        size_t recv_ind_count, \
        value_type* recv_val_buf, \
        size_t recv_val_count, \
        ccl::reduction reduction, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream);

#define API_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(index_type, value_type) \
    template ccl::communicator::coll_request_t CCL_API ccl::communicator::sparse_allreduce( \
        const index_type& send_ind_buf, \
        size_t send_ind_count, \
        const value_type& send_val_buf, \
        size_t send_val_count, \
        index_type& recv_ind_buf, \
        size_t recv_ind_count, \
        value_type& recv_val_buf, \
        size_t recv_val_count, \
        ccl::reduction reduction, \
        const ccl::coll_attr* attr, \
        const ccl::stream_t& stream);

#endif //TODO
/**
 * Core types generators
 */
#define DEVICE_COMM_INTERFACE_COLL_DECLARATION__VOID \
\
    virtual ccl::request_t allgatherv(const void* send_buf, \
                                      size_t send_count, \
                                      void* recv_buf, \
                                      const ccl::vector_class<size_t>& recv_counts, \
                                      ccl::datatype dtype, \
                                      ccl::stream::impl_value_t& stream, \
                                      const ccl::allgatherv_attr& attr, \
                                      const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t allgatherv(const void* send_buf, \
                                      size_t send_count, \
                                      const ccl::vector_class<void*>& recv_bufs, \
                                      const ccl::vector_class<size_t>& recv_counts, \
                                      ccl::datatype dtype, \
                                      ccl::stream::impl_value_t& stream, \
                                      const ccl::allgatherv_attr& attr, \
                                      const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t allreduce(const void* send_buf, \
                                     void* recv_buf, \
                                     size_t count, \
                                     ccl::datatype dtype, \
                                     ccl::reduction reduction, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::allreduce_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoall(const void* send_buf, \
                                    void* recv_buf, \
                                    size_t count, \
                                    ccl::datatype dtype, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::alltoall_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) = 0; \
    virtual ccl::request_t alltoall(const ccl::vector_class<void*>& send_buf, \
                                    const ccl::vector_class<void*>& recv_buf, \
                                    size_t count, \
                                    ccl::datatype dtype, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::alltoall_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoallv(const void* send_buf, \
                                     const ccl::vector_class<size_t>& send_counts, \
                                     void* recv_buf, \
                                     const ccl::vector_class<size_t>& recv_counts, \
                                     ccl::datatype dtype, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::alltoallv_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoallv(const ccl::vector_class<void*>& send_bufs, \
                                     const ccl::vector_class<size_t>& send_counts, \
                                     const ccl::vector_class<void*>& recv_bufs, \
                                     const ccl::vector_class<size_t>& recv_counts, \
                                     ccl::datatype dtype, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::alltoallv_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t bcast(void* buf, \
                                 size_t count, \
                                 ccl::datatype dtype, \
                                 size_t root, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::broadcast_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t reduce(const void* send_buf, \
                                  void* recv_buf, \
                                  size_t count, \
                                  ccl::datatype dtype, \
                                  ccl::reduction reduction, \
                                  size_t root, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::reduce_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t reduce_scatter(const void* send_buf, \
                                          void* recv_buf, \
                                          size_t recv_count, \
                                          ccl::datatype dtype, \
                                          ccl::reduction reduction, \
                                          ccl::stream::impl_value_t& stream, \
                                          const reduce_scatter_attr& attr, \
                                          const ccl::vector_class<ccl::event>& deps = {}) = 0;

#define DEVICE_COMM_INTERFACE_SPARSE_DECLARATION__VOID \
\
    virtual ccl::request_t sparse_allreduce(const void* send_ind_buf, \
                                            size_t send_ind_count, \
                                            const void* send_val_buf, \
                                            size_t send_val_count, \
                                            void* recv_ind_buf, \
                                            size_t recv_ind_count, \
                                            void* recv_val_buf, \
                                            size_t recv_val_count, \
                                            ccl::datatype index_dtype, \
                                            ccl::datatype value_dtype, \
                                            ccl::reduction reduction, \
                                            ccl::stream::impl_value_t& stream, \
                                            const ccl::sparse_allreduce_attr& attr, \
                                            const ccl::vector_class<ccl::event>& deps = {}) = 0;

#define DEVICE_COMM_INTERFACE_COLL_DECLARATION(type) \
\
    virtual ccl::request_t allgatherv(const type* send_buf, \
                                      size_t send_count, \
                                      type* recv_buf, \
                                      const ccl::vector_class<size_t>& recv_counts, \
                                      ccl::stream::impl_value_t& stream, \
                                      const ccl::allgatherv_attr& attr, \
                                      const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t allgatherv(const type* send_buf, \
                                      size_t send_count, \
                                      ccl::vector_class<type*>& recv_bufs, \
                                      const ccl::vector_class<size_t>& recv_counts, \
                                      ccl::stream::impl_value_t& stream, \
                                      const ccl::allgatherv_attr& attr, \
                                      const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t allreduce(const type* send_buf, \
                                     type* recv_buf, \
                                     size_t count, \
                                     ccl::reduction reduction, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::allreduce_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoall(const type* send_buf, \
                                    type* recv_buf, \
                                    size_t count, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::alltoall_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoall(const ccl::vector_class<type*>& send_buf, \
                                    const ccl::vector_class<type*>& recv_buf, \
                                    size_t count, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::alltoall_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoallv(const type* send_buf, \
                                     const ccl::vector_class<size_t>& send_counts, \
                                     type* recv_buf, \
                                     const ccl::vector_class<size_t>& recv_counts, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::alltoallv_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoallv(const ccl::vector_class<type*>& send_bufs, \
                                     const ccl::vector_class<size_t>& send_counts, \
                                     const ccl::vector_class<type*>& recv_bufs, \
                                     const ccl::vector_class<size_t>& recv_counts, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::alltoallv_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t bcast(type* buf, \
                                 size_t count, \
                                 size_t root, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::broadcast_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t reduce(const type* send_buf, \
                                  type* recv_buf, \
                                  size_t count, \
                                  ccl::reduction reduction, \
                                  size_t root, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::reduce_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t reduce_scatter(const type* send_buf, \
                                          type* recv_buf, \
                                          size_t recv_count, \
                                          ccl::reduction reduction, \
                                          ccl::stream::impl_value_t& stream, \
                                          const ccl::reduce_scatter_attr& attr, \
                                          const ccl::vector_class<ccl::event>& deps) = 0;

#define DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(index_type, value_type) \
\
    virtual ccl::request_t sparse_allreduce(const index_type* send_ind_buf, \
                                            size_t send_ind_count, \
                                            const value_type* send_val_buf, \
                                            size_t send_val_count, \
                                            index_type* recv_ind_buf, \
                                            size_t recv_ind_count, \
                                            value_type* recv_val_buf, \
                                            size_t recv_val_count, \
                                            ccl::reduction reduction, \
                                            ccl::stream::impl_value_t& stream, \
                                            const ccl::sparse_allreduce_attr& attr, \
                                            const ccl::vector_class<ccl::event>& deps = {}) = 0;

#define DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(type) \
\
    virtual ccl::request_t allgatherv(const type& send_buf, \
                                      size_t send_count, \
                                      type& recv_buf, \
                                      const ccl::vector_class<size_t>& recv_counts, \
                                      ccl::stream::impl_value_t& stream, \
                                      const ccl::allgatherv_attr& attr, \
                                      const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t allgatherv( \
        const type& send_buf, \
        size_t send_count, \
        ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_bufs, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t allreduce(const type& send_buf, \
                                     type& recv_buf, \
                                     size_t count, \
                                     ccl::reduction reduction, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::allreduce_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoall(const type& send_buf, \
                                    type& recv_buf, \
                                    size_t count, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::alltoall_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoall( \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& send_buf, \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_buf, \
        size_t count, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoallv(const type& send_buf, \
                                     const ccl::vector_class<size_t>& send_counts, \
                                     type& recv_buf, \
                                     const ccl::vector_class<size_t>& recv_counts, \
                                     ccl::stream::impl_value_t& stream, \
                                     const ccl::alltoallv_attr& attr, \
                                     const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t alltoallv( \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& send_bufs, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_bufs, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t bcast(type& buf, \
                                 size_t count, \
                                 size_t root, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::broadcast_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t reduce(const type& send_buf, \
                                  type& recv_buf, \
                                  size_t count, \
                                  ccl::reduction reduction, \
                                  size_t root, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::reduce_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps = {}) = 0; \
\
    virtual ccl::request_t reduce_scatter(const type& send_buf, \
                                          type& recv_buf, \
                                          size_t recv_count, \
                                          ccl::reduction reduction, \
                                          ccl::stream::impl_value_t& stream, \
                                          const ccl::reduce_scatter_attr& attr, \
                                          const ccl::vector_class<ccl::event>& deps = {}) = 0;

#define DEVICE_COMM_INTERFACE_SPARSE_CLASS_DECLARATION(index_type, value_type) \
\
    virtual ccl::request_t sparse_allreduce(const index_type& send_ind_buf, \
                                            size_t send_ind_count, \
                                            const value_type& send_val_buf, \
                                            size_t send_val_count, \
                                            index_type& recv_ind_buf, \
                                            size_t recv_ind_count, \
                                            value_type& recv_val_buf, \
                                            size_t recv_val_count, \
                                            ccl::reduction reduction, \
                                            ccl::stream::impl_value_t& stream, \
                                            const ccl::sparse_allreduce_attr& attr, \
                                            const ccl::vector_class<ccl::event>& deps = {}) = 0;

/**
 * Specific coll instantiation
 */
#define DEVICE_COMM_INTERFACE_COLL_DEFINITION__VOID \
\
    ccl::request_t allgatherv(const void* send_buf, \
                              size_t send_count, \
                              void* recv_buf, \
                              const ccl::vector_class<size_t>& recv_counts, \
                              ccl::datatype dtype, \
                              ccl::stream::impl_value_t& stream, \
                              const ccl::allgatherv_attr& attr, \
                              const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allgatherv_impl( \
            send_buf, send_count, recv_buf, recv_counts, dtype, stream, attr, deps); \
    } \
\
    ccl::request_t allgatherv(const void* send_buf, \
                              size_t send_count, \
                              const ccl::vector_class<void*>& recv_bufs, \
                              const ccl::vector_class<size_t>& recv_counts, \
                              ccl::datatype dtype, \
                              ccl::stream::impl_value_t& stream, \
                              const ccl::allgatherv_attr& attr, \
                              const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allgatherv_impl( \
            send_buf, send_count, recv_bufs, recv_counts, dtype, stream, attr, deps); \
    } \
\
    ccl::request_t allreduce(const void* send_buf, \
                             void* recv_buf, \
                             size_t count, \
                             ccl::datatype dtype, \
                             ccl::reduction reduction, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::allreduce_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allreduce_impl( \
            send_buf, recv_buf, count, dtype, reduction, stream, attr, deps); \
    } \
\
    ccl::request_t alltoall(const void* send_buf, \
                            void* recv_buf, \
                            size_t count, \
                            ccl::datatype dtype, \
                            ccl::stream::impl_value_t& stream, \
                            const ccl::alltoall_attr& attr, \
                            const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->alltoall_impl(send_buf, recv_buf, count, dtype, stream, attr, deps); \
    } \
    ccl::request_t alltoall(const ccl::vector_class<void*>& send_buf, \
                            const ccl::vector_class<void*>& recv_buf, \
                            size_t count, \
                            ccl::datatype dtype, \
                            ccl::stream::impl_value_t& stream, \
                            const ccl::alltoall_attr& attr, \
                            const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->alltoall_impl(send_buf, recv_buf, count, dtype, stream, attr, deps); \
    } \
\
    ccl::request_t alltoallv(const void* send_buf, \
                             const ccl::vector_class<size_t>& send_counts, \
                             void* recv_buf, \
                             const ccl::vector_class<size_t>& recv_counts, \
                             ccl::datatype dtype, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::alltoallv_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->alltoallv_impl( \
            send_buf, send_counts, recv_buf, recv_counts, dtype, stream, attr, deps); \
    } \
    ccl::request_t alltoallv(const ccl::vector_class<void*>& send_bufs, \
                             const ccl::vector_class<size_t>& send_counts, \
                             const ccl::vector_class<void*>& recv_bufs, \
                             const ccl::vector_class<size_t>& recv_counts, \
                             ccl::datatype dtype, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::alltoallv_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->alltoallv_impl( \
            send_bufs, send_counts, recv_bufs, recv_counts, dtype, stream, attr, deps); \
    } \
\
    ccl::request_t bcast(void* buf, \
                         size_t count, \
                         ccl::datatype dtype, \
                         size_t root, \
                         ccl::stream::impl_value_t& stream, \
                         const ccl::broadcast_attr& attr, \
                         const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->broadcast_impl(buf, count, dtype, root, stream, attr, deps); \
    } \
\
    ccl::request_t reduce(const void* send_buf, \
                          void* recv_buf, \
                          size_t count, \
                          ccl::datatype dtype, \
                          ccl::reduction reduction, \
                          size_t root, \
                          ccl::stream::impl_value_t& stream, \
                          const ccl::reduce_attr& attr, \
                          const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->reduce_impl( \
            send_buf, recv_buf, count, dtype, reduction, root, stream, attr, deps); \
    } \
\
    ccl::request_t reduce_scatter(const void* send_buf, \
                                  void* recv_buf, \
                                  size_t recv_count, \
                                  ccl::datatype dtype, \
                                  ccl::reduction reduction, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::reduce_scatter_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->reduce_scatter_impl( \
            send_buf, recv_buf, recv_count, dtype, reduction, stream, attr, deps); \
    }

#define DEVICE_COMM_INTERFACE_SPARSE_DEFINITION__VOID \
\
    ccl::request_t sparse_allreduce(const void* send_ind_buf, \
                                    size_t send_ind_count, \
                                    const void* send_val_buf, \
                                    size_t send_val_count, \
                                    void* recv_ind_buf, \
                                    size_t recv_ind_count, \
                                    void* recv_val_buf, \
                                    size_t recv_val_count, \
                                    ccl::datatype index_dtype, \
                                    ccl::datatype value_dtype, \
                                    ccl::reduction reduction, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::sparse_allreduce_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->sparse_allreduce_impl(send_ind_buf, \
                                                 send_ind_count, \
                                                 send_val_buf, \
                                                 send_val_count, \
                                                 recv_ind_buf, \
                                                 recv_ind_count, \
                                                 recv_val_buf, \
                                                 recv_val_count, \
                                                 index_dtype, \
                                                 value_dtype, \
                                                 reduction, \
                                                 stream, \
                                                 attr, \
                                                 deps); \
    }

#define DEVICE_COMM_INTERFACE_COLL_DEFINITION(type) \
\
    ccl::request_t allgatherv(const type* send_buf, \
                              size_t send_count, \
                              type* recv_buf, \
                              const ccl::vector_class<size_t>& recv_counts, \
                              ccl::stream::impl_value_t& stream, \
                              const ccl::allgatherv_attr& attr, \
                              const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allgatherv_impl( \
            send_buf, send_count, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t allgatherv(const type* send_buf, \
                              size_t send_count, \
                              ccl::vector_class<type*>& recv_buf, \
                              const ccl::vector_class<size_t>& recv_counts, \
                              ccl::stream::impl_value_t& stream, \
                              const ccl::allgatherv_attr& attr, \
                              const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allgatherv_impl( \
            send_buf, send_count, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t allreduce(const type* send_buf, \
                             type* recv_buf, \
                             size_t count, \
                             ccl::reduction reduction, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::allreduce_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allreduce_impl( \
            send_buf, recv_buf, count, reduction, stream, attr, deps); \
    } \
\
    ccl::request_t alltoall(const type* send_buf, \
                            type* recv_buf, \
                            size_t count, \
                            ccl::stream::impl_value_t& stream, \
                            const ccl::alltoall_attr& attr, \
                            const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->alltoall_impl(send_buf, recv_buf, count, stream, attr, deps); \
    } \
    ccl::request_t alltoall(const ccl::vector_class<type*>& send_buf, \
                            const ccl::vector_class<type*>& recv_buf, \
                            size_t count, \
                            ccl::stream::impl_value_t& stream, \
                            const ccl::alltoall_attr& attr, \
                            const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->alltoall_impl(send_buf, recv_buf, count, stream, attr, deps); \
    } \
\
    ccl::request_t alltoallv(const type* send_buf, \
                             const ccl::vector_class<size_t>& send_counts, \
                             type* recv_buf, \
                             const ccl::vector_class<size_t>& recv_counts, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::alltoallv_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->alltoallv_impl( \
            send_buf, send_counts, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t alltoallv(const ccl::vector_class<type*>& send_bufs, \
                             const ccl::vector_class<size_t>& send_counts, \
                             const ccl::vector_class<type*>& recv_bufs, \
                             const ccl::vector_class<size_t>& recv_counts, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::alltoallv_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->alltoallv_impl( \
            send_bufs, send_counts, recv_bufs, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t bcast(type* buf, \
                         size_t count, \
                         size_t root, \
                         ccl::stream::impl_value_t& stream, \
                         const ccl::broadcast_attr& attr, \
                         const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->broadcast_impl(buf, count, root, stream, attr, deps); \
    } \
\
    ccl::request_t reduce(const type* send_buf, \
                          type* recv_buf, \
                          size_t count, \
                          ccl::reduction reduction, \
                          size_t root, \
                          ccl::stream::impl_value_t& stream, \
                          const ccl::reduce_attr& attr, \
                          const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->reduce_impl( \
            send_buf, recv_buf, count, reduction, root, stream, attr, deps); \
    } \
\
    ccl::request_t reduce_scatter(const type* send_buf, \
                                  type* recv_buf, \
                                  size_t recv_count, \
                                  ccl::reduction reduction, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::reduce_scatter_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->reduce_scatter_impl( \
            send_buf, recv_buf, recv_count, reduction, stream, attr, deps); \
    }

#define DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(index_type, value_type) \
\
    ccl::request_t sparse_allreduce(const index_type* send_ind_buf, \
                                    size_t send_ind_count, \
                                    const value_type* send_val_buf, \
                                    size_t send_val_count, \
                                    index_type* recv_ind_buf, \
                                    size_t recv_ind_count, \
                                    value_type* recv_val_buf, \
                                    size_t recv_val_count, \
                                    ccl::reduction reduction, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::sparse_allreduce_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->sparse_allreduce_impl(send_ind_buf, \
                                                 send_ind_count, \
                                                 send_val_buf, \
                                                 send_val_count, \
                                                 recv_ind_buf, \
                                                 recv_ind_count, \
                                                 recv_val_buf, \
                                                 recv_val_count, \
                                                 reduction, \
                                                 stream, \
                                                 attr, \
                                                 deps); \
    }

#define DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(type) \
\
    ccl::request_t allgatherv(const type& send_buf, \
                              size_t send_count, \
                              type& recv_buf, \
                              const ccl::vector_class<size_t>& recv_counts, \
                              ccl::stream::impl_value_t& stream, \
                              const ccl::allgatherv_attr& attr, \
                              const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allgatherv_impl( \
            send_buf, send_count, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t allgatherv(const type& send_buf, \
                              size_t send_count, \
                              ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_buf, \
                              const ccl::vector_class<size_t>& recv_counts, \
                              ccl::stream::impl_value_t& stream, \
                              const ccl::allgatherv_attr& attr, \
                              const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allgatherv_impl( \
            send_buf, send_count, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t allreduce(const type& send_buf, \
                             type& recv_buf, \
                             size_t count, \
                             ccl::reduction reduction, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::allreduce_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->allreduce_impl( \
            send_buf, recv_buf, count, reduction, stream, attr, deps); \
    } \
\
    ccl::request_t alltoall(const type& send_buf, \
                            type& recv_buf, \
                            size_t count, \
                            ccl::stream::impl_value_t& stream, \
                            const ccl::alltoall_attr& attr, \
                            const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->alltoall_impl(send_buf, recv_buf, count, stream, attr, deps); \
    } \
    ccl::request_t alltoall(const ccl::vector_class<ccl::reference_wrapper_class<type>>& send_buf, \
                            const ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_buf, \
                            size_t count, \
                            ccl::stream::impl_value_t& stream, \
                            const ccl::alltoall_attr& attr, \
                            const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->alltoall_impl(send_buf, recv_buf, count, stream, attr, deps); \
    } \
\
    ccl::request_t alltoallv(const type& send_buf, \
                             const ccl::vector_class<size_t>& send_counts, \
                             type& recv_buf, \
                             const ccl::vector_class<size_t>& recv_counts, \
                             ccl::stream::impl_value_t& stream, \
                             const ccl::alltoallv_attr& attr, \
                             const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->alltoallv_impl( \
            send_buf, send_counts, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t alltoallv( \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->alltoallv_impl( \
            send_buf, send_counts, recv_buf, recv_counts, stream, attr, deps); \
    } \
\
    ccl::request_t bcast(type& buf, \
                         size_t count, \
                         size_t root, \
                         ccl::stream::impl_value_t& stream, \
                         const ccl::broadcast_attr& attr, \
                         const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->broadcast_impl(buf, count, root, stream, attr, deps); \
    } \
\
    ccl::request_t reduce(const type& send_buf, \
                          type& recv_buf, \
                          size_t count, \
                          ccl::reduction reduction, \
                          size_t root, \
                          ccl::stream::impl_value_t& stream, \
                          const ccl::reduce_attr& attr, \
                          const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->reduce_impl( \
            send_buf, recv_buf, count, reduction, root, stream, attr, deps); \
    } \
\
    ccl::request_t reduce_scatter(const type& send_buf, \
                                  type& recv_buf, \
                                  size_t recv_count, \
                                  ccl::reduction reduction, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::reduce_scatter_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps) override { \
        return get_impl()->reduce_scatter_impl( \
            send_buf, recv_buf, recv_count, reduction, stream, attr, deps); \
    }

#define DEVICE_COMM_INTERFACE_SPARSE_CLASS_DEFINITION(index_type, value_type) \
\
    ccl::request_t sparse_allreduce(const index_type& send_ind_buf, \
                                    size_t send_ind_count, \
                                    const value_type& send_val_buf, \
                                    size_t send_val_count, \
                                    index_type& recv_ind_buf, \
                                    size_t recv_ind_count, \
                                    value_type& recv_val_buf, \
                                    size_t recv_val_count, \
                                    ccl::reduction reduction, \
                                    ccl::stream::impl_value_t& stream, \
                                    const ccl::sparse_allreduce_attr& attr, \
                                    const ccl::vector_class<ccl::event>& deps = {}) override { \
        return get_impl()->sparse_allreduce_impl(send_ind_buf, \
                                                 send_ind_count, \
                                                 send_val_buf, \
                                                 send_val_count, \
                                                 recv_ind_buf, \
                                                 recv_ind_count, \
                                                 recv_val_buf, \
                                                 recv_val_count, \
                                                 reduction, \
                                                 stream, \
                                                 attr, \
                                                 deps); \
    }

/**
 * Coll implementations
 */
#define DEVICE_COMM_IMPL_DECLARATION \
    ccl::request_t allgatherv_impl(const void* send_buf, \
                                   size_t send_count, \
                                   void* recv_buf, \
                                   const ccl::vector_class<size_t>& recv_counts, \
                                   ccl::datatype dtype, \
                                   ccl::stream::impl_value_t& stream, \
                                   const ccl::allgatherv_attr& attr, \
                                   const ccl::vector_class<ccl::event>& deps); \
    ccl::request_t allgatherv_impl(const void* send_buf, \
                                   size_t send_count, \
                                   const ccl::vector_class<void*>& recv_bufs, \
                                   const ccl::vector_class<size_t>& recv_counts, \
                                   ccl::datatype dtype, \
                                   ccl::stream::impl_value_t& stream, \
                                   const ccl::allgatherv_attr& attr, \
                                   const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t allgatherv_impl(const buffer_type* send_buf, \
                                   size_t send_count, \
                                   buffer_type* recv_buf, \
                                   const ccl::vector_class<size_t>& recv_counts, \
                                   ccl::stream::impl_value_t& stream, \
                                   const ccl::allgatherv_attr& attr, \
                                   const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t allgatherv_impl(const buffer_type* send_buf, \
                                   size_t send_count, \
                                   ccl::vector_class<buffer_type*>& recv_buf, \
                                   const ccl::vector_class<size_t>& recv_counts, \
                                   ccl::stream::impl_value_t& stream, \
                                   const ccl::allgatherv_attr& attr, \
                                   const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t allreduce_impl(const void* send_buf, \
                                  void* recv_buf, \
                                  size_t count, \
                                  ccl::datatype dtype, \
                                  ccl::reduction reduction, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::allreduce_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t allreduce_impl(const buffer_type* send_buf, \
                                  buffer_type* recv_buf, \
                                  size_t count, \
                                  ccl::reduction reduction, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::allreduce_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t alltoall_impl(const void* send_buf, \
                                 void* recv_buf, \
                                 size_t count, \
                                 ccl::datatype dtype, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::alltoall_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps); \
    ccl::request_t alltoall_impl(const ccl::vector_class<void*>& send_buf, \
                                 const ccl::vector_class<void*>& recv_buf, \
                                 size_t count, \
                                 ccl::datatype dtype, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::alltoall_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t alltoall_impl(const buffer_type* send_buf, \
                                 buffer_type* recv_buf, \
                                 size_t count, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::alltoall_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t alltoall_impl(const ccl::vector_class<buffer_type*>& send_buf, \
                                 const ccl::vector_class<buffer_type*>& recv_buf, \
                                 size_t count, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::alltoall_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t alltoallv_impl(const void* send_buf, \
                                  const ccl::vector_class<size_t>& send_counts, \
                                  void* recv_buf, \
                                  const ccl::vector_class<size_t>& recv_counts, \
                                  ccl::datatype dtype, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::alltoallv_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
    ccl::request_t alltoallv_impl(const ccl::vector_class<void*>& send_buf, \
                                  const ccl::vector_class<size_t>& send_counts, \
                                  ccl::vector_class<void*> recv_buf, \
                                  const ccl::vector_class<size_t>& recv_counts, \
                                  ccl::datatype dtype, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::alltoallv_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t alltoallv_impl(const ccl::vector_class<buffer_type*>& send_buf, \
                                  const ccl::vector_class<size_t>& send_counts, \
                                  const ccl::vector_class<buffer_type*>& recv_buf, \
                                  const ccl::vector_class<size_t>& recv_counts, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::alltoallv_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t alltoallv_impl(const buffer_type* send_buf, \
                                  const ccl::vector_class<size_t>& send_counts, \
                                  buffer_type* recv_buf, \
                                  const ccl::vector_class<size_t>& recv_counts, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::alltoallv_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t broadcast_impl(void* buf, \
                                  size_t count, \
                                  ccl::datatype dtype, \
                                  size_t root, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::broadcast_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t broadcast_impl(buffer_type* buf, \
                                  size_t count, \
                                  size_t root, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::broadcast_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t reduce_impl(const void* send_buf, \
                               void* recv_buf, \
                               size_t count, \
                               ccl::datatype dtype, \
                               ccl::reduction reduction, \
                               size_t root, \
                               ccl::stream::impl_value_t& stream, \
                               const ccl::reduce_attr& attr, \
                               const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t reduce_impl(const buffer_type* send_buf, \
                               buffer_type* recv_buf, \
                               size_t count, \
                               ccl::reduction reduction, \
                               size_t root, \
                               ccl::stream::impl_value_t& stream, \
                               const ccl::reduce_attr& attr, \
                               const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t reduce_scatter_impl(const void* send_buf, \
                                       void* recv_buf, \
                                       size_t recv_count, \
                                       ccl::datatype dtype, \
                                       ccl::reduction reduction, \
                                       ccl::stream::impl_value_t& stream, \
                                       const ccl::reduce_scatter_attr& attr, \
                                       const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t reduce_scatter_impl(const buffer_type* send_buf, \
                                       buffer_type* recv_buf, \
                                       size_t recv_count, \
                                       ccl::reduction reduction, \
                                       ccl::stream::impl_value_t& stream, \
                                       const ccl::reduce_scatter_attr& attr, \
                                       const ccl::vector_class<ccl::event>& deps);

#define DEVICE_COMM_IMPL_SPARSE_DECLARATION \
    ccl::request_t sparse_allreduce_impl(const void* send_ind_buf, \
                                         size_t send_ind_count, \
                                         const void* send_val_buf, \
                                         size_t send_val_count, \
                                         void* recv_ind_buf, \
                                         size_t recv_ind_count, \
                                         void* recv_val_buf, \
                                         size_t recv_val_count, \
                                         ccl::datatype index_dtype, \
                                         ccl::datatype value_dtype, \
                                         ccl::reduction reduction, \
                                         ccl::stream::impl_value_t& stream, \
                                         const ccl::sparse_allreduce_attr& attr, \
                                         const ccl::vector_class<ccl::event>& deps); \
    template <class index_type, class value_type> \
    ccl::request_t sparse_allreduce_impl(const index_type* send_ind_buf, \
                                         size_t send_ind_count, \
                                         const value_type* send_val_buf, \
                                         size_t send_val_count, \
                                         index_type* recv_ind_buf, \
                                         size_t recv_ind_count, \
                                         value_type* recv_val_buf, \
                                         size_t recv_val_count, \
                                         ccl::reduction reduction, \
                                         ccl::stream::impl_value_t& stream, \
                                         const ccl::sparse_allreduce_attr& attr, \
                                         const ccl::vector_class<ccl::event>& deps);

#define DEVICE_COMM_IMPL_CLASS_DECLARATION \
    template <class buffer_type> \
    ccl::request_t allgatherv_impl(const buffer_type& send_buf, \
                                   size_t send_count, \
                                   buffer_type& recv_buf, \
                                   const ccl::vector_class<size_t>& recv_counts, \
                                   ccl::stream::impl_value_t& stream, \
                                   const ccl::allgatherv_attr& attr, \
                                   const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t allgatherv_impl( \
        const buffer_type& send_buf, \
        size_t send_count, \
        ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t allreduce_impl(const buffer_type& send_buf, \
                                  buffer_type& recv_buf, \
                                  size_t count, \
                                  ccl::reduction reduction, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::allreduce_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t alltoall_impl(const buffer_type& send_buf, \
                                 buffer_type& recv_buf, \
                                 size_t count, \
                                 ccl::stream::impl_value_t& stream, \
                                 const ccl::alltoall_attr& attr, \
                                 const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t alltoall_impl( \
        const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf, \
        const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf, \
        size_t count, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t alltoallv_impl(const buffer_type& send_buf, \
                                  const ccl::vector_class<size_t>& send_counts, \
                                  buffer_type& recv_buf, \
                                  const ccl::vector_class<size_t>& recv_counts, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::alltoallv_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t alltoallv_impl( \
        const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t broadcast_impl(buffer_type& buf, \
                                  size_t count, \
                                  size_t root, \
                                  ccl::stream::impl_value_t& stream, \
                                  const ccl::broadcast_attr& attr, \
                                  const ccl::vector_class<ccl::event>& deps); \
    template <class buffer_type> \
    ccl::request_t reduce_impl(const buffer_type& send_buf, \
                               buffer_type& recv_buf, \
                               size_t count, \
                               ccl::reduction reduction, \
                               size_t root, \
                               ccl::stream::impl_value_t& stream, \
                               const ccl::reduce_attr& attr, \
                               const ccl::vector_class<ccl::event>& deps); \
\
    template <class buffer_type> \
    ccl::request_t reduce_scatter_impl(const buffer_type& send_buf, \
                                       buffer_type& recv_buf, \
                                       size_t recv_count, \
                                       ccl::reduction reduction, \
                                       ccl::stream::impl_value_t& stream, \
                                       const ccl::reduce_scatter_attr& attr, \
                                       const ccl::vector_class<ccl::event>& deps);

#define DEVICE_COMM_IMPL_SPARSE_CLASS_DECLARATION \
    template <class index_type, class value_type> \
    ccl::request_t sparse_allreduce_impl(const index_type& send_ind_buf, \
                                         size_t send_ind_count, \
                                         const value_type& send_val_buf, \
                                         size_t send_val_count, \
                                         index_type& recv_ind_buf, \
                                         size_t recv_ind_count, \
                                         value_type& recv_val_buf, \
                                         size_t recv_val_count, \
                                         ccl::reduction reduction, \
                                         ccl::stream::impl_value_t& stream, \
                                         const ccl::sparse_allreduce_attr& attr, \
                                         const ccl::vector_class<ccl::event>& deps);

/**
 * Force intantiations
 */
#define DEVICE_COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(comm_class, type) \
    template ccl::request_t comm_class::allgatherv_impl( \
        const type& send_buf, \
        size_t send_count, \
        type& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::allreduce_impl(const type& send_buf, \
                                                       type& recv_buf, \
                                                       size_t count, \
                                                       ccl::reduction reduction, \
                                                       ccl::stream::impl_value_t& stream, \
                                                       const ccl::allreduce_attr& attr, \
                                                       const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::alltoall_impl(const type& send_buf, \
                                                      type& recv_buf, \
                                                      size_t count, \
                                                      ccl::stream::impl_value_t& stream, \
                                                      const ccl::alltoall_attr& attr, \
                                                      const ccl::vector_class<ccl::event>& deps); \
\
    ccl::request_t alltoall_impl( \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& send_buf, \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_buf, \
        size_t count, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::alltoallv_impl<type>( \
        const type& send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        type& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
    template ccl::request_t comm_class::alltoallv_impl<type>( \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<ccl::reference_wrapper_class<type>>& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::broadcast_impl(type& buf, \
                                                       size_t count, \
                                                       size_t root, \
                                                       ccl::stream::impl_value_t& stream, \
                                                       const ccl::broadcast_attr& attr, \
                                                       const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::reduce_impl(const type& send_buf, \
                                                    type& recv_buf, \
                                                    size_t count, \
                                                    ccl::reduction reduction, \
                                                    size_t root, \
                                                    ccl::stream::impl_value_t& stream, \
                                                    const ccl::reduce_attr& attr, \
                                                    const ccl::vector_class<ccl::event>& deps);

#define DEVICE_COMM_INTERFACE_COLL_INSTANTIATIONS(comm_class, type) \
\
    template ccl::request_t comm_class::allgatherv_impl( \
        const type* send_buf, \
        size_t send_count, \
        type* recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::allreduce_impl(const type* send_buf, \
                                                       type* recv_buf, \
                                                       size_t count, \
                                                       ccl::reduction reduction, \
                                                       ccl::stream::impl_value_t& stream, \
                                                       const ccl::allreduce_attr& attr, \
                                                       const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::alltoall_impl(const type* send_buf, \
                                                      type* recv_buf, \
                                                      size_t count, \
                                                      ccl::stream::impl_value_t& stream, \
                                                      const ccl::alltoall_attr& attr, \
                                                      const ccl::vector_class<ccl::event>& deps); \
    template ccl::request_t comm_class::alltoall_impl(const ccl::vector_class<type*>& send_buf, \
                                                      const ccl::vector_class<type*>& recv_buf, \
                                                      size_t count, \
                                                      ccl::stream::impl_value_t& stream, \
                                                      const ccl::alltoall_attr& attr, \
                                                      const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::alltoallv_impl<type>( \
        const type* send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        type* recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
    template ccl::request_t comm_class::alltoallv_impl<type>( \
        const ccl::vector_class<type*>& send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<type*>& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream::impl_value_t& stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::broadcast_impl(type* buf, \
                                                       size_t count, \
                                                       size_t root, \
                                                       ccl::stream::impl_value_t& stream, \
                                                       const ccl::broadcast_attr& attr, \
                                                       const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::request_t comm_class::reduce_impl(const type* send_buf, \
                                                    type* recv_buf, \
                                                    size_t count, \
                                                    ccl::reduction reduction, \
                                                    size_t root, \
                                                    ccl::stream::impl_value_t& stream, \
                                                    const ccl::reduce_attr& attr, \
                                                    const ccl::vector_class<ccl::event>& deps);

#define DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION( \
    comm_class, index_type, value_type) \
    template ccl::request_t comm_class::sparse_allreduce_impl( \
        const index_type* send_ind_buf, \
        size_t send_ind_count, \
        const value_type* send_val_buf, \
        size_t send_val_count, \
        index_type* recv_ind_buf, \
        size_t recv_ind_count, \
        value_type* recv_val_buf, \
        size_t recv_val_count, \
        ccl::reduction reduction, \
        ccl::stream::impl_value_t& stream, \
        const ccl::sparse_allreduce_attr& attr, \
        const ccl::vector_class<ccl::event>& deps);

#define DEVICE_COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION( \
    comm_class, index_type, value_type) \
    template ccl::request_t comm_class::sparse_allreduce_impl( \
        const index_type& send_ind_buf, \
        size_t send_ind_count, \
        const value_type& send_val_buf, \
        size_t send_val_count, \
        index_type& recv_ind_buf, \
        size_t recv_ind_count, \
        value_type& recv_val_buf, \
        size_t recv_val_count, \
        ccl::reduction reduction, \
        ccl::stream::impl_value_t& stream, \
        const ccl::sparse_allreduce_attr& attr, \
        const ccl::vector_class<ccl::event>& deps);
