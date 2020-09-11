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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

class request;
class kvs_interface;
class host_communicator;

/**
 * A host communicator that permits communication operations
 * Has no defined public constructor.
 * Use ccl::environment::create_communicator for communicator objects creation.
 */
class communicator final
        : public ccl_api_base_movable<communicator, direct_access_policy, host_communicator> {
public:
    using request_t = ccl::request_t;
    using coll_request_t = request_t;

    using base_t = ccl_api_base_movable<communicator, direct_access_policy, host_communicator>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    communicator() = delete;
    communicator(communicator& other) = delete;
    communicator(communicator&& other);
    communicator operator=(communicator& other) = delete;
    communicator& operator=(communicator&& other);
    ~communicator() noexcept;

    /**
     * Retrieves the rank in a communicator
     * @return rank corresponding to communicator object
     */
    size_t rank() const;

    /**
     * Retrieves the number of rank in a communicator
     * @return number of the ranks
     */
    size_t size() const;

    communicator split(const comm_split_attr& attr);

    /**
     * Allgatherv is a collective communication operation that collects data from all ranks within a communicator
     * into a single buffer or vector of buffers, one per rank. Different ranks can contribute segments of different sizes.
     * The resulting data in the output buffer(s) must be the same for each rank.
     */

    /**
     * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
     * @param send_count number of elements of type @c dtype in @c send_buf
     * @param recv_buf [out] the buffer to store gathered result, should be large enough to hold values from all ranks
     * @param recv_counts array with number of elements of type @c dtype to be received from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t allgatherv(const void* send_buf,
                         size_t send_count,
                         void* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const allgatherv_attr& attr = default_allgatherv_attr);

    /**
     * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
     * @param send_count number of elements of type @c dtype in @c send_buf
     * @param recv_bufs [out] array of buffers to store gathered result, one buffer per each rank
     * @param recv_counts array with number of elements of type @c dtype to be received from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t allgatherv(const void* send_buf,
                         size_t send_count,
                         const vector_class<void*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const allgatherv_attr& attr = default_allgatherv_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
     * @param send_count number of elements of type @c BufferType in @c send_buf
     * @param recv_buf [out] the buffer to store gathered result, should be large enough to hold values from all ranks
     * @param recv_counts array with number of elements of type @c BufferType to be received from each rank
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         BufferType* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         const allgatherv_attr& attr = default_allgatherv_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
     * @param send_count number of elements of type @c BufferType in @c send_buf
     * @param recv_bufs [out] array of buffers to store gathered result, one buffer per each rank
     * @param recv_counts array with number of elements of type @c BufferType to be received from each rank
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         const vector_class<BufferType*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         const allgatherv_attr& attr = default_allgatherv_attr);

    /**
     * Allreduce is a collective communication operation that makes global reduction operation
     * on values from all ranks of communicator and distributes result back to all ranks.
     */

    /**
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
     * @param count number of elements of type @c dtype in @c send_buf and @c recv_buf
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t allreduce(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        datatype dtype,
                        reduction rtype,
                        const allreduce_attr& attr = default_allreduce_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c BufferType that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
     * @param count number of elements of type @c BufferType in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t allreduce(const BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t count,
                        reduction rtype,
                        const allreduce_attr& attr = default_allreduce_attr);

    /**
     * Alltoall is a collective communication operation in which each rank
     * sends distinct equal-sized blocks of data to each rank.
     * The j-th block of @c send_buf sent from the i-th rank is received by the j-th rank
     * and is placed in the i-th block of @c recvbuf.
     */

    /**
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be sent
     * @param recv_buf [out] the buffer to store received result, should be large enough
     * to hold values from all ranks, i.e. at least @c comm_size * @c count
     * @param count number of elements of type @c dtype to be send to or to received from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoall(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       datatype dtype,
                       const alltoall_attr& attr = default_alltoall_attr);

    /**
     * @param send_bufs array of buffers with local data to be sent, one buffer per each rank
     * @param recv_bufs [out] array of buffers to store received result, one buffer per each rank
     * @param count number of elements of type @c dtype to be send to or to received from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoall(const vector_class<void*>& send_buf,
                       const vector_class<void*>& recv_buf,
                       size_t count,
                       datatype dtype,
                       const alltoall_attr& attr = default_alltoall_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c BufferType that stores local data to be sent
     * @param recv_buf [out] the buffer to store received result, should be large enough
     * to hold values from all ranks, i.e. at least @c comm_size * @c count
     * @param count number of elements to be send to or to received from each rank
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoall(const BufferType* send_buf,
                       BufferType* recv_buf,
                       size_t count,
                       const alltoall_attr& attr = default_alltoall_attr);

    /**
     * Type safety version:
     * @param send_bufs array of buffers with local data to be sent, one buffer per each rank
     * @param recv_bufs [out] array of buffers to store received result, one buffer per each rank
     * @param count number of elements to be send to or to received from each rank
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoall(const vector_class<BufferType*>& send_buf,
                       const vector_class<BufferType*>& recv_buf,
                       size_t count,
                       const alltoall_attr& attr = default_alltoall_attr);

    /**
     * Alltoall is a collective communication operation in which each rank
     * sends distinct blocks of data to each rank. Block sizes may differ.
     * The j-th block of @c send_buf sent from the i-th rank is received by the j-th rank
     * and is placed in the i-th block of @c recvbuf.
     */

    /**
     * @param send_buf the buffer with elements of @c dtype that stores local blocks to be sent to each rank
     * @param send_counts array with number of elements of type @c dtype in send blocks for each rank
     * @param recv_buf [out] the buffer to store received result, should be large enough to hold blocks from all ranks
     * @param recv_counts array with number of elements of type @c dtype in receive blocks from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoallv(const void* send_buf,
                        const vector_class<size_t>& send_counts,
                        void* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        const alltoallv_attr& attr = default_alltoallv_attr);

    /**
     * @param send_bufs array of buffers to store send blocks, one buffer per each rank
     * @param send_counts array with number of elements of type @c dtype in send blocks for each rank
     * @param recv_bufs [out] array of buffers to store receive blocks, one buffer per each rank
     * @param recv_counts array with number of elements of type @c dtype in receive blocks from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoallv(const vector_class<void*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<void*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        const alltoallv_attr& attr = default_alltoallv_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with elements of @c BufferType that stores local blocks to be sent to each rank
     * @param send_counts array with number of elements of type @c BufferType in send blocks for each rank
     * @param recv_buf [out] the buffer to store received result, should be large enough to hold blocks from all ranks
     * @param recv_counts array with number of elements of type @c BufferType in receive blocks from each rank
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoallv(const BufferType* send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferType* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        const alltoallv_attr& attr = default_alltoallv_attr);

    /**
     * Type safety version:
     * @param send_bufs array of buffers to store send blocks, one buffer per each rank
     * @param send_counts array with number of elements of type @c BufferType in send blocks for each rank
     * @param recv_bufs [out] array of buffers to store receive blocks, one buffer per each rank
     * @param recv_counts array with number of elements of type @c BufferType in receive blocks from each rank
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoallv(const vector_class<BufferType*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<BufferType*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        const alltoallv_attr& attr = default_alltoallv_attr);

    /**
     * Barrier synchronization across all ranks of communicator.
     * Completes after all ranks in the communicator have called it.
     *
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t barrier(const barrier_attr& attr = default_barrier_attr);

    /**
     * Broadcast is collective communication operation that broadcasts data
     * from one rank of communicator (denoted as root) to all other ranks.
     */

    /**
     * @param buf [in,out] the buffer with @c count elements of @c dtype
     * serves as send buffer for root and as receive buffer for other ranks
     * @param count number of elements of type @c dtype in @c buf
     * @param dtype datatype of elements in @c buf
     * @param root the rank that broadcasts @c buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t broadcast(void* buf,
                        size_t count,
                        datatype dtype,
                        size_t root,
                        const broadcast_attr& attr = default_broadcast_attr);

    /**
     * Type safety version:
     * @param buf [in,out] the buffer with @c count elements of @c BufferType
     * serves as send buffer for root and as receive buffer for other ranks
     * @param count number of elements of type @c BufferType in @c buf
     * @param root the rank that broadcasts @c buf
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t broadcast(BufferType* buf,
                        size_t count,
                        size_t root,
                        const broadcast_attr& attr = default_broadcast_attr);

    /**
     * Reduce is a collective communication operation that makes global reduction operation
     * on values from all ranks of communicator and returns result to root rank.
     */

    /**
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf.
     * Used by the @c root rank only, ignored by other ranks.
     * @param count number of elements of type @c dtype in @c send_buf and @c recv_buf
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param root the rank that gets the result of reduction
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t reduce(const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction rtype,
                     size_t root,
                     const reduce_attr& attr = default_reduce_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c BufferType that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf.
     * Used by the @c root rank only, ignored by other ranks.
     * @param count number of elements of type @c BufferType in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param root the rank that gets the result of reduction
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t reduce(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t count,
                     reduction rtype,
                     size_t root,
                     const reduce_attr& attr = default_reduce_attr);

    /**
     * Reduce-scatter is a collective communication operation that makes global reduction operation
     * on values from all ranks of communicator and scatters result in blocks back to all ranks.
     */

    /**
     * @param send_buf the buffer with @c comm_size * @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store result block containing @c recv_count elements of type @c dtype
     * @param recv_count number of elements of type @c dtype in receive block
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t reduce_scatter(const void* send_buf,
                             void* recv_buf,
                             size_t recv_count,
                             datatype dtype,
                             reduction rtype,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr);

    /**
     * Type safety version:
     * @param send_buf the buffer with @c comm_size * @c count elements of @c BufferType that stores local data to be reduced
     * @param recv_buf [out] the buffer to store result block containing @c recv_count elements of type @c BufferType
     * @param recv_count number of elements of type @c BufferType in receive block
     * @param rtype type of reduction operation to be applied
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t reduce_scatter(const BufferType* send_buf,
                             BufferType* recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr);

    /**
     * Sparse allreduce is a collective communication operation that makes global reduction operation
     * on sparse buffers from all ranks of communicator and distributes result back to all ranks.
     * Sparse buffers are defined by separate index and value buffers.
     *
     * WARNING: sparse_allreduce is currently considered experimental, so the API may change!
     */

    /**
     * @param send_ind_buf the buffer of indices with @c send_ind_count elements of type @c ind_dtype
     * @param send_int_count number of elements of type @c ind_type @c send_ind_buf
     * @param send_val_buf the buffer of values with @c send_val_count elements of type @c val_dtype
     * @param send_val_count number of elements of type @c val_type @c send_val_buf
     * @param recv_ind_buf [out] the buffer to store reduced indices, unused
     * @param recv_ind_count [out] number of elements in @c recv_ind_buf, unused
     * @param recv_val_buf [out] the buffer to store reduced values, unused
     * @param recv_val_count [out] number of elements in @c recv_val_buf, unused
     * @param ind_dtype datatype of elements in @c send_ind_buf and @c recv_ind_buf
     * @param val_dtype datatype of elements in @c send_val_buf and @c recv_val_buf
     * @param rtype type of reduction operation to be applied
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t sparse_allreduce(const void* send_ind_buf,
                               size_t send_ind_count,
                               const void* send_val_buf,
                               size_t send_val_count,
                               void* recv_ind_buf,
                               size_t recv_ind_count,
                               void* recv_val_buf,
                               size_t recv_val_count,
                               ccl::datatype ind_dtype,
                               ccl::datatype val_dtype,
                               reduction rtype,
                               const sparse_allreduce_attr& attr = default_sparse_allreduce_attr);

    /**
     * Type safety version:
     * @param send_ind_buf the buffer of indices with @c send_ind_count elements of type @c ind_dtype
     * @param send_int_count number of elements of type @c ind_type @c send_ind_buf
     * @param send_val_buf the buffer of values with @c send_val_count elements of type @c val_dtype
     * @param send_val_count number of elements of type @c val_type @c send_val_buf
     * @param recv_ind_buf [out] the buffer to store reduced indices, unused
     * @param recv_ind_count [out] number of elements in @c recv_ind_buf, unused
     * @param recv_val_buf [out] the buffer to store reduced values, unused
     * @param recv_val_count [out] number of elements in @c recv_val_buf, unused
     * @param rtype type of reduction operation to be applied
     * @param attr optional attributes to customize operation
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class IndexBufferType,
              class ValueBufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<ValueBufferType>(),
                                              request_t>::type>
    request_t sparse_allreduce(const IndexBufferType* send_ind_buf,
                               size_t send_ind_count,
                               const ValueBufferType* send_val_buf,
                               size_t send_val_count,
                               IndexBufferType* recv_ind_buf,
                               size_t recv_ind_count,
                               ValueBufferType* recv_val_buf,
                               size_t recv_val_count,
                               reduction rtype,
                               const sparse_allreduce_attr& attr = default_sparse_allreduce_attr);

private:
    friend class environment;
    explicit communicator(impl_value_t&& impl);

    static communicator create_communicator();
    static communicator create_communicator(const size_t size, shared_ptr_class<kvs_interface> kvs);
    static communicator create_communicator(const size_t size,
                                            const size_t rank,
                                            shared_ptr_class<kvs_interface> kvs);
}; // class communicator

} // namespace ccl
