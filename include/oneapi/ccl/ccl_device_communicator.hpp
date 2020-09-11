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

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
namespace ccl {
class request;
class kvs_interface;
using rank_t = size_t;

struct communicator_interface;
/**
 * A device communicator that permits device communication operations
 * Has no defined public constructor.
 * Use ccl::environment::create_device_communicator for communicator objects creation.
 */
class device_communicator final : public ccl_api_base_movable<device_communicator,
                                                              direct_access_policy,
                                                              communicator_interface,
                                                              std::shared_ptr> {
public:
    using base_t = ccl_api_base_movable<device_communicator,
                                        direct_access_policy,
                                        communicator_interface,
                                        std::shared_ptr>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    /**
     * Type allows to get underlying device type,
     * which was used as communicator construction argument
     */
    using ccl_device_t = typename unified_device_type::ccl_native_t;

    /**
     * Declare communicator device context native type
     */
    using ccl_context_t = typename unified_device_context_type::ccl_native_t;

    using request_t = ccl::request_t;
    using coll_request_t = request_t;

    device_communicator(device_communicator&& src);
    device_communicator& operator=(device_communicator&& src);
    ~device_communicator();

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

    /**
     * Retrieves underlying device, which was used as communicator construction argument
     */
    ccl_device_t get_device();

    /**
     * Retrieves underlying context, which was used as communicator construction argument
     */
    ccl_context_t get_context();

    template <class... attr_value_pair_t>
    stream create_stream(attr_value_pair_t&&... avps) {
        // return stream::create_stream_from_attr(get_device(), get_context(), std::forward<attr_value_pair_t>(avps)...);
        throw;
    }

    device_communicator split(const device_comm_split_attr& attr);

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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t allgatherv(const void* send_buf,
                         size_t send_count,
                         void* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         stream op_stream = default_stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

    /**
     * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
     * @param send_count number of elements of type @c dtype in @c send_buf
     * @param recv_bufs [out] array of buffers to store gathered result, one buffer per each rank
     * @param recv_counts array with number of elements of type @c dtype to be received from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t allgatherv(const void* send_buf,
                         size_t send_count,
                         const vector_class<void*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         stream op_stream = default_stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});
    /**
     * Type safety version:
     * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
     * @param send_count number of elements of type @c BufferType in @c send_buf
     * @param recv_buf [out] the buffer to store gathered result, should be large enough to hold values from all ranks
     * @param recv_counts array with number of elements of type @c BufferType to be received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         BufferType* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         stream op_stream = default_stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
     * @param send_count number of elements of type @c BufferType in @c send_buf
     * @param recv_bufs [out] array of buffers to store gathered result, one buffer per each rank
     * @param recv_counts array with number of elements of type @c BufferType to be received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         vector_class<BufferType*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         stream op_stream = default_stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
     * @param send_count number of elements of type @c dtype in @c send_buf
     * @param recv_buf [out] the buffer to store gathered result, should be large enough to hold values from all ranks
     * @param recv_counts array with number of elements of type @c dtype to be received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t allgatherv(const BufferObjectType& send_buf,
                         size_t send_count,
                         BufferObjectType& recv_buf,
                         const vector_class<size_t>& recv_counts,
                         stream op_stream = default_stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
     * @param send_count number of elements of type @c dtype in @c send_buf
     * @param recv_bufs [out] array of buffers to store gathered result, one buffer per each rank
     * @param recv_counts array with number of elements of type @c dtype to be received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t allgatherv(const BufferObjectType& send_buf,
                         size_t send_count,
                         vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         stream op_stream = default_stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t allreduce(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        datatype dtype,
                        reduction rtype,
                        stream op_stream = default_stream,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c BufferType that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
     * @param count number of elements of type @c BufferType in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t allreduce(const BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t count,
                        reduction rtype,
                        stream op_stream = default_stream,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
     * @param count number of elements of type @c dtype in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t allreduce(const BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t count,
                        reduction rtype,
                        stream op_stream = default_stream,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoall(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       datatype dtype,
                       stream op_stream = default_stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

    /**
     * @param send_bufs array of buffers with local data to be sent, one buffer per each rank
     * @param recv_bufs [out] array of buffers to store received result, one buffer per each rank
     * @param count number of elements of type @c dtype to be send to or to received from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoall(const vector_class<void*>& send_buf,
                       const vector_class<void*>& recv_buf,
                       size_t count,
                       datatype dtype,
                       stream op_stream = default_stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c BufferType that stores local data to be sent
     * @param recv_buf [out] the buffer to store received result, should be large enough
     * to hold values from all ranks, i.e. at least @c comm_size * @c count
     * @param count number of elements to be send to or to received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoall(const BufferType* send_buf,
                       BufferType* recv_buf,
                       size_t count,
                       stream op_stream = default_stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_bufs array of buffers with local data to be sent, one buffer per each rank
     * @param recv_bufs [out] array of buffers to store received result, one buffer per each rank
     * @param count number of elements to be send to or to received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoall(const vector_class<BufferType*>& send_buf,
                       const vector_class<BufferType*>& recv_buf,
                       size_t count,
                       stream op_stream = default_stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be sent
     * @param recv_buf [out] the buffer to store received result, should be large enough
     * to hold values from all ranks, i.e. at least @c comm_size * @c count
     * @param count number of elements to be send to or to received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t alltoall(const BufferObjectType& send_buf,
                       BufferObjectType& recv_buf,
                       size_t count,
                       stream op_stream = default_stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_bufs array of buffers with local data to be sent, one buffer per each rank
     * @param recv_bufs [out] array of buffers to store received result, one buffer per each rank
     * @param count number of elements to be send to or to received from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
                       const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                       size_t count,
                       stream op_stream = default_stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

    /**
     * Alltoallv is a collective communication operation in which each rank
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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoallv(const void* send_buf,
                        const vector_class<size_t>& send_counts,
                        void* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        stream op_stream = default_stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

    /**
     * @param send_bufs array of buffers to store send blocks, one buffer per each rank
     * @param send_counts array with number of elements of type @c dtype in send blocks for each rank
     * @param recv_bufs [out] array of buffers to store receive blocks, one buffer per each rank
     * @param recv_counts array with number of elements of type @c dtype in receive blocks from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t alltoallv(const vector_class<void*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<void*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        stream op_stream = default_stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with elements of @c BufferType that stores local blocks to be sent to each rank
     * @param send_counts array with number of elements of type @c BufferType in send blocks for each rank
     * @param recv_buf [out] the buffer to store received result, should be large enough to hold blocks from all ranks
     * @param recv_counts array with number of elements of type @c BufferType in receive blocks from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoallv(const BufferType* send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferType* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        stream op_stream = default_stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_bufs array of buffers to store send blocks, one buffer per each rank
     * @param send_counts array with number of elements of type @c BufferType in send blocks for each rank
     * @param recv_bufs [out] array of buffers to store receive blocks, one buffer per each rank
     * @param recv_counts array with number of elements of type @c BufferType in receive blocks from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t alltoallv(const vector_class<BufferType*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<BufferType*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        stream op_stream = default_stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with elements of @c dtype that stores local blocks to be sent to each rank
     * @param send_counts array with number of elements of type @c dtype in send blocks for each rank
     * @param recv_buf [out] the buffer to store received result, should be large enough to hold blocks from all ranks
     * @param recv_counts array with number of elements of type @c dtype in receive blocks from each rank
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t alltoallv(const BufferObjectType& send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferObjectType& recv_buf,
                        const vector_class<size_t>& recv_counts,
                        stream op_stream = default_stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_bufs array of buffers to store send blocks, one buffer per each rank
     * @param send_counts array with number of elements of type @c dtype in send blocks for each rank
     * @param recv_bufs [out] array of buffers to store receive blocks, one buffer per each rank
     * @param recv_counts array with number of elements of type @c dtype in receive blocks from each rank
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        stream op_stream = default_stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

    /**
     * Barrier synchronization across all ranks of communicator.
     * Completes after all ranks in the communicator have called it.
     *
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t barrier(stream op_stream = default_stream,
                      const barrier_attr& attr = default_barrier_attr,
                      const vector_class<event>& deps = {});

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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t broadcast(void* buf,
                        size_t count,
                        datatype dtype,
                        size_t root,
                        stream op_stream = default_stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param buf [in,out] the buffer with @c count elements of @c BufferType
     * serves as send buffer for root and as receive buffer for other ranks
     * @param count number of elements of type @c BufferType in @c buf
     * @param root the rank that broadcasts @c buf
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t broadcast(BufferType* buf,
                        size_t count,
                        size_t root,
                        stream op_stream = default_stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param buf [in,out] the buffer with @c count elements of @c dtype
     * serves as send buffer for root and as receive buffer for other ranks
     * @param count number of elements of type @c dtype in @c buf
     * @param root the rank that broadcasts @c buf
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t broadcast(BufferObjectType& buf,
                        size_t count,
                        size_t root,
                        stream op_stream = default_stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t reduce(const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction rtype,
                     size_t root,
                     stream op_stream = default_stream,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c BufferType that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf.
     * Used by the @c root rank only, ignored by other ranks.
     * @param count number of elements of type @c BufferType in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param root the rank that gets the result of reduction
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
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
                     stream op_stream = default_stream,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf.
     * Used by the @c root rank only, ignored by other ranks.
     * @param count number of elements of type @c dtype in @c send_buf and @c recv_buf
     * @param dtype datatype of elements in @c send_buf and @c recv_buf
     * @param rtype type of reduction operation to be applied
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t reduce(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t count,
                     reduction rtype,
                     size_t root,
                     stream op_stream = default_stream,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

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
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    request_t reduce_scatter(const void* send_buf,
                             void* recv_buf,
                             size_t recv_count,
                             datatype dtype,
                             reduction rtype,
                             stream op_stream = default_stream,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c comm_size * @c count elements of @c BufferType that stores local data to be reduced
     * @param recv_buf [out] the buffer to store result block containing @c recv_count elements of type @c BufferType
     * @param recv_count number of elements of type @c BufferType in receive block
     * @param rtype type of reduction operation to be applied
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferType,
              class = typename std::enable_if<ccl::is_native_type_supported<BufferType>(),
                                              request_t>::type>
    request_t reduce_scatter(const BufferType* send_buf,
                             BufferType* recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             stream op_stream = default_stream,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

    /**
     * Type safety version:
     * @param send_buf the buffer with @c comm_size * @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store result block containing @c recv_count elements of type @c dtype
     * @param recv_count number of elements of type @c dtype in receive block
     * @param rtype type of reduction operation to be applied
     * @param op_stream op_stream associated with the operation
     * @param attr optional attributes to customize operation
     * @param deps optional vector of events that the operation should depend on
     * @return @ref ccl::request_t object to track the progress of the operation
     */
    template <class BufferObjectType,
              class = typename std::enable_if<ccl::is_class_supported<BufferObjectType>(),
                                              request_t>::type>
    request_t reduce_scatter(const BufferObjectType& send_buf,
                             BufferObjectType& recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             stream op_stream = default_stream,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

private:
    friend class environment;
    friend class comm_group;

    device_communicator(impl_value_t&& impl);

    // factory methods
    template <class DeviceType, class ContextType>
    static vector_class<device_communicator> create_device_communicators(
        const size_t comm_size,
        const vector_class<DeviceType>& local_devices,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs);

    template <class DeviceType, class ContextType>
    static vector_class<device_communicator> create_device_communicators(
        const size_t comm_size,
        const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs);

    template <class DeviceType, class ContextType>
    static vector_class<device_communicator> create_device_communicators(
        const size_t comm_size,
        const map_class<rank_t, DeviceType>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs);
};

} // namespace ccl
#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
