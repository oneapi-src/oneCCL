/*
 Copyright 2016-2019 Intel Corporation
 
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
#include <memory>
#include <ostream>

#include "ccl_types.hpp"
#include "ccl_type_traits.hpp"

class ccl_comm;
class ccl_stream;

namespace ccl
{

class communicator;
class stream;

/**
 * Type @c communicator_t allows to operate communicator object in RAII manner
 */
using communicator_t = std::unique_ptr<ccl::communicator>;

/**
 * Type @c stream_t allows to operate stream object in RAII manner
 */
using stream_t = std::unique_ptr<ccl::stream>;

/**
 * ccl environment singleton
 */
class environment
{
public:
    ~environment();

    /**
     * Retrieves the unique ccl environment object
     * and makes the first-time initialization
     * of ccl library
     */
    static environment& instance();

    /**
     * Enables job scalability policy
     * @param callback of @c ccl_resize_fn_t type, which enables scalability policy
     * (@c nullptr enables default behavior)
     */
    void set_resize_fn(ccl_resize_fn_t callback);

    /**
     * Creates a new communicator according to @c attr parameters
     * or creates a copy of global communicator, if @c attr is @c nullptr(default)
     * @param attr
     */
    communicator_t create_communicator(const ccl::comm_attr* attr = nullptr) const;

    /**
     * Creates a new ccl stream of @c type with @c native stream
     * @param type the @c ccl::stream_type and may be @c cpu or @c sycl (if configured)
     * @param native_stream the existing handle of stream
     */
    stream_t create_stream(ccl::stream_type type = ccl::stream_type::cpu, void* native_stream = nullptr) const;

    /**
     * Retrieves the current version
     */
    ccl_version_t get_version() const;
private:
    environment();
};

/**
 * A request interface that allows the user to track collective operation progress
 */
class request
{
public:
    /**
     * Blocking wait for collective operation completion
     */
    virtual void wait() = 0;

    /**
     * Non-blocking check for collective operation completion
     * @retval true if the operations has been completed
     * @retval false if the operations has not been completed
     */
    virtual bool test() = 0;

    virtual ~request() = default;
};

/**
 * A stream object is an abstraction over CPU/GPU streams
 * Has no defined public constructor. Use ccl::environment::create_stream
 * for stream objects creation
 */
class stream
{
public:
    /**
     * stream is not copyable
     */
    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;

    /**
     * stream is movable
     */
    stream(stream&&) = default;
    stream& operator=(stream&&) = default;

private:
    friend class communicator;
    friend class environment;
    stream();
    stream(ccl::stream_type type, void* native_stream);

    std::shared_ptr<ccl_stream> stream_impl;
};

/**
 * A communicator that permits collective operations
 * Has no defined public constructor. Use ccl::environment::create_communicator
 * for communicator objects creation
 */
class communicator
{
public:

    /**
     * Type allows to operate request interface in RAII manner
     */
    using coll_request_t = std::unique_ptr<request>;

    /**
     * Retrieves the rank of the current process in a communicator
     * @return rank of the current process
     */
    size_t rank();

    /**
     * Retrieves the number of processes in a communicator
     * @return number of the processes
     */
    size_t size();

    /**
     * Gathers @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be gathered
     * @param send_count number of elements of type @c dtype in @c send_buf
     * @param recv_buf [out] the buffer to store gathered result on the @c each process, must have the same dimension
     * as @c buf. Used by the @c root process only, ignored by other processes
     * @param recv_counts array with number of elements received by each process
     * @param dtype data type of elements in the buffer @c buf and @c recv_buf
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    coll_request_t allgatherv(const void* send_buf, size_t send_count,
                              void* recv_buf, const size_t* recv_counts,
                              ccl::data_type dtype,
                              const ccl::coll_attr* attr = nullptr,
                              const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Gathers @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_buf the buffer with @c count elements of @c buffer_type that stores local data to be gathered
     * @param send_count number of elements of type @c buffer_type in @c send_buf
     * @param recv_buf [out] the buffer to store gathered result on the @c each process, must have the same dimension
     * as @c buf. Used by the @c root process only, ignored by other processes
     * @param recv_counts array with number of elements received by each process
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_type,
             class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t allgatherv(const buffer_type* send_buf, size_t send_count,
                              buffer_type* recv_buf, const size_t* recv_counts,
                              const ccl::coll_attr* attr = nullptr,
                              const ccl::stream_t& stream = ccl::stream_t());
    /**
     * Type safety version:
     * Gathers @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_buf the buffer of @c buffer_container_type with @c count elements that stores local data to be gathered
     * @param send_count number of elements in @c send_buf
     * @param recv_buf [out] the buffer of @c buffer_container_type to store gathered result on the @c each process, must have the same dimension
     * as @c buf. Used by the @c root process only, ignored by other processes
     * @param recv_counts array with number of elements received by each process
     * @param attr optional attributes that customize operation
     * @return @ref ccl::request object that can be used to track the progress of the operation
     */
    template<class buffer_container_type,
             class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t allgatherv(const buffer_container_type& send_buf, size_t send_count,
                              buffer_container_type& recv_buf, const size_t* recv_counts,
                              const ccl::coll_attr* attr = nullptr,
                              const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Reduces @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] - the buffer to store reduced result , must have the same dimension
     * as @c buf.
     * @param count number of elements of type @c dtype in @c buf
     * @param dtype data type of elements in the buffer @c buf and @c recv_buf
     * @param reduction type of reduction operation to be applied
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    coll_request_t allreduce(const void* send_buf, void* recv_buf,
                             size_t count, ccl::data_type dtype,
                             ccl::reduction reduction,
                             const ccl::coll_attr* attr = nullptr,
                             const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Reduces @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_buf the buffer with @c count elements of @c buffer_type that stores local data to be reduced
     * @param recv_buf [out] - the buffer to store reduced result , must have the same dimension
     * as @c buf.
     * @param count number of elements of type @c buffer_type in @c buf
     * @param reduction type of reduction operation to be applied
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_type,
             class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t allreduce(const buffer_type* send_buf,
                             buffer_type* recv_buf,
                             size_t count,
                             ccl::reduction reduction,
                             const ccl::coll_attr* attr = nullptr,
                             const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Reduces @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_buf the buffer of @c buffer_container_type with @c count elements that stores local data to be reduced
     * @param recv_buf [out] - the buffer of @c buffer_container_type to store reduced result, must have the same dimension
     * as @c buf.
     * @param count number of elements in @c send_buf
     * @param reduction type of reduction operation to be applied
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_container_type,
             class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t allreduce(const buffer_container_type& send_buf,
                             buffer_container_type& recv_buf,
                             size_t count,
                             ccl::reduction reduction,
                             const ccl::coll_attr* attr = nullptr,
                             const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Each process sends distinct data to each of the receivers. The j-th block sent from process i is received
     * by process j and is placed in the i-th block of recvbuf.
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data
     * @param recv_buf [out] - the buffer to store received result , must have the N * dimension
     * of @c buf, where N - communicator size.
     * @param count number of elements to send / receive from each process
     * @param dtype data type of elements in the buffer @c buf and @c recv_buf
     * @param attr optional attributes that customize operation
     * @return @ref ccl::request object that can be used to track the progress of the operation
     */
    coll_request_t alltoall(const void* send_buf, void* recv_buf,
                            size_t count, ccl::data_type dtype,
                            const ccl::coll_attr* attr = nullptr,
                            const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Each process sends distinct data to each of the receivers. The j-th block sent from process i is received
     * by process j and is placed in the i-th block of recvbuf.
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data
     * @param recv_buf [out] - the buffer to store received result , must have the N * dimension
     * of @c buf, where N - communicator size.
     * @param count number of elements to send / receive from each process
     * @param dtype data type of elements in the buffer @c buf and @c recv_buf
     * @param attr optional attributes that customize operation
     * @return @ref ccl::request object that can be used to track the progress of the operation
     */
    template<class buffer_type,
        class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t alltoall(const buffer_type* send_buf,
                            buffer_type* recv_buf,
                            size_t count,
                            const ccl::coll_attr* attr = nullptr,
                            const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Each process sends distinct data to each of the receivers. The j-th block sent from process i is received
     * by process j and is placed in the i-th block of recvbuf.
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data
     * @param recv_buf [out] - the buffer to store received result , must have the N * dimension
     * of @c buf, where N - communicator size.
     * @param count number of elements to send / receive from each process
     * @param dtype data type of elements in the buffer @c buf and @c recv_buf
     * @param attr optional attributes that customize operation
     * @return @ref ccl::request object that can be used to track the progress of the operation
     */
    template<class buffer_container_type,
        class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t alltoall(const buffer_container_type& send_buf,
                            buffer_container_type& recv_buf,
                            size_t count,
                            const ccl::coll_attr* attr = nullptr,
                            const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Collective operation that blocks each process until every process have reached it
     */
    void barrier(const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Broadcasts @c buf from the @c root process to other processes in a communicator
     * @param buf [in,out] the buffer with @c count elements of @c dtype to be transmitted
     * if the rank of the communicator is equal to @c root or to be received by other ranks
     * @param count number of elements of type @c dtype in @c buf
     * @param dtype data type of elements in the buffer @c buf
     * @param root the rank of the process that will transmit @c buf
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    coll_request_t bcast(void* buf, size_t count,
                         ccl::data_type dtype,
                         size_t root,
                         const ccl::coll_attr* attr = nullptr,
                         const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Broadcasts @c buf from the @c root process to other processes in a communicator
     * @param buf [in,out] the buffer with @c count elements of @c buffer_type to be transmitted
     * if the rank of the communicator is equal to @c root or to be received by other ranks
     * @param count number of elements of type @c buffer_type in @c buf
     * @param root the rank of the process that will transmit @c buf
     * @param attr optional attributes that customize operation
     * @return @ref coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_type,
             class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t bcast(buffer_type* buf, size_t count,
                         size_t root,
                         const ccl::coll_attr* attr = nullptr,
                         const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Broadcasts @c buf from the @c root process to other processes in a communicator
     * @param buf [in,out] the  buffer of @c buffer_container_type with @c count elements to be transmitted
     * if the rank of the communicator is equal to @c root or to be received by other ranks
     * @param count number of elements in @c buf
     * @param root the rank of the process that will transmit @c buf
     * @param attr optional attributes that customize operation
     * @return @ref coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_container_type,
             class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t bcast(buffer_container_type& buf, size_t count,
                         size_t root,
                         const ccl::coll_attr* attr = nullptr,
                         const ccl::stream_t& stream = ccl::stream_t());


    /**
     * Reduces @c buf on all process in the communicator and stores result in @c recv_buf
     * on the @c root process
     * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result on the @c root process, must have the same dimension
     * as @c buf. Used by the @c root process only, ignored by other processes
     * @param count number of elements of type @c dtype in @c buf
     * @param dtype data type of elements in the buffer @c buf and @c recv_buf
     * @param reduction type of reduction operation to be applied
     * @param root the rank of the process that will held result of reduction
     * @param attr optional attributes that customize operation
     * @return @ref coll_request_t object that can be used to track the progress of the operation
     */
    coll_request_t reduce(const void* send_buf, void* recv_buf,
                          size_t count,
                          ccl::data_type dtype,
                          ccl::reduction reduction,
                          size_t root,
                          const ccl::coll_attr* attr = nullptr,
                          const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Reduces @c buf on all process in the communicator and stores result in @c recv_buf
     * on the @c root process
     * @param send_buf the buffer with @c count elements of @c buffer_type that stores local data to be reduced
     * @param recv_buf [out] the buffer to store reduced result on the @c root process, must have the same dimension
     * as @c buf. Used by the @c root process only, ignored by other processes
     * @param count number of elements of type @c buffer_type in @c buf
     * @param reduction type of reduction operation to be applied
     * @param root the rank of the process that will held result of reduction
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_type,
             class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t reduce(const buffer_type* send_buf, buffer_type* recv_buf,
                          size_t count,
                          ccl::reduction reduction,
                          size_t root,
                          const ccl::coll_attr* attr = nullptr,
                          const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Reduces @c buf on all process in the communicator and stores result in @c recv_buf
     * on the @c root process
     * @param send_buf the buffer of @c buffer_container_type with @c count elements that stores local data to be reduced
     * @param recv_buf [out] the buffer of @c buffer_container_type to store reduced result on the @c root process, must have the same dimension
     * as @c buf. Used by the @c root process only, ignored by other processes
     * @param count number of elements of type @c buffer_type in @c buf
     * @param reduction type of reduction operation to be applied
     * @param root the rank of the process that will held result of reduction
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template<class buffer_container_type,
             class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t reduce(const buffer_container_type& send_buf, buffer_container_type& recv_buf,
                          size_t count,
                          ccl::reduction reduction,
                          size_t root,
                          const ccl::coll_attr* attr = nullptr,
                          const ccl::stream_t& stream = ccl::stream_t());


    /**
     * Reduces sparse @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_ind_buf the buffer of indices with @c send_ind_count elements of @c index_dtype
     * @param send_int_count number of elements of type @c index_type @c send_ind_buf
     * @param send_val_buf the buffer of values with @c send_val_count elements of @c value_dtype
     * @param send_val_count number of elements of type @c value_type @c send_val_buf
     * @param recv_ind_buf [out] the buffer to store reduced indices, must have the same dimension as @c send_ind_buf
     * @param recv_ind_count [out] the amount of reduced indices
     * @param recv_val_buf [out] the buffer to store reduced values, must have the same dimension as @c send_val_buf
     * @param recv_val_count [out] the amount of reduced values
     * @param index_dtype index type of elements in the buffer @c send_ind_buf and @c recv_ind_buf
     * @param value_dtype data type of elements in the buffer @c send_val_buf and @c recv_val_buf
     * @param reduction type of reduction operation to be applied
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    coll_request_t sparse_allreduce(const void* send_ind_buf, size_t send_ind_count,
                                    const void* send_val_buf, size_t send_val_count,
                                    void** recv_ind_buf, size_t* recv_ind_count,
                                    void** recv_val_buf, size_t* recv_val_count,
                                    ccl::data_type index_dtype,
                                    ccl::data_type value_dtype,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr = nullptr,
                                    const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Reduces sparse @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_ind_buf the buffer of indices with @c send_ind_count elements of @c index_buffer_type
     * @param send_int_count number of elements of type @c index_buffer_type @c send_ind_buf
     * @param send_val_buf the buffer of values with @c send_val_count elements of @c value_buffer_type
     * @param send_val_count number of elements of type @c value_buffer_type @c send_val_buf
     * @param recv_ind_buf [out] the buffer to store reduced indices, must have the same dimension as @c send_ind_buf
     * @param recv_ind_count [out] the amount of reduced indices
     * @param recv_val_buf [out] the buffer to store reduced values, must have the same dimension as @c send_val_buf
     * @param recv_val_count [out] the amount of reduced values
     * @param reduction type of reduction operation to be applied
     * @param attr optional attributes that customize operation
     * @return @ref ccl::request object that can be used to track the progress of the operation
     */
    template<class index_buffer_type,
             class value_buffer_type,
             class = typename std::enable_if<ccl::is_native_type_supported<value_buffer_type>()>::type>
    coll_request_t sparse_allreduce(const index_buffer_type* send_ind_buf, size_t send_ind_count,
                                    const value_buffer_type* send_val_buf, size_t send_val_count,
                                    index_buffer_type** recv_ind_buf, size_t* recv_ind_count,
                                    value_buffer_type** recv_val_buf, size_t* recv_val_count,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr = nullptr,
                                    const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Reduces sparse @c buf on all process in the communicator and stores result in @c recv_buf
     * on each process
     * @param send_ind_buf the buffer of @c index_buffer_container_type of indices with @c send_ind_count elements
     * @param send_int_count number of elements of @c send_ind_buf
     * @param send_val_buf the buffer of @c value_buffer_container_type of values with @c send_val_count elements
     * @param send_val_count number of elements of in @c send_val_buf
     * @param recv_ind_buf [out] the buffer of @c index_buffer_container_type to store reduced indices, must have the same dimension as @c send_ind_buf
     * @param recv_ind_count [out] the amount of reduced indices
     * @param recv_val_buf [out] the buffer of @c value_buffer_container_type to store reduced values, must have the same dimension as @c send_val_buf
     * @param recv_val_count [out] the amount of reduced values
     * @param reduction type of reduction operation to be applied
     * @param attr optional attributes that customize operation
     * @return @ref ccl::request object that can be used to track the progress of the operation
     */
    template<class index_buffer_container_type,
             class value_buffer_container_type,
             class = typename std::enable_if<ccl::is_class_supported<value_buffer_container_type>()>::type>
    coll_request_t sparse_allreduce(const index_buffer_container_type& send_ind_buf, size_t send_ind_count,
                                    const value_buffer_container_type& send_val_buf, size_t send_val_count,
                                    index_buffer_container_type** recv_ind_buf, size_t* recv_ind_count,
                                    value_buffer_container_type** recv_val_buf, size_t* recv_val_count,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr = nullptr,
                                    const ccl::stream_t& stream = ccl::stream_t());

private:
    friend class environment;

    /**
     * Creates ccl communicator as a copy of global communicator
     */
    communicator();

    /**
     * Creates a new communicator according to @c attr parameters
     * @param attr
     */
    explicit communicator(const ccl::comm_attr* attr);

    std::shared_ptr<ccl_comm> comm_impl;
};

}
