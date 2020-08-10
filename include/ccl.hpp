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
#include <memory>
#include <ostream>

#include "ccl_types.hpp"
#include "ccl_type_traits.hpp"

class ccl_comm;
class ccl_stream;

namespace ccl {

class communicator;
class stream;
struct communicator_interface;

struct host_attr_impl;
class ccl_host_attr;

#ifdef MULTI_GPU_SUPPORT
struct gpu_comm_attr;
class comm_group;
struct device_attr_impl;
class ccl_device_attr;
#endif
/**
 * Type @c communicator_t allows to operate communicator object in RAII manner
 */
using communicator_t = std::unique_ptr<ccl::communicator>;
using shared_communicator_t = std::shared_ptr<ccl::communicator>;

/**
 * Type @c stream_t allows to operate stream object in RAII manner
 */
using stream_t = std::unique_ptr<ccl::stream>;

/**
 * Class @c ccl_host_attr allows to configure host wire-up communicator creation parametes
 */
class ccl_host_attr {
public:
    friend class ccl_device_attr;
    friend struct communicator_interface_dispatcher;
    friend class environment;

    virtual ~ccl_host_attr() noexcept;

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <ccl_host_attributes attrId,
              class Value,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type>
    Value set_value(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <ccl_host_attributes attrId>
    const typename ccl_host_attributes_traits<attrId>::type& get_value() const;

protected:
    ccl_host_attr(const ccl_host_attr& src);

private:
    ccl_host_attr(const ccl_version_t& library_version,
                  const ccl_host_comm_attr_t& core = ccl_host_comm_attr_t(),
                  ccl_version_t api_version = ccl_version_t{ CCL_MAJOR_VERSION,
                                                             CCL_MINOR_VERSION,
                                                             CCL_UPDATE_VERSION,
                                                             CCL_PRODUCT_STATUS,
                                                             CCL_PRODUCT_BUILD_DATE,
                                                             CCL_PRODUCT_FULL });

    std::unique_ptr<host_attr_impl> pimpl;
};

using comm_attr_t = std::shared_ptr<ccl_host_attr>;

#ifdef MULTI_GPU_SUPPORT
using comm_group_t = std::shared_ptr<comm_group>;
/**
 * Class @c ccl_device_attr allows to configure device communicator creation parametes
 */
class ccl_device_attr : public ccl_host_attr {
public:
    friend class comm_group;
    friend struct communicator_interface_dispatcher;

    using base_t = ccl_host_attr;
    ~ccl_device_attr() noexcept;

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <
        ccl_device_attributes attrId,
        class Value,
        class = typename std::enable_if<
            std::is_same<typename ccl_device_attributes_traits<attrId>::type, Value>::value>::type>
    Value set_value(Value&& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <ccl_device_attributes attrId>
    const typename ccl_device_attributes_traits<attrId>::type& get_value() const;

private:
    ccl_device_attr(const ccl_host_attr& src);
    std::unique_ptr<device_attr_impl> pimpl;
};

using device_comm_attr_t = std::shared_ptr<ccl_device_attr>;
#endif //MULTI_GPU_SUPPORT

/**
 * ccl environment singleton
 */
class environment {
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
    communicator_t create_communicator(const ccl::comm_attr_t& attr = ccl::comm_attr_t()) const;

    /**
     * Creates a new ccl stream from @stream_native_type
     * @param native_stream the existing handle of stream
     */
    template <class stream_native_type,
              class = typename std::enable_if<is_stream_supported<stream_native_type>()>::type>
    stream_t create_stream(stream_native_type& native_stream);

    stream_t create_stream() const;

    /**
     * Retrieves the current version
     */
    ccl_version_t get_version() const;

    /**
     * Created @attr, which used to create host from @environment
     */
    comm_attr_t create_host_comm_attr(
        const ccl_host_comm_attr_t& attr = ccl_host_comm_attr_t()) const;

#ifdef MULTI_GPU_SUPPORT
    /**
     * Creates a new device group, which is entrance point of device communicator creation
     */
    comm_group_t create_comm_group(
        size_t current_device_group_size,
        size_t process_device_group_size,
        ccl::shared_communicator_t parent_comm = ccl::shared_communicator_t());

#endif //MULTI_GPU_SUPPORT
private:
    environment();
};

/**
 * A request interface that allows the user to track collective operation progress
 */
class request {
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
 * Helper functions to create custom datatype.
 */
ccl::datatype datatype_create(const ccl::datatype_attr* attr);
void datatype_free(ccl::datatype dtype);
size_t datatype_get_size(ccl::datatype dtype);

/**
 * A stream object is an abstraction over CPU/GPU streams
 * Has no defined public constructor. Use ccl::environment::create_stream
 * for stream objects creation
 */
class stream {
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

    using impl_t = std::shared_ptr<ccl_stream>;

private:
    friend class communicator;
    friend class environment;
    stream(impl_t&& impl);

    impl_t stream_impl;
};

/**
 * A communicator that permits collective operations
 * Has no defined public constructor. Use ccl::environment::create_communicator or ccl::comm_group
 * for communicator objects creation
 */
class communicator final {
public:
    ~communicator();

    /**
     * Type allows to operate request interface in RAII manner
     */
    using coll_request_t = std::unique_ptr<request>;

    /**
     * Retrieves the rank of the current process in a communicator
     * @return rank of the current process
     */
    size_t rank() const;

    /**
     * Retrieves the number of processes in a communicator
     * @return number of the processes
     */
    size_t size() const;

#ifdef MULTI_GPU_SUPPORT
    /**
     * Type allows to get underlying device type,
     * which was used as communicator construction argument
     */
    using device_native_reference_t = typename unified_device_type::native_reference_t;

    /**
     * Retrieves underlying device, which was used as communicator construction argument
     */
    device_native_reference_t get_device();

    /**
     * Retrieves logically determined devices topology based on hardware preferred
     * devices topology. Can be overriden during communicator creation phase
     */
    ccl::device_group_split_type get_device_group_split_type() const;

    ccl::device_topology_type get_topology_class() const;

    device_comm_attr_t get_device_attr() const;
#endif

    comm_attr_t get_host_attr() const;
    /**
     * Retrieves status of wiring-up progress of communicators in group
     * After all expected communicators are created in parent comm_group,
     * then wiring-up phase is automatically executed
     * and all communicator object will go in ready status
     */
    bool is_ready() const;

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
    coll_request_t allgatherv(const void* send_buf,
                              size_t send_count,
                              void* recv_buf,
                              const size_t* recv_counts,
                              ccl::datatype dtype,
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
    template <class buffer_type,
              class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t allgatherv(const buffer_type* send_buf,
                              size_t send_count,
                              buffer_type* recv_buf,
                              const size_t* recv_counts,
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
    template <
        class buffer_container_type,
        class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t allgatherv(const buffer_container_type& send_buf,
                              size_t send_count,
                              buffer_container_type& recv_buf,
                              const size_t* recv_counts,
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
    coll_request_t allreduce(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::datatype dtype,
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
    template <class buffer_type,
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
    template <
        class buffer_container_type,
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
    coll_request_t alltoall(const void* send_buf,
                            void* recv_buf,
                            size_t count,
                            ccl::datatype dtype,
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
    template <class buffer_type,
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
    template <
        class buffer_container_type,
        class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t alltoall(const buffer_container_type& send_buf,
                            buffer_container_type& recv_buf,
                            size_t count,
                            const ccl::coll_attr* attr = nullptr,
                            const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Each process sends distinct data to each of the receivers. The j-th block sent from process i is received
     * by process j and is placed in the i-th block of recvbuf. Block sizes may differ.
     * @param send_buf the buffer with elements of @c dtype that stores local data
     * @param send_counts array with number of elements send to each process
     * @param recv_buf [out] the buffer to store received result from  each process
     * @param recv_counts array with number of elements received from each process
     * @param dtype data type of elements in the buffer @c send_buf and @c recv_buf
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    coll_request_t alltoallv(const void* send_buf,
                             const size_t* send_counts,
                             void* recv_buf,
                             const size_t* recv_counts,
                             ccl::datatype dtype,
                             const ccl::coll_attr* attr = nullptr,
                             const ccl::stream_t& stream = ccl::stream_t());

    /**
     * Type safety version:
     * Each process sends distinct data to each of the receivers. The j-th block sent from process i is received
     * by process j and is placed in the i-th block of recvbuf. Block sizes may differ.
     * @param send_buf the buffer with elements of @c dtype that stores local data
     * @param send_counts array with number of elements send to each process
     * @param recv_buf [out] the buffer to store received result from  each process
     * @param recv_counts array with number of elements received from each process
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template <class buffer_type,
              class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t alltoallv(const buffer_type* send_buf,
                             const size_t* send_counts,
                             buffer_type* recv_buf,
                             const size_t* recv_counts,
                             const ccl::coll_attr* attr = nullptr,
                             const ccl::stream_t& stream = ccl::stream_t());
    /**
     * Type safety version:
     * Each process sends distinct data to each of the receivers. The j-th block sent from process i is received
     * by process j and is placed in the i-th block of recvbuf. Block sizes may differ.
     * @param send_buf the buffer with elements of @c dtype that stores local data
     * @param send_counts array with number of elements send to each process
     * @param recv_buf [out] the buffer to store received result from  each process
     * @param recv_counts array with number of elements received from each process
     * @param attr optional attributes that customize operation
     * @return @ref ccl::communicator::coll_request_t object that can be used to track the progress of the operation
     */
    template <
        class buffer_container_type,
        class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t alltoallv(const buffer_container_type& send_buf,
                             const size_t* send_counts,
                             buffer_container_type& recv_buf,
                             const size_t* recv_counts,
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
    coll_request_t bcast(void* buf,
                         size_t count,
                         ccl::datatype dtype,
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
    template <class buffer_type,
              class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t bcast(buffer_type* buf,
                         size_t count,
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
    template <
        class buffer_container_type,
        class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t bcast(buffer_container_type& buf,
                         size_t count,
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
    coll_request_t reduce(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          ccl::datatype dtype,
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
    template <class buffer_type,
              class = typename std::enable_if<ccl::is_native_type_supported<buffer_type>()>::type>
    coll_request_t reduce(const buffer_type* send_buf,
                          buffer_type* recv_buf,
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
    template <
        class buffer_container_type,
        class = typename std::enable_if<ccl::is_class_supported<buffer_container_type>()>::type>
    coll_request_t reduce(const buffer_container_type& send_buf,
                          buffer_container_type& recv_buf,
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
    coll_request_t sparse_allreduce(const void* send_ind_buf,
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
    template <
        class index_buffer_type,
        class value_buffer_type,
        class = typename std::enable_if<ccl::is_native_type_supported<value_buffer_type>()>::type>
    coll_request_t sparse_allreduce(const index_buffer_type* send_ind_buf,
                                    size_t send_ind_count,
                                    const value_buffer_type* send_val_buf,
                                    size_t send_val_count,
                                    index_buffer_type* recv_ind_buf,
                                    size_t recv_ind_count,
                                    value_buffer_type* recv_val_buf,
                                    size_t recv_val_count,
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
    template <class index_buffer_container_type,
              class value_buffer_container_type,
              class = typename std::enable_if<
                  ccl::is_class_supported<value_buffer_container_type>()>::type>
    coll_request_t sparse_allreduce(const index_buffer_container_type& send_ind_buf,
                                    size_t send_ind_count,
                                    const value_buffer_container_type& send_val_buf,
                                    size_t send_val_count,
                                    index_buffer_container_type& recv_ind_buf,
                                    size_t recv_ind_count,
                                    value_buffer_container_type& recv_val_buf,
                                    size_t recv_val_count,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr = nullptr,
                                    const ccl::stream_t& stream = ccl::stream_t());

private:
    friend class environment;
    friend class comm_group;

    explicit communicator(std::shared_ptr<communicator_interface> impl);

    /**
     * Holds specific-implementation details of communicator
     */
    std::shared_ptr<communicator_interface> pimpl;
};
} // namespace ccl
#ifdef MULTI_GPU_SUPPORT
#include "ccl_gpu_modules.h"
#include "gpu_communicator.hpp"
#endif
