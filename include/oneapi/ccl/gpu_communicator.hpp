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

#ifdef MULTI_GPU_SUPPORT
#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl_types.hpp'"
#endif

namespace ccl {
struct gpu_comm_attr;
struct communicator_interface;

#if DEPRECATED
class communicator final {
public:
    /**
     * Type allows to get underlying device type,
     * which was used as communicator construction argument
     */
    using device_native_reference_t = typename unified_device_type::ccl_native_t;

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

    /**
     * Retrieves underlying device, which was used as communicator constrution argument
     */
    device_native_reference_t get_device();

    /**
     * Retrieves status of wiring-up progress of communicators in group
     * After all expected communicators are created in parent comm_group,
     * then wiring-up phase is automatically executed
     * and all communicator object will go in ready status
     */
    bool is_ready() const;

    /**
     * Retrieves logically determined devices topology based on hardware preferred
     * devices topology. Can be overriden during communicator creation phase
     */
    ccl::device_group_split_type get_topology_type() const;

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
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream);

private:
    friend class environment;
    friend class comm_group;

    explicit communicator(std::shared_ptr<communicator_interface> impl);

    /**
     * Holds specific-implementation details of communicator
     */
    std::shared_ptr<communicator_interface> pimpl;
};
#endif //DEPRECATED
} // namespace ccl

#endif //MULTI_GPU_SUPPORT
