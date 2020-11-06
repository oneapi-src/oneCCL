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
class event;
class kvs_interface;
using rank_t = size_t;

struct communicator_interface;
/**
 * A device communicator that permits device communication operations
 * Has no defined public constructor.
 * Use ccl::environment::create_device_communicator for communicator objects creation.
 */
class communicator final : public ccl_api_base_movable<communicator,
                                                              direct_access_policy,
                                                              communicator_interface,
                                                              std::shared_ptr> {
public:
    using base_t = ccl_api_base_movable<communicator,
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

    using coll_request_t = ccl::event;

    communicator(communicator&& src);
    communicator& operator=(communicator&& src);
    ~communicator();

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

    communicator split(const comm_split_attr& attr);

private:
    friend class environment;
    friend class comm_group;
    friend struct impl_dispatch;

    template<cl_backend_type type>
    friend struct comm_impl_dispatch_selector;

    communicator(impl_value_t&& impl);

    // factory methods
    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators(
        size_t comm_size,
        const vector_class<DeviceType>& local_devices,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators(
        size_t comm_size,
        const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators(
        size_t comm_size,
        const map_class<rank_t, DeviceType>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs);

    static communicator create_communicator();
    static communicator create_communicator(size_t size,
                                            shared_ptr_class<kvs_interface> kvs);
    static communicator create_communicator(size_t size,
                                            size_t rank,
                                            shared_ptr_class<kvs_interface> kvs);
};

} // namespace ccl
