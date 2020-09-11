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
#include <utility>
#include <vector>

#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "oneapi/ccl/ccl_datatype_attr_ids.hpp"
#include "oneapi/ccl/ccl_datatype_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_datatype_attr.hpp"

#include "oneapi/ccl/ccl_event_attr_ids.hpp"
#include "oneapi/ccl/ccl_event_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "oneapi/ccl/ccl_kvs.hpp"

#include "oneapi/ccl/ccl_request.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "oneapi/ccl/ccl_communicator.hpp"
#include "oneapi/ccl/ccl_device_communicator.hpp"

namespace ccl {

/**
 * CCL environment singleton
 */
class environment {
public:
    ~environment();

    /**
     * Retrieves the unique environment object
     * and makes the first-time initialization of CCL library
     */
    static environment& instance();

    /**
     * Retrieves the library version
     */
    ccl::library_version get_library_version() const;

    /**
     * Creates @attr which used to register custom datatype
     */
    template <class... attr_value_pair_t>
    datatype_attr create_datatype_attr(attr_value_pair_t&&... avps) const {
        static_assert(sizeof...(avps) > 0, "At least one argument must be specified");
        auto attr = create_postponed_api_type<datatype_attr>();
        int expander[]{ (attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return attr;
    }

    /**
     * Registers custom datatype to be used in communication operations
     * @param attr datatype attributes
     * @return datatype handle
     */
    ccl::datatype register_datatype(const ccl::datatype_attr& attr);

    /**
     * Deregisters custom datatype
     * @param dtype custom datatype handle
     */
    void deregister_datatype(ccl::datatype dtype);

    /**
     * Retrieves a datatype size in bytes
     * @param dtype datatype handle
     * @return datatype size
     */
    size_t get_datatype_size(ccl::datatype dtype) const;

    /**
     * Enables job scalability policy
     * @param callback of @c ccl_resize_fn_t type, which enables scalability policy
     * (@c nullptr enables default behavior)
     */
    //void set_resize_fn(ccl_resize_fn_t callback);

    /**
     * Creates a main key-value store.
     * It's address should be distributed using out of band communication mechanism
     * and be used to create key-value stores on other ranks.
     * @return kvs object
     */
    shared_ptr_class<kvs> create_main_kvs() const;

    /**
     * Creates a new key-value store from main kvs address
     * @param addr address of main kvs
     * @return kvs object
     */
    shared_ptr_class<kvs> create_kvs(const kvs::address_type& addr) const;

    /**
     * Creates a new host communicator with externally provided size, rank and kvs.
     * Implementation is platform specific and non portable.
     * @return host communicator
     */
    communicator create_communicator() const;

    /**
     * Creates a new host communicator with user supplied size and kvs.
     * Rank will be assigned automatically.
     * @param size user-supplied total number of ranks
     * @param kvs key-value store for ranks wire-up
     * @return host communicator
     */
    communicator create_communicator(const size_t size, shared_ptr_class<kvs_interface> kvs) const;

    /**
     * Creates a new host communicator with user supplied size, rank and kvs.
     * @param size user-supplied total number of ranks
     * @param rank user-supplied rank
     * @param kvs key-value store for ranks wire-up
     * @return host communicator
     */
    communicator create_communicator(const size_t size,
                                     const size_t rank,
                                     shared_ptr_class<kvs_interface> kvs) const;

    template <class coll_attribute_type, class... attr_value_pair_t>
    coll_attribute_type create_operation_attr(attr_value_pair_t&&... avps) const {
        auto op_attr = create_postponed_api_type<coll_attribute_type>();
        int expander[]{ (op_attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return op_attr;
    }

    /**
     * Creates @attr which used to split host communicator
     */
    template <class... attr_value_pair_t>
    comm_split_attr create_comm_split_attr(attr_value_pair_t&&... avps) const {
        auto split_attr = create_postponed_api_type<comm_split_attr>();
        int expander[]{ (split_attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return split_attr;
    }

#ifdef CCL_ENABLE_SYCL
    device_communicator create_single_device_communicator(
        const size_t comm_size,
        const size_t rank,
        const cl::sycl::device& device,
        shared_ptr_class<kvs_interface> kvs) const;
#endif

//     device_communicator create_single_device_communicator(const size_t world_size,
//                                      const size_t rank,
//                                      cl::sycl::queue queue,
//                                      shared_ptr_class<kvs_interface> kvs) const;

//     template<class DeviceSelectorType>
//     device_communicator create_single_device_communicator(const size_t world_size,
//                                      const size_t rank,
//                                      const DeviceSelectorType& selector,
//                                      shared_ptr_class<kvs_interface> kvs) const
//     {
//         return create_single_device_communicator(world_size, rank, cl::sycl::device(selector), kvs);
//     }

// #endif
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

    template <class... attr_value_pair_t>
    device_comm_split_attr create_device_comm_split_attr(attr_value_pair_t&&... avps) const {
        auto split_attr = create_postponed_api_type<device_comm_split_attr>();
        int expander[]{ (split_attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return split_attr;
    }

    /**
     * Creates a new device communicators with user supplied size, device indices and kvs.
     * Ranks will be assigned automatically.
     * @param comm_size user-supplied total number of ranks
     * @param local_devices user-supplied device objects for local ranks
     * @param context context containing the devices
     * @param kvs key-value store for ranks wire-up
     * @return vector of device communicators
     */
    template <class DeviceType, class ContextType>
    vector_class<device_communicator> create_device_communicators(
        const size_t comm_size,
        const vector_class<DeviceType>& local_devices,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const;

    /**
     * Creates a new device communicators with user supplied size, ranks, device indices and kvs.
     * @param comm_size user-supplied total number of ranks
     * @param local_rank_device_map user-supplied mapping of local ranks on devices
     * @param context context containing the devices
     * @param kvs key-value store for ranks wire-up
     * @return vector of device communicators
     */
    template <class DeviceType, class ContextType>
    vector_class<device_communicator> create_device_communicators(
        const size_t comm_size,
        const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const;

    template <class DeviceType, class ContextType>
    vector_class<device_communicator> create_device_communicators(
        const size_t comm_size,
        const map_class<rank_t, DeviceType>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const;

    /**
     * Splits device communicators according to attributes.
     * @param attrs split attributes for local communicators
     * @return vector of device communicators
     */
    vector_class<device_communicator> split_device_communicators(
        const vector_class<pair_class<device_communicator, device_comm_split_attr>>& attrs) const;

    /**
     * Creates a new stream from @native_stream_type
     * @param native_stream the existing handle of stream
     * @return stream object
     */
    template <class native_stream_type,
              class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
    stream create_stream(native_stream_type& native_stream);

    template <class native_stream_type,
              class native_context_type,
              class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
    stream create_stream(native_stream_type& native_stream, native_context_type& native_ctx);

    template <class... attr_value_pair_t>
    stream create_stream_from_attr(typename unified_device_type::ccl_native_t device,
                                   attr_value_pair_t&&... avps) {
        stream str = create_postponed_api_type<stream>(device);
        int expander[]{ (str.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        str.build_from_params();
        return str;
    }

    template <class... attr_value_pair_t>
    stream create_stream_from_attr(typename unified_device_type::ccl_native_t device,
                                   typename unified_device_context_type::ccl_native_t context,
                                   attr_value_pair_t&&... avps) {
        stream str = create_postponed_api_type<stream>(device, context);
        int expander[]{ (str.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        str.build_from_params();
        return str;
    }

    /**
     * Creates a new event from @native_event_type
     * @param native_event the existing handle of event
     * @return event object
     */
    template <class event_type,
              class = typename std::enable_if<is_event_supported<event_type>()>::type>
    event create_event(event_type& native_event);

    template <class event_handle_type,
              class = typename std::enable_if<is_event_supported<event_handle_type>()>::type>
    event create_event(event_handle_type native_event_handle,
                       typename unified_device_context_type::ccl_native_t context);

    template <class event_type, class... attr_value_pair_t>
    event create_event_from_attr(event_type& native_event_handle,
                                 typename unified_device_context_type::ccl_native_t context,
                                 attr_value_pair_t&&... avps) {
        event ev = create_postponed_api_type<event>(native_event_handle, context);
        int expander[]{ (ev.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        ev.build_from_params();
        return ev;
    }

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

private:
    environment();

    template <class ccl_api_type, class... args_type>
    ccl_api_type create_postponed_api_type(args_type... args) const;
};

} // namespace ccl
