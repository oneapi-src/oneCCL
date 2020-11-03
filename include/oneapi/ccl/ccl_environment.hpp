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

#include "oneapi/ccl/ccl_context_attr_ids.hpp"
#include "oneapi/ccl/ccl_context_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_context.hpp"

#include "oneapi/ccl/ccl_datatype_attr_ids.hpp"
#include "oneapi/ccl/ccl_datatype_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_datatype_attr.hpp"

#include "oneapi/ccl/ccl_device_attr_ids.hpp"
#include "oneapi/ccl/ccl_device_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_device.hpp"

#include "oneapi/ccl/ccl_kvs.hpp"

#include "oneapi/ccl/ccl_event.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "oneapi/ccl/ccl_communicator.hpp"

#include "oneapi/ccl/ccl_exception.hpp"

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

    ccl::library_version get_library_version() const;

    template <class... attr_value_pair_t>
    datatype_attr create_datatype_attr(attr_value_pair_t&&... avps) const {
        static_assert(sizeof...(avps) > 0, "At least one argument must be specified");
        auto attr = create_postponed_api_type<datatype_attr>();
        int expander[]{ (attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return attr;
    }

    ccl::datatype register_datatype(const ccl::datatype_attr& attr);
    void deregister_datatype(ccl::datatype dtype);
    size_t get_datatype_size(ccl::datatype dtype) const;

    shared_ptr_class<kvs> create_main_kvs() const;
    shared_ptr_class<kvs> create_kvs(const kvs::address_type& addr) const;

    device create_device(empty_t empty) const;

    template <class native_device_type,
              class = typename std::enable_if<is_device_supported<native_device_type>()>::type>
    device create_device(native_device_type&& native_device) const;

    template <class... attr_value_pair_t>
    device create_device_from_attr(typename unified_device_type::ccl_native_t dev,
                                   attr_value_pair_t&&... avps) const {
        device str = create_postponed_api_type<device>(dev);
        int expander[]{ (str.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        str.build_from_params();
        return str;
    }

    context create_context(empty_t empty) const;

    template <class native_device_contex_type,
              class = typename std::enable_if<is_device_supported<native_device_contex_type>()>::type>
    context create_context(native_device_contex_type&& native_device_context) const;

    template <class... attr_value_pair_t>
    context create_context_from_attr(typename unified_device_context_type::ccl_native_t ctx,
                                   attr_value_pair_t&&... avps) const {
        context str = create_postponed_api_type<context>(ctx);
        int expander[]{ (str.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        str.build_from_params();
        return str;
    }

    template <class coll_attribute_type, class... attr_value_pair_t>
    coll_attribute_type create_operation_attr(attr_value_pair_t&&... avps) const {
        auto op_attr = create_postponed_api_type<coll_attribute_type>();
        int expander[]{ (op_attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return op_attr;
    }

    template <class event_type,
            class = typename std::enable_if<is_event_supported<event_type>()>::type>
    event create_event(event_type& native_event) {
        return event::create_from_native(native_event);
    }

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


#ifdef CCL_ENABLE_SYCL
    communicator create_single_device_communicator(
        size_t comm_size,
        size_t rank,
        const cl::sycl::device& device,
        const cl::sycl::context& context,
        shared_ptr_class<kvs_interface> kvs) const;
#endif

    template <class... attr_value_pair_t>
    comm_split_attr create_comm_split_attr(attr_value_pair_t&&... avps) const {
        auto split_attr = create_postponed_api_type<comm_split_attr>();
        int expander[]{ (split_attr.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
        (void)expander;
        return split_attr;
    }

    communicator create_communicator() const;
    communicator create_communicator(size_t size, shared_ptr_class<kvs_interface> kvs) const;
    communicator create_communicator(size_t size,
                                     size_t rank,
                                     shared_ptr_class<kvs_interface> kvs) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicators(
        size_t comm_size,
        const vector_class<DeviceType>& local_devices,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicators(
        size_t comm_size,
        const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicators(
        size_t comm_size,
        const map_class<rank_t, DeviceType>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const;

    vector_class<communicator> split_device_communicators(
        const vector_class<pair_class<communicator, comm_split_attr>>& attrs) const;

private:
    environment();

    template <class ccl_api_type, class... args_type>
    ccl_api_type create_postponed_api_type(args_type... args) const;
};

} /* ccl */
