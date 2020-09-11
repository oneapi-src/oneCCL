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
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_event_attr_ids.hpp"
#include "oneapi/ccl/ccl_event_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "common/event/event.hpp"

namespace ccl {

template <class event_type, class... attr_value_pair_t>
event event::create_event_from_attr(event_type& native_event_handle,
                                    typename unified_device_context_type::ccl_native_t context,
                                    attr_value_pair_t&&... avps) {
    ccl::library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    event str{ event::impl_value_t(new event::impl_t(native_event_handle, context, ret)) };
    int expander[]{ (str.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
    (void)expander;
    str.build_from_params();

    return str;
}

template <class event_handle_type, typename T>
event event::create_event(event_handle_type native_event_handle,
                          typename unified_device_context_type::ccl_native_t context) {
    ccl::library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    event str{ event::impl_value_t(new event::impl_t(native_event_handle, context, ret)) };
    str.build_from_params();

    return str;
}

template <class event_type, typename T>
event event::create_event(event_type& native_event) {
    ccl::library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    return { event::impl_value_t(new event::impl_t(native_event, ret)) };
}

template <event_attr_id attrId>
CCL_API const typename details::ccl_api_type_attr_traits<event_attr_id, attrId>::return_type&
event::get() const {
    return get_impl()->get_attribute_value(
        details::ccl_api_type_attr_traits<event_attr_id, attrId>{});
}

template<event_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename ccl::details::ccl_api_type_attr_traits<ccl::event_attr_id, attrId>::return_type event::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, details::ccl_api_type_attr_traits<event_attr_id, attrId>{});
}

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_EVENT_CREATION_FORCE_INSTANTIATION(native_event_type) \
    template CCL_API ccl::event ccl::event::create_event(native_event_type& native_event);

#define API_EVENT_CREATION_EXT_FORCE_INSTANTIATION(native_event_handle_type) \
    template CCL_API ccl::event ccl::event::create_event( \
        native_event_handle_type handle, \
        typename unified_device_context_type::ccl_native_t native_ctx);

#define API_EVENT_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    template CCL_API typename ccl::details::ccl_api_type_attr_traits<ccl::event_attr_id, \
                                                                     IN_attrId>::return_type \
    ccl::event::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_EVENT_FORCE_INSTANTIATION_GET(IN_attrId) \
    template CCL_API const typename ccl::details:: \
        ccl_api_type_attr_traits<ccl::event_attr_id, IN_attrId>::return_type& \
        ccl::event::get<IN_attrId>() const;

#define API_EVENT_FORCE_INSTANTIATION(IN_attrId, IN_Value) \
    API_EVENT_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    API_EVENT_FORCE_INSTANTIATION_GET(IN_attrId)

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
