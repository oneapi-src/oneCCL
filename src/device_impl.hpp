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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_device_attr_ids.hpp"
#include "oneapi/ccl/ccl_device_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_device.hpp"

#include "common/device/device.hpp"

namespace ccl {

template <class device_type, class... attr_value_pair_t>
CCL_API device device::create_device_from_attr(device_type& native_device_handle,
                                       attr_value_pair_t&&... avps) {
    ccl::library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    device str{ device::impl_value_t(new device::impl_t(native_device_handle, ret)) };
    int expander[]{ (str.template set<attr_value_pair_t::idx()>(avps.val()), 0)... };
    (void)expander;
    str.build_from_params();

    return str;
}

template <class device_type, typename T>
CCL_API device device::create_device(device_type&& native_device) {
    ccl::library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    return { device::impl_value_t(new device::impl_t(std::forward<device_type>(native_device), ret)) };
}

template <device_attr_id attrId>
CCL_API const typename details::ccl_api_type_attr_traits<device_attr_id, attrId>::return_type&
device::get() const {
    return get_impl()->get_attribute_value(
        details::ccl_api_type_attr_traits<device_attr_id, attrId>{});
}

template<device_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename ccl::details::ccl_api_type_attr_traits<ccl::device_attr_id, attrId>::return_type device::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, details::ccl_api_type_attr_traits<device_attr_id, attrId>{});
}

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_DEVICE_CREATION_FORCE_INSTANTIATION(native_device_type) \
    template CCL_API ccl::device ccl::device::create_device(native_device_type&& dev); \
    template CCL_API ccl::device ccl::device::create_device(native_device_type& dev);

#define API_DEVICE_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    template CCL_API typename ccl::details::ccl_api_type_attr_traits<ccl::device_attr_id, \
                                                                     IN_attrId>::return_type \
    ccl::device::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_DEVICE_FORCE_INSTANTIATION_GET(IN_attrId) \
    template CCL_API const typename ccl::details:: \
        ccl_api_type_attr_traits<ccl::device_attr_id, IN_attrId>::return_type& \
        ccl::device::get<IN_attrId>() const;

#define API_DEVICE_FORCE_INSTANTIATION(IN_attrId, IN_Value) \
    API_DEVICE_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    API_DEVICE_FORCE_INSTANTIATION_GET(IN_attrId)
