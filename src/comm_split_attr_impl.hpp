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
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

namespace ccl {

/**
 * comm_split_attr attributes definition
 */
template <comm_split_attr_id attrId, class Value>
CCL_API Value comm_split_attr::set(const Value& v) {
    return get_impl()->set_attribute_value(
        v, details::ccl_host_split_traits<comm_split_attr_id, attrId>{});
}

template <comm_split_attr_id attrId>
CCL_API const typename details::ccl_host_split_traits<comm_split_attr_id, attrId>::type&
comm_split_attr::get() const {
    return get_impl()->get_attribute_value(
        details::ccl_host_split_traits<comm_split_attr_id, attrId>{});
}

template <comm_split_attr_id attrId>
CCL_API bool comm_split_attr::is_valid() const noexcept {
    return get_impl()->is_valid<attrId>();
}

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

/**
 * device_comm_split_attr attributes definition
 */
template <comm_split_attr_id attrId, class Value>
CCL_API Value device_comm_split_attr::set(const Value& v) {
    return get_impl()->set_attribute_value(
        v, details::ccl_device_split_traits<comm_split_attr_id, attrId>{});
}

template <comm_split_attr_id attrId>
CCL_API const typename details::ccl_device_split_traits<comm_split_attr_id, attrId>::type&
device_comm_split_attr::get() const {
    return get_impl()->get_attribute_value(
        details::ccl_device_split_traits<comm_split_attr_id, attrId>{});
}

template <comm_split_attr_id attrId>
CCL_API bool device_comm_split_attr::is_valid() const noexcept {
    return get_impl()->is_valid<attrId>();
}

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

} // namespace ccl
