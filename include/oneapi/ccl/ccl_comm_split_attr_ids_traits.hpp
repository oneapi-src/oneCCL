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

namespace details {

/**
 * Host-specific traits
 */
template <class attr, attr id>
struct ccl_host_split_traits {};

template <>
struct ccl_host_split_traits<comm_split_attr_id, comm_split_attr_id::version> {
    using type = ccl::library_version;
};

template <>
struct ccl_host_split_traits<comm_split_attr_id, comm_split_attr_id::color> {
    using type = int;
};

template <>
struct ccl_host_split_traits<comm_split_attr_id, comm_split_attr_id::group> {
    using type = ccl_group_split_type;
};

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

/**
 * Device-specific traits
 */
template <class attr, attr id>
struct ccl_device_split_traits {};

template <>
struct ccl_device_split_traits<comm_split_attr_id, comm_split_attr_id::version> {
    using type = ccl::library_version;
};

template <>
struct ccl_device_split_traits<comm_split_attr_id, comm_split_attr_id::color> {
    using type = int;
};

template <>
struct ccl_device_split_traits<comm_split_attr_id, comm_split_attr_id::group> {
    using type = device_group_split_type;
};

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

} // namespace details

} // namespace ccl
