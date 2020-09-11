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

#if defined(MULTI_GPU_SUPPORT)
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl/backend/Intel_level0.hpp>
//static cl::sycl::vector_class<cl::sycl::device> gpu_sycl_devices;
#endif

#include "oneapi/ccl/native_device_api/l0/utils.hpp"

namespace native {
namespace details {
static ccl_device_driver::device_ptr get_runtime_device_impl(const ccl::device_index_type& path) {
    return get_platform().get_device(path);
}
} // namespace details

template <class DeviceType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type = 0>
CCL_API ccl_device_driver::device_ptr get_runtime_device(const DeviceType& device) {
    static_assert(std::is_same<typename ccl::unified_device_type::handle_t, DeviceType>::value,
                  "Unsupported 'DeviceType'");
    size_t driver_idx = 0; // limitation for OPENCL/SYCL
    size_t device_id = 0;
#ifdef CCL_ENABLE_SYCL
    device_id = native::details::get_sycl_device_id(device);
#endif
    ccl::device_index_type path(driver_idx, device_id, ccl::unused_index_value);

    return details::get_runtime_device_impl(path);
}

template <class DeviceType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type = 0>
CCL_API ccl_device_driver::device_ptr get_runtime_device(const DeviceType& device) {
    return details::get_runtime_device_impl(device);
}

} // namespace native

template native::ccl_device_driver::device_ptr native::get_runtime_device(
    const ccl::device_index_type& path);

#ifdef CCL_ENABLE_SYCL
template native::ccl_device_driver::device_ptr native::get_runtime_device(
    const cl::sycl::device& device);
#endif

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
