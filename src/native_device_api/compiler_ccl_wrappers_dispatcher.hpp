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

#if defined(CCL_ENABLE_ZE)
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/native_device_api/l0/declarations.hpp"
#include "oneapi/ccl/type_traits.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl/backend/level_zero.hpp>
#include "common/utils/sycl_utils.hpp"
//static cl::sycl::vector_class<cl::sycl::device> gpu_sycl_devices;
#endif

#include "oneapi/ccl/native_device_api/l0/utils.hpp"
#include "common/log/log.hpp"

namespace native {
namespace detail {
static ccl_device_driver::device_ptr get_runtime_device_impl(const ccl::device_index_type& path) {
    return get_platform().get_device(path);
}

} // namespace detail

template <class DeviceType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type = 0>
/*CCL_API*/ ccl_device_driver::device_ptr get_runtime_device(const DeviceType& device) {
    static_assert(std::is_same<typename ccl::unified_device_type::ccl_native_t, DeviceType>::value,
                  "Unsupported 'DeviceType'");
    size_t driver_idx = 0; // limitation for OPENCL/SYCL
    size_t device_id = 0;
    size_t subdevice_id = 0;
#ifdef CCL_ENABLE_SYCL
    device_id = native::detail::get_sycl_device_id(device);
    subdevice_id = native::detail::get_sycl_subdevice_id(device);
#endif
    ccl::device_index_type path(driver_idx, device_id, subdevice_id);

    return detail::get_runtime_device_impl(path);
}

template <class DeviceType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type = 0>
/*CCL_API*/ ccl_device_driver::device_ptr get_runtime_device(const DeviceType& device) {
    return detail::get_runtime_device_impl(device);
}

template <class ContextType>
/*CCL_API*/ ccl_driver_context_ptr get_runtime_context(const ContextType& ctx) {
#ifdef CCL_ENABLE_SYCL
    static_assert(
        std::is_same<typename std::remove_cv<ContextType>::type, cl::sycl::context>::value,
        "Invalid ContextType");
    auto l0_handle_ptr = sycl::get_native<ccl::utils::get_level_zero_backend()>(ctx);
    if (!l0_handle_ptr) {
        CCL_THROW("failed for sycl context: handle is nullptr");
    }
    auto& drivers = get_platform().get_drivers();
    assert(drivers.size() == 1 && "Only one driver supported for L0 at now");
    return drivers.begin()->second->create_context_from_handle(l0_handle_ptr);
#else
    return ctx;
#endif
}
} // namespace native

template native::ccl_device_driver::device_ptr native::get_runtime_device(
    const ccl::device_index_type& path);

template native::ccl_driver_context_ptr native::get_runtime_context(
    const ccl::unified_context_type::ccl_native_t& ctx);

#ifdef CCL_ENABLE_SYCL
template native::ccl_device_driver::device_ptr native::get_runtime_device(
    const cl::sycl::device& device);
#endif

#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
