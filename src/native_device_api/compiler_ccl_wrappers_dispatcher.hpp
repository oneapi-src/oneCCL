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

#include "native_device_api/export_api.hpp"
#include "ccl_type_traits.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl/backend/Intel_level0.hpp>
static cl::sycl::vector_class<cl::sycl::device> gpu_sycl_devices;
#endif

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
    static_assert(std::is_same<typename ccl::unified_device_type::device_t, DeviceType>::value,
                  "Unsupported 'DeviceType'");
    size_t driver_idx = 0; // limitation for OPENCL/SYCL
    size_t device_id = std::numeric_limits<size_t>::max();
    ccl::device_index_type path(driver_idx, device_id, ccl::unused_index_value);

#ifdef CCL_ENABLE_SYCL

    if (!device.is_gpu()) {
        throw std::runtime_error(
            std::string("get_runtime_device failed for sycl device: it is not gpu!"));
    }

    // extract native handle L0
    auto l0_handle_ptr = device.template get_native<cl::sycl::backend::level0>();
    if (!l0_handle_ptr) {
        throw std::runtime_error(
            std::string("get_runtime_device failed for sycl device: handle is nullptr!"));
    }

    ze_device_properties_t device_properties;
    device_properties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
    ze_result_t ret = zeDeviceGetProperties(l0_handle_ptr, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("zeDeviceGetProperties failed, error: ") +
                                 native::to_string(ret));
    }

    //use deviceId to return native device
    device_id = device_properties.deviceId;

    //TODO only device not subdevices
    std::get<ccl::device_index_enum::device_index_id>(path) = device_id;
#endif
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
