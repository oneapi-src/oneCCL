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
#include "oneapi/ccl/native_device_api/l0/utils.hpp"

#if defined(MULTI_GPU_SUPPORT)
#include "oneapi/ccl/native_device_api/l0/device.hpp"

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
//#ifdef CCL_ENABLE_SYCL
#include <CL/sycl/backend/Intel_level0.hpp>
//static cl::sycl::vector_class<cl::sycl::device> gpu_sycl_devices;
#endif

namespace native {
namespace details {

adjacency_matrix::adjacency_matrix(std::initializer_list<typename base::value_type> init)
        : base(init) {}

cross_device_rating binary_p2p_rating_calculator(const native::ccl_device& lhs,
                                                 const native::ccl_device& rhs,
                                                 size_t weight) {
    return property_p2p_rating_calculator(lhs, rhs, 1);
}

#ifdef CCL_ENABLE_SYCL
size_t get_sycl_device_id(const cl::sycl::device& device) {
    if (!device.is_gpu()) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - failed for sycl device: it is not gpu!");
    }

    size_t device_id = std::numeric_limits<size_t>::max();

    // extract native handle L0
    auto l0_handle_ptr = device.template get_native<cl::sycl::backend::level0>();
    if (!l0_handle_ptr) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - failed for sycl device: handle is nullptr!");
    }

    ze_device_properties_t device_properties;
    device_properties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
    ze_result_t ret = zeDeviceGetProperties(l0_handle_ptr, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string(__FUNCTION__) +
            " - zeDeviceGetProperties failed, error: " + native::to_string(ret));
    }

    //use deviceId to return native device
    device_id = device_properties.deviceId;

    //TODO only device not subdevices
    return device_id;
}
#endif
} // namespace details
} // namespace native
#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
