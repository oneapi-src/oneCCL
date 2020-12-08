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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "common/log/log.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

namespace ccl {

#ifdef CCL_ENABLE_SYCL
/*
CCL_API generic_device_type<cl_backend_type::dpcpp_sycl>::generic_device_type(
    device_index_type id,
    cl::sycl::info::device_type type)
        : device() {
    LOG_DEBUG("Try to find SYCL device by index: ",
              id,
              ", type: ",
              static_cast<typename std::underlying_type<cl::sycl::info::device_type>::type>(type));

    auto platforms = cl::sycl::platform::get_platforms();
    LOG_DEBUG("Found CL plalforms: ", platforms.size());
    auto platform_it =
        std::find_if(platforms.begin(), platforms.end(), [](const cl::sycl::platform& pl) {
            return pl.get_info<cl::sycl::info::platform::name>().find("Level-Zero") !=
                   std::string::npos;
            //or platform.get_backend() == cl::sycl::backend::level_zero
        });
    if (platform_it == platforms.end()) {
        std::stringstream ss;
        ss << "cannot find Level-Zero platform. Supported platforms are:\n";
        for (const auto& pl : platforms) {
            ss << "Platform:\nprofile: " << pl.get_info<cl::sycl::info::platform::profile>()
               << "\nversion: " << pl.get_info<cl::sycl::info::platform::version>()
               << "\nname: " << pl.get_info<cl::sycl::info::platform::name>()
               << "\nvendor: " << pl.get_info<cl::sycl::info::platform::vendor>();
        }

        throw std::runtime_error(std::string("Cannot find device by id: ") + ccl::to_string(id) +
                                 ", reason:\n" + ss.str());
    }

    LOG_DEBUG("Platform:\nprofile: ",
              platform_it->get_info<cl::sycl::info::platform::profile>(),
              "\nversion: ",
              platform_it->get_info<cl::sycl::info::platform::version>(),
              "\nname: ",
              platform_it->get_info<cl::sycl::info::platform::name>(),
              "\nvendor: ",
              platform_it->get_info<cl::sycl::info::platform::vendor>());
    auto devices = platform_it->get_devices(type);

    LOG_DEBUG("Found devices: ", devices.size());
    auto it =
        std::find_if(devices.begin(), devices.end(), [id](const cl::sycl::device& dev) -> bool {
#ifdef MULTI_GPU_SUPPORT
            //TODO -S-
            return id == native::get_runtime_device(dev)->get_device_path();
#endif
            (void)id;
            return true;
        });
    if (it == devices.end()) {
        std::stringstream ss;
        ss << "cannot find requested device. Supported devices are:\n";
        for (const auto& dev : devices) {
            ss << "Device:\nname: " << dev.get_info<cl::sycl::info::device::name>()
               << "\nvendor: " << dev.get_info<cl::sycl::info::device::vendor>()
               << "\nversion: " << dev.get_info<cl::sycl::info::device::version>()
               << "\nprofile: " << dev.get_info<cl::sycl::info::device::profile>();
        }

        throw std::runtime_error(std::string("Cannot find device by id: ") + ccl::to_string(id) +
                                 ", reason:\n" + ss.str());
    }
    device = *it;
}

generic_device_type<cl_backend_type::dpcpp_sycl>::generic_device_type(const cl::sycl::device& in_device)
        : device(in_device) {}

device_index_type generic_device_type<cl_backend_type::dpcpp_sycl>::get_id() const {
    //TODO -S-
#ifdef MULTI_GPU_SUPPORT
    return native::get_runtime_device(device)->get_device_path();
#endif
    return device_index_type{};
}

typename generic_device_type<cl_backend_type::dpcpp_sycl>::ccl_native_t&
generic_device_type<cl_backend_type::dpcpp_sycl>::get() noexcept {
    return device;
}
*/
#else
#ifdef MULTI_GPU_SUPPORT
// #else
// #error "No compute runtime is configured"
#endif
#endif
} // namespace ccl
