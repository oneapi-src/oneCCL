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
#include <string>
#include <type_traits>
#include "common/utils/enums.hpp"

namespace native {
enum class gpu_types : size_t {
    REAL_GPU,
    VIRTUAL_GPU,
    CONCURRENT_GPU,
    CONCURRENT_REAL_GPU = CONCURRENT_GPU,
    CONCURRENT_VIRT_GPU,

    IPC_GPU,
    IPC_SOURCE_REAL_GPU = IPC_GPU,
    IPC_SOURCE_VIRT_GPU,
    IPC_DESTINATION_GPU,

    SCALING_PROXY_GPU_TYPES,
    NUMA_PROXY_GPU_TYPES = SCALING_PROXY_GPU_TYPES,

    NUMA_PROXY_REAL_GPU = NUMA_PROXY_GPU_TYPES,
    NUMA_PROXY_VIRTUAL_GPU,

    SCALE_UP_GPU_TYPES,
    SCALE_UP_REAL_GPU = SCALE_UP_GPU_TYPES,
    SCALE_UP_VIRTUAL_GPU,

    MIX_SCALE_UP_NUMA_TYPES,
    MIX_SCALE_UP_NUMA_REAL = MIX_SCALE_UP_NUMA_TYPES,
    MIX_SCALE_UP_NUMA_VIRTUAL,

    SCALE_OUT_GPU_TYPES,
    SCALE_OUT_REAL_GPU = SCALE_OUT_GPU_TYPES,
    SCALE_OUT_VIRTUAL_GPU,

    MIX_SCALE_OUT_NUMA_TYPES,
    MIX_SCALE_OUT_NUMA_REAL = MIX_SCALE_OUT_NUMA_TYPES,
    MIX_SCALE_OUT_NUMA_VIRTUAL,

    MIX_SCALE_OUT_SCALE_UP_TYPES,
    MIX_SCALE_OUT_SCALE_UP_REAL = MIX_SCALE_OUT_SCALE_UP_TYPES,
    MIX_SCALE_OUT_SCALE_UP_VIRTUAL,

    MIX_UNIVERSAL_TYPES,
    MIX_UNIVERSAL_REAL = MIX_UNIVERSAL_TYPES,
    MIX_UNIVERSAL_VIRTUAL,

    MAX_TYPE
};

using gpu_type_names =
    ::utils::enum_to_str<static_cast<typename std::underlying_type<gpu_types>::type>(
        gpu_types::MAX_TYPE)>;
inline std::string to_string(gpu_types type) {
    return gpu_type_names({ "REAL_GPU",
                            "VIRTUAL_GPU",
                            "CONCURRENT_REAL_GPU",
                            "CONCURRENT_VIRT_GPU",
                            "SOURCE_IPC_REAL_GPU",
                            "SOURCE_IPC_VIRT_GPU",
                            "DESTINATION_IPC_GPU",
                            "NUMA_REAL_PROXY",
                            "NUMA_VIRT_PROXY",
                            "SUP_REAL_PROXY",
                            "SUP_VIRT_PROXY",
                            "MIX_SUP_NUMA_REAL",
                            "MIX_SUP_NUMA_VIRTUAL",
                            "SOUT_REAL_PROXY",
                            "SOUT_VIRT_PROXY",
                            "MIX_SOUT_NUMA_REAL",
                            "MIX_SIUT_NUMA_VIRT",
                            "MIX_SOUT_SUP_REAL",
                            "MIX_SOUT_SUP_VIRT",
                            "MIX_UNIVERSAL_REAL",
                            "MIX_UNIVERSAL_VIRT" })
        .choose(type, "INVALID_VALUE");
}

constexpr inline gpu_types operator+(gpu_types a,
                                     typename std::underlying_type<gpu_types>::type b) {
    return static_cast<gpu_types>(static_cast<typename std::underlying_type<gpu_types>::type>(a) +
                                  static_cast<typename std::underlying_type<gpu_types>::type>(b));
}

// devices
template <class device_t>
using device_t_ptr = std::shared_ptr<device_t>;
} // namespace native
