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
#include <vector>
#include "oneapi/ccl/native_device_api/l0/device.hpp"

namespace native {
namespace utils {
size_t serialize_ipc_handles(const std::vector<ccl_device::device_ipc_memory_handle>& in_ipc_memory,
                             std::vector<uint8_t>& out_raw_data,
                             size_t out_raw_data_initial_offset_bytes);
/*
    size_t deserialize_ipc_handles(const std::vector<ccl_device::device_ipc_memory_handle>& in_ipc_memory,
                                 std::vector<uint8_t>& out_raw_data,
                                 size_t out_raw_data_initial_offset_bytes);*/
} // namespace utils
} // namespace native
