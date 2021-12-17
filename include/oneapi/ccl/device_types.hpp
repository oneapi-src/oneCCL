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
#error "Do not include this file directly. Please include 'ccl_types.hpp'"
#endif

namespace ccl {
/* TODO
 * Push the following code into something similar with 'ccl_device_types.hpp'
 */

using process_id = size_t;
using host_id = std::string;

#ifdef CCL_ENABLE_ZE
constexpr size_t CCL_GPU_DEVICES_AFFINITY_MASK_SIZE = 4;
using device_mask_t = std::bitset<CCL_GPU_DEVICES_AFFINITY_MASK_SIZE>;
using process_aggregated_device_mask_t = std::map<process_id, device_mask_t>;
using cluster_aggregated_device_mask_t = std::map<host_id, process_aggregated_device_mask_t>;
#endif

using index_type = uint32_t;
static constexpr index_type unused_index_value = std::numeric_limits<index_type>::max(); //TODO

//TODO implement class instead
using device_index_type = std::tuple<index_type, index_type, index_type>;
enum device_index_enum { driver_index_id, device_index_id, subdevice_index_id };
std::string to_string(const device_index_type& device_id);
device_index_type from_string(const std::string& device_id_str);

using device_indices_type = std::multiset<device_index_type>;
using process_device_indices_type = std::map<process_id, device_indices_type>;
using cluster_device_indices_type = std::map<host_id, process_device_indices_type>;

std::string to_string(const device_indices_type& indices);
std::string to_string(const process_device_indices_type& indices);
std::string to_string(const cluster_device_indices_type& indices);

struct empty_t {};

template <cl_backend_type config_backend>
struct backend_info {};

template <cl_backend_type config_backend>
struct generic_device_type {};

template <cl_backend_type config_backend>
struct generic_context_type {};

template <cl_backend_type config_backend>
struct generic_platform_type {};

template <cl_backend_type config_backend>
struct generic_stream_type {};

template <cl_backend_type config_backend>
struct generic_event_type {};

template <class type>
struct api_type_info {
    static constexpr bool is_supported() {
        return false;
    }
    static constexpr bool is_class() {
        return false;
    }
};

#define API_CLASS_TYPE_INFO(api_type) \
    template <> \
    struct api_type_info<api_type> { \
        static constexpr bool is_supported() { \
            return true; \
        } \
        static constexpr bool is_class() { \
            return std::is_class<api_type>::value; \
        } \
    };
} // namespace ccl

std::ostream& operator<<(std::ostream& out, const ccl::device_index_type&);
std::ostream& operator>>(std::ostream& out, const ccl::device_index_type&);
