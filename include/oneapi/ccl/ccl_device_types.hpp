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

/** Device topology group. */
typedef enum {
    device_group = 0,
    thread_group = 1,
    process_group = 2,

    ccl_topology_group_last_value
} ccl_topology_group_t;

enum device_topology_type { undetermined = -1, ring, a2a, last_class_value };

// TODO: tmp mapping
#define ring_algo_class               device_topology_type::ring
#define a2a_algo_class                device_topology_type::a2a
#define ccl_topology_class_last_value device_topology_type::last_class_value

using process_id = size_t;
using host_id = std::string;

#ifdef MULTI_GPU_SUPPORT
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

using device_indices_t = std::multiset<device_index_type>;
using process_device_indices_t = std::map<process_id, device_indices_t>;
using cluster_device_indices_t = std::map<host_id, process_device_indices_t>;

template <int sycl_feature_enabled>
struct generic_device_type {};

template <int sycl_feature_enabled>
struct generic_device_context_type {};

template <int sycl_feature_enabled>
struct generic_platform_type {};

template <int sycl_feature_enabled>
struct generic_stream_type {};

template <int sycl_feature_enabled>
struct generic_event_type {};
} // namespace ccl

std::ostream& operator<<(std::ostream& out, const ccl::device_index_type&);
std::ostream& operator>>(std::ostream& out, const ccl::device_index_type&);
