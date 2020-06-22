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

#ifdef MULTI_GPU_SUPPORT
#ifndef CCL_PRODUCT_FULL
    #error "Do not include this file directly. Please include 'ccl_types.hpp'"
#endif

namespace ccl
{
/* TODO
 * Push the following code into something similar with 'ccl_device_types.hpp'
 */
enum device_topology_type
{
    device_group_ring,
    device_group_torn_apart_ring,
    thread_group_ring,
    thread_group_torn_apart_ring,
    allied_process_group_ring,
    process_group_torn_apart_ring,
    a2a_device_group,
    a2a_thread_group,
    a2a_allied_process_group,

    last_value
};

#define SUPPORTED_HW_TOPOLOGIES_DECL_LIST       ccl::device_topology_type::device_group_ring,               \
                                                ccl::device_topology_type::device_group_torn_apart_ring,    \
                                                ccl::device_topology_type::thread_group_ring,               \
                                                ccl::device_topology_type::thread_group_torn_apart_ring,    \
                                                ccl::device_topology_type::allied_process_group_ring,       \
                                                ccl::device_topology_type::process_group_torn_apart_ring,   \
                                                ccl::device_topology_type::a2a_device_group,                \
                                                ccl::device_topology_type::a2a_thread_group,                \
                                                ccl::device_topology_type::a2a_allied_process_group

#define DEVICE_GROUP_TOPOLOGIES_DECL_LIST       ccl::device_topology_type::device_group_ring,               \
                                                ccl::device_topology_type::device_group_torn_apart_ring,    \
                                                ccl::device_topology_type::a2a_device_group

#define THREAD_GROUP_TOPOLOGIES_DECL_LIST       ccl::device_topology_type::thread_group_ring,               \
                                                ccl::device_topology_type::thread_group_torn_apart_ring,    \
                                                ccl::device_topology_type::a2a_thread_group

#define PROCESS_GROUP_TOPOLOGIES_DECL_LIST      ccl::device_topology_type::allied_process_group_ring,       \
                                                ccl::device_topology_type::process_group_torn_apart_ring,   \
                                                ccl::device_topology_type::a2a_allied_process_group

enum device_topology_class
{
    ring_class = ring_algo_class,
    a2a_class = a2a_algo_class,

    last_class_value = ccl_topology_class_last_value
};

enum device_topology_group
{
    dev_group = device_group,
    thread_dev_group = thread_group,
    process_dev_group = process_group,

    last_group_value = ccl_topology_group_last_value
};

template <device_topology_type topology>
constexpr device_topology_class topology_to_class()
{
    return std::tuple_element<topology, std::tuple<std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::ring_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::ring_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::ring_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::ring_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::ring_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::ring_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::a2a_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::a2a_class>,
                                                   std::integral_constant<ccl::device_topology_class, ccl::device_topology_class::a2a_class>>>::type::value;

}

template <device_topology_type topology>
constexpr device_topology_group topology_to_group()
{
    return std::tuple_element<topology,
                std::tuple<std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::thread_dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::thread_dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::process_dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::process_dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::thread_dev_group>,
                           std::integral_constant<ccl::device_topology_group,
                                                  ccl::device_topology_group::process_dev_group>
                           >>::type::value;

}
template<ccl_device_attributes attrId>
struct ccl_device_attributes_traits {};

using process_id = size_t;
using host_id = std::string;

using device_mask_t = std::bitset<CCL_GPU_DEVICES_AFFINITY_MASK_SIZE>;
using process_aggregated_device_mask_t = std::map<process_id, device_mask_t>;
using cluster_aggregated_device_mask_t = std::map<host_id, process_aggregated_device_mask_t>;

using index_type = uint32_t;
static constexpr index_type unused_index_value = std::numeric_limits<index_type>::max(); //TODO
//TODO implement class instead
using device_index_type = std::tuple<index_type, index_type, index_type>;
enum device_index_enum
{
    driver_index_id,
    device_index_id,
    subdevice_index_id
};
std::string to_string(const device_index_type& device_id);
device_index_type from_string(const std::string& device_id_str);

using device_indices_t = std::multiset<device_index_type>;
using process_device_indices_t = std::map<process_id, device_indices_t>;
using cluster_device_indices_t = std::map<host_id, process_device_indices_t>;

template<int sycl_feature_enabled>
struct generic_device_type {};

template<int sycl_feature_enabled>
struct generic_device_context_type {};
}

std::ostream& operator<<(std::ostream& out, const ccl::device_index_type&);
std::ostream& operator>>(std::ostream& out, const ccl::device_index_type&);

#endif //MULTI_GPU_SUPPORT
