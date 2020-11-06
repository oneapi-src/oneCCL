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
#include <list>

#include "common/comm/l0/device_community.hpp"
#include "common/comm/l0/gpu_device_types.hpp"
#include "common/comm/l0/topology/topology_creator.hpp"
#include "common/comm/l0/topology/topology_declarations.hpp"
#include "oneapi/ccl/native_device_api/l0/utils.hpp"

class device_group_router;
#define DEVICE_GROUP_WEIGHT  9
#define THREAD_GROUP_WEIGHT  5
#define PROCESS_GROUP_WEIGHT 2

namespace native {
struct process_group_context;
struct thread_group_context;
struct device_group_context;
struct device_storage;
struct ccl_device;

namespace details {

adjacency_matrix create_adjacency_matrix_for_devices(
    const ccl_device_driver::devices_storage_type& devices,
    p2p_rating_function ping);

void fill_adjacency_matrix_for_single_device_in_devices(
    const native::ccl_device& lhs_device,
    const ccl::device_index_type& lhs_index,
    const ccl_device_driver::devices_storage_type& devices,
    adjacency_matrix& matrix,
    p2p_rating_function ping);

void fill_adjacency_matrix_for_single_device_in_devices_by_cond(
    const native::ccl_device& lhs_device,
    const ccl::device_index_type& lhs_index,
    const ccl_device_driver::devices_storage_type& devices,
    adjacency_matrix& matrix,
    p2p_rating_function ping,
    std::function<bool(const ccl::device_index_type&)> rhs_filter =
        std::function<bool(const ccl::device_index_type&)>());

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::device_indices_t& device_indexes);
plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::process_device_indices_t& per_process_device_indexes);
plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::process_aggregated_device_mask_t& per_process_device_masks);

plain_graph_list graph_list_resolver(const adjacency_matrix& matrix,
                                     const ccl::device_indices_t& device_indexes);
plain_graph_list graph_list_resolver(
    const adjacency_matrix& matrix,
    const ccl::process_device_indices_t& per_process_device_indexes,
    details::p2p_rating_function ping);

plain_graph_list graph_list_resolver(
    const adjacency_matrix& matrix,
    const ccl::process_aggregated_device_mask_t& per_process_device_masks);

bool check_graph_a2a_capable(const plain_graph& graph,
                             const adjacency_matrix& matrix,
                             std::ostream& out);

plain_graph_list merge_graph_lists_stable(const std::list<plain_graph_list>& lists,
                                          details::p2p_rating_function ping,
                                          bool brake_on_incompatible = false);

colored_plain_graph_list merge_graph_lists_stable(const std::list<colored_plain_graph_list>& lists,
                                                  details::p2p_rating_function ping,
                                                  bool brake_on_incompatible = false);
colored_plain_graph_list merge_graph_lists_stable_for_process(
    const std::list<colored_plain_graph_list>& lists,
    details::p2p_rating_function ping,
    bool to_right,
    size_t& merged_process_count);

size_t property_p2p_rating_calculator(const native::ccl_device& lhs,
                                      const native::ccl_device& rhs,
                                      size_t weight);

void reset_color(colored_plain_graph_list& list, color_t new_color);
} // namespace details
std::ostream& operator<<(std::ostream& out, const details::adjacency_matrix& matrix);

} // namespace native
