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
#include "common/comm/l0/topology/topology_construction_utils.hpp"
#if 0
namespace native
{
// First aggregate all devices into from different threads into plain vector represent devices from one host
// take action for simple enumerator on this plain vector
// check against connectivity from plain vector from other processes
// add IPC wrapper!

//use collcected devices count from cluster per hostname & processes to take offset for current devices!!!!
class allied_process_group_ring_topology
{
    size_t process_index;
    size_t process_count;
    process_group_context& context;
    device_storage& devices;
    size_t device_cluster_rank_offset;
    size_t device_cluster_size;

public:
    static constexpr ccl::group_split_type type()
    {
        return ccl::group_split_type::cluster;
    }

    static constexpr const char* name()
    {
        return "process_group_ring_creator";
    }

    allied_process_group_ring_topology(size_t process_idx,
                                       size_t process_nums,
                                       process_group_context &ctx, device_storage& devs,
                                       size_t cluster_rank_offset, size_t cluster_size);
    static std::pair<size_t, size_t>
        calculate_rank_offset_with_size(size_t process_id,
                                        const std::string& host_id,
                                        const ccl::cluster_aggregated_device_mask_t& cluster_affinity_mask);

    static size_t default_property_p2p_rating_calculator(const ccl_device &lhs, const ccl_device &rhs);
    static detail::adjacency_matrix build_p2p_capability_matrix(std::ostream& out,
                                                          const ccl::process_aggregated_device_mask_t &node_device_masks,
                                                          detail::p2p_rating_function ping =
                                                                        default_property_p2p_rating_calculator);
    static detail::adjacency_matrix build_p2p_capability_matrix(std::ostream& out,
                                                          const ccl::process_device_indices_type& node_device_indices,
                                                           detail::p2p_rating_function ping =
                                                                        default_property_p2p_rating_calculator);
    bool build(std::ostream& out,
               const ccl::process_aggregated_device_mask_t& per_thread_device_masks,
               const std::vector<ccl::device_mask_t>& ipc_device_indices,
               const detail::adjacency_matrix& matrix,
               detail::p2p_rating_function ping = default_property_p2p_rating_calculator);

    bool build(std::ostream& out,
               const ccl::process_device_indices_type& per_thread_device_indices,
               const std::vector<ccl::device_indices_type>& ipc_device_indices,
               const detail::adjacency_matrix& matrix,
               detail::p2p_rating_function ping = default_property_p2p_rating_calculator);

    bool build_all(std::ostream& out,
                   const ccl::process_device_indices_type& per_thread_device_indices,
                   const std::vector<ccl::device_indices_type>& ipc_device_indices,
                   const detail::adjacency_matrix& matrix,
                   detail::p2p_rating_function ping = default_property_p2p_rating_calculator);
private:
    bool build_specific(std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const detail::plain_graph& graph);
    bool build_specific(std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const detail::plain_graph_list& graph_list);
    bool build_specific(std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const ccl::device_indices_type& scaleout_device_indices,
                        const detail::plain_graph_list& graph_list);

    bool build_specific_colored(std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const ccl::process_device_indices_type& ipc_device_indices,
                        detail::colored_plain_graph& graph,
                        const std::map<size_t, size_t>& process_device_rank_offset);
    bool build_specific_scale_up(std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const ccl::process_device_indices_type& ipc_device_indices,
                        detail::colored_plain_graph_list& graph_list,
                        const std::map<size_t, size_t>& process_device_rank_offset);
    bool build_specific_scale_up_out(std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const ccl::process_device_indices_type& scaleout_device_indices,
                        const ccl::process_device_indices_type& ipc_device_indices,
                        detail::colored_plain_graph_list& graph_list,
                        const std::map<size_t, size_t>& process_device_rank_offset);

    detail::plain_graph_list
            create_my_process_graphs(std::ostream& out,
                                     const ccl::process_device_indices_type& per_thread_device_indices,
                                     const detail::adjacency_matrix& matrix,
                                     detail::p2p_rating_function ping = default_property_p2p_rating_calculator);

    detail::global_sorted_plain_graphs collect_cluster_plain_graphs(std::ostream& out,
                                                                  std::shared_ptr<ccl::communicator> comm,
                                                                  size_t process_index,
                                                                  const detail::plain_graph_list& my_process_graph);
    detail::global_sorted_colored_plain_graphs
                    collect_cluster_colored_plain_graphs(std::ostream& out,
                                                         std::shared_ptr<ccl::communicator> comm,
                                                         size_t process_index,
                                                         const detail::colored_plain_graph_list& my_process_graph);

    virtual detail::global_plain_graphs merge_allied_nodes_plain_graphs(std::ostream& out,
                                                                 const ccl::cluster_device_indices_type &cluster_indices,
                                                                 size_t process_index,
                                                                 const detail::global_sorted_plain_graphs& cluster_graphs,
                                                                 detail::p2p_rating_function ping = default_property_p2p_rating_calculator);
    virtual detail::global_colored_plain_graphs
                    merge_allied_nodes_in_colored_plain_graphs(std::ostream& out,
                                                               const ccl::cluster_device_indices_type &cluster_indices,
                                                               size_t process_index,
                                                               size_t process_count,
                                                               const detail::global_sorted_colored_plain_graphs& cluster_graphs,
                                                               detail::p2p_rating_function ping = default_property_p2p_rating_calculator);

    detail::plain_graph_list resize_merged_graphs_for_process(size_t process_index,
                                                               const detail::global_plain_graphs& merged_cluster_graphs,
                                                               const detail::plain_graph_list& original_graph_list,
                                                               std::ostream& out);
    detail::colored_plain_graph_list
                    resize_merged_colored_graphs_for_process(
                                        size_t process_index,
                                        size_t process_count,
                                        const detail::global_colored_plain_graphs& merged_cluster_graphs,
                                        const detail::colored_plain_graph_list& original_graph_list,
                                        std::ostream& out);

    virtual ccl::process_device_indices_type
                    create_scaleout_devices_in_graphs_for_process(
                                        size_t process_index,
                                        size_t cluster_size,
                                        detail::global_sorted_plain_graphs& cluster_graphs,
                                        std::ostream& out);
    virtual ccl::process_device_indices_type
                    create_scaleout_devices_in_colored_graphs_for_process(
                                        size_t process_index,
                                        size_t cluster_size,
                                        detail::global_sorted_colored_plain_graphs& cluster_graphs,
                                        detail::global_sorted_colored_plain_graphs& initial_cluster_graphs,
                                        std::ostream& out);
    virtual ccl::process_device_indices_type
                    create_ipc_devices_in_colored_graphs_for_process(
                                        size_t process_idx,
                                        size_t cluster_size,
                                        detail::global_sorted_colored_plain_graphs& cluster_graphs,
                                        detail::global_sorted_colored_plain_graphs& initial_cluster_graphs,
                                        std::ostream& out);

    detail::global_sorted_plain_graphs global_graph_list_resolver(const detail::adjacency_matrix& matrix,
                                                       const ccl::process_device_indices_type& per_process_device_indexes,
                                                       const ccl::process_device_indices_type& foreign_processes_device_indexes,
                                                       detail::p2p_rating_function ping);
};
}
#endif
