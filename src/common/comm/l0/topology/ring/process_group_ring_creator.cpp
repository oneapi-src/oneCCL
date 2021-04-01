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
#include "common/comm/l0/topology/ring/ring_construction_utils.hpp"
#include "common/comm/l0/topology/ring/device_group_ring_creator.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

#include "common/comm/l0/topology/topology_serializer.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"

namespace native {

allied_process_group_ring_topology::allied_process_group_ring_topology(
    size_t process_idx,
    size_t process_nums,
    process_group_context& ctx,
    device_storage& devs,
    size_t cluster_rank_offset,
    size_t cluster_size,
    const ccl::context_comm_addr& comm_addr)
        : process_index(process_idx),
          process_count(process_nums),
          context(ctx),
          devices(devs),
          device_cluster_rank_offset(cluster_rank_offset),
          device_cluster_size(cluster_size),
          ctx_comm_addr(comm_addr) {}

size_t allied_process_group_ring_topology::default_property_p2p_rating_calculator(
    const ccl_device& lhs,
    const ccl_device& rhs) {
    return detail::property_p2p_rating_calculator(lhs, rhs, PROCESS_GROUP_WEIGHT);
}

std::pair<size_t, size_t> allied_process_group_ring_topology::calculate_rank_offset_with_size(
    size_t process_id,
    const std::string& host_id,
    const ccl::cluster_aggregated_device_mask_t& cluster_affinity_mask) {
    auto from_begin = [](const ccl::process_aggregated_device_mask_t& processes) ->
        typename ccl::process_aggregated_device_mask_t::const_iterator {
            return processes.begin();
        };
    auto from_my_rank = [process_id](const ccl::process_aggregated_device_mask_t& processes) ->
        typename ccl::process_aggregated_device_mask_t::const_iterator {
            return processes.lower_bound(process_id);
        };

    auto till_my_rank = from_my_rank;
    auto till_end = [](const ccl::process_aggregated_device_mask_t& processes) ->
        typename ccl::process_aggregated_device_mask_t::const_iterator {
            return processes.end();
        };

    auto device_summator =
        [](size_t part_sum,
           const ccl::process_aggregated_device_mask_t::value_type& mask) -> size_t {
        return part_sum + mask.second.count();
    };

    auto left_rank_summator =
        [from_begin, till_my_rank, device_summator](
            size_t part_sum,
            const typename ccl::cluster_aggregated_device_mask_t::value_type& processes_pair)
        -> size_t {
        return std::accumulate(from_begin(processes_pair.second),
                               till_my_rank(processes_pair.second),
                               part_sum,
                               device_summator);
    };
    auto right_rank_summator =
        [from_my_rank, till_end, device_summator](
            size_t part_sum,
            const typename ccl::cluster_aggregated_device_mask_t::value_type& processes_pair)
        -> size_t {
        return std::accumulate(from_my_rank(processes_pair.second),
                               till_end(processes_pair.second),
                               part_sum,
                               device_summator);
    };
    auto rank_summator =
        [from_begin, till_end, device_summator](
            size_t part_sum,
            const typename ccl::cluster_aggregated_device_mask_t::value_type& processes_pair)
        -> size_t {
        return std::accumulate(from_begin(processes_pair.second),
                               till_end(processes_pair.second),
                               part_sum,
                               device_summator);
    };

    //calculate ranks offset: summ of devices for each process for each node
    //TODO node sorted by lexicographic comparison
    auto my_node_it = cluster_affinity_mask.find(host_id);

    size_t my_node_rank_devices_offset =
        std::accumulate(cluster_affinity_mask.begin(), my_node_it, 0, rank_summator);
    my_node_rank_devices_offset = std::accumulate(
        my_node_it, std::next(my_node_it), my_node_rank_devices_offset, left_rank_summator);

    size_t cluster_devices_count = std::accumulate(
        my_node_it, std::next(my_node_it), my_node_rank_devices_offset, right_rank_summator);
    cluster_devices_count = std::accumulate(
        my_node_it, cluster_affinity_mask.end(), cluster_devices_count, rank_summator);
    return { my_node_rank_devices_offset, cluster_devices_count };
}

detail::adjacency_matrix allied_process_group_ring_topology::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::process_aggregated_device_mask_t& node_device_masks,
    detail::p2p_rating_function ping) {
    ccl::process_device_indices_type per_process_device_indices;
    for (const auto& mask : node_device_masks) {
        per_process_device_indices.insert(
            { mask.first, ccl_device_driver::get_device_indices(mask.second) });
    }

    return build_p2p_capability_matrix(out, per_process_device_indices, ping);
}

detail::adjacency_matrix allied_process_group_ring_topology::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::process_device_indices_type& node_device_indices,
    detail::p2p_rating_function ping) {
    // Build adjacency matrix with P2P capability:
    // Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // element values - is a weight of P2P activity: 0 means - devices are not connected
    // If values is not 0 - than two devies can be combined together

    detail::adjacency_matrix ring_p2p_matrix;
    if (node_device_indices.empty()) {
        out << "No indices nothing to build" << std::endl;
        return ring_p2p_matrix;
    }

    out << "Build adjacency matrix by: " << allied_process_group_ring_topology::name() << std::endl;
    out << "Processes count: " << node_device_indices.size() << "\t";
    out << "Delegate to thread group ring" << std::endl;
    return thread_group_ring_topology::build_p2p_capability_matrix(out, node_device_indices, ping);
}

bool allied_process_group_ring_topology::build_all(
    std::ostream& out,
    const ccl::process_device_indices_type& per_thread_device_indices,
    const detail::adjacency_matrix& matrix,
    detail::p2p_rating_function ping) {
    const std::string& threads_indices_str = ccl::to_string(per_thread_device_indices);
    LOG_DEBUG("\n/************* \"",
              allied_process_group_ring_topology::name(),
              "\" for threads: ",
              context.process_device_topology.size(),
              "*************/\n");

    LOG_DEBUG("Process: ", process_index, ", threads indices: ", threads_indices_str);
    out << "Build process group device graphs, from threads: " << per_thread_device_indices.size()
        << ", threads indices: \n"
        << threads_indices_str << std::endl;

    detail::plain_graph_list my_rings =
        create_my_process_graphs(per_thread_device_indices, matrix, ping);
    size_t size = my_rings.size();
    LOG_DEBUG("Resolved graphs count: ", size, ", process_index: ", process_index);
    if (!size) {
        out << "Cannot build any ring" << std::endl;
        return false;
    }

    {
        const std::string& graph_to_str = detail::to_string(my_rings);
        out << "Graph for process: " << process_index << "\n";
        out << graph_to_str << std::endl;

        LOG_DEBUG("Graph for process: ", process_index, " resolved:\n", graph_to_str);
    }

    out << "Transform graph to colored with process color: " << process_index << "\n";
    detail::colored_plain_graph_list my_colored_ring =
        detail::create_colored(my_rings, process_index);

    detail::global_sorted_colored_plain_graphs global_graphs;
    context.collect_cluster_colored_plain_graphs(my_colored_ring, global_graphs);

    std::map<size_t, size_t> process_device_rank_offset;
    size_t accumulated_offset = 0;

    out << "Print ranks offset in cluster for global graphs: " << global_graphs.size() << std::endl;
    for (typename detail::global_sorted_colored_plain_graphs::value_type& process_graphs :
         global_graphs) {
        size_t process_num = process_graphs.first;
        const detail::colored_plain_graph_list& proc_graphs = process_graphs.second;

        process_device_rank_offset[process_num] = accumulated_offset; //offset for iter process
        out << "Process idx: " << process_num << ", rank_offset: " << accumulated_offset
            << std::endl;
        for (const detail::colored_plain_graph& graph : proc_graphs) {
            accumulated_offset += graph.size();
        }
    }

    out << "Cluster device size: " << accumulated_offset << std::endl;
    detail::global_colored_plain_graphs merged_cluster_graphs =
        merge_allied_nodes_in_colored_plain_graphs(
            out, context.cluster_gpu_indices, process_index, process_count, global_graphs, ping);

    const std::string& merged_cluster_graphs_str = detail::to_string(merged_cluster_graphs);
    LOG_INFO("Cluster merged graphs process idx: ",
             process_index,
             " result:\n",
             merged_cluster_graphs_str);
    out << "Cluster merged graphs result on process idx: " << process_index << std::endl;
    out << merged_cluster_graphs_str << std::endl;

    detail::colored_plain_graph_list my_merged_rings = resize_merged_colored_graphs_for_process(
        process_index, process_count, merged_cluster_graphs, my_colored_ring, out);

    const std::string& my_merged_rings_str = detail::to_string(my_merged_rings);
    LOG_INFO("Resized merged graph list on process idx: ",
             process_index,
             " result:\n",
             my_merged_rings_str);
    out << "Resized merged graph list on process idx: " << process_index << std::endl;

    out << "Notify merged graphs changes for cluster\n";
    detail::global_sorted_colored_plain_graphs global_merged_graphs;
    context.collect_cluster_colored_plain_graphs(my_merged_rings, global_merged_graphs);

    ccl::process_device_indices_type scaleout_devices =
        create_scaleout_devices_in_colored_graphs_for_process(
            process_index, process_count, global_merged_graphs, global_graphs, out);
    const std::string& scaleout_devices_str = ccl::to_string(scaleout_devices);
    LOG_INFO("Collected scaleout devices on process: ",
             process_index,
             " result:\n",
             scaleout_devices_str);
    out << "Collected scaleout_devices: \n";
    out << scaleout_devices_str << std::endl;

    ccl::process_device_indices_type ipc_devices = create_ipc_devices_in_colored_graphs_for_process(
        process_index, process_count, global_merged_graphs, global_graphs, out);
    const std::string& ipc_devices_str = ccl::to_string(ipc_devices);
    LOG_INFO("Collected IPC devices on process: ", process_index, " result:\n", ipc_devices_str);
    out << "Collected ipc_devices: \n";
    out << ipc_devices_str << std::endl;

    // enumerate as usual
    if (scaleout_devices.empty()) {
        size_t size = my_merged_rings.size();
        out << "Resolved graphs count: " << size << "\n";
        if (!size) {
            out << "Cannot build any ring" << std::endl;
            return false;
        }
        else if (size == 1) // whole ring
        {
            return build_specific_colored(out,
                                          per_thread_device_indices,
                                          ipc_devices,
                                          *my_merged_rings.begin(),
                                          process_device_rank_offset);
        }
        //torn-apart ring
        return build_specific_scale_up(out,
                                       per_thread_device_indices,
                                       ipc_devices,
                                       my_merged_rings,
                                       process_device_rank_offset);
    }
    else if (ipc_devices.empty()) {
        //pure scale-out
        return build_specific_scale_out_only(out,
                                             per_thread_device_indices,
                                             scaleout_devices,
                                             my_merged_rings,
                                             process_device_rank_offset);
    }
    else {
        throw std::runtime_error(
            "torn-apart ring with scaleout\n"
            "return build_specific_scale_up_out(out, per_thread_device_indices,\n"
            "scaleout_devices, ipc_devices,\n"
            "my_merged_rings, process_device_rank_offset)\n"
            "UNSUPPORTED");
    }
    return false;
}

detail::plain_graph_list allied_process_group_ring_topology::create_my_process_graphs(
    const ccl::process_device_indices_type& per_thread_device_indices,
    const detail::adjacency_matrix& matrix,
    detail::p2p_rating_function ping) {
    return detail::graph_list_resolver(matrix, per_thread_device_indices, ping);
}
detail::global_sorted_plain_graphs allied_process_group_ring_topology::collect_cluster_plain_graphs(
    std::ostream& out,
    std::shared_ptr<ccl::host_communicator> comm,
    size_t process_index,
    const detail::plain_graph_list& my_process_graph) {
    using namespace detail::serialize;

    out << "Collect cluster plain graphs, my process index: " << process_index
        << ", graphs count: " << my_process_graph.size() << std::endl;

    std::vector<size_t> recv_process_indices_counts(comm->size(), 1);
    device_path_serializable::raw_data_t my_serialized_graph =
        device_path_serializer::serialize_indices(my_process_graph);

    size_t send_count = my_serialized_graph.size();
    std::vector<size_t> receive_process_graph_sizes(comm->size());

    out << "Ask graph lists sizes by process index: " << process_index
        << ", serialized size: " << send_count << std::endl;
    ccl::stream::impl_value_t empty_stream{};
    auto req = comm->allgatherv_impl(&send_count,
                                     1,
                                     receive_process_graph_sizes.data(),
                                     recv_process_indices_counts,
                                     empty_stream,
                                     ccl::default_allgatherv_attr,
                                     {});

    req.wait();
    size_t global_graph_data_size =
        std::accumulate(receive_process_graph_sizes.begin(), receive_process_graph_sizes.end(), 0);

    device_path_serializable::raw_data_t global_serialized_graph;
    try {
        out << "Send graph list by process index: " << process_index
            << ", serialized size: " << send_count << std::endl;

        global_serialized_graph.resize(global_graph_data_size);
        req = comm->allgatherv_impl(reinterpret_cast<void*>(my_serialized_graph.data()),
                                    send_count,
                                    reinterpret_cast<void*>(global_serialized_graph.data()),
                                    receive_process_graph_sizes,
                                    ccl::datatype::int8,
                                    empty_stream,
                                    ccl::default_allgatherv_attr,
                                    {});
        req.wait();
    }
    catch (const std::exception& ex) {
        out << "Cannot submit global-serialized-graph requests " << ex.what() << std::endl;
        out << "Memory required for hostnames size: " << global_graph_data_size << " bytes\n";
        abort();
    }

    size_t deserialized_bytes = 0;
    size_t offset_bytes = 0;
    detail::global_sorted_plain_graphs global_ret;

    out << "Deserialize graph_lists" << std::endl;
    for (size_t i = 0; i < static_cast<size_t>(comm->size()); i++) {
        detail::plain_graph_list graph = device_path_deserializer::deserialize_graph_list_indices(
            global_serialized_graph, deserialized_bytes, offset_bytes);
        out << "Process index: " << i << ", deserialized bytes: " << deserialized_bytes
            << ", by offset: " << offset_bytes << std::endl;

        global_ret.emplace(i, std::move(graph));
    }

    out << "Global graph deserialized on process: " << process_index << std::endl;
    return global_ret;
}

detail::global_sorted_colored_plain_graphs
allied_process_group_ring_topology::collect_cluster_colored_plain_graphs(
    std::ostream& out,
    std::shared_ptr<ccl::host_communicator> comm,
    size_t process_index,
    const detail::colored_plain_graph_list& my_process_graph) {
    using namespace detail::serialize;

    out << "Collect cluster colored plain graphs, my process index: " << process_index
        << ", graphs count: " << my_process_graph.size() << std::endl;

    std::vector<size_t> recv_process_indices_counts(comm->size(), 1);
    device_path_serializable::raw_data_t my_serialized_graph =
        device_path_serializer::serialize_indices(my_process_graph);

    size_t send_count = my_serialized_graph.size();
    std::vector<size_t> receive_process_graph_sizes(comm->size());

    out << "Ask graph lists sizes by process index: " << process_index
        << ", serialized size: " << send_count << std::endl;
    ccl::stream::impl_value_t empty_stream{};
    auto req = comm->allgatherv_impl(&send_count,
                                     1,
                                     receive_process_graph_sizes.data(),
                                     recv_process_indices_counts,
                                     empty_stream,
                                     ccl::default_allgatherv_attr,
                                     {});

    req.wait();
    size_t global_graph_data_size =
        std::accumulate(receive_process_graph_sizes.begin(), receive_process_graph_sizes.end(), 0);

    device_path_serializable::raw_data_t global_serialized_graph;
    try {
        out << "Send graph list by process index: " << process_index
            << ", serialized size: " << send_count << std::endl;

        global_serialized_graph.resize(global_graph_data_size);
        req = comm->allgatherv_impl(reinterpret_cast<void*>(my_serialized_graph.data()),
                                    send_count,
                                    reinterpret_cast<void*>(global_serialized_graph.data()),
                                    receive_process_graph_sizes,
                                    ccl::datatype::int8,
                                    empty_stream,
                                    ccl::default_allgatherv_attr,
                                    {});
        req.wait();
    }
    catch (const std::exception& ex) {
        out << "Cannot submit global-serialized-graph requests " << ex.what() << std::endl;
        out << "Memory required for hostnames size: " << global_graph_data_size << " bytes\n";
        abort();
    }

    size_t deserialized_bytes = 0;
    size_t offset_bytes = 0;
    detail::global_sorted_colored_plain_graphs global_ret;

    out << "Deserialize colored_graph_lists" << std::endl;
    for (size_t i = 0; i < static_cast<size_t>(comm->size()); i++) {
        detail::colored_plain_graph_list graph =
            device_path_deserializer::deserialize_colored_graph_list_indices(
                global_serialized_graph, deserialized_bytes, offset_bytes);
        out << "Process index: " << i << ", deserialized bytes: " << deserialized_bytes
            << ", by offset: " << offset_bytes << std::endl;

        global_ret.emplace(i, std::move(graph));
    }

    out << "Global colored_graph deserialized on process: " << process_index << std::endl;
    return global_ret;
}

detail::global_plain_graphs allied_process_group_ring_topology::merge_allied_nodes_plain_graphs(
    std::ostream& out,
    const ccl::cluster_device_indices_type& cluster_indices,
    size_t process_index,
    const detail::global_sorted_plain_graphs& cluster_graphs,
    detail::p2p_rating_function ping) {
    out << "Merge global graphs from processes: " << cluster_graphs.size() << std::endl;
    detail::global_plain_graphs ret;
    for (const auto& host_process_id_pair : cluster_indices) {
        const ccl::host_id& hostname = host_process_id_pair.first;

        //iterate over all allied processes on the same host
        const ccl::process_device_indices_type& processes = host_process_id_pair.second;
        out << "Try to merge graphs for host: " << hostname
            << ", allied processes count: " << processes.size() << std::endl;

        //collect graphs for all allied processes in lists for merge trying
        std::list<detail::plain_graph_list> tmp_allied_processes_graphs;
        for (const auto& process_val : processes) {
            auto process_id = process_val.first;
            auto process_graph_list_it = cluster_graphs.find(process_id);
            if (process_graph_list_it == cluster_graphs.end()) {
                out << "Cannot find process id: " << process_id << ", for hostname: " << hostname
                    << ", in cluster graphs\n";
                std::stringstream ss;
                ss << out.rdbuf();
                throw std::runtime_error(std::string("Cannot merge custer graphs. Log:\n") +
                                         ss.str());
            }
            tmp_allied_processes_graphs.emplace_back(process_graph_list_it->second);
        }

        //merge and set result for all allied processes
        for (const auto& process_val : processes) {
            //merge_lists is stable, let's my process graph list at first in merge result
            std::list<detail::plain_graph_list> rotated = tmp_allied_processes_graphs;
            /* TODO rotate ? */
            auto process_index = process_val.first;

            auto new_begin_it = rotated.begin();
            std::advance(new_begin_it, process_index);
            std::rotate(rotated.begin(), new_begin_it, rotated.end());

            ret.push_back(
                std::make_pair(process_val.first, detail::merge_graph_lists_stable(rotated, ping)));
        }

        out << "graph merged into list, size: " << ret.size() << std::endl;
    }
    return ret;
}

detail::global_colored_plain_graphs
allied_process_group_ring_topology::merge_allied_nodes_in_colored_plain_graphs(
    std::ostream& out,
    const ccl::cluster_device_indices_type& cluster_indices,
    size_t process_index,
    size_t process_count,
    const detail::global_sorted_colored_plain_graphs& cluster_graphs,
    detail::p2p_rating_function ping) {
    out << "Merge global colored graphs from processes: " << cluster_graphs.size() << std::endl;
    detail::global_colored_plain_graphs ret;
    for (const auto& host_process_id_pair : cluster_indices) {
        const ccl::host_id& hostname = host_process_id_pair.first;

        //iterate over all allied processes on the same host
        const ccl::process_device_indices_type& processes = host_process_id_pair.second;
        out << "Try to merge colored graphs for host: " << hostname
            << ", allied processes count: " << processes.size() << std::endl;

        //collect graphs for all allied processes in lists for merge trying
        std::list<detail::colored_plain_graph_list> tmp_allied_processes_graphs;

        size_t terminator_process_index = 0; // TODO LIMITATION on MAX PROCESSES COUNT
        for (const auto& process_val : processes) {
            auto process_id = process_val.first;
            auto process_graph_list_it = cluster_graphs.find(process_id);
            if (process_graph_list_it == cluster_graphs.end()) {
                out << "Cannot find process id: " << process_id << ", for hostname: " << hostname
                    << ", in cluster graphs\n";
                std::stringstream ss;
                ss << out.rdbuf();

                assert(false);
                throw std::runtime_error(std::string("Cannot merge colored custer graphs. Log:\n") +
                                         ss.str());
            }
            tmp_allied_processes_graphs.emplace_back(process_graph_list_it->second);

            terminator_process_index = std::max(process_val.first, terminator_process_index);
        }

        terminator_process_index++;
        out << "terminator_process_index: " << terminator_process_index;

        //merge and set result for all allied processes
        for (const auto& process_val : processes) {
            //merge_lists is stable, let's my process graph list at first in merge result
            auto process_index = process_val.first;

            //turn right
            auto new_begin_it = tmp_allied_processes_graphs.begin();
            std::advance(new_begin_it, process_index);
            std::list<detail::colored_plain_graph_list> to_right_part(
                new_begin_it, tmp_allied_processes_graphs.end());

            //use terminator!
            if (processes.size() != 1) {
                if (process_index == processes.size() - 1) {
                    //set terminator for right side
                    detail::colored_plain_graph_list terminated_list =
                        *tmp_allied_processes_graphs.begin();
                    reset_color(terminated_list, terminator_process_index);
                    to_right_part.push_back(std::move(terminated_list));
                }
            }

            size_t merged_from_right = 0;
            detail::colored_plain_graph_list to_right =
                detail::merge_graph_lists_stable_for_process(
                    to_right_part, ping, true, merged_from_right);
            if (to_right.empty()) //i am the rightest process
            {
                to_right = *new_begin_it;
            }

            //turn left
            size_t merged_from_left = 0;
            auto new_end_it = tmp_allied_processes_graphs.begin();
            std::advance(new_end_it, process_index + 1);
            std::list<detail::colored_plain_graph_list> to_left_part(
                tmp_allied_processes_graphs.begin(), new_end_it);
            std::reverse(to_left_part.begin(), to_left_part.end());
            if (to_left_part.empty()) {
                to_left_part.push_back(to_right);
            }
            else {
                *to_left_part.begin() = to_right;
            }

            //use terminator!
            if (processes.size() != 1) {
                if (process_index == 0) {
                    //set terminator for right side
                    detail::colored_plain_graph_list terminated_list =
                        *tmp_allied_processes_graphs.rbegin();
                    reset_color(terminated_list, terminator_process_index);
                    to_left_part.push_back(std::move(terminated_list));
                }
            }
            for (auto& graph : to_left_part) {
                std::reverse(graph.begin(), graph.end());
            }
            *to_left_part.begin() = to_right;

            detail::colored_plain_graph_list to_left_right =
                detail::merge_graph_lists_stable_for_process(
                    to_left_part, ping, false, merged_from_left);
            ret.push_back(std::make_pair(process_val.first, to_left_right));
        }

        out << "colored graph merged into list, size: " << ret.size() << std::endl;
    }
    return ret;
}

detail::plain_graph_list allied_process_group_ring_topology::resize_merged_graphs_for_process(
    size_t process_index,
    const detail::global_plain_graphs& merged_cluster_graphs,
    const detail::plain_graph_list& original_graph_list,
    std::ostream& out) {
    out << "remove foreign chains from my merged graphs for process idx: " << process_index << "\n";
    auto it =
        std::find_if(merged_cluster_graphs.begin(),
                     merged_cluster_graphs.end(),
                     [process_index](const typename detail::global_plain_graphs::value_type& val) {
                         return val.first == process_index;
                     });
    if (it == merged_cluster_graphs.end()) {
        out << "Cannot find process: " << process_index
            << " in merged_cluster_graphs with size: " << merged_cluster_graphs.size() << std::endl;
        std::stringstream ss;
        ss << out.rdbuf();
        assert(false);
        throw std::runtime_error(std::string("Cannot resize custer graphs. Log:\n") + ss.str());
    }

    detail::plain_graph_list my_merged_rings_copy = it->second;
    {
        size_t new_size = my_merged_rings_copy.size();
        size_t old_size = original_graph_list.size();

        out << "Check ring sizes, before: " << old_size << ", after: " << new_size << std::endl;
        if (old_size > new_size) {
            abort();
        }

        auto merged_erased_range_it = my_merged_rings_copy.begin();
        std::advance(merged_erased_range_it, old_size);
        my_merged_rings_copy.erase(merged_erased_range_it, my_merged_rings_copy.end());
    }
    return my_merged_rings_copy;
}

detail::colored_plain_graph_list
allied_process_group_ring_topology::resize_merged_colored_graphs_for_process(
    size_t process_index,
    size_t process_size,
    const detail::global_colored_plain_graphs& merged_cluster_graphs,
    const detail::colored_plain_graph_list& original_graph_list,
    std::ostream& out) {
    out << "remove foreign chains from my colored merged graphs for process idx: " << process_index
        << "\n";
    auto it = std::find_if(
        merged_cluster_graphs.begin(),
        merged_cluster_graphs.end(),
        [process_index](const typename detail::global_colored_plain_graphs::value_type& val) {
            return val.first == process_index;
        });
    if (it == merged_cluster_graphs.end()) {
        out << "Cannot find process: " << process_index
            << " in merged_cluster_graphs with size: " << merged_cluster_graphs.size() << std::endl;
        std::stringstream ss;
        ss << out.rdbuf();
        throw std::runtime_error(std::string("Cannot resize colored custer graphs. Log:\n") +
                                 ss.str());
    }

    detail::colored_plain_graph_list my_merged_rings_copy = it->second;
    {
        size_t new_size = my_merged_rings_copy.size();
        size_t old_size = original_graph_list.size();

        out << "Check ring sizes, before: " << old_size << ", after: " << new_size << std::endl;
        if (old_size > new_size) {
            abort();
        }

        auto merged_erased_range_it = my_merged_rings_copy.begin();
        std::advance(merged_erased_range_it, old_size);
        my_merged_rings_copy.erase(merged_erased_range_it, my_merged_rings_copy.end());
    }

    //sort graphs by process id
    /*
    for(auto& graph : my_merged_rings_copy)
    {
        std::stable_sort(graph.begin(), graph.end(), [process_index, process_size]
                                                        (const detail::colored_idx& lhs,
                                                         const detail::colored_idx& rhs)
        {
            //size_t right_index = (process_index + 1 ) % process_size;
            //size_t left_index = ( process_index == 0 ?  process_size : process_index - 1);
            return (lhs.first < rhs.first); //stable sort by color!
        });
    }
*/
    return my_merged_rings_copy;
}

ccl::process_device_indices_type
allied_process_group_ring_topology::create_scaleout_devices_in_graphs_for_process(
    size_t process_idx,
    size_t cluster_size,
    detail::global_sorted_plain_graphs& cluster_graphs,
    std::ostream& out) {
    size_t left_process_idx = (process_idx == 0 ? cluster_size - 1 : process_idx - 1);
    size_t right_process_idx = ((process_idx + 1) % cluster_size);

    out << "Create scaleout devices for process: (" << process_idx << "/" << cluster_size << ")"
        << ", left_process_idx: " << left_process_idx
        << ", right_process_idx: " << right_process_idx << std::endl;

    ccl::process_device_indices_type scaleout_devices;
    auto me = cluster_graphs.find(process_idx)->second;

    if (process_idx > left_process_idx) {
        auto lhs = cluster_graphs.find(left_process_idx)->second;
        auto find_shared_graph_it = std::find(lhs.begin(), lhs.end(), *me.begin());
        if (find_shared_graph_it == lhs.end()) {
            const ccl::device_index_type& scaleout = *(lhs.rbegin()->rbegin());
            out << "scaleout candidate from Lhs: " << scaleout << std::endl;
            me.insert(me.begin(), { { scaleout } });
            scaleout_devices[left_process_idx] = { scaleout };
        }
    }

    if (process_idx < right_process_idx) {
        auto rhs = cluster_graphs.find(right_process_idx)->second;
        auto find_shared_graph_it = std::find(rhs.begin(), rhs.end(), *me.rbegin());
        if (find_shared_graph_it == rhs.end()) {
            const ccl::device_index_type& scaleout = *(rhs.begin()->begin());
            out << "scaleout candidate from Rhs: " << scaleout << std::endl;
            me.insert(me.end(), { { scaleout } });
            scaleout_devices[right_process_idx] = { scaleout };
        }
    }

    return scaleout_devices;
}

ccl::process_device_indices_type
allied_process_group_ring_topology::create_scaleout_devices_in_colored_graphs_for_process(
    size_t process_idx,
    size_t cluster_size,
    detail::global_sorted_colored_plain_graphs& cluster_graphs,
    detail::global_sorted_colored_plain_graphs& initial_cluster_graphs,
    std::ostream& out)

{
    using optional_process = std::pair<bool, size_t>;

    optional_process left_process_idx =
        std::make_pair(true, (process_idx == 0 ? cluster_size - 1 : process_idx - 1));
    optional_process right_process_idx = std::make_pair(true, ((process_idx + 1) % cluster_size));

    out << "Create scaleout devices for process: (" << process_idx << "/" << cluster_size << ")"
        << ", left_process_idx: " << left_process_idx.second
        << ", right_process_idx: " << right_process_idx.second << std::endl;

    ccl::process_device_indices_type scaleout_devices;
    // process corner cases
    if (left_process_idx == right_process_idx) {
        //two processes
        if (process_idx > left_process_idx.second) {
            left_process_idx.first = false; //do not process left
        }
        else {
            right_process_idx.first = false; //do not process right
        }
    }

    if (left_process_idx.second == process_idx and process_idx == right_process_idx.second) {
        return scaleout_devices; //nothing to scaleout
    }

    auto& me = cluster_graphs.find(process_idx)->second;

    std::unique_ptr<size_t> color_to_find(new size_t);
    auto find_in_list_by_color =
        [&color_to_find](const detail::colored_plain_graph& graph) -> bool {
        auto it = std::find_if(
            graph.begin(), graph.end(), [&color_to_find](const detail::colored_idx& idx) {
                return (idx.color == *color_to_find);
            });
        return it != graph.end();
    };

    if (left_process_idx.first) {
        // find lhs in my graphs
        *color_to_find = left_process_idx.second;
        if (process_idx == 0) {
            //use terminate
            *color_to_find = cluster_size;
        }

        if (std::find_if(me.begin(), me.end(), find_in_list_by_color) == me.end()) {
            //add scaleout device
            auto lhs_it = initial_cluster_graphs.find(left_process_idx.second);
            if (lhs_it == initial_cluster_graphs.end()) {
                assert(false && "lhs process doesn't exist");
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - invalid cluster_graph: " + "no process by id: " +
                                         std::to_string(left_process_idx.second));
            }

            const auto& lhs = lhs_it->second;
            if (lhs.empty()) {
                assert(false && "lhs process graph is empty ");
                throw std::runtime_error(
                    std::string(__FUNCTION__) + " - invalid cluster_graph: empty list " +
                    "for process by id: " + std::to_string(left_process_idx.second));
            }
            const ccl::device_index_type& scaleout = (lhs.rbegin()->rbegin())->index;
            out << "scaleout candidate from Lhs: " << scaleout << std::endl;
            //me.insert(me.begin(), { {left_process_idx.second, scaleout}});
            scaleout_devices[left_process_idx.second] = { scaleout };
        }
    }

    if (right_process_idx.first) {
        // find rhs in my graphs
        *color_to_find = right_process_idx.second;
        if (process_idx == cluster_size - 1) {
            //use terminate
            *color_to_find = cluster_size;
        }

        if (std::find_if(me.begin(), me.end(), find_in_list_by_color) == me.end()) {
            //add scaleout device
            auto rhs_it = initial_cluster_graphs.find(right_process_idx.second);
            if (rhs_it == initial_cluster_graphs.end()) {
                assert(false && "rhs process doesn't exist");
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - invalid cluster_graph: " + "no process by id: " +
                                         std::to_string(right_process_idx.second));
            }

            const auto& rhs = rhs_it->second;
            if (rhs.empty()) {
                assert(false && "rhs process graph is empty ");
                throw std::runtime_error(
                    std::string(__FUNCTION__) + " - invalid cluster_graph: empty list " +
                    "for process by id: " + std::to_string(right_process_idx.second));
            }
            const ccl::device_index_type& scaleout = (rhs.begin()->begin())->index;
            out << "scaleout candidate from Lhs: " << scaleout << std::endl;
            //me.insert(me.end(), {{right_process_idx.second, scaleout}});
            scaleout_devices[right_process_idx.second] = { scaleout };
        }
    }

    return scaleout_devices;
}

ccl::process_device_indices_type
allied_process_group_ring_topology::create_ipc_devices_in_colored_graphs_for_process(
    size_t process_idx,
    size_t cluster_size,
    detail::global_sorted_colored_plain_graphs& cluster_graphs,
    detail::global_sorted_colored_plain_graphs& initial_cluster_graphs,
    std::ostream& out) {
    (void)initial_cluster_graphs;

    using optional_process = std::pair<bool, size_t>;

    optional_process left_process_idx =
        std::make_pair(true, (process_idx == 0 ? cluster_size /* - 1 */ : process_idx - 1));
    optional_process right_process_idx =
        std::make_pair(true, process_idx + 1 /*((process_idx + 1) % cluster_size)*/);

    out << "Create IPC devices for process: (" << process_idx << "/" << cluster_size << ")"
        << ", left_process_idx: " << left_process_idx.second
        << ", right_process_idx: " << right_process_idx.second << std::endl;

    ccl::process_device_indices_type ipc_devices;
    // process corner cases
    /*
    if (left_process_idx == right_process_idx) {
        //two processes
        if (process_idx > left_process_idx.second) {
            left_process_idx.first = false; //do not process left
        }
        else {
            right_process_idx.first = false; //do not process right
        }
    }
    */
    if (left_process_idx.second == process_idx and process_idx == right_process_idx.second) {
        return ipc_devices; //nothing to ipc
    }

    auto& me = cluster_graphs.find(process_idx)->second;

    std::unique_ptr<size_t> color_to_find(new size_t);
    std::vector<detail::colored_idx> devices_to_remember;

    //TODO limitation: all graphs ipc devices would be merged into one vector
    auto filter_list_by_color =
        [&color_to_find, &devices_to_remember](const detail::colored_plain_graph& graph) -> void {
        std::copy_if(graph.begin(),
                     graph.end(),
                     std::back_inserter(devices_to_remember),
                     [&color_to_find](const detail::colored_idx& idx) {
                         return (idx.color == *color_to_find);
                     });
    };

    if (left_process_idx.first) {
        // find lhs color in my graphs
        *color_to_find = left_process_idx.second;
        devices_to_remember.clear();
        if (process_idx == 0) {
            //use terminate
            *color_to_find = cluster_size;
        }

        //fill ipc devices candidates in devices_to_remember
        std::for_each(me.begin(), me.end(), filter_list_by_color);
        if (!devices_to_remember.empty()) {
            const ccl::device_index_type& ipc = devices_to_remember.rbegin()->index;
            out << "ipc candidate from LHS: " << ipc << ", color: " << left_process_idx.second
                << std::endl;
            ipc_devices[left_process_idx.second] = { ipc };
        }
    }

    if (right_process_idx.first) {
        // find rhs in my graphs
        *color_to_find = right_process_idx.second;
        devices_to_remember.clear();
        if (process_idx == cluster_size - 1) {
            //use terminate
            *color_to_find = cluster_size;
        }

        //fill ipc devices candidates in devices_to_remember
        std::for_each(me.begin(), me.end(), filter_list_by_color);
        if (!devices_to_remember.empty()) {
            const ccl::device_index_type& ipc = devices_to_remember.begin()->index;
            out << "ipc candidate from RHS: " << ipc << ", color: " << right_process_idx.second
                << std::endl;
            ipc_devices[right_process_idx.second] = { ipc };
        }
    }

    return ipc_devices;
}

// Well tested topology creator
bool allied_process_group_ring_topology::build_specific_colored(
    std::ostream& out,
    const ccl::process_device_indices_type& per_thread_device_indices,
    const ccl::process_device_indices_type& ipc_device_indices,
    detail::colored_plain_graph& id_ring,
    const std::map<size_t, size_t>& process_device_rank_offset) {
    //continuous ring, without scale-up devices
    //processes connected using IPC devices
    //Rank = Index
    constexpr ccl::device_topology_type topology_type = ccl::device_topology_type::ring;

    out << "Start building topology: " << ::to_string(topology_type) << ", for colored graph:\n"
        << detail::to_string(id_ring) << std::endl;

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;

    // get in-process devices rank offset in cluster map
    auto offset_it = process_device_rank_offset.find(process_index);
    if (offset_it == process_device_rank_offset.end()) {
        assert(false && "");
    }

    size_t device_rank_offset = offset_it->second;
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // let's start IPC devices search & creation
    // TODO
    // We need upgrade algo for detection IPC destination devices, which belong to specific thread in process
    // Currently the final thread in list owns IPC device
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    out << "global rank offset: " << device_rank_offset << std::endl;
    auto& ctx_per_thread_data = context.process_device_topology;
    auto topology_comm_addr = ctx_comm_addr;
    topology_comm_addr.comm_size = device_cluster_size;

    // remember ring first position, which has not termination color, but actual process id
    // It's because after merge id_rings routine we got merged id_rings
    // `merged` means:
    // from left side of list we have IPC devices for left process relation to current process
    // from right side of list we also have IPC devices for right process relation to current process
    // SO, ir_rings starts NOT from existing process devices
    auto local_proc_ring_start =
        std::find_if(id_ring.begin(), id_ring.end(), [this](native::detail::colored_idx& val) {
            //return (val.color != process_count); / / first not terminator index
            return (val.color == process_index); // first not terminator index
        });
    auto id_ring_begin = id_ring.begin();
    size_t distance = std::distance(id_ring_begin, local_proc_ring_start);

    LOG_DEBUG("apply index builder for local thread context, threads count: ",
              ctx_per_thread_data.size(),
              ", process indices ring offset: ",
              distance);
    for (auto per_thread_it = ctx_per_thread_data.begin();
         per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it) {
        size_t thread_id = per_thread_it->first; // first
        const auto& thread_dev_indices = per_thread_device_indices.find(thread_id)->second;

        /**Initialize empty topologies**/
        if (context.get_process_topology<topology_type>(process_index, thread_id)
                .closed_rings.empty()) {
            context.get_process_topology<topology_type>(process_index, thread_id)
                .set_topology(
                    std::make_shared<device_community<topology_type>>(topology_comm_addr));
        }

        // Get reference on OUT-enumerated devices array
        auto& out_indexed_devices =
            context.get_process_topology<topology_type>(process_index,
                                                        thread_id)
                .get_topology()
                ->get_device_storage(); // just second

        // Get IN-non-enumerated devices for current thread
        std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
            devices.thread_gpu_comms.find(thread_id)->second;

        //allocate IPC devices pool(if needed)
        LOG_DEBUG("LIMITATION: Allocate IPC pool for the LAST thread: ",
                  thread_id,
                  ", ipc_device_indices cound: ",
                  ipc_device_indices.size());
        detail::cluster_ipc_devices_pool ipc_comms;
        if (thread_id ==
            ctx_per_thread_data.size() - 1) //TODO only final thread owns IPC devies at now
        {
            ipc_comms =
                detail::create_filtered_ipc_destination_gpu_comms<group_id(), topology_type>(
                    id_ring,
                    ipc_device_indices,
                    process_index,
                    process_count,
                    context,
                    devices,
                    *non_indexed_plain_devices);
        }
        LOG_DEBUG("Create indexer builder for process index: ",
                  process_index,
                  ", process_count: ",
                  process_count,
                  ", device rank offset: ",
                  device_rank_offset,
                  ", thread devices: ",
                  thread_dev_indices.size(),
                  ", ipc_device_indices count: ",
                  ipc_device_indices.size());

        // Rank builder operaes on IN-non-enumarated devices:
        // 1) find its position (using ID comparison & color) in id_ring
        // 2) calculate operational rank as position offset  in id_ring
        // 3) calcuate operations size as whole id_ring size
        // 4) adjust ranks & size to actual using `device_rank_offset` specifically for process
        // 5) put enumerated device into OUT-enumerated devices array `out_indexed_devices`
        auto rank_builder = create_device_functor<
            detail::smart_ring_indexer<group_id(), topology_type, process_group_context>>(
            id_ring,
            process_index,
            process_count,
            device_rank_offset,
            ipc_comms.size()
                ? 1
                : 0, /* TODO self closed ring - only one id_ring for all:   prev_proc: me_proc: next_proc   - prev = next, exclude one*/
            devices,
            out_indexed_devices,
            ipc_device_indices,
            ccl::process_device_indices_type{},
            local_proc_ring_start,
            context);
        //start indexer
        ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

        detail::printer<group_id(), topology_type> p;
        ccl_tuple_for_each(out_indexed_devices, p);

        {
            std::stringstream ss;
            ss << "Indexer result for devices in thread idx (" << thread_id << "/"
               << ctx_per_thread_data.size() << "):\n"
               << p.to_string() << std::endl;
            const std::string& str = ss.str();
            LOG_DEBUG(str);
            out << str;
        }
    }

    out << "\nStart ring builder" << std::endl;
    LOG_DEBUG("Start ring builder for threads: ", ctx_per_thread_data.size());
    for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
         current_thread_idx++) {
        // find max rank in current thread device list
        auto& indexed_devices_for_current_thread =
            context.get_process_topology<topology_type>(process_index, current_thread_idx)
                .get_topology()
                ->get_device_storage();
        const auto& curr_real =
            detail::get_device_with_min_rank<ccl_gpu_comm, group_id(), topology_type>(
                indexed_devices_for_current_thread, id_ring);
        const auto& curr_virt =
            detail::get_device_with_min_rank<ccl_virtual_gpu_comm, group_id(), topology_type>(
                indexed_devices_for_current_thread, id_ring);

        size_t tg_max_rank = std::max({ std::get<0>(curr_real), std::get<0>(curr_virt) });

        // find thread, which will connect to current thread max rank with next_rank
        size_t next_rank = (tg_max_rank + 1) % id_ring.size();

        {
            std::stringstream ss;

            ss << "Current thread: " << current_thread_idx
               << ", max rank candidates: " << std::get<0>(curr_real) << ", "
               << std::get<0>(curr_virt) << ", selected max rank: " << tg_max_rank
               << ", expected next_rank: " << next_rank << std::endl;
            const std::string str = ss.str();
            LOG_DEBUG(str);
            out << str;
        }

        //Find in local threads at first
        bool find_in_current_process = false;
        for (size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size();
             next_thread_id++) {
            if (next_thread_id == current_thread_idx) {
                // wrong thread, get next
                continue;
            }

            // search next_rank in that thread
            auto& next_thread_ring_topology =
                context.get_process_topology<topology_type>(process_index, next_thread_id)
                    .get_topology()
                    ->get_device_storage();
            const auto& real =
                detail::get_device_with_max_rank<ccl_gpu_comm, group_id(), topology_type>(
                    next_thread_ring_topology, id_ring);
            const auto& virt =
                detail::get_device_with_max_rank<ccl_virtual_gpu_comm, group_id(), topology_type>(
                    next_thread_ring_topology, id_ring);

            if (next_rank != std::min({ std::get<0>(real), std::get<0>(virt) })) {
                // wrong thread, get next
                continue;
            }

            {
                std::stringstream ss;
                ss << "next thread: " << next_thread_id
                   << ", min rank candidates: " << std::get<0>(real) << ", " << std::get<0>(virt)
                   << std::endl;

                const std::string str = ss.str();
                LOG_DEBUG(str);
                out << str;
            }

            find_in_current_process = true;
            out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id
                << ")" << std::endl;
            if (next_rank == std::get<0>(real)) {
                auto locker =
                    detail::add_concurrent_locker_device<ccl_gpu_comm, group_id(), topology_type>(
                        next_rank, 0, real, devices, indexed_devices_for_current_thread);
                out << "Added real locker by index: " << next_rank
                    << ", for thread idx: " << current_thread_idx << ":\n"
                    << locker->to_string() << std::endl;
            }
            else if (next_rank == std::get<0>(virt)) {
                auto locker = detail::
                    add_concurrent_locker_device<ccl_virtual_gpu_comm, group_id(), topology_type>(
                        next_rank, 0, virt, devices, indexed_devices_for_current_thread);
                out << "Added virtual locker by index: " << next_rank
                    << ", for thread idx: " << current_thread_idx << ":\n"
                    << locker->to_string() << std::endl;
            }
            else {
                assert(false && "unknown device type");
                std::ostringstream ss;
                ss << out.rdbuf();
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - unknown device type. Log:\n" + ss.str());
            }
        }

        /*-S-
        if (!find_in_current_process)
        {
            abort();
        }*/
        (void)find_in_current_process;
        /*//if not find in process local threads - use IPC to find
        if (!find_in_current_process and !ipc_comms.empty())
        {
            out << "Find IPC device\n";
            bool find = false;
            for (const auto& process_ipc_comms : ipc_comms)
            {
                indexed_device_container<ccl_ipc_gpu_comm>& curr_locker_map =
                        std::get<ccl_ipc_gpu_comm::type_idx()>(*indexed_devices_for_current_thread);
                auto ipc_it = process_ipc_comms.second.find(next_rank);
                if(ipc_it == process_ipc_comms.second.end())
                {
                    out << "skip process index: " << process_ipc_comms.first << std::endl;
                    continue;
                }
                find = true;
                out << "Lock IPC ring for threads (" << current_thread_idx << " <-> xxx\")" << std::endl;
                const auto& comm_addr = ipc_it->second->template get_comm_data<type(), topology_type>();
                curr_locker_map.insert({comm_addr.rank, ipc_it->second});
                out << "Added locker for thread idx: " << current_thread_idx  <<":\n" << ipc_it->second->to_string() << std::endl;
            }
            if (!find)
            {
                std::stringstream ss;
                ss << out.rdbuf();
                std::cerr << "Cannot find IPC deice by rank: " << next_rank << "\nPrevious log:\n" << ss.str() <<"\nAbort Program" << std::endl;
                abort();
            }
            //upgrade left gpu device to IPC SOURCE type
            if (!ipc_comms.empty()/ *has another IPC Device* / and current_thread_idx == 0 / * left comm is IPC comm for last process* / )
            {
                const auto& real = detail::get_device_with_max_rank<ccl_gpu_comm, type(), topology_type>(*indexed_devices_for_current_thread, id_ring);
                const auto& virt = detail::get_device_with_max_rank<ccl_virtual_gpu_comm, type(), topology_type>(*indexed_devices_for_current_thread, id_ring);
                size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                out << "Upgrade thread id: " << current_thread_idx
                    << " GPU by rank: " << left_ipc_source_rank
                    << " to IPC SOURCE GPU" << std::endl;
                if(left_ipc_source_rank == std::get<0>(real))
                {
                    auto locker =
                            detail::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                  type(), topology_type>(next_rank,
                                                                                 0,
                                                                                 real,
                                                                                 devices,
                                                                                 *indexed_devices_for_current_thread);
                    out << "Upgrage REAL to IPC_REAL_SOURCE locker by rank: " << next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
                else if (left_ipc_source_rank == std::get<0>(virt))
                {
                    auto locker =
                            detail::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  type(), topology_type>(next_rank,
                                                                                 0,
                                                                                 virt,
                                                                                 devices,
                                                                                 *indexed_devices_for_current_thread);
                    out << "Upgrage VIRTUAL to IPC_VIRT_SOURCE locker by rank: " << next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
            }
        }
        */
    }

    {
        //print topology
        for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
             current_thread_idx++) {
            const auto& indexed_devices_for_current_thread =
                context.get_process_topology<topology_type>(process_index, current_thread_idx)
                    .get_topology()
                    ->get_device_storage();

            detail::printer<group_id(), topology_type> p;
            ccl_tuple_for_each(indexed_devices_for_current_thread, p);
            std::stringstream ss;
            ss << "Builder result for devices in thread idx (" << current_thread_idx << "/"
               << ctx_per_thread_data.size() << "):\n"
               << p.to_string() << std::endl;
            const std::string& str = ss.str();
            LOG_DEBUG(str);
            out << str;
        }
    }
    return true;
}

bool allied_process_group_ring_topology::build_specific_scale_up(
    std::ostream& out,
    const ccl::process_device_indices_type& per_thread_device_indices,
    const ccl::process_device_indices_type& ipc_device_indices,
    detail::colored_plain_graph_list& graph_list,
    const std::map<size_t, size_t>& process_device_rank_offset) {
    constexpr ccl::device_topology_type class_id = ccl::device_topology_type::ring;

    out << "Start building topology: " << ::to_string(group_id())
        << ", for colored graphs: " << graph_list.size() << "\n"
        << detail::to_string(graph_list) << std::endl;

    auto& ctx_per_thread_data = context.process_device_topology;
    out << "\nStart gpu comm transformation scaling role for graph list count: "
        << graph_list.size() << std::endl;
    std::set<ccl::device_index_type> created_scaleup_indices;

    // allocate IPC devices pool(by demand)
    detail::cluster_ipc_devices_pool ipc_comms;
    size_t ring_index = 0;

    // let's start scaling devices search & creation
    for (auto id_ring_it = graph_list.begin(); id_ring_it != graph_list.end(); ++id_ring_it) {
        const auto& id_ring = *id_ring_it;
        for (const auto& per_thread : per_thread_device_indices) {
            size_t thread_id = per_thread.first;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices.thread_gpu_comms.find(thread_id)->second;

            // create device comm wrappers
            // 1) upgrade last devices in list up to scaling proxy type: numa
            auto last_graph_item = id_ring.rbegin();
            for (; last_graph_item != id_ring.rend(); ++last_graph_item) {
                detail::color_t process = last_graph_item->color;
                ccl::device_index_type last_in_graph_index = last_graph_item->index;

                if (process != process_index) {
                    out << "thread: " << thread_id
                        << " detect device wit foreign color: " << *last_graph_item << std::endl;
                    continue;
                }
                if (per_thread.second.find(last_in_graph_index) != per_thread.second.end()) {
                    out << "thread: " << thread_id
                        << " wants to create scaling device by idx: " << last_in_graph_index
                        << std::endl;
                    if (created_scaleup_indices.find(last_in_graph_index) !=
                        created_scaleup_indices.end()) {
                        out << "skip existing scaling device candidate by: " << last_in_graph_index
                            << std::endl;
                        continue;
                    }

                    size_t inserted_device_type_index = detail::role_mod::inject_numa_device<
                        group_id(),
                        class_id,
                        process_group_context,
                        ccl_virtual_gpu_comm, /* `virtual` is better candiate */
                        ccl_gpu_comm>(
                        *non_indexed_plain_devices, last_in_graph_index, context, devices);
                    if (inserted_device_type_index == std::numeric_limits<size_t>::max()) {
                        assert(false && "Unsupported device type in topology creation");
                        std::ostringstream ss;
                        ss << out.rdbuf();
                        throw std::runtime_error(
                            std::string("Unsupported device type in topology creation. Log:\n") +
                            ss.str());
                    }

                    out << "Inject numa device by order: " << inserted_device_type_index
                        << "\nby idx: " << last_in_graph_index << std::endl;
                    created_scaleup_indices.insert(last_in_graph_index);

                    break;
                }
            }

            //2) create IPC wrappers
            //TODO THE last id_ring from graph_list AND the last thread should process id_ring for IPC device creation here
            //BUT we cannot determine 'last' ring for 'last thread' here in pretty way.
            //So We need to extend 'color' by process_id and thread_id together instead process_id single one
            if (std::next(id_ring_it, 1) == graph_list.end()) {
                if (thread_id ==
                    ctx_per_thread_data.size() - 1) //TODO only final thread owns IPC devies at now
                {
                    ipc_comms =
                        detail::create_filtered_ipc_destination_gpu_comms<group_id(), class_id>(
                            *id_ring_it,
                            ipc_device_indices,
                            process_index,
                            process_count,
                            context,
                            devices,
                            *non_indexed_plain_devices);
                }
            }
        }
    }

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t /*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    auto offset_it = process_device_rank_offset.find(process_index);
    if (offset_it == process_device_rank_offset.end()) {
        assert(false && "");
    }

    accumulated_index_offset_for_graph = offset_it->second;
    auto topology_comm_addr = ctx_comm_addr;
    topology_comm_addr.comm_size = device_cluster_size;

    out << "global rank offset: " << accumulated_index_offset_for_graph << std::endl;

    for (auto& id_ring : graph_list) {
        auto local_proc_ring_start =
            std::find_if(id_ring.begin(), id_ring.end(), [this](native::detail::colored_idx& val) {
                //return (val.color != process_count && val.color != native::detail::marked_color); / / first not terminator index
                return val.color == process_index;
            });

        if (local_proc_ring_start == id_ring.end()) {
            out << "graph fully processes: " << detail::to_string(id_ring) << ", take next"
                << std::endl;
            continue;
        }

        size_t index_offset = accumulated_index_offset_for_graph;
        for (auto per_thread_it = ctx_per_thread_data.begin();
             per_thread_it != ctx_per_thread_data.end();
             ++per_thread_it) {
            size_t thread_id = per_thread_it->first; //first

            /** Initialize empty context**/
            std::shared_ptr<device_community<class_id>> out_indexed_devices;
            if (graph_list.size() == 1) {
                if (context.get_process_topology<class_id>(process_index, thread_id)
                        .closed_rings.empty()) {
                    context.get_process_topology<class_id>(process_index, thread_id)
                        .set_topology(
                            std::make_shared<device_community<class_id>>(topology_comm_addr));
                }

                out_indexed_devices =
                    context.get_process_topology<class_id>(process_index, thread_id)
                        .get_topology(ring_index);
            }
            else {
                if (context.get_process_topology<class_id>(process_index, thread_id)
                        .torn_apart_rings.empty()) {
                    context.get_process_topology<class_id>(process_index, thread_id)
                        .set_additiona_topology(
                            std::make_shared<device_community<class_id>>(topology_comm_addr));
                }

                out_indexed_devices =
                    context.get_process_topology<class_id>(process_index, thread_id)
                        .get_additiona_topology(ring_index);
            }

            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id
                << ", index offset: " << index_offset << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices.thread_gpu_comms.find(thread_id)->second;

            auto rank_builder = create_device_functor<
                detail::smart_ring_indexer<group_id(), class_id, process_group_context>>(
                id_ring,
                process_index,
                process_count,
                index_offset,
                0,
                devices,
                out_indexed_devices->get_device_storage(),
                ipc_device_indices,
                ccl::process_device_indices_type{},
                local_proc_ring_start,
                context);

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            detail::printer<group_id(), class_id> p;
            ccl_tuple_for_each(out_indexed_devices->get_device_storage(), p);
            out << "Indexer result for devices in thread idx (" << thread_id << "/"
                << ctx_per_thread_data.size() << "):\n"
                << p.to_string() << std::endl;

            accumulated_index_offset_for_graph +=
                rank_builder.get_functor().get_marked_indices_count();
            out << "\nIndexer for graph num: " << graph_num
                << ", finished. imarked_indices: " << accumulated_index_offset_for_graph << "\n";
        }
        index_offset_for_graphs[graph_num] = index_offset;
        graph_num++;
    }

    out << "Created IPC devices for processes: " << ipc_comms.size()
        << ", for cluster_size: " << device_cluster_size
        << ", with device_cluster_rank_offset: " << device_cluster_rank_offset << "\n";
    for (const auto& process_ipc : ipc_comms) {
        out << "prx: " << process_ipc.first << std::endl;
        for (const auto& ipc : process_ipc.second) {
            out << "{ rank: " << ipc.first << ", comm: " << ipc.second->to_string() << "}\n";
        }
        out << std::endl;
    }

    out << "\nStart ring builder for graphs count: " << graph_list.size() << std::endl;
    graph_num = 0;
    for (const auto& id_ring : graph_list) {
        out << "\nStart ring builder for graph num: " << graph_num << std::endl;
        for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
             current_thread_idx++) {
            // find max rank in current thread device list
            std::shared_ptr<device_community<class_id>> indexed_topology;
            if (graph_list.size() == 1) {
                indexed_topology =
                    context.get_process_topology<class_id>(process_index, current_thread_idx)
                        .get_topology(ring_index);
            }
            else {
                indexed_topology =
                    context.get_process_topology<class_id>(process_index, current_thread_idx)
                        .get_additiona_topology(ring_index);
            }

            auto& indexed_devices_for_current_thread = indexed_topology->get_device_storage();
            const auto& curr_real =
                detail::get_device_with_min_rank<ccl_gpu_comm, group_id(), class_id>(
                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_virt =
                detail::get_device_with_min_rank<ccl_virtual_gpu_comm, group_id(), class_id>(
                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_real = detail::
                get_device_with_min_rank<ccl_numa_proxy<ccl_gpu_comm>, group_id(), class_id>(
                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_virt =
                detail::get_device_with_min_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>,
                                                 group_id(),
                                                 class_id>(indexed_devices_for_current_thread,
                                                           id_ring);

            size_t tg_max_rank = std::max({ std::get<0>(curr_real),
                                            std::get<0>(curr_virt),
                                            std::get<0>(curr_scale_real),
                                            std::get<0>(curr_scale_virt) });

            // find thread, which will connect to current thread max rank with next_rank
            size_t next_rank = (tg_max_rank + 1) % id_ring.size();
            out << "Current thread: " << current_thread_idx
                << ", max rank candidates: " << std::get<0>(curr_real) << ", "
                << std::get<0>(curr_virt) << ", " << std::get<0>(curr_scale_real) << ", "
                << std::get<0>(curr_scale_virt) << ", selected max rank: " << tg_max_rank
                << ", expected next_rank: " << next_rank << std::endl;

            //Find in local threads at first
            for (size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size();
                 next_thread_id++) {
                if (next_thread_id == current_thread_idx) {
                    // wrong thread, get next
                    continue;
                }

                // search next_rank in that thread
                std::shared_ptr<device_community<class_id>> next_indexed_topology;
                if (graph_list.size() == 1) {
                    next_indexed_topology =
                        context.get_process_topology<class_id>(process_index, next_thread_id)
                            .get_topology(ring_index);
                }
                else {
                    next_indexed_topology =
                        context.get_process_topology<class_id>(process_index, next_thread_id)
                            .get_additiona_topology(ring_index);
                }

                auto& next_thread_ring_topology = next_indexed_topology->get_device_storage();
                const auto& real =
                    detail::get_device_with_max_rank<ccl_gpu_comm, group_id(), class_id>(
                        next_thread_ring_topology, id_ring);
                const auto& virt =
                    detail::get_device_with_max_rank<ccl_virtual_gpu_comm, group_id(), class_id>(
                        next_thread_ring_topology, id_ring);
                const auto& scale_real =
                    detail::get_device_with_max_rank<ccl_numa_proxy<ccl_gpu_comm>,
                                                     group_id(),
                                                     class_id>(next_thread_ring_topology, id_ring);
                const auto& scale_virt =
                    detail::get_device_with_max_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>,
                                                     group_id(),
                                                     class_id>(next_thread_ring_topology, id_ring);
                if (next_rank != std::min({ std::get<0>(real),
                                            std::get<0>(virt),
                                            std::get<0>(scale_real),
                                            std::get<0>(scale_virt) })) {
                    // wrong thread, get next
                    continue;
                }

                out << "next thread: " << next_thread_id
                    << ", min rank candidates: " << std::get<0>(real) << ", " << std::get<0>(virt)
                    << ", " << std::get<0>(scale_real) << ", " << std::get<0>(scale_virt)
                    << std::endl;

                out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id
                    << ")" << std::endl;

                if (next_rank == std::get<0>(real)) {
                    auto locker =
                        detail::add_concurrent_locker_device<ccl_gpu_comm, group_id(), class_id>(
                            next_rank, 0, real, devices, indexed_devices_for_current_thread);
                    out << "Added real locker by index: " << next_rank
                        << ", for thread idx: " << current_thread_idx << ":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(virt)) {
                    auto locker = detail::
                        add_concurrent_locker_device<ccl_virtual_gpu_comm, group_id(), class_id>(
                            next_rank, 0, virt, devices, indexed_devices_for_current_thread);
                    out << "Added virtual locker by index: " << next_rank
                        << ", for thread idx: " << current_thread_idx << ":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(scale_real)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup  real proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
                else if (next_rank == std::get<0>(scale_virt)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup virtual proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
                /*else
                {
                    assert(false && "unknown device type");
                    std::ostringstream ss;
                    ss << out.rdbuf();
                    throw std::runtime_error(std::string(__FUNCTION__) + " - unknown device type. Log:\n" +
                                             ss.str());
                }*/
            }

            //if not find in process local threads - use IPC to find
            /*if (!find_in_current_process and !ipc_comms.empty())
            {
                out << "Find IPC device\n";
                bool find = false;
                for (const auto& process_ipc_comms : ipc_comms)
                {
                    indexed_device_container<ccl_ipc_gpu_comm>& curr_locker_map =
                            std::get<ccl_ipc_gpu_comm::type_idx()>(indexed_devices_for_current_thread);

                    auto ipc_it = process_ipc_comms.second.find(next_rank);
                    if(ipc_it == process_ipc_comms.second.end())
                    {
                        out << "skip process index: " << process_ipc_comms.first << std::endl;
                        continue;
                    }
                    find = true;
                    out << "Lock IPC ring for threads (" << current_thread_idx << " <-> xxx\")" << std::endl;
                    const auto& comm_addr = ipc_it->second->template get_comm_data<type(), group_id()>();
                    curr_locker_map.insert({comm_addr.rank, ipc_it->second});
                    out << "Added locker for thread idx: " << current_thread_idx  <<":\n" << ipc_it->second->to_string() << std::endl;
                }

                if (!find)
                {
                    std::stringstream ss;
                    ss << out.rdbuf();
                    std::cerr << "Cannot find IPC deice by rank: " << next_rank << "\nPrevious log:\n" << ss.str() <<"\nAbort Program" << std::endl;
                    abort();
                }

                //upgrade left gpu device to IPC SOURCE type
                if ( current_thread_idx == 0 / * left comm is IPC comm for last process* / )
                {
                    const auto& real = detail::get_device_with_max_rank<ccl_gpu_comm, type(), group_id()>(indexed_devices_for_current_thread, id_ring);
                    const auto& virt = detail::get_device_with_max_rank<ccl_virtual_gpu_comm, type(), group_id()>(indexed_devices_for_current_thread, id_ring);

                    size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                    out << "Upgrade thread id: " << current_thread_idx
                        << " GPU by rank: " << left_ipc_source_rank
                        << " to IPC SOURCE GPU" << std::endl;

                    if(left_ipc_source_rank == std::get<0>(real))
                    {
                        auto locker =
                                    detail::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                        type(), group_id()>(next_rank,
                                                                                   0,
                                                                                   real,
                                                                                   devices,indexed_devices_for_current_thread);
                        out << "Upgrage REAL to IPC_REAL_SOURCE locker by rank: " << next_rank
                            << ", for thread idx: " << current_thread_idx  <<":\n"
                            << locker->to_string() << std::endl;
                    }
                    else if (left_ipc_source_rank == std::get<0>(virt))
                    {
                        auto locker =
                                detail::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  type(), group_id()>(next_rank,
                                                                                 0,
                                                                                 virt,
                                                                                 devices,indexed_devices_for_current_thread);
                        out << "Upgrage VIRTUAL to IPC_VIRT_SOURCE locker by rank: " << next_rank
                            << ", for thread idx: " << current_thread_idx  <<":\n"
                            << locker->to_string() << std::endl;
                    }
                }
            }
            */
        }
        graph_num++;
    }

    {
        //print topology
        for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
             current_thread_idx++) {
            std::shared_ptr<device_community<class_id>> indexed_topology;
            if (graph_list.size() == 1) {
                indexed_topology =
                    context.get_process_topology<class_id>(process_index, current_thread_idx)
                        .get_topology(ring_index);
            }
            else {
                indexed_topology =
                    context.get_process_topology<class_id>(process_index, current_thread_idx)
                        .get_additiona_topology(ring_index);
            }

            detail::printer<group_id(), class_id> p;
            ccl_tuple_for_each(indexed_topology->get_device_storage(), p);
            std::stringstream ss;
            ss << "Builder result for devices in thread idx (" << current_thread_idx << "/"
               << ctx_per_thread_data.size() << "):\n"
               << p.to_string() << std::endl;
            const std::string& str = ss.str();
            LOG_DEBUG(str);
            out << str;
        }
    }
    return true;
}

bool allied_process_group_ring_topology::build_specific_scale_out_only(
    std::ostream& out,
    const ccl::process_device_indices_type& per_thread_device_indices,
    const ccl::process_device_indices_type& scaleout_device_indices,
    detail::colored_plain_graph_list& graph_list,
    const std::map<size_t, size_t>& process_device_rank_offset) {
    constexpr ccl::device_topology_type class_id = ccl::device_topology_type::ring;

    out << "Start building topology: " << ::to_string(group_id())
        << ", for colored graphs: " << graph_list.size() << "\n"
        << detail::to_string(graph_list) << std::endl;

    auto& ctx_per_thread_data = context.process_device_topology;
    out << "\nStart gpu comm transformation scaling role for graph list count: "
        << graph_list.size() << std::endl;
    std::set<ccl::device_index_type> created_numa_indices;
    size_t ring_index = 0;

    // let's start scaling devices search & creation
    for (auto id_ring_it = graph_list.begin(); id_ring_it != graph_list.end(); ++id_ring_it) {
        const auto& id_ring = *id_ring_it;
        for (const auto& per_thread : per_thread_device_indices) {
            size_t thread_id = per_thread.first;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices.thread_gpu_comms.find(thread_id)->second;

            // create device comm wrappers
            // 1) upgrade last devices in list up to scaling proxy type: numa
            if (graph_list.size() == 1) {
                //no numa for single graph
                break;
            }

            auto last_graph_item = id_ring.rbegin();
            for (; last_graph_item != id_ring.rend(); ++last_graph_item) {
                detail::color_t process = last_graph_item->color;
                ccl::device_index_type last_in_graph_index = last_graph_item->index;

                if (process != process_index) {
                    out << "thread: " << thread_id
                        << " detect device wit foreign color: " << *last_graph_item << std::endl;
                    continue;
                }
                if (per_thread.second.find(last_in_graph_index) != per_thread.second.end()) {
                    out << "thread: " << thread_id
                        << " wants to create scaling device by idx: " << last_in_graph_index
                        << std::endl;
                    if (created_numa_indices.find(last_in_graph_index) !=
                        created_numa_indices.end()) {
                        out << "skip existing scaling device candidate by: " << last_in_graph_index
                            << std::endl;
                        continue;
                    }

                    size_t inserted_device_type_index = detail::role_mod::inject_numa_device<
                        group_id(),
                        class_id,
                        process_group_context,
                        ccl_virtual_gpu_comm, /* `virtual` is better candiate */
                        ccl_gpu_comm>(
                        *non_indexed_plain_devices, last_in_graph_index, context, devices);
                    if (inserted_device_type_index == std::numeric_limits<size_t>::max()) {
                        assert(false && "Unsupported device type in topology creation");
                        std::ostringstream ss;
                        ss << out.rdbuf();
                        throw std::runtime_error(
                            std::string("Unsupported device type in topology creation. Log:\n") +
                            ss.str());
                    }

                    out << "Inject numa device by order: " << inserted_device_type_index
                        << "\nby idx: " << last_in_graph_index << std::endl;
                    created_numa_indices.insert(last_in_graph_index);

                    break;
                }
            }

            //TODO No IPC devices
        }
    }

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t /*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    auto offset_it = process_device_rank_offset.find(process_index);
    if (offset_it == process_device_rank_offset.end()) {
        assert(false && "");
    }

    accumulated_index_offset_for_graph = offset_it->second;
    auto topology_comm_addr = ctx_comm_addr;
    topology_comm_addr.comm_size = device_cluster_size;

    out << "global rank offset: " << accumulated_index_offset_for_graph << std::endl;

    for (auto id_ring_it = graph_list.begin(); id_ring_it != graph_list.end(); ++id_ring_it) {
        auto& id_ring = *id_ring_it;
        auto local_proc_ring_start = std::find_if(
            id_ring.begin(), id_ring.end(), [this](const native::detail::colored_idx& val) {
                //return (val.color != process_count && val.color != native::detail::marked_color); / / first not terminator index
                return val.color == process_index;
            });

        if (local_proc_ring_start == id_ring.end()) {
            out << "graph fully processes: " << detail::to_string(id_ring) << ", take next"
                << std::endl;
            continue;
        }

        size_t index_offset = accumulated_index_offset_for_graph;
        for (auto per_thread_it = ctx_per_thread_data.begin();
             per_thread_it != ctx_per_thread_data.end();
             ++per_thread_it) {
            size_t thread_id = per_thread_it->first; //first

            /** Initialize empty context**/
            std::shared_ptr<device_community<class_id>> out_indexed_devices;
            if (context.get_process_topology<class_id>(process_index, thread_id)
                    .torn_apart_rings.empty()) {
                context.get_process_topology<class_id>(process_index, thread_id)
                    .set_additiona_topology(
                        std::make_shared<device_community<class_id>>(topology_comm_addr));
            }

            out_indexed_devices = context.get_process_topology<class_id>(process_index, thread_id)
                                      .get_additiona_topology(ring_index);

            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id
                << ", index offset: " << index_offset << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices.thread_gpu_comms.find(thread_id)->second;

            auto rank_builder = create_device_functor<
                detail::smart_ring_indexer<group_id(), class_id, process_group_context>>(
                id_ring,
                process_index,
                process_count,
                index_offset,
                0,
                devices,
                out_indexed_devices->get_device_storage(),
                ccl::process_device_indices_type{},
                ccl::process_device_indices_type{},
                local_proc_ring_start,
                context);

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            // Inject Scale out devices for the last thread
            if (std::next(id_ring_it, 1) == graph_list.end()) {
                if (thread_id ==
                    ctx_per_thread_data.size() - 1) //TODO only final thread owns IPC devies at now
                {
                    size_t inserted_device_type_index = detail::role_mod::inject_scaleout_device<
                        group_id(),
                        class_id,
                        process_group_context,
                        ccl_gpu_scaleup_proxy<ccl_numa_proxy<ccl_gpu_comm>>,
                        ccl_gpu_scaleup_proxy<ccl_numa_proxy<ccl_virtual_gpu_comm>>,
                        ccl_gpu_scaleup_proxy<ccl_gpu_comm>,
                        ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>,
                        ccl_numa_proxy<ccl_gpu_comm>,
                        ccl_numa_proxy<ccl_virtual_gpu_comm>,
                        ccl_virtual_gpu_comm,
                        ccl_gpu_comm>(out_indexed_devices->get_device_storage(),
                                      id_ring_it->begin()->index,
                                      context,
                                      devices);
                    if (inserted_device_type_index != std::numeric_limits<size_t>::max()) {
                        out << "Inject scaleout device by order: " << inserted_device_type_index
                            << "\nby idx: " << id_ring_it->begin()->to_string() << std::endl;
                    }
                }
            }

            detail::printer<group_id(), class_id> p;
            ccl_tuple_for_each(out_indexed_devices->get_device_storage(), p);
            out << "Indexer result for devices in thread idx (" << thread_id << "/"
                << ctx_per_thread_data.size() << "):\n"
                << p.to_string() << std::endl;

            accumulated_index_offset_for_graph +=
                rank_builder.get_functor().get_marked_indices_count();
            out << "\nIndexer for graph num: " << graph_num
                << ", finished. imarked_indices: " << accumulated_index_offset_for_graph << "\n";
        }
        index_offset_for_graphs[graph_num] = index_offset;
        graph_num++;
    }

    out << "\nStart ring builder for graphs count: " << graph_list.size() << std::endl;
    graph_num = 0;
    for (const auto& id_ring : graph_list) {
        out << "\nStart ring builder for graph num: " << graph_num << std::endl;
        for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
             current_thread_idx++) {
            // find max rank in current thread device list
            std::shared_ptr<device_community<class_id>> indexed_topology;
            indexed_topology =
                context.get_process_topology<class_id>(process_index, current_thread_idx)
                    .get_additiona_topology(ring_index);

            auto& indexed_devices_for_current_thread = indexed_topology->get_device_storage();
            const auto& curr_real =
                detail::get_device_with_min_rank<ccl_gpu_comm, group_id(), class_id>(
                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_virt =
                detail::get_device_with_min_rank<ccl_virtual_gpu_comm, group_id(), class_id>(
                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_real = detail::
                get_device_with_min_rank<ccl_numa_proxy<ccl_gpu_comm>, group_id(), class_id>(
                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_virt =
                detail::get_device_with_min_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>,
                                                 group_id(),
                                                 class_id>(indexed_devices_for_current_thread,
                                                           id_ring);

            size_t tg_max_rank = std::max({ std::get<0>(curr_real),
                                            std::get<0>(curr_virt),
                                            std::get<0>(curr_scale_real),
                                            std::get<0>(curr_scale_virt) });

            // find thread, which will connect to current thread max rank with next_rank
            size_t next_rank = (tg_max_rank + 1) % id_ring.size();
            out << "Current thread: " << current_thread_idx
                << ", max rank candidates: " << std::get<0>(curr_real) << ", "
                << std::get<0>(curr_virt) << ", " << std::get<0>(curr_scale_real) << ", "
                << std::get<0>(curr_scale_virt) << ", selected max rank: " << tg_max_rank
                << ", expected next_rank: " << next_rank << std::endl;

            //Find in local threads at first
            for (size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size();
                 next_thread_id++) {
                if (next_thread_id == current_thread_idx) {
                    // wrong thread, get next
                    continue;
                }

                // search next_rank in that thread
                std::shared_ptr<device_community<class_id>> next_indexed_topology;
                next_indexed_topology =
                    context.get_process_topology<class_id>(process_index, next_thread_id)
                        .get_additiona_topology(ring_index);

                auto& next_thread_ring_topology = next_indexed_topology->get_device_storage();
                const auto& real =
                    detail::get_device_with_max_rank<ccl_gpu_comm, group_id(), class_id>(
                        next_thread_ring_topology, id_ring);
                const auto& virt =
                    detail::get_device_with_max_rank<ccl_virtual_gpu_comm, group_id(), class_id>(
                        next_thread_ring_topology, id_ring);
                const auto& scale_real =
                    detail::get_device_with_max_rank<ccl_numa_proxy<ccl_gpu_comm>,
                                                     group_id(),
                                                     class_id>(next_thread_ring_topology, id_ring);
                const auto& scale_virt =
                    detail::get_device_with_max_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>,
                                                     group_id(),
                                                     class_id>(next_thread_ring_topology, id_ring);
                if (next_rank != std::min({ std::get<0>(real),
                                            std::get<0>(virt),
                                            std::get<0>(scale_real),
                                            std::get<0>(scale_virt) })) {
                    // wrong thread, get next
                    continue;
                }

                out << "next thread: " << next_thread_id
                    << ", min rank candidates: " << std::get<0>(real) << ", " << std::get<0>(virt)
                    << ", " << std::get<0>(scale_real) << ", " << std::get<0>(scale_virt)
                    << std::endl;

                out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id
                    << ")" << std::endl;

                if (next_rank == std::get<0>(real)) {
                    auto locker =
                        detail::add_concurrent_locker_device<ccl_gpu_comm, group_id(), class_id>(
                            next_rank, 0, real, devices, indexed_devices_for_current_thread);
                    out << "Added real locker by index: " << next_rank
                        << ", for thread idx: " << current_thread_idx << ":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(virt)) {
                    auto locker = detail::
                        add_concurrent_locker_device<ccl_virtual_gpu_comm, group_id(), class_id>(
                            next_rank, 0, virt, devices, indexed_devices_for_current_thread);
                    out << "Added virtual locker by index: " << next_rank
                        << ", for thread idx: " << current_thread_idx << ":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(scale_real)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup  real proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
                else if (next_rank == std::get<0>(scale_virt)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup virtual proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
                /*else
                {
                    assert(false && "unknown device type");
                    std::ostringstream ss;
                    ss << out.rdbuf();
                    throw std::runtime_error(std::string(__FUNCTION__) + " - unknown device type. Log:\n" +
                                             ss.str());
                }*/
            }

            //if not find in process local threads - use IPC to find
            /*if (!find_in_current_process and !ipc_comms.empty())
            {
                out << "Find IPC device\n";
                bool find = false;
                for (const auto& process_ipc_comms : ipc_comms)
                {
                    indexed_device_container<ccl_ipc_gpu_comm>& curr_locker_map =
                            std::get<ccl_ipc_gpu_comm::type_idx()>(indexed_devices_for_current_thread);

                    auto ipc_it = process_ipc_comms.second.find(next_rank);
                    if(ipc_it == process_ipc_comms.second.end())
                    {
                        out << "skip process index: " << process_ipc_comms.first << std::endl;
                        continue;
                    }
                    find = true;
                    out << "Lock IPC ring for threads (" << current_thread_idx << " <-> xxx\")" << std::endl;
                    const auto& comm_addr = ipc_it->second->template get_comm_data<type(), group_id()>();
                    curr_locker_map.insert({comm_addr.rank, ipc_it->second});
                    out << "Added locker for thread idx: " << current_thread_idx  <<":\n" << ipc_it->second->to_string() << std::endl;
                }

                if (!find)
                {
                    std::stringstream ss;
                    ss << out.rdbuf();
                    std::cerr << "Cannot find IPC deice by rank: " << next_rank << "\nPrevious log:\n" << ss.str() <<"\nAbort Program" << std::endl;
                    abort();
                }

                //upgrade left gpu device to IPC SOURCE type
                if ( current_thread_idx == 0 / * left comm is IPC comm for last process* / )
                {
                    const auto& real = detail::get_device_with_max_rank<ccl_gpu_comm, type(), group_id()>(indexed_devices_for_current_thread, id_ring);
                    const auto& virt = detail::get_device_with_max_rank<ccl_virtual_gpu_comm, type(), group_id()>(indexed_devices_for_current_thread, id_ring);

                    size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                    out << "Upgrade thread id: " << current_thread_idx
                        << " GPU by rank: " << left_ipc_source_rank
                        << " to IPC SOURCE GPU" << std::endl;

                    if(left_ipc_source_rank == std::get<0>(real))
                    {
                        auto locker =
                                    detail::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                        type(), group_id()>(next_rank,
                                                                                   0,
                                                                                   real,
                                                                                   devices,indexed_devices_for_current_thread);
                        out << "Upgrage REAL to IPC_REAL_SOURCE locker by rank: " << next_rank
                            << ", for thread idx: " << current_thread_idx  <<":\n"
                            << locker->to_string() << std::endl;
                    }
                    else if (left_ipc_source_rank == std::get<0>(virt))
                    {
                        auto locker =
                                detail::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  type(), group_id()>(next_rank,
                                                                                 0,
                                                                                 virt,
                                                                                 devices,indexed_devices_for_current_thread);
                        out << "Upgrage VIRTUAL to IPC_VIRT_SOURCE locker by rank: " << next_rank
                            << ", for thread idx: " << current_thread_idx  <<":\n"
                            << locker->to_string() << std::endl;
                    }
                }
            }
            */
        }
        graph_num++;
    }

    {
        //print topology
        for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
             current_thread_idx++) {
            std::shared_ptr<device_community<class_id>> indexed_topology;
            indexed_topology =
                context.get_process_topology<class_id>(process_index, current_thread_idx)
                    .get_additiona_topology(ring_index);

            detail::printer<group_id(), class_id> p;
            ccl_tuple_for_each(indexed_topology->get_device_storage(), p);
            std::stringstream ss;
            ss << "Builder result for devices in thread idx (" << current_thread_idx << "/"
               << ctx_per_thread_data.size() << "):\n"
               << p.to_string() << std::endl;
            const std::string& str = ss.str();
            LOG_DEBUG(str);
            out << str;
        }
    }
    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
bool allied_process_group_ring_topology::build_specific(std::ostream& out,
                                                        const ccl::process_device_indices_type& per_thread_device_indices,
                                                        const detail::plain_graph_list& graph_list)
{
     constexpr ccl::group_split_type topology_type =
                                        ccl::group_split_type::process_group_torn_apart_ring;

    out << "Start building topology: " << ::to_string(topology_type)
        << ", for graphs: " << graph_list.size() << "\n";
    for (const auto& graph : graph_list)
    {
        out << "\n\t{";
        for(const auto& id : graph)
        {
            out << id << ", ";
        }
        out << "},";
    }

    auto& ctx_per_thread_data = context.process_device_topology;
    out << "\nStart gpu comm transformation scael-up for graph list count: "
        << graph_list.size() << std::endl;
    std::set<ccl::device_index_type> created_scaleup_indices;

    // let's start scale-up devices search & creation
    for (const auto& id_ring : graph_list)
    {
        for(const auto& per_thread : per_thread_device_indices)
        {
            size_t thread_id = per_thread.first;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                devices.thread_gpu_comms.find(thread_id)->second;
            // create device comm wrappers and upgrade last devices in list up to scale_up_proxy type
            const ccl::device_index_type& last_in_graph_index = *id_ring.rbegin();
            if (per_thread.second.find(last_in_graph_index) != per_thread.second.end())
            {
                out << "thread: " << thread_id << " wants to create scale_up device by idx: "
                    << last_in_graph_index << std::endl;
                if (created_scaleup_indices.find(last_in_graph_index) != created_scaleup_indices.end())
                {
                    out << "skip existing scale_up device candidate by: " << last_in_graph_index << std::endl;
                    continue;
                }

                auto scale_virt = detail::add_numa_proxy_device<ccl_virtual_gpu_comm, topology_type>(
                                                                        *non_indexed_plain_devices,
                                                                        last_in_graph_index,
                                                                        context,
                                                                        devices);
                if (scale_virt)
                {
                    created_scaleup_indices.insert(last_in_graph_index);
                    out << "added scaleup virtual device: " << scale_virt->to_string()
                        << ", by idx: " << last_in_graph_index << std::endl;
                }
                else
                {
                    auto scale_real = detail::add_numa_proxy_device<ccl_gpu_comm, topology_type>(
                                                                        *non_indexed_plain_devices,
                                                                        last_in_graph_index,
                                                                        context,
                                                                        devices);
                    if (scale_real)
                    {
                        created_scaleup_indices.insert(last_in_graph_index);
                        out << "added scaleup real device: " << scale_real->to_string()
                            << ", by idx: " << last_in_graph_index << std::endl;
                    }
                    else
                    {
                        assert(false && "Unsupported device type in torn-apart ring creation");
                        std::ostringstream ss;
                        ss << out.rdbuf();
                        throw std::runtime_error(std::string("Unsupported device type in torn-apart ring creation. Log:\n") +
                                                 ss.str());
                    }
                }
            }
        }
    }

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    detail::ipc_devices_pool ipc_comms;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t/*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    for (const auto& id_ring : graph_list)
    {
        detail::id_thread_table assigned_ids;  //device_id -> thread_id

        std::vector<detail::marked_idx> marked_id_ring = detail::create_marked(id_ring);  // marked graph

        size_t index_offset = accumulated_index_offset_for_graph;
        for (auto per_thread_it = ctx_per_thread_data.begin(); per_thread_it != ctx_per_thread_data.end();
            ++per_thread_it)
        {
            size_t thread_id = per_thread_it->first;        //first
            /** Initialize empty context**/
            std::shared_ptr<device_community<topology_type>> out_indexed_devices;
            if (graph_list.size() == 1) {
                if (context.get_process_topology<topology_type>(process_index, thread_id)
                        .closed_rings.empty()) {
                    context.get_process_topology<topology_type>(process_index, thread_id)
                        .set_topology(
                            std::make_shared<device_community<topology_type>>(topology_comm_addr));
                }

                out_indexed_devices =
                    context.get_process_topology<topology_type>(process_index, thread_id)
                        .get_topology(ring_index);
            }
            else {
                if (context.get_process_topology<topology_type>(process_index, thread_id)
                        .torn_apart_rings.empty()) {
                    context.get_process_topology<topology_type>(process_index, thread_id)
                        .set_additiona_topology(
                            std::make_shared<device_community<topology_type>>(topology_comm_addr));
                }

                out_indexed_devices =
                    context.get_process_topology<topology_type>(process_index, thread_id)
                        .get_additiona_topology(ring_index);
            }



            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices.thread_gpu_comms.find(thread_id)->second;

            // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
            auto rank_builder =
                    create_device_functor<detail::graph_ring_indexer_unique_index_ext<topology_type>>(marked_id_ring,
                                                                                      assigned_ids,
                                                                                      thread_id,
                                                                                      out_indexed_devices,
                                                                                      index_offset + device_cluster_rank_offset,
                                                                                      0,
                                                                                      0);
//                                                                                    device_cluster_size

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            detail::printer<type(), topology_type> p;
            ccl_tuple_for_each(out_indexed_devices, p);
            out << "Indexer result for devices in thread idx ("
                << thread_id << "/" << ctx_per_thread_data.size() << "):\n"
                << p.to_string() << std::endl;

            accumulated_index_offset_for_graph += rank_builder.get_functor().get_marked_indices_count();
            out << "\nIndexer for graph num: " << graph_num
                << ", finished. imarked_indices: " << accumulated_index_offset_for_graph <<"\n";
        }
        index_offset_for_graphs[graph_num] = index_offset;


        out << "\nStart gpu comm transformation ipc for graph num: "
            << graph_num << std::endl;

        //allocate IPC devices pool with rank from unassigned IDs
        detail::ipc_devices_pool tmp_ipc_comms =
                        detail::create_ipc_gpu_comms<type(), topology_type>(assigned_ids, id_ring, devices,
                                                                     device_cluster_size,
                                                                     device_cluster_rank_offset);
        out << "Created Tmp IPC devices: " << tmp_ipc_comms.size()
            << ", for cluster_size: " << device_cluster_size
            << ", with device_cluster_rank_offset: " << device_cluster_rank_offset << "\n";
        for (const auto& ipc : tmp_ipc_comms)
        {
            out << "{ rank: " << ipc.first << ", comm: " << ipc.second->to_string() << "}\n";
        }
        ipc_comms.insert(tmp_ipc_comms.begin(), tmp_ipc_comms.end());
        graph_num ++;
    }



    out << "\nStart ring builder for graphs count: " << graph_list.size() << std::endl;
    graph_num = 0;
    for (const auto& id_ring : graph_list)
    {
        out << "\nStart ring builder for graph num: " << graph_num << std::endl;
        for(size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size(); current_thread_idx++)
        {
            // find max rank in current thread device list
            auto& indexed_devices_for_current_thread =
                    context.get_process_topology<topology_type>(process_index,
                                                                current_thread_idx).get_topology()->get_device_storage();
            const auto& curr_real =
                    detail::get_device_with_min_rank<ccl_gpu_comm, type(), topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_virt =
                    detail::get_device_with_min_rank<ccl_virtual_gpu_comm, type(), topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_real =
                    detail::get_device_with_min_rank<ccl_numa_proxy<ccl_gpu_comm>, type(), topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_virt =
                    detail::get_device_with_min_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>, type(), topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);

            size_t tg_max_rank = std::max({std::get<0>(curr_real), std::get<0>(curr_virt),
                                           std::get<0>(curr_scale_real), std::get<0>(curr_scale_virt)});

            // find thread, which will connect to current thread max rank with next_rank
            size_t next_rank = (tg_max_rank + 1 ) % id_ring.size();
            out << "Current thread: " << current_thread_idx << ", max rank candidates: "
                << std::get<0>(curr_real) << ", " << std::get<0>(curr_virt) << ", "
                << std::get<0>(curr_scale_real) << ", " << std::get<0>(curr_scale_virt)
                << ", selected max rank: " << tg_max_rank
                << ", expected next_rank: " << next_rank << std::endl;

            //Find in local threads at first
            bool find_in_current_process = false;
            for(size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size(); next_thread_id++)
            {
                if( next_thread_id == current_thread_idx)
                {
                    // wrong thread, get next
                    continue;
                }

                // search next_rank in that thread
                auto& next_thread_ring_topology =
                        context.get_process_topology<topology_type>(process_index,
                                                                    next_thread_id)->get_device_storage();
                const auto& real =
                        detail::get_device_with_max_rank<ccl_gpu_comm, type(), topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& virt =
                        detail::get_device_with_max_rank<ccl_virtual_gpu_comm, type(), topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& scale_real =
                        detail::get_device_with_max_rank<ccl_numa_proxy<ccl_gpu_comm>, type(), topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& scale_virt =
                        detail::get_device_with_max_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>, type(), topology_type>(
                                                            next_thread_ring_topology, id_ring);
                if (next_rank != std::min({std::get<0>(real), std::get<0>(virt),
                                           std::get<0>(scale_real), std::get<0>(scale_virt)}))
                {
                    // wrong thread, get next
                    continue;
                }

                out << "next thread: " << next_thread_id << ", min rank candidates: "
                    << std::get<0>(real) << ", " << std::get<0>(virt) << ", "
                    << std::get<0>(scale_real) << ", " << std::get<0>(scale_virt) << std::endl;

                find_in_current_process = true;
                out << "Lock ring for threads ("
                    << current_thread_idx << " <-> "<< next_thread_id << ")" << std::endl;

                if (next_rank == std::get<0>(real))
                {
                    auto locker =
                        detail::add_concurrent_locker_device<ccl_gpu_comm, type(), topology_type>(next_rank,
                                                                                       0,
                                                                                       real,
                                                                                       devices,indexed_devices_for_current_thread);
                    out << "Added real locker by index: " << next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(virt))
                {
                    auto locker =
                        detail::add_concurrent_locker_device<ccl_virtual_gpu_comm, type(), topology_type>(next_rank,
                                                                                               0,
                                                                                               virt,
                                                                                               devices,indexed_devices_for_current_thread);
                    out << "Added virtual locker by index: " << next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(scale_real))
                {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup  real proxy in current thread: " << current_thread_idx << std::endl;
                }
                else if (next_rank == std::get<0>(scale_virt))
                {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup virtual proxy in current thread: " << current_thread_idx << std::endl;
                }
                /*else
                {
                    assert(false && "unknown device type");
                    std::ostringstream ss;
                    ss << out.rdbuf();
                    throw std::runtime_error(std::string(__FUNCTION__) + " - unknown device type. Log:\n" +
                                             ss.str());
                }*/
            }

            //if not find in process local threads - use IPC to find
            if (!find_in_current_process and !ipc_comms.empty())
            {
                indexed_device_container<ccl_ipc_gpu_comm>& curr_locker_map =
                            std::get<ccl_ipc_gpu_comm::type_idx()>(indexed_devices_for_current_thread);

                out << "Lock IPC ring for threads (" << current_thread_idx << " <-> xxx\")" << std::endl;
                auto ipc_it = ipc_comms.find(next_rank);
                if(ipc_it == ipc_comms.end())
                {
                    std::stringstream ss;
                    ss << out.rdbuf();
                    std::cerr << "Cannot find IPC deice by rank: " << next_rank << "\nPrevious log:\n" << ss.str() <<"\nAbort Program" << std::endl;
                    abort();
                }
                const auto& comm_addr = ipc_it->second->template get_comm_data<type(), topology_type>();
                curr_locker_map.insert({comm_addr.rank, ipc_it->second});
                out << "Added locker for thread idx: " << current_thread_idx  <<":\n" << ipc_it->second->to_string() << std::endl;
            }

            //upgrade left gpu device to IPC SOURCE type
            if (!ipc_comms.empty() /*has another IPC Device*/ and current_thread_idx == 0 /* left comm is IPC comm for last process*/ )
            {
                const auto& real = detail::get_device_with_max_rank<ccl_gpu_comm, type(), topology_type>(indexed_devices_for_current_thread, id_ring);
                const auto& virt = detail::get_device_with_max_rank<ccl_virtual_gpu_comm, type(), topology_type>(indexed_devices_for_current_thread, id_ring);

                size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                out << "Upgrade thread id: " << current_thread_idx
                    << " GPU by rank: " << left_ipc_source_rank
                    << " to IPC SOURCE GPU" << std::endl;

                if(left_ipc_source_rank == std::get<0>(real))
                {
                    auto locker =
                                detail::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                    type(), topology_type>(next_rank,
                                                                                   0,
                                                                                   real,
                                                                                   devices,indexed_devices_for_current_thread);
                    out << "Upgrage REAL to IPC_REAL_SOURCE locker by rank: " << next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
                else if (left_ipc_source_rank == std::get<0>(virt))
                {
                    auto locker =
                                detail::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  type(), topology_type>(next_rank,
                                                                                 0,
                                                                                 virt,
                                                                                 devices,indexed_devices_for_current_thread);
                    out << "Upgrage VIRTUAL to IPC_VIRT_SOURCE locker by rank: " << next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
            }
            graph_num++;
        }
    }
    return true;
}

bool allied_process_group_ring_topology::build_specific(std::ostream& out,
                                                        const ccl::process_device_indices_type& per_thread_device_indices,
                                                        const ccl::device_indices_type& scaleout_device_indices,
                                                        const detail::plain_graph_list& graph_list)
{
    out << "TODO: Not implemented";
    return false;
}

bool allied_process_group_ring_topology::build_specific_scale_up_out(
                        std::ostream& out,
                        const ccl::process_device_indices_type& per_thread_device_indices,
                        const ccl::process_device_indices_type& scaleout_device_indices,
                        const ccl::process_device_indices_type& ipc_device_indices,
                        detail::colored_plain_graph_list& graph_list,
                        const std::map<size_t, size_t>& process_device_rank_offset)
{
    out << "TODO: Not implemented";
    return false;
}
detail::global_sorted_plain_graphs
        allied_process_group_ring_topology::global_graph_list_resolver(
                                const detail::adjacency_matrix& matrix,
                                const ccl::process_device_indices_type& per_process_device_indexes,
                                const ccl::process_device_indices_type& foreign_processes_device_indexes,
                                detail::p2p_rating_function ping)
{
    detail::global_sorted_plain_graphs global_graph_list;

    {
        detail::plain_graph_list my_process_list = detail::graph_list_resolver(matrix,
                                                                                 per_process_device_indexes,
                                                                                 ping);
        global_graph_list.emplace(process_index, std::move(my_process_list));
    }

    /*                        my_process_list
     *  <---unknown ring------> [ <> <> ] <---unknown ring---->
     *
     * [------------------------------------------------------]
                        - global size-
     *
     * [<><><>]                 [ <> <> ]               [<><><>]
     *
     * left_index:[<><><>]    my_index:[ <> <> ]    right_index:[<><><>]
     *                ||                  |   |                   | |
     * >______________||__________________|   |___________________| |_____>
     *                   local_comm_group_1     local_comm_group_2     local_i
     */



    return global_graph_list;
}
#endif
} // namespace native
