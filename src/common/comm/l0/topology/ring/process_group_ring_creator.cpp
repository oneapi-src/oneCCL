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

#if 0
namespace native
{

allied_process_group_ring_topology::allied_process_group_ring_topology(size_t process_idx,
                                                                       size_t process_nums,
                                                                       process_group_context &ctx,
                                                                       device_storage& devs,
                                                                       size_t cluster_rank_offset, size_t cluster_size) :
 process_index(process_idx),
 process_count(process_nums),
 context(ctx),
 devices(devs),
 device_cluster_rank_offset(cluster_rank_offset),
 device_cluster_size(cluster_size)
{
}

size_t
    allied_process_group_ring_topology::default_property_p2p_rating_calculator(const ccl_device &lhs,
                                                                               const ccl_device &rhs)
{
    return details::property_p2p_rating_calculator(lhs, rhs, PROCESS_GROUP_WEIGHT);
}

std::pair<size_t, size_t>
allied_process_group_ring_topology::calculate_rank_offset_with_size(size_t process_id,
                                        const std::string& host_id,
                                        const ccl::cluster_aggregated_device_mask_t& cluster_affinity_mask)
{
        auto from_begin = [] (const ccl::process_aggregated_device_mask_t& processes) -> typename ccl::process_aggregated_device_mask_t::const_iterator
                          {
                              return processes.begin();
                          };
        auto from_my_rank = [process_id] (const ccl::process_aggregated_device_mask_t& processes) -> typename ccl::process_aggregated_device_mask_t::const_iterator
                            {
                                return processes.lower_bound(process_id);
                            };

        auto till_my_rank = from_my_rank;
        auto till_end = [](const ccl::process_aggregated_device_mask_t& processes) -> typename ccl::process_aggregated_device_mask_t::const_iterator
                        {
                            return processes.end();
                        };

        auto device_summator = [](size_t part_sum, const ccl::process_aggregated_device_mask_t::value_type& mask) ->size_t
                               {
                                    return part_sum + mask.second.count();
                               };

        auto left_rank_summator = [from_begin, till_my_rank, device_summator]
                                  (size_t part_sum, const typename ccl::cluster_aggregated_device_mask_t::value_type& processes_pair) -> size_t
                                  {
                                      return std::accumulate(from_begin(processes_pair.second), till_my_rank(processes_pair.second), part_sum, device_summator);
                                  };
        auto right_rank_summator = [from_my_rank, till_end, device_summator]
                                   (size_t part_sum, const typename ccl::cluster_aggregated_device_mask_t::value_type& processes_pair) -> size_t
                                   {
                                       return std::accumulate(from_my_rank(processes_pair.second), till_end(processes_pair.second), part_sum, device_summator);
                                   };
        auto rank_summator = [from_begin, till_end, device_summator]
                             (size_t part_sum, const typename ccl::cluster_aggregated_device_mask_t::value_type& processes_pair) -> size_t
                             {
                                 return std::accumulate(from_begin(processes_pair.second), till_end(processes_pair.second), part_sum, device_summator);
                             };

        //calculate ranks offset: summ of devices for each process for each node
        //TODO node sorted by lexicographic comparison
        auto my_node_it = cluster_affinity_mask.find(host_id);

        size_t my_node_rank_devices_offset = std::accumulate(cluster_affinity_mask.begin(),
                                                             my_node_it,
                                                             0,
                                                             rank_summator);
        my_node_rank_devices_offset = std::accumulate(my_node_it,
                                                      std::next(my_node_it),
                                                      my_node_rank_devices_offset,
                                                      left_rank_summator);

        size_t cluster_devices_count = std::accumulate(my_node_it,
                                                      std::next(my_node_it),
                                                      my_node_rank_devices_offset,
                                                      right_rank_summator);
        cluster_devices_count = std::accumulate(my_node_it,
                                                cluster_affinity_mask.end(),
                                                cluster_devices_count,
                                                rank_summator);
        return {my_node_rank_devices_offset, cluster_devices_count};
    }


details::adjacency_matrix
        allied_process_group_ring_topology::build_p2p_capability_matrix(std::ostream& out,
                                                                        const ccl::process_aggregated_device_mask_t& node_device_masks,
                                                                        details::p2p_rating_function ping)
{
    ccl::process_device_indices_t per_process_device_indices;
    for(const auto& mask : node_device_masks)
    {
        per_process_device_indices.insert({mask.first, ccl_device_driver::get_device_indices(mask.second)});
    }

    return build_p2p_capability_matrix(out, per_process_device_indices,
                                       ping);
}

details::adjacency_matrix
    allied_process_group_ring_topology::build_p2p_capability_matrix(std::ostream& out,
                                                                    const ccl::process_device_indices_t& node_device_indices,
                                                                    details::p2p_rating_function ping)
{
    // Build adjacency matrix with P2P capability:
    // Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // element values - is a weight of P2P activity: 0 means - devices are not connected
    // If values is not 0 - than two devies can be combined together

    details::adjacency_matrix ring_p2p_matrix;
    if (node_device_indices.empty())
    {
        out << "No indices nothing to build" << std::endl;
        return ring_p2p_matrix;
    }

    out << "Build adjacency matrix by: " << allied_process_group_ring_topology::name() << std::endl;
    out << "Processes count: " << node_device_indices.size() << "\t";
    out << "Delegate to thread group ring" << std::endl;
    return thread_group_ring_topology::build_p2p_capability_matrix(out,
                                                                   node_device_indices,
                                                                   ping);
}

bool allied_process_group_ring_topology::build(std::ostream& out,
                                                const ccl::process_aggregated_device_mask_t& per_thread_device_masks,
                                                const std::vector<ccl::device_mask_t>& ipc_device_mask,
                                                const details::adjacency_matrix& matrix,
                                                details::p2p_rating_function ping)
{

    ccl::process_device_indices_t per_thread_device_indices;
    for(const auto& mask : per_thread_device_masks)
    {
        per_thread_device_indices.insert({mask.first, ccl_device_driver::get_device_indices(mask.second)});
    }

    std::vector<ccl::device_indices_t> ipc_device_indices;
    for(const auto& mask : ipc_device_mask)
    {
        ipc_device_indices.push_back(ccl_device_driver::get_device_indices(mask));
    }
    return build(out, per_thread_device_indices, ipc_device_indices, matrix, ping);
}

bool allied_process_group_ring_topology::build(std::ostream& out,
               const ccl::process_device_indices_t& per_thread_device_indices,
               const std::vector<ccl::device_indices_t>& ipc_device_indices,
               const details::adjacency_matrix& matrix,
               details::p2p_rating_function ping)
{
    out << "\n/************* \"" << allied_process_group_ring_topology::name()
        << "\" for threads: " << context.process_device_topology.size()
        << "*************/\n" << std::endl;

    // let's emulate process as thread, because topology builder is similar with thread topology
    ccl::process_device_indices_t full_device_indices = per_thread_device_indices;
    size_t max_current_thread_id = per_thread_device_indices.rbegin()->first;
    out << "Assign specific-mock thread id for ipc_devices, count: "
        << ipc_device_indices.size() << std::endl;

    std::vector<size_t> ipc_mock_threads;
    for(size_t i = 0 ; i < ipc_device_indices.size(); i ++)
    {
        size_t mock_thread_id = max_current_thread_id + i + 1; // emulate next thread

        ipc_mock_threads.push_back(mock_thread_id);
        full_device_indices.insert({mock_thread_id, ipc_device_indices[i]});
        out << "{" << mock_thread_id << " for ";
        for (const ccl::device_index_type& idx : ipc_device_indices[i])
        {
             out << idx << ", ";
        }
        out << "}, ";
    }
    out << std::endl;

    // build ring, based on p2p for device hw id
    out << "Resolve device graph" << std::endl;
    details::plain_graph_list id_rings = graph_list_resolver(matrix, full_device_indices, ping);
    size_t size = id_rings.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size)
    {
        out << "Cannot build any ring" << std::endl;
        return false;
    }
    else if (id_rings.size() == 1) // whole ring
    {
        return build_specific(out, full_device_indices, *id_rings.begin());
    }

    //torn-apart ring
    return build_specific(out, full_device_indices, id_rings);
}

bool allied_process_group_ring_topology::build_all(std::ostream& out,
                                                  const ccl::process_device_indices_t& per_thread_device_indices,
                                                  const std::vector<ccl::device_indices_t>& ipc_device_indices,
                                                  const details::adjacency_matrix& matrix,
                                                  details::p2p_rating_function ping)
{
    out << "\n/************* \"" << allied_process_group_ring_topology::name()
        << "\" for threads: " << context.process_device_topology.size()
        << "*************/\n" << std::endl;

    details::plain_graph_list my_rings = create_my_process_graphs(out,
                                                                  per_thread_device_indices,
                                                                  matrix,
                                                                  ping);
    size_t size = my_rings.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size)
    {
        out << "Cannot build any ring" << std::endl;
        return false;
    }

    out << "Graph for process: " << process_index << "\n";
    out << details::to_string(my_rings) << std::endl;

    out << "Transform graph to colored with process color: " << process_index << "\n";
    details::colored_plain_graph_list my_colored_ring = details::create_colored(my_rings, process_index);

    details::global_sorted_colored_plain_graphs global_graphs =
                                    collect_cluster_colored_plain_graphs(out,
                                                                 context.get_communicator(),
                                                                 process_index, my_colored_ring);

    std::map<size_t, size_t> process_device_rank_offset;
    size_t accumulated_offset = 0;
    for (typename details::global_sorted_colored_plain_graphs::value_type& process_graphs : global_graphs)
    {
        size_t process_num = process_graphs.first;
        const details::colored_plain_graph_list& proc_graphs = process_graphs.second;

        process_device_rank_offset[process_num] = accumulated_offset;  //offset for iter process
        out << "Process idx: " << process_num
            << ", rank_offset: " << accumulated_offset << std::endl;
        for (const details::colored_plain_graph& graph : proc_graphs)
        {
            accumulated_offset += graph.size();
        }
    }

    out << "Cluster device size: " << accumulated_offset << std::endl;
    details::global_colored_plain_graphs merged_cluster_graphs =
                                    merge_allied_nodes_in_colored_plain_graphs(out,
                                                                    context.cluster_gpu_indices,
                                                                    process_index, process_count,
                                                                    global_graphs,
                                                                    ping);

    out << "Cluster merged graphs result on process idx: " << process_index << std::endl;
    out << details::to_string(merged_cluster_graphs) << std::endl;

    details::colored_plain_graph_list my_merged_rings =
            resize_merged_colored_graphs_for_process(process_index, process_count,
                                                     merged_cluster_graphs,
                                                     my_colored_ring, out);

    out << "Resized merged graph list on process idx: " << process_index << std::endl;
    out << details::to_string(my_merged_rings) << std::endl;

    out << "Notify merged graphs changes for cluster\n";
    details::global_sorted_colored_plain_graphs global_merged_graphs =
                                    collect_cluster_colored_plain_graphs(out,
                                                                 context.get_communicator(),
                                                                 process_index, my_merged_rings);

    ccl::process_device_indices_t scaleout_devices =
                        create_scaleout_devices_in_colored_graphs_for_process(
                                                                process_index,
                                                                process_count,
                                                                global_merged_graphs,
                                                                global_graphs,
                                                                out);
    out << "Collected scaleout devices: \n";
    for (const auto& pair_idx : scaleout_devices)
    {
        out << "{process: " << pair_idx.first
            << ", device: ";
        for (const auto& idx : pair_idx.second)
        {
            out << idx << ", ";
        }
        out <<"}, ";
    }
    out << std::endl;

    ccl::process_device_indices_t ipc_devices =
                        create_ipc_devices_in_colored_graphs_for_process(
                                                                process_index,
                                                                process_count,
                                                                global_merged_graphs,
                                                                global_graphs,
                                                                out);
    out << "Collected ipc_devices: \n";
    for (const auto& pair_idx : ipc_devices)
    {
        out << "{process: " << pair_idx.first
            << ", device: ";
        for (const auto& idx : pair_idx.second)
        {
            out << idx << ", ";
        }
        out <<"}, ";
    }
    out << std::endl;

    my_merged_rings = global_merged_graphs.find(process_index)->second;
    out << "Final process idx: " << process_index
        << ", has got colored graphs count: " << my_merged_rings.size() << std::endl;
    out << details::to_string(my_merged_rings) << std::endl;

    // enumerate as usual
    if (scaleout_devices.empty())
    {
        size_t size = my_merged_rings.size();
        out << "Resolved graphs count: " << size << "\n";
        if (!size)
        {
            out << "Cannot build any ring" << std::endl;
            return false;
        }
        else if (size == 1) // whole ring
        {
            return build_specific_colored(out, per_thread_device_indices,
                                          ipc_devices, *my_merged_rings.begin(),
                                          process_device_rank_offset);
        }
        //torn-apart ring
        return build_specific_scale_up(out, per_thread_device_indices,
                              ipc_devices, my_merged_rings,
                              process_device_rank_offset);
    }
    //torn-apart ring with scaleout
    return build_specific_scale_up_out(out, per_thread_device_indices,
                          scaleout_devices, ipc_devices,
                          my_merged_rings, process_device_rank_offset);
}

details::plain_graph_list
        allied_process_group_ring_topology::create_my_process_graphs(
                                std::ostream& out,
                                const ccl::process_device_indices_t& per_thread_device_indices,
                                const details::adjacency_matrix& matrix,
                                details::p2p_rating_function ping)
{
    out << "Build device graphs, from threads: " << per_thread_device_indices.size() << std::endl;
    return details::graph_list_resolver(matrix, per_thread_device_indices, ping);

}
details::global_sorted_plain_graphs
        allied_process_group_ring_topology::collect_cluster_plain_graphs(std::ostream& out,
                                                                         std::shared_ptr<ccl::communicator> comm,
                                                                         size_t process_index,
                                                                         const details::plain_graph_list& my_process_graph)
{
    using namespace details::serialize;

    out << "Collect cluster plain graphs, my process index: " << process_index
        << ", graphs count: " << my_process_graph.size() << std::endl;

    std::vector<size_t> recv_process_indices_counts(comm->size(), 1);
    device_path_serializable::raw_data_t my_serialized_graph =
            device_path_serializer::serialize_indices(my_process_graph);

    size_t send_count = my_serialized_graph.size();
    std::vector<size_t> receive_process_graph_sizes(comm->size());

    //std::vector<ccl::communicator::coll_request_t> requests;
    out << "Ask graph lists sizes by process index: " << process_index
        << ", serialized size: " << send_count << std::endl;
    ccl::communicator::coll_request_t req =
                comm->allgatherv(&send_count, 1,
                                 receive_process_graph_sizes.data(),
                                 recv_process_indices_counts.data());

    req->wait();
    size_t global_graph_data_size = std::accumulate(receive_process_graph_sizes.begin(),
                                                    receive_process_graph_sizes.end(),
                                                    0);

    device_path_serializable::raw_data_t global_serialized_graph;
    try
    {
        out << "Send graph list by process index: " << process_index
            << ", serialized size: " << send_count << std::endl;

        global_serialized_graph.resize(global_graph_data_size);
        req = comm->allgatherv(reinterpret_cast<char*>(my_serialized_graph.data()), send_count,
                               reinterpret_cast<char*>(global_serialized_graph.data()),
                               receive_process_graph_sizes.data());
        req->wait();
    }
    catch(const std::exception& ex)
    {
        out << "Cannot submit global-serialized-graph requests " << ex.what() << std::endl;
        out << "Memory required for hostnames size: " << global_graph_data_size << " bytes\n";
        abort();
    }

    size_t deserialized_bytes = 0;
    size_t offset_bytes = 0;
    details::global_sorted_plain_graphs global_ret;

    out << "Deserialize graph_lists" << std::endl;
    for(size_t i = 0; i < comm->size(); i++)
    {
        details::plain_graph_list graph =
                device_path_deserializer::deserialize_graph_list_indices(global_serialized_graph,
                                                                         deserialized_bytes,
                                                                         offset_bytes);
        out << "Process index: " << i << ", deserialized bytes: " << deserialized_bytes
            << ", by offset: " << offset_bytes << std::endl;

        global_ret.emplace(i, std::move(graph));
    }

    out << "Global graph deserialized on process: " << process_index << std::endl;
    return global_ret;
}

details::global_sorted_colored_plain_graphs
        allied_process_group_ring_topology::collect_cluster_colored_plain_graphs(
                                                    std::ostream& out,
                                                    std::shared_ptr<ccl::communicator> comm,
                                                    size_t process_index,
                                                    const details::colored_plain_graph_list& my_process_graph)
{
    using namespace details::serialize;

    out << "Collect cluster colored plain graphs, my process index: " << process_index
        << ", graphs count: " << my_process_graph.size() << std::endl;

    std::vector<size_t> recv_process_indices_counts(comm->size(), 1);
    device_path_serializable::raw_data_t my_serialized_graph =
            device_path_serializer::serialize_indices(my_process_graph);

    size_t send_count = my_serialized_graph.size();
    std::vector<size_t> receive_process_graph_sizes(comm->size());

    //std::vector<ccl::communicator::coll_request_t> requests;
    out << "Ask graph lists sizes by process index: " << process_index
        << ", serialized size: " << send_count << std::endl;
    ccl::communicator::coll_request_t req =
                comm->allgatherv(&send_count, 1,
                                 receive_process_graph_sizes.data(),
                                 recv_process_indices_counts.data());

    req->wait();
    size_t global_graph_data_size = std::accumulate(receive_process_graph_sizes.begin(),
                                                    receive_process_graph_sizes.end(),
                                                    0);

    device_path_serializable::raw_data_t global_serialized_graph;
    try
    {
        out << "Send graph list by process index: " << process_index
            << ", serialized size: " << send_count << std::endl;

        global_serialized_graph.resize(global_graph_data_size);
        req = comm->allgatherv(reinterpret_cast<char*>(my_serialized_graph.data()), send_count,
                               reinterpret_cast<char*>(global_serialized_graph.data()),
                               receive_process_graph_sizes.data());
        req->wait();
    }
    catch(const std::exception& ex)
    {
        out << "Cannot submit global-serialized-graph requests " << ex.what() << std::endl;
        out << "Memory required for hostnames size: " << global_graph_data_size << " bytes\n";
        abort();
    }

    size_t deserialized_bytes = 0;
    size_t offset_bytes = 0;
    details::global_sorted_colored_plain_graphs global_ret;

    out << "Deserialize colored_graph_lists" << std::endl;
    for(size_t i = 0; i < comm->size(); i++)
    {
        details::colored_plain_graph_list graph =
                device_path_deserializer::deserialize_colored_graph_list_indices(global_serialized_graph,
                                                                                 deserialized_bytes,
                                                                                 offset_bytes);
        out << "Process index: " << i << ", deserialized bytes: " << deserialized_bytes
            << ", by offset: " << offset_bytes << std::endl;

        global_ret.emplace(i, std::move(graph));
    }

    out << "Global colored_graph deserialized on process: " << process_index << std::endl;
    return global_ret;
}


details::global_plain_graphs
        allied_process_group_ring_topology::merge_allied_nodes_plain_graphs(std::ostream& out,
                                                                            const ccl::cluster_device_indices_t &cluster_indices,
                                                                            size_t process_index,
                                                                            const details::global_sorted_plain_graphs& cluster_graphs,
                                                                            details::p2p_rating_function ping)
{
    out << "Merge global graphs from processes: " << cluster_graphs.size() << std::endl;
    details::global_plain_graphs ret;
    for (const auto &host_process_id_pair : cluster_indices)
    {
        const ccl::host_id& hostname = host_process_id_pair.first;

        //iterate over all allied processes on the same host
        const ccl::process_device_indices_t& processes = host_process_id_pair.second;
        out << "Try to merge graphs for host: " << hostname << ", allied processes count: "
            << processes.size() << std::endl;

        //collect graphs for all allied processes in lists for merge trying
        std::list<details::plain_graph_list> tmp_allied_processes_graphs;
        for (const auto& process_val : processes)
        {
            auto process_id = process_val.first;
            auto process_graph_list_it = cluster_graphs.find(process_id);
            if (process_graph_list_it == cluster_graphs.end())
            {
                out << "Cannot find process id: " << process_id <<", for hostname: " << hostname
                    << ", in cluster graphs\n";
                std::stringstream ss;
                ss << out.rdbuf();
                throw std::runtime_error(std::string("Cannot merge custer graphs. Log:\n") +
                                                 ss.str());
            }
            tmp_allied_processes_graphs.emplace_back(process_graph_list_it->second);
        }

        //merge and set result for all allied processes
        for (const auto& process_val : processes)
        {
            //merge_lists is stable, let's my process graph list at first in merge result
            std::list<details::plain_graph_list> rotated = tmp_allied_processes_graphs;
            /* TODO rotate ? */
            auto process_index = process_val.first;

            auto new_begin_it = rotated.begin();
            std::advance(new_begin_it, process_index);
            std::rotate(rotated.begin(), new_begin_it, rotated.end());

            ret.push_back(std::make_pair(process_val.first,
                                         details::merge_graph_lists_stable(rotated,
                                                                           ping)));
        }

        out << "graph merged into list, size: " << ret.size() << std::endl;
    }
    return ret;
}

details::global_colored_plain_graphs
        allied_process_group_ring_topology::merge_allied_nodes_in_colored_plain_graphs(
                                                std::ostream& out,
                                                const ccl::cluster_device_indices_t &cluster_indices,
                                                size_t process_index,
                                                size_t process_count,
                                                const details::global_sorted_colored_plain_graphs& cluster_graphs,
                                                details::p2p_rating_function ping)
{
    out << "Merge global colored graphs from processes: " << cluster_graphs.size() << std::endl;
    details::global_colored_plain_graphs ret;
    for (const auto &host_process_id_pair : cluster_indices)
    {
        const ccl::host_id& hostname = host_process_id_pair.first;

        //iterate over all allied processes on the same host
        const ccl::process_device_indices_t& processes = host_process_id_pair.second;
        out << "Try to merge colored graphs for host: " << hostname << ", allied processes count: "
            << processes.size() << std::endl;

        //collect graphs for all allied processes in lists for merge trying
        std::list<details::colored_plain_graph_list> tmp_allied_processes_graphs;

        size_t terminator_process_index = 0;// TODO LIMITATION on MAX PROCESSES COUNT
        for (const auto& process_val : processes)
        {
            auto process_id = process_val.first;
            auto process_graph_list_it = cluster_graphs.find(process_id);
            if (process_graph_list_it == cluster_graphs.end())
            {
                out << "Cannot find process id: " << process_id <<", for hostname: " << hostname
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

        terminator_process_index ++;
        out << "terminator_process_index: " << terminator_process_index;

        //merge and set result for all allied processes
        for (const auto& process_val : processes)
        {
            //merge_lists is stable, let's my process graph list at first in merge result
            auto process_index = process_val.first;

            //turn right
            auto new_begin_it = tmp_allied_processes_graphs.begin();
            std::advance(new_begin_it, process_index);
            std::list<details::colored_plain_graph_list> to_right_part(new_begin_it,
                                                                  tmp_allied_processes_graphs.end());

            //use terminator!
            if(processes.size() != 1)
            {
                if (process_index == processes.size() - 1)
                {
                    //set terminator for right side
                    details::colored_plain_graph_list terminated_list = *tmp_allied_processes_graphs.begin();
                    reset_color(terminated_list, terminator_process_index);
                    to_right_part.push_back(std::move(terminated_list));
                }
            }

            size_t merged_from_right = 0;
            details::colored_plain_graph_list to_right =
                    details::merge_graph_lists_stable_for_process(to_right_part, ping,
                                                                  true, merged_from_right);
            if (to_right.empty())   //i am the rightest process
            {
                to_right = *new_begin_it;
            }


            //turn left
            size_t merged_from_left = 0;
            auto new_end_it = tmp_allied_processes_graphs.begin();
            std::advance(new_end_it, process_index + 1);
            std::list<details::colored_plain_graph_list> to_left_part(tmp_allied_processes_graphs.begin(),
                                                                 new_end_it);
            std::reverse(to_left_part.begin(), to_left_part.end());
            if(to_left_part.empty())
            {
                to_left_part.push_back(to_right);
            }
            else
            {
                *to_left_part.begin() = to_right;
            }

            //use terminator!
            if(processes.size() != 1)
            {
                if (process_index == 0)
                {
                    //set terminator for right side
                    details::colored_plain_graph_list terminated_list = *tmp_allied_processes_graphs.rbegin();
                    reset_color(terminated_list, terminator_process_index);
                    to_left_part.push_back(std::move(terminated_list));
                }
            }
            for (auto &graph : to_left_part)
            {
                std::reverse(graph.begin(), graph.end());
            }
            *to_left_part.begin() = to_right;

            details::colored_plain_graph_list to_left_right =
                    details::merge_graph_lists_stable_for_process(to_left_part, ping,
                                                                  false, merged_from_left);
            ret.push_back(std::make_pair(process_val.first,
                                         to_left_right));
        }

        out << "colored graph merged into list, size: " << ret.size() << std::endl;
    }
    return ret;
}

details::plain_graph_list
        allied_process_group_ring_topology::resize_merged_graphs_for_process(
                                                    size_t process_index,
                                                    const details::global_plain_graphs& merged_cluster_graphs,
                                                    const details::plain_graph_list& original_graph_list,
                                                    std::ostream& out)
{
    out << "remove foreign chains from my merged graphs for process idx: " << process_index <<"\n";
    auto it = std::find_if(merged_cluster_graphs.begin(), merged_cluster_graphs.end(),
                           [process_index] (const typename details::global_plain_graphs::value_type& val)
                           {
                               return val.first == process_index;
                           });
    if (it == merged_cluster_graphs.end())
    {
        out << "Cannot find process: " << process_index << " in merged_cluster_graphs with size: "
            << merged_cluster_graphs.size() << std::endl;
        std::stringstream ss;
        ss << out.rdbuf();
        assert(false);
        throw std::runtime_error(std::string("Cannot resize custer graphs. Log:\n") +
                                                 ss.str());
    }

    details::plain_graph_list my_merged_rings_copy = it->second;
    {
        size_t new_size = my_merged_rings_copy.size();
        size_t old_size = original_graph_list.size();

        out << "Check ring sizes, before: " << old_size << ", after: " << new_size << std::endl;
        if (old_size > new_size)
        {
            abort();
        }

        auto merged_erased_range_it = my_merged_rings_copy.begin();
        std::advance(merged_erased_range_it, old_size);
        my_merged_rings_copy.erase(merged_erased_range_it, my_merged_rings_copy.end());
    }
    return my_merged_rings_copy;
}

details::colored_plain_graph_list
        allied_process_group_ring_topology::resize_merged_colored_graphs_for_process(
                                            size_t process_index,
                                            size_t process_size,
                                            const details::global_colored_plain_graphs& merged_cluster_graphs,
                                            const details::colored_plain_graph_list& original_graph_list,
                                            std::ostream& out)
{
    out << "remove foreign chains from my colored merged graphs for process idx: " << process_index <<"\n";
    auto it = std::find_if(merged_cluster_graphs.begin(), merged_cluster_graphs.end(),
              [process_index] (const typename details::global_colored_plain_graphs::value_type& val)
              {
                   return val.first == process_index;
              });
    if (it == merged_cluster_graphs.end())
    {
        out << "Cannot find process: " << process_index << " in merged_cluster_graphs with size: "
            << merged_cluster_graphs.size() << std::endl;
        std::stringstream ss;
        ss << out.rdbuf();
        throw std::runtime_error(std::string("Cannot resize colored custer graphs. Log:\n") +
                                                 ss.str());
    }

    details::colored_plain_graph_list my_merged_rings_copy = it->second;
    {
        size_t new_size = my_merged_rings_copy.size();
        size_t old_size = original_graph_list.size();

        out << "Check ring sizes, before: " << old_size << ", after: " << new_size << std::endl;
        if (old_size > new_size)
        {
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
                                                        (const details::colored_idx& lhs,
                                                         const details::colored_idx& rhs)
        {
            //size_t right_index = (process_index + 1 ) % process_size;
            //size_t left_index = ( process_index == 0 ?  process_size : process_index - 1);
            return (lhs.first < rhs.first); //stable sort by color!
        });
    }
*/
    return my_merged_rings_copy;
}

ccl::process_device_indices_t
        allied_process_group_ring_topology::create_scaleout_devices_in_graphs_for_process(
                                                        size_t process_idx,
                                                        size_t cluster_size,
                                                        details::global_sorted_plain_graphs& cluster_graphs,
                                                        std::ostream& out)
{
    size_t left_process_idx = (process_idx == 0
                               ? cluster_size - 1 : process_idx - 1);
    size_t right_process_idx = ((process_idx + 1) % cluster_size);

    out << "Create scaleout devices for process: (" << process_idx << "/" << cluster_size << ")"
        << ", left_process_idx: " << left_process_idx
        << ", right_process_idx: " << right_process_idx << std::endl;

    ccl::process_device_indices_t scaleout_devices;
    auto me = cluster_graphs.find(process_idx)->second;

    if (process_idx > left_process_idx)
    {
        auto lhs = cluster_graphs.find(left_process_idx)->second;
        auto find_shared_graph_it = std::find(lhs.begin(), lhs.end(), *me.begin());
        if (find_shared_graph_it == lhs.end())
        {
            const ccl::device_index_type& scaleout = *(lhs.rbegin()->rbegin());
            out << "scaleout candidate from Lhs: " << scaleout << std::endl;
            me.insert(me.begin(), {{scaleout}});
            scaleout_devices[left_process_idx]= {scaleout};
        }
    }

    if (process_idx < right_process_idx)
    {
        auto rhs = cluster_graphs.find(right_process_idx)->second;
        auto find_shared_graph_it = std::find(rhs.begin(), rhs.end(), *me.rbegin());
        if (find_shared_graph_it == rhs.end())
        {
            const ccl::device_index_type& scaleout = *(rhs.begin()->begin());
            out << "scaleout candidate from Rhs: " << scaleout << std::endl;
            me.insert(me.end(), {{scaleout}});
            scaleout_devices[right_process_idx] = {scaleout};
        }
    }

    return scaleout_devices;
}

ccl::process_device_indices_t
                allied_process_group_ring_topology::create_scaleout_devices_in_colored_graphs_for_process(
                                        size_t process_idx,
                                        size_t cluster_size,
                                        details::global_sorted_colored_plain_graphs& cluster_graphs,
                                        details::global_sorted_colored_plain_graphs& initial_cluster_graphs,
                                        std::ostream& out)

{
    using optional_process = std::pair<bool, size_t>;

    optional_process left_process_idx = std::make_pair(true,
                                                       (process_idx == 0
                                                       ? cluster_size - 1 : process_idx - 1));
    optional_process right_process_idx = std::make_pair(true,
                                                        ((process_idx + 1) % cluster_size));

    out << "Create scaleout devices for process: (" << process_idx << "/" << cluster_size << ")"
        << ", left_process_idx: " << left_process_idx.second
        << ", right_process_idx: " << right_process_idx.second << std::endl;

    ccl::process_device_indices_t scaleout_devices;
    // process corner cases
    if(left_process_idx == right_process_idx)
    {
        //two processes
        if (process_idx > left_process_idx.second)
        {
            left_process_idx.first = false; //do not process left
        }
        else
        {
            right_process_idx.first = false; //do not process right
        }
    }

    if(left_process_idx.second == process_idx and process_idx == right_process_idx.second)
    {
        return scaleout_devices;    //nothing to scaleout
    }


    auto& me = cluster_graphs.find(process_idx)->second;

    std::unique_ptr<size_t> color_to_find(new size_t);
    auto find_in_list_by_color = [&color_to_find](const details::colored_plain_graph& graph) -> bool
    {
        auto it = std::find_if(graph.begin(), graph.end(), [&color_to_find](const details::colored_idx& idx)
        {
            return (idx.color == *color_to_find);
        });
        return it != graph.end();
    };

    if (left_process_idx.first)
    {
        // find lhs in my graphs
        *color_to_find = left_process_idx.second;
        if (process_idx == 0)
        {
            //use terminate
            *color_to_find = cluster_size;
        }

        if (std::find_if(me.begin(), me.end(), find_in_list_by_color) == me.end())
        {
            //add scaleout device
            auto lhs_it = initial_cluster_graphs.find(left_process_idx.second);
            if (lhs_it == initial_cluster_graphs.end())
            {
                assert(false && "lhs process doesn't exist");
                throw std::runtime_error(std::string(__FUNCTION__) + " - invalid cluster_graph: " +
                                         "no process by id: " + std::to_string(left_process_idx.second));
            }

            const auto& lhs = lhs_it->second;
            if(lhs.empty())
            {
                assert(false && "lhs process graph is empty ");
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - invalid cluster_graph: empty list " +
                                         "for process by id: " + std::to_string(left_process_idx.second));

            }
            const ccl::device_index_type& scaleout = (lhs.rbegin()->rbegin())->index;
            out << "scaleout candidate from Lhs: " << scaleout << std::endl;
            //me.insert(me.begin(), { {left_process_idx.second, scaleout}});
            scaleout_devices[left_process_idx.second] = {scaleout};
        }
    }

    if (right_process_idx.first)
    {
        // find rhs in my graphs
        *color_to_find = right_process_idx.second;
        if (process_idx == cluster_size - 1)
        {
            //use terminate
            *color_to_find = cluster_size;
        }

        if (std::find_if(me.begin(), me.end(), find_in_list_by_color) == me.end())
        {
            //add scaleout device
            auto rhs_it = initial_cluster_graphs.find(right_process_idx.second);
            if (rhs_it == initial_cluster_graphs.end())
            {
                assert(false && "rhs process doesn't exist");
                throw std::runtime_error(std::string(__FUNCTION__) + " - invalid cluster_graph: " +
                                         "no process by id: " + std::to_string(right_process_idx.second));
            }

            const auto& rhs = rhs_it->second;
            if(rhs.empty())
            {
                assert(false && "rhs process graph is empty ");
                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - invalid cluster_graph: empty list " +
                                         "for process by id: " + std::to_string(right_process_idx.second));

            }
            const ccl::device_index_type& scaleout = (rhs.begin()->begin())->index;
            out << "scaleout candidate from Lhs: " << scaleout << std::endl;
            //me.insert(me.end(), {{right_process_idx.second, scaleout}});
            scaleout_devices[right_process_idx.second] = {scaleout};
        }
    }

    return scaleout_devices;
}

ccl::process_device_indices_t
                allied_process_group_ring_topology::create_ipc_devices_in_colored_graphs_for_process(
                                        size_t process_idx,
                                        size_t cluster_size,
                                        details::global_sorted_colored_plain_graphs& cluster_graphs,
                                        details::global_sorted_colored_plain_graphs& initial_cluster_graphs,
                                        std::ostream& out)
{
    (void)initial_cluster_graphs;

    using optional_process = std::pair<bool, size_t>;

    optional_process left_process_idx = std::make_pair(true,
                                                       (process_idx == 0
                                                       ? cluster_size - 1 : process_idx - 1));
    optional_process right_process_idx = std::make_pair(true,
                                                        ((process_idx + 1) % cluster_size));

    out << "Create scaleout devices for process: (" << process_idx << "/" << cluster_size << ")"
        << ", left_process_idx: " << left_process_idx.second
        << ", right_process_idx: " << right_process_idx.second << std::endl;

    ccl::process_device_indices_t ipc_devices;
    // process corner cases
    if(left_process_idx == right_process_idx)
    {
        //two processes
        if (process_idx > left_process_idx.second)
        {
            left_process_idx.first = false; //do not process left
        }
        else
        {
            right_process_idx.first = false; //do not process right
        }
    }

    if(left_process_idx.second == process_idx and process_idx == right_process_idx.second)
    {
        return ipc_devices;    //nothing to ipc
    }


    auto& me = cluster_graphs.find(process_idx)->second;

    std::unique_ptr<size_t> color_to_find(new size_t);
    std::vector<details::colored_idx> devices_to_remember;

    //TODO limitation: all graphs ipc devices would be merged into one vector
    auto filter_list_by_color =
    [&color_to_find, &devices_to_remember] (const details::colored_plain_graph& graph) -> void
    {
        std::copy_if(graph.begin(), graph.end(), std::back_inserter(devices_to_remember),
                     [&color_to_find](const details::colored_idx& idx)
        {
            return (idx.color == *color_to_find);
        });
    };

    if (left_process_idx.first)
    {
        // find lhs color in my graphs
        *color_to_find = left_process_idx.second;
        devices_to_remember.clear();
        if (process_idx == 0)
        {
            //use terminate
            *color_to_find = cluster_size;
        }

        //fill ipc devices candidates in devices_to_remember
        std::for_each(me.begin(), me.end(), filter_list_by_color);
        if (!devices_to_remember.empty())
        {
            const ccl::device_index_type& ipc = devices_to_remember.rbegin()->index;
            out << "ipc candidate from Lhs: " << ipc << std::endl;
            ipc_devices[left_process_idx.second] = {ipc};
        }
    }

    if (right_process_idx.first)
    {
        // find rhs in my graphs
        *color_to_find = right_process_idx.second;
        devices_to_remember.clear();
        if (process_idx == cluster_size - 1)
        {
            //use terminate
            *color_to_find = cluster_size;
        }

        //fill ipc devices candidates in devices_to_remember
        std::for_each(me.begin(), me.end(), filter_list_by_color);
        if (!devices_to_remember.empty())
        {
            const ccl::device_index_type& ipc = devices_to_remember.begin()->index;
            out << "ipc candidate from rhs: " << ipc << std::endl;
            ipc_devices[right_process_idx.second] = {ipc};
        }
    }

    return ipc_devices;
}

bool allied_process_group_ring_topology::build_specific(std::ostream& out,
                                                        const ccl::process_device_indices_t& per_thread_device_indices,
                                                        const details::plain_graph& id_ring)
{
    constexpr ccl::device_group_split_type topology_type = ccl::device_group_split_type::cluster;

    out << "Start building topology: " << ::to_string(topology_type) << ", for graph:\n";
    for (const auto& id : id_ring)
    {
        out << id << ", ";
    }

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    auto& ctx_per_thread_data = context.process_device_topology;
    details::id_thread_table assigned_ids;
    std::vector<details::marked_idx> marked_id_ring = details::create_marked(id_ring);
    for (auto per_thread_it = ctx_per_thread_data.begin(); per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it)
    {
        size_t thread_id = per_thread_it->first;        // first
        auto& out_indexed_devices =
                context.get_process_topology<topology_type>(process_index,
                                                            thread_id)->get_device_storage(); // just second

        std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices.thread_gpu_comms.find(thread_id)->second;

        auto rank_builder =
                    create_device_functor<details::graph_ring_indexer_ext<topology_type>>(marked_id_ring,
                                                                                          assigned_ids,
                                                                                          thread_id,
                                                                                          out_indexed_devices,
                                                                                          0,
                                                                                          device_cluster_rank_offset,
                                                                                          device_cluster_size);
        ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

        details::printer<topology_type> p;
        ccl_tuple_for_each(*non_indexed_plain_devices, p);
        out << "Indexer result for devices in thread idx ("
            << thread_id << "/" << ctx_per_thread_data.size() << "):\n"
            << p.to_string() << std::endl;
    }

    //allocate IPC devices pool with rank from unassigned IDs
    details::ipc_devices_pool ipc_comms =
                    details::create_ipc_gpu_comms<topology_type>(assigned_ids, id_ring, devices,
                                                                 device_cluster_size,
                                                                 device_cluster_rank_offset);
    out << "Created IPC devices: " << ipc_comms.size() << ", for cluster_size: " << device_cluster_size
        << ", with device_cluster_rank_offset: " << device_cluster_rank_offset << "\n";
    for (const auto& ipc : ipc_comms)
    {
        out << "{ rank: " << ipc.first << ", comm: " << ipc.second->to_string() << "}\n";
    }

    out << "\nStart ring builder" << std::endl;
    for(size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size(); current_thread_idx++)
    {
         // find max rank in current thread device list
        auto& indexed_devices_for_current_thread =
                    context.get_process_topology<topology_type>(process_index,
                                                                current_thread_idx)->get_device_storage();
        const auto& curr_real = details::get_device_with_min_rank<ccl_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);
        const auto& curr_virt = details::get_device_with_min_rank<ccl_virtual_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);

        size_t tg_max_rank = std::max({std::get<0>(curr_real), std::get<0>(curr_virt)});

        // find thread, which will connect to current thread max rank with next_rank
        size_t next_rank = (tg_max_rank + 1 ) % id_ring.size();

        out << "Current thread: " << current_thread_idx << ", max rank candidates: "
            << std::get<0>(curr_real) << ", " << std::get<0>(curr_virt)
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
            const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(next_thread_ring_topology, id_ring);
            const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(next_thread_ring_topology, id_ring);

            if (next_rank != std::min({std::get<0>(real), std::get<0>(virt)}))
            {
                // wrong thread, get next
                continue;
            }

            out << "next thread: " << next_thread_id << ", min rank candidates: "
                << std::get<0>(real) << ", " << std::get<0>(virt) << std::endl;

            find_in_current_process = true;
            out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id << ")" << std::endl;
            if (next_rank == std::get<0>(real))
            {
                auto locker =
                    details::add_concurrent_locker_device<ccl_gpu_comm, topology_type>(next_rank,
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
                    details::add_concurrent_locker_device<ccl_virtual_gpu_comm, topology_type>(next_rank,
                                                                                               0,
                                                                                               virt,
                                                                                               devices,indexed_devices_for_current_thread);
                out << "Added virtual locker by index: " << next_rank
                    << ", for thread idx: " << current_thread_idx  <<":\n"
                    << locker->to_string() << std::endl;
            }
            else
            {
                assert(false && "unknown device type");
                std::ostringstream ss;
                ss << out.rdbuf();
                throw std::runtime_error(std::string(__FUNCTION__) + " - unknown device type. Log:\n" +
                                         ss.str());
            }
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
            const auto& comm_addr = ipc_it->second->template get_comm_data<topology_type>();
            curr_locker_map.insert({comm_addr.rank, ipc_it->second});
            out << "Added locker for thread idx: " << current_thread_idx  <<":\n" << ipc_it->second->to_string() << std::endl;
        }

        //upgrade left gpu device to IPC SOURCE type
        if (!ipc_comms.empty()/*has another IPC Device*/ and current_thread_idx == 0 /* left comm is IPC comm for last process*/ )
        {
            const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);
            const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);

            size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
            out << "Upgrade thread id: " << current_thread_idx
                << " GPU by rank: " << left_ipc_source_rank
                << " to IPC SOURCE GPU" << std::endl;

            if(left_ipc_source_rank == std::get<0>(real))
            {
                auto locker =
                            details::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                  topology_type>(next_rank,
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
                            details::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  topology_type>(next_rank,
                                                                                 0,
                                                                                 virt,
                                                                                 devices,indexed_devices_for_current_thread);
                out << "Upgrage VIRTUAL to IPC_VIRT_SOURCE locker by rank: " << next_rank
                    << ", for thread idx: " << current_thread_idx  <<":\n"
                    << locker->to_string() << std::endl;
            }
        }
    }
    return true;
}

bool allied_process_group_ring_topology::build_specific_colored(std::ostream& out,
                        const ccl::process_device_indices_t& per_thread_device_indices,
                        const ccl::process_device_indices_t& ipc_device_indices,
                        details::colored_plain_graph& id_ring,
                        const std::map<size_t, size_t>& process_device_rank_offset)
{
    //continuous ring, without scale-up devices
    //processes connected using IPC devices
    //Rank = Index
    constexpr ccl::device_group_split_type topology_type = ccl::device_group_split_type::cluster;

    out << "Start building topology: " << ::to_string(topology_type) << ", for colored graph:\n"
        << details::to_string(id_ring) << std::endl;

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;

    auto offset_it = process_device_rank_offset.find(process_index);
    if (offset_it == process_device_rank_offset.end())
    {
        assert(false && "");
    }

    size_t device_rank_offset = offset_it->second;

    out << "global rank offset: " << device_rank_offset << std::endl;
    auto& ctx_per_thread_data = context.process_device_topology;
    for (auto per_thread_it = ctx_per_thread_data.begin(); per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it)
    {
        size_t thread_id = per_thread_it->first;        // first
        auto& out_indexed_devices =
                context.get_process_topology<topology_type>(process_index,
                                                            thread_id)->get_device_storage(); // just second

        std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices.thread_gpu_comms.find(thread_id)->second;

        //allocate IPC devices pool(if needed)
        details::cluster_ipc_devices_pool ipc_comms =
                    details::create_filtered_ipc_destination_gpu_comms<topology_type>(
                                            id_ring,
                                            ipc_device_indices,
                                            process_index,
                                            process_count,
                                            devices,
                                            *non_indexed_plain_devices);

        auto rank_builder =
                    create_device_functor<details::smart_ring_indexer<topology_type>>(
                                            id_ring,
                                            process_index,
                                            process_count,
                                            device_rank_offset,
                                            devices,
                                            out_indexed_devices,
                                            ipc_device_indices,
                                            ccl::process_device_indices_t{});
        //start indexer
        ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

        details::printer<topology_type> p;
        ccl_tuple_for_each(*non_indexed_plain_devices, p);
        out << "Indexer result for devices in thread idx ("
            << thread_id << "/" << ctx_per_thread_data.size() << "):\n"
            << p.to_string() << std::endl;
    }

    out << "\nStart ring builder" << std::endl;
    for(size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size(); current_thread_idx++)
    {
        // find max rank in current thread device list
        auto& indexed_devices_for_current_thread =
                    context.get_process_topology<topology_type>(process_index,
                                                                current_thread_idx)->get_device_storage();
        const auto& curr_real =
                    details::get_device_with_min_rank<ccl_gpu_comm, topology_type>(
                                        indexed_devices_for_current_thread, id_ring);
        const auto& curr_virt =
                    details::get_device_with_min_rank<ccl_virtual_gpu_comm, topology_type>(
                                        indexed_devices_for_current_thread, id_ring);

        size_t tg_max_rank = std::max({std::get<0>(curr_real), std::get<0>(curr_virt)});

        // find thread, which will connect to current thread max rank with next_rank
        size_t next_rank = (tg_max_rank + 1 ) % id_ring.size();

        out << "Current thread: " << current_thread_idx << ", max rank candidates: "
            << std::get<0>(curr_real) << ", " << std::get<0>(curr_virt)
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
                    details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(
                                        next_thread_ring_topology, id_ring);
            const auto& virt =
                    details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(
                                        next_thread_ring_topology, id_ring);

            if (next_rank != std::min({std::get<0>(real), std::get<0>(virt)}))
            {
                // wrong thread, get next
                continue;
            }

            out << "next thread: " << next_thread_id << ", min rank candidates: "
                << std::get<0>(real) << ", " << std::get<0>(virt) << std::endl;

            find_in_current_process = true;
            out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id << ")" << std::endl;
            if (next_rank == std::get<0>(real))
            {
                auto locker =
                    details::add_concurrent_locker_device<ccl_gpu_comm, topology_type>(next_rank,
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
                    details::add_concurrent_locker_device<ccl_virtual_gpu_comm, topology_type>(next_rank,
                                                                                               0,
                                                                                               virt,
                                                                                               devices,indexed_devices_for_current_thread);
                out << "Added virtual locker by index: " << next_rank
                    << ", for thread idx: " << current_thread_idx  <<":\n"
                    << locker->to_string() << std::endl;
            }
            else
            {
                assert(false && "unknown device type");
                std::ostringstream ss;
                ss << out.rdbuf();
                throw std::runtime_error(std::string(__FUNCTION__) + " - unknown device type. Log:\n" +
                                         ss.str());
            }
        }

        if (find_in_current_process)
        {
            abort();
        }
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
                const auto& comm_addr = ipc_it->second->template get_comm_data<topology_type>();
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
                const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(*indexed_devices_for_current_thread, id_ring);
                const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(*indexed_devices_for_current_thread, id_ring);
                size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                out << "Upgrade thread id: " << current_thread_idx
                    << " GPU by rank: " << left_ipc_source_rank
                    << " to IPC SOURCE GPU" << std::endl;
                if(left_ipc_source_rank == std::get<0>(real))
                {
                    auto locker =
                            details::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                  topology_type>(next_rank,
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
                            details::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  topology_type>(next_rank,
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
    return true;
}

bool allied_process_group_ring_topology::build_specific(std::ostream& out,
                                                        const ccl::process_device_indices_t& per_thread_device_indices,
                                                        const details::plain_graph_list& graph_list)
{
     constexpr ccl::device_group_split_type topology_type =
                                        ccl::device_group_split_type::process_group_torn_apart_ring;

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

                auto scale_virt = details::add_numa_proxy_device<ccl_virtual_gpu_comm, topology_type>(
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
                    auto scale_real = details::add_numa_proxy_device<ccl_gpu_comm, topology_type>(
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
    details::ipc_devices_pool ipc_comms;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t/*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    for (const auto& id_ring : graph_list)
    {
        details::id_thread_table assigned_ids;  //device_id -> thread_id

        std::vector<details::marked_idx> marked_id_ring = details::create_marked(id_ring);  // marked graph

        size_t index_offset = accumulated_index_offset_for_graph;
        for (auto per_thread_it = ctx_per_thread_data.begin(); per_thread_it != ctx_per_thread_data.end();
            ++per_thread_it)
        {
            size_t thread_id = per_thread_it->first;        //first
            auto& out_indexed_devices =
                    context.get_process_topology<topology_type>(process_index,
                                                            thread_id)->get_device_storage(); //just second

            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices.thread_gpu_comms.find(thread_id)->second;

            // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
            auto rank_builder =
                    create_device_functor<details::graph_ring_indexer_unique_index_ext<topology_type>>(marked_id_ring,
                                                                                      assigned_ids,
                                                                                      thread_id,
                                                                                      out_indexed_devices,
                                                                                      index_offset + device_cluster_rank_offset,
                                                                                      0,
                                                                                      0);
//                                                                                    device_cluster_size

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            details::printer<topology_type> p;
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
        details::ipc_devices_pool tmp_ipc_comms =
                        details::create_ipc_gpu_comms<topology_type>(assigned_ids, id_ring, devices,
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
                                                                current_thread_idx)->get_device_storage();
            const auto& curr_real =
                    details::get_device_with_min_rank<ccl_gpu_comm, topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_virt =
                    details::get_device_with_min_rank<ccl_virtual_gpu_comm, topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_real =
                    details::get_device_with_min_rank<ccl_numa_proxy<ccl_gpu_comm>, topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_virt =
                    details::get_device_with_min_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>, topology_type>(
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
                        details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& virt =
                        details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& scale_real =
                        details::get_device_with_max_rank<ccl_numa_proxy<ccl_gpu_comm>, topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& scale_virt =
                        details::get_device_with_max_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>, topology_type>(
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
                        details::add_concurrent_locker_device<ccl_gpu_comm, topology_type>(next_rank,
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
                        details::add_concurrent_locker_device<ccl_virtual_gpu_comm, topology_type>(next_rank,
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
                const auto& comm_addr = ipc_it->second->template get_comm_data<topology_type>();
                curr_locker_map.insert({comm_addr.rank, ipc_it->second});
                out << "Added locker for thread idx: " << current_thread_idx  <<":\n" << ipc_it->second->to_string() << std::endl;
            }

            //upgrade left gpu device to IPC SOURCE type
            if (!ipc_comms.empty() /*has another IPC Device*/ and current_thread_idx == 0 /* left comm is IPC comm for last process*/ )
            {
                const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);
                const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);

                size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                out << "Upgrade thread id: " << current_thread_idx
                    << " GPU by rank: " << left_ipc_source_rank
                    << " to IPC SOURCE GPU" << std::endl;

                if(left_ipc_source_rank == std::get<0>(real))
                {
                    auto locker =
                                details::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                    topology_type>(next_rank,
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
                                details::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  topology_type>(next_rank,
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

bool allied_process_group_ring_topology::build_specific_scale_up(std::ostream& out,
                        const ccl::process_device_indices_t& per_thread_device_indices,
                        const ccl::process_device_indices_t& ipc_device_indices,
                        details::colored_plain_graph_list& graph_list,
                        const std::map<size_t, size_t>& process_device_rank_offset)
{
    constexpr ccl::device_group_split_type topology_type =
                                        ccl::device_group_split_type::process_group_torn_apart_ring;

    out << "Start building topology: " << ::to_string(topology_type)
        << ", for colored graphs: " << graph_list.size() << "\n";
    out << details::to_string(graph_list) << std::endl;

    auto& ctx_per_thread_data = context.process_device_topology;
    out << "\nStart gpu comm transformation scale-up for graph list count: "
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
            details::color_t process;
            ccl::device_index_type last_in_graph_index;
            auto tmp = *id_ring.rbegin();
            process = tmp.color;
            last_in_graph_index = tmp.index;
            if (per_thread.second.find(last_in_graph_index) != per_thread.second.end())
            {
                assert(process == process_index);
                out << "thread: " << thread_id << " wants to create scale_up device by idx: "
                    << last_in_graph_index << std::endl;
                if (created_scaleup_indices.find(last_in_graph_index) != created_scaleup_indices.end())
                {
                    out << "skip existing scale_up device candidate by: " << last_in_graph_index << std::endl;
                    continue;
                }

                auto scale_virt = details::add_numa_proxy_device<ccl_virtual_gpu_comm, topology_type>(
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
                    auto scale_real = details::add_numa_proxy_device<ccl_gpu_comm, topology_type>(
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
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t/*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    auto offset_it = process_device_rank_offset.find(process_index);
    if (offset_it == process_device_rank_offset.end())
    {
        assert(false && "");
    }

    accumulated_index_offset_for_graph = offset_it->second;

    out << "global rank offset: " << accumulated_index_offset_for_graph << std::endl;

    for (auto& id_ring : graph_list)
    {
        size_t index_offset = accumulated_index_offset_for_graph;
        for (auto per_thread_it = ctx_per_thread_data.begin();
             per_thread_it != ctx_per_thread_data.end();
             ++per_thread_it)
        {
            size_t thread_id = per_thread_it->first;        //first
            auto& out_indexed_devices =
                    context.get_process_topology<topology_type>(process_index,
                                                            thread_id)->get_device_storage(); //just second

            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices.thread_gpu_comms.find(thread_id)->second;

            //allocate IPC devices pool(if needed)
            details::cluster_ipc_devices_pool ipc_comms =
                    details::create_filtered_ipc_destination_gpu_comms<topology_type>(
                                            id_ring,
                                            ipc_device_indices,
                                            process_index,
                                            process_count,
                                            devices,
                                            *non_indexed_plain_devices);

            auto rank_builder =
                    create_device_functor<details::smart_ring_indexer<topology_type>>(
                                            id_ring,
                                            process_index,
                                            process_count,
                                            index_offset,
                                            devices,
                                            out_indexed_devices,
                                            ipc_device_indices,
                                            ccl::process_device_indices_t{});

            // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
           /* auto rank_builder =
                    create_device_functor<details::colored_graph_ring_indexer<topology_type>>(id_ring,
                                                                                      thread_id,
                                                                                      process_index,
                                                                                      out_indexed_devices,
                                                                                      0,
                                                                                      0,
                                                                                      index_offset);

*/
            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            details::printer<topology_type> p;
            ccl_tuple_for_each(out_indexed_devices, p);
            out << "Indexer result for devices in thread idx ("
                << thread_id << "/" << ctx_per_thread_data.size() << "):\n"
                << p.to_string() << std::endl;

            accumulated_index_offset_for_graph += rank_builder.get_functor().get_marked_indices_count();
            out << "\nIndexer for graph num: " << graph_num
                << ", finished. imarked_indices: " << accumulated_index_offset_for_graph <<"\n";
        }
        index_offset_for_graphs[graph_num] = index_offset;
        graph_num ++;
    }

    //allocate IPC devices pool with rank from unassigned IDs
    details::cluster_ipc_devices_pool ipc_comms =
                    details::create_ipc_gpu_comms<topology_type>(graph_list, process_index, devices,
                                                                 device_cluster_size,
                                                                 device_cluster_rank_offset);
    out << "Created IPC devices for processes: " << ipc_comms.size() << ", for cluster_size: " << device_cluster_size
        << ", with device_cluster_rank_offset: " << device_cluster_rank_offset << "\n";
    for (const auto& process_ipc : ipc_comms)
    {
        out << "prx: " << process_ipc.first << std::endl;
        for (const auto& ipc : process_ipc.second)
        {
            out << "{ rank: " << ipc.first << ", comm: " << ipc.second->to_string() << "}\n";
        }
        out <<  std::endl;
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
                                                                current_thread_idx)->get_device_storage();
            const auto& curr_real =
                    details::get_device_with_min_rank<ccl_gpu_comm, topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_virt =
                    details::get_device_with_min_rank<ccl_virtual_gpu_comm, topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_real =
                    details::get_device_with_min_rank<ccl_numa_proxy<ccl_gpu_comm>, topology_type>(
                                                    indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_virt =
                    details::get_device_with_min_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>, topology_type>(
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
                        details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& virt =
                        details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& scale_real =
                        details::get_device_with_max_rank<ccl_numa_proxy<ccl_gpu_comm>, topology_type>(
                                                            next_thread_ring_topology, id_ring);
                const auto& scale_virt =
                        details::get_device_with_max_rank<ccl_numa_proxy<ccl_virtual_gpu_comm>, topology_type>(
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
                        details::add_concurrent_locker_device<ccl_gpu_comm, topology_type>(next_rank,
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
                        details::add_concurrent_locker_device<ccl_virtual_gpu_comm, topology_type>(next_rank,
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
                    const auto& comm_addr = ipc_it->second->template get_comm_data<topology_type>();
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
                if ( current_thread_idx == 0 /* left comm is IPC comm for last process*/ )
                {
                    const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);
                    const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(indexed_devices_for_current_thread, id_ring);

                    size_t left_ipc_source_rank = std::min({std::get<0>(real), std::get<0>(virt)});
                    out << "Upgrade thread id: " << current_thread_idx
                        << " GPU by rank: " << left_ipc_source_rank
                        << " to IPC SOURCE GPU" << std::endl;

                    if(left_ipc_source_rank == std::get<0>(real))
                    {
                        auto locker =
                                    details::add_ipc_source_locker_device<ccl_gpu_comm,
                                                                        topology_type>(next_rank,
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
                                details::add_ipc_source_locker_device<ccl_virtual_gpu_comm,
                                                                  topology_type>(next_rank,
                                                                                 0,
                                                                                 virt,
                                                                                 devices,indexed_devices_for_current_thread);
                        out << "Upgrage VIRTUAL to IPC_VIRT_SOURCE locker by rank: " << next_rank
                            << ", for thread idx: " << current_thread_idx  <<":\n"
                            << locker->to_string() << std::endl;
                    }
                }
            }
            graph_num++;
        }
    }
    return true;
}

bool allied_process_group_ring_topology::build_specific(std::ostream& out,
                                                        const ccl::process_device_indices_t& per_thread_device_indices,
                                                        const ccl::device_indices_t& scaleout_device_indices,
                                                        const details::plain_graph_list& graph_list)
{
    out << "TODO: Not implemented";
    return false;
}

bool allied_process_group_ring_topology::build_specific_scale_up_out(
                        std::ostream& out,
                        const ccl::process_device_indices_t& per_thread_device_indices,
                        const ccl::process_device_indices_t& scaleout_device_indices,
                        const ccl::process_device_indices_t& ipc_device_indices,
                        details::colored_plain_graph_list& graph_list,
                        const std::map<size_t, size_t>& process_device_rank_offset)
{
    out << "TODO: Not implemented";
    return false;
}
details::global_sorted_plain_graphs
        allied_process_group_ring_topology::global_graph_list_resolver(
                                const details::adjacency_matrix& matrix,
                                const ccl::process_device_indices_t& per_process_device_indexes,
                                const ccl::process_device_indices_t& foreign_processes_device_indexes,
                                details::p2p_rating_function ping)
{
    details::global_sorted_plain_graphs global_graph_list;

    {
        details::plain_graph_list my_process_list = details::graph_list_resolver(matrix,
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

}
#endif
