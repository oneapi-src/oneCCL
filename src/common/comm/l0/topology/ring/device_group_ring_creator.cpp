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

namespace native
{

device_group_ring_topology::device_group_ring_topology(device_group_context& comm,
                                                       device_storage& devs) :
 context(comm),
 devices_factory(devs)
{
}

size_t device_group_ring_topology::default_property_p2p_rating_calculator(const ccl_device &lhs,
                                                                          const ccl_device &rhs)
{
    return details::property_p2p_rating_calculator(lhs, rhs, DEVICE_GROUP_WEIGHT);
}

details::adjacency_matrix
device_group_ring_topology::build_p2p_capability_matrix(std::ostream& out,
                                                        const ccl::device_indices_t &group_device_indices,
                                                        details::p2p_rating_function ping)
{
    // Build adjacency matrix between devices using `ping` function:
    // Default ping function is checking P2P access capabilities in a way:
    // 1) Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // 2) Matrix element values - is P2P access score: 0 means - devices are not connected
    // If values is not 0 - than two devies can access either together

    out << "Build adjacency matrix by: " << device_group_ring_topology::name()
        << " - group indices count: " << group_device_indices.size() << std::endl;

    //Request for alldevices in all allied processes on the node
    std::shared_ptr<ccl_device_platform> platform_ptr =
                    ccl_device_platform::create(group_device_indices);

    size_t driver_idx = 0; //TODO
    ccl_device_platform::driver_ptr driver = platform_ptr->get_driver(driver_idx);// MUST EXIST
    return details::create_adjacency_matrix_for_devices(driver->get_devices(),
                                                        ping);
}


details::adjacency_matrix
device_group_ring_topology::build_p2p_capability_matrix(std::ostream& out,
                                                        const ccl::device_mask_t &group_device_masks,
                                                        details::p2p_rating_function ping)
{
    // Build adjacency matrix between devices using `ping` function:
    // Default ping function is checking P2P access capabilities in a way:
    // 1) Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // 2) Matrix element values - is P2P access score: 0 means - devices are not connected
    // If values is not 0 - than two devies can access either together

    out << "Group mask mask: " << group_device_masks << std::endl;
    return build_p2p_capability_matrix(out,
                                       native::ccl_device_driver::get_device_indices(group_device_masks),
                                       ping);
}

bool device_group_ring_topology::build(std::ostream& out,
                                        const ccl::context_comm_addr& comm_addr,
                                        const ccl::device_indices_t &group_device_indices,
                                        const details::adjacency_matrix& matrix)
{
    out << "\n/*************\"" << device_group_ring_topology::name()
        << "\"*************/\n" << std::endl;

    out << "Build device graph: " << std::endl;
    details::plain_graph_list id_rings = graph_list_resolver(matrix, group_device_indices);

    size_t size = id_rings.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size)
    {
        out << "Cannot build any ring" << std::endl;
        return false;
    }
    else if (id_rings.size() == 1) // whole ring
    {
        return build_specific(out, comm_addr, group_device_indices, *id_rings.begin());
    }

    //torn-apart ring
    return build_specific(out, comm_addr, group_device_indices, id_rings);
}

bool device_group_ring_topology::build(std::ostream& out, const ccl::context_comm_addr& comm_addr,
                                       const ccl::device_mask_t &group_device_masks,
                                       const details::adjacency_matrix& matrix)
{
    return build(out, comm_addr, native::ccl_device_driver::get_device_indices(group_device_masks), matrix);
}

bool device_group_ring_topology::build_specific(std::ostream& out,
                                                const ccl::context_comm_addr& comm_addr,
                                                const ccl::device_indices_t& group_device_indices,
                                                const details::plain_graph& graph)
{
    constexpr ccl::device_topology_type topology_type = ccl::device_topology_type::device_group_ring;

    out << "Start building topology: " << ::to_string(topology_type) << ", for graph:\n";
    for (const auto& id : graph)
    {
        out << id << ", ";
    }

    size_t thread_id = comm_addr.thread_idx;
    auto ring_device_topology = std::make_shared<device_community<topology_type>>(comm_addr);

    out << "\nStart indexer for thread: " << thread_id << std::endl;
    details::id_thread_table assigned_ids;
    std::vector<details::marked_idx> marked_id_ring = details::create_marked(graph);
    auto rank_builder =
                    create_device_functor<details::graph_ring_indexer<topology_type>>(marked_id_ring,
                                                                                      assigned_ids,
                                                                                      thread_id,
                                                                                      ring_device_topology->get_device_storage_ptr());
    std::shared_ptr<specific_plain_device_storage> group_gpu_comms =
                            devices_factory.create_devices_by_indices(thread_id, group_device_indices);

    ccl_tuple_for_each(*group_gpu_comms, rank_builder);

    details::printer<topology_type> p;
    ccl_tuple_for_each(*group_gpu_comms, p);
    out << "Indexer result: \n" << p.to_string();

    out << "\nFinished building topology: " << ::to_string(topology_type) << std::endl;

    //remember
    std::get<topology_type>(context.device_topology) = ring_device_topology;
    return true;
}

bool device_group_ring_topology::build_specific(std::ostream& out,
                                                const ccl::context_comm_addr& comm_addr,
                                                const ccl::device_indices_t& group_device_indices,
                                                const details::plain_graph_list& graph_list)
{
    constexpr ccl::device_topology_type topology_type = ccl::device_topology_type::device_group_torn_apart_ring;

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

    size_t thread_id = comm_addr.thread_idx;
    auto ring_device_topology = std::make_shared<device_community<topology_type>>(comm_addr);

    size_t graph_num = 0;
    size_t index_offset = 0;
    for (const auto& graph : graph_list)
    {
        out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id << std::endl;
        details::id_thread_table assigned_ids;
        std::vector<details::marked_idx> marked_id_ring = details::create_marked(graph);
        auto rank_builder =
                        create_device_functor<details::graph_ring_indexer_unique_index<topology_type>>(marked_id_ring,
                                                                                          assigned_ids,
                                                                                          thread_id,
                                                                                          ring_device_topology->get_device_storage_ptr(),
                                                                                          index_offset,
                                                                                          0,
                                                                                          0);
        // create plainn devices list for single graph indices
        ccl::device_indices_t graph_device_indices(graph.begin(), graph.end());
        std::shared_ptr<specific_plain_device_storage> group_gpu_comms =
                                devices_factory.create_devices_by_indices(thread_id, graph_device_indices);

        // create device comm wrappers and upgrade last devices in list up to scale_up_proxy type
        const ccl::device_index_type& last_in_graph_index = *graph.rbegin();
        auto scale_virt =
                details::add_scaleup_device<ccl_virtual_gpu_comm, topology_type>(
                                                                        *group_gpu_comms,
                                                                        last_in_graph_index,
                                                                        context,
                                                                        devices_factory);
        if (scale_virt)
        {
            out << "added scaleup virtual device: " << scale_virt->to_string()
                << ", by idx: " << last_in_graph_index << std::endl;
        }
        else
        {
            auto scale_real = details::add_scaleup_device<ccl_gpu_comm, topology_type>(
                                                                        *group_gpu_comms,
                                                                        last_in_graph_index,
                                                                        context,
                                                                        devices_factory);
            if (scale_real)
            {
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

        ccl_tuple_for_each(*group_gpu_comms, rank_builder);

        details::printer<topology_type> p;
        ccl_tuple_for_each(*group_gpu_comms, p);
        out << "\nIndexer for graph num: " << graph_num++ << ", result: \n" << p.to_string();

        index_offset += graph.size();
    }

    out << "\nFinished building topology: " << ::to_string(topology_type) << std::endl;

    //remember
    std::get<topology_type>(context.device_topology) = ring_device_topology;
    return true;
}
}
