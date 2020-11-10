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

namespace native {

device_group_ring_topology::device_group_ring_topology(device_group_context& comm,
                                                       device_storage& devs)
        : context(comm),
          devices_factory(devs) {}

size_t device_group_ring_topology::default_property_p2p_rating_calculator(const ccl_device& lhs,
                                                                          const ccl_device& rhs) {
    return detail::property_p2p_rating_calculator(lhs, rhs, DEVICE_GROUP_WEIGHT);
}

detail::adjacency_matrix device_group_ring_topology::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::device_indices_type& group_device_indices,
    detail::p2p_rating_function ping) {
    // Build adjacency matrix between devices using `ping` function:
    // Default ping function is checking P2P access capabilities in a way:
    // 1) Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // 2) Matrix element values - is P2P access score: 0 means - devices are not connected
    // If values is not 0 - than two devies can access either together

    out << "Build adjacency matrix by: " << device_group_ring_topology::name()
        << " - group indices count: " << group_device_indices.size() << std::endl;

    //Request for alldevices in all allied processes on the node
    return get_platform().calculate_device_access_metric(group_device_indices, ping);
}

detail::adjacency_matrix device_group_ring_topology::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::device_mask_t& group_device_masks,
    detail::p2p_rating_function ping) {
    // Build adjacency matrix between devices using `ping` function:
    // Default ping function is checking P2P access capabilities in a way:
    // 1) Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // 2) Matrix element values - is P2P access score: 0 means - devices are not connected
    // If values is not 0 - than two devies can access either together

    out << "Group mask mask: " << group_device_masks << std::endl;
    return build_p2p_capability_matrix(
        out, native::ccl_device_driver::get_device_indices(group_device_masks), ping);
}

bool device_group_ring_topology::build(std::ostream& out,
                                       const ccl::context_comm_addr& comm_addr,
                                       const ccl::device_indices_type& group_device_indices,
                                       const detail::adjacency_matrix& matrix) {
    out << "\n/*************\"" << device_group_ring_topology::name() << "\"*************/\n"
        << std::endl;

    out << "Resolve device graph" << std::endl;
    detail::plain_graph_list id_rings = graph_list_resolver(matrix, group_device_indices);

    size_t size = id_rings.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size) {
        out << "Cannot build any ring" << std::endl;
        return false;
    }
    else if (id_rings.size() == 1) // whole ring, each device is accessible, no CPU copy here
    {
        return build_specific(out, comm_addr, group_device_indices, *id_rings.begin(), matrix);
    }

    /* torn-apart ring:
     * there are inaccessible devices in group - need to insert broadcast devices wrappers for
     * CPU RAM copying
     */
    return build_scale_up_specific(out, comm_addr, group_device_indices, id_rings, matrix);
}

bool device_group_ring_topology::build(std::ostream& out,
                                       const ccl::context_comm_addr& comm_addr,
                                       const ccl::device_mask_t& group_device_masks,
                                       const detail::adjacency_matrix& matrix) {
    return build(
        out, comm_addr, native::ccl_device_driver::get_device_indices(group_device_masks), matrix);
}

template <ccl::device_topology_type class_id>
bool device_group_ring_topology::build_specific_topology(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::device_indices_type& group_device_indices,
    const detail::plain_graph& graph) {
    out << "Start building topology: " << ::to_string(class_id) << ", for graph:\n";
    out << detail::to_string(graph);

    size_t thread_id = comm_addr.thread_idx;
    auto topology_comm_addr = comm_addr;
    topology_comm_addr.comm_size = graph.size();
    auto device_topology = std::make_shared<device_community<class_id>>(topology_comm_addr);

    out << "\nStart indexer for thread: " << thread_id << std::endl;
    detail::id_thread_table assigned_ids;
    std::vector<detail::marked_idx> marked_id_ring = detail::create_marked(graph);
    auto rank_builder = create_device_functor<detail::graph_ring_indexer<group_id(), class_id>>(
        marked_id_ring, assigned_ids, thread_id, device_topology->get_device_storage());
    std::shared_ptr<specific_plain_device_storage> group_gpu_comms =
        devices_factory.create_devices_by_indices(thread_id, group_device_indices);

    ccl_tuple_for_each(*group_gpu_comms, rank_builder);

    detail::printer<group_id(), class_id> p;
    ccl_tuple_for_each(*group_gpu_comms, p);
    out << "Indexer result: \n" << p.to_string();

    out << "\nFinished building topology: " << ::to_string(class_id) << std::endl;

    //remember
    context.device_topology.get_community<class_id>().set_topology(device_topology);
    return true;
}

bool device_group_ring_topology::build_specific(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::device_indices_type& group_device_indices,
    const detail::plain_graph& graph,
    const detail::adjacency_matrix& matrix) {
    bool result = build_specific_topology<ccl::device_topology_type::ring>(
        out, comm_addr, group_device_indices, graph);
    /*
    // check a2a possibility
    bool a2a_capable = detail::check_graph_a2a_capable(graph, matrix,out);
    if (a2a_capable)
    {
        // a2a should starts from real device
        // if do not reset, than it continue creation from existing ring devices
        devices_factory.reset(comm_addr.thread_idx);      <--- AVOID because affects thread_group

        a2a_capable =
            build_specific_topology<ccl::group_split_type::a2a_device_group>(out,
                                                                                 comm_addr,
                                                                                 group_device_indices,
                                                                                 graph);
    }

    return result || a2a_capable;*/
    return result;
}

template <ccl::device_topology_type class_id>
bool device_group_ring_topology::build_scale_up_specific_topology(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::device_indices_type& group_device_indices,
    const detail::plain_graph_list& graph_list) {
    out << "Start building topology: " << ::to_string(class_id)
        << ", for graphs: " << graph_list.size() << "\n";
    out << detail::to_string(graph_list);

    size_t thread_id = comm_addr.thread_idx;
    size_t graph_num = 0;
    size_t index_offset = 0;

    // create all required device wrappers
    // these wrappers would be used for ALL context at the next iteration
    ccl::device_indices_type total_device_indices;
    for (const auto& graph : graph_list) {
        total_device_indices.insert(graph.begin(), graph.end());
    }
    std::shared_ptr<specific_plain_device_storage> initial_group_gpu_comms =
        devices_factory.create_devices_by_indices(thread_id, total_device_indices);
    //set lobal devices size to topology
    auto topology_comm_addr = comm_addr;
    topology_comm_addr.comm_size = total_device_indices.size();
    auto device_topology = std::make_shared<device_community<class_id>>(topology_comm_addr);

    // make copy for wrappers, because other context should work with original structure
    // but current context modified it (transform some wrappers into new scale_up_wrapper type)
    std::shared_ptr<specific_plain_device_storage> group_gpu_comms =
        std::make_shared<specific_plain_device_storage>(*initial_group_gpu_comms);
    for (const auto& graph : graph_list) {
        out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id
            << std::endl;

        detail::id_thread_table assigned_ids;
        std::vector<detail::marked_idx> marked_id_ring = detail::create_marked(graph);
        auto rank_builder =
            create_device_functor<detail::graph_ring_indexer_unique_index<group_id(), class_id>>(
                marked_id_ring,
                assigned_ids,
                thread_id,
                device_topology->get_device_storage(),
                index_offset,
                0,
                0);
        // promote real-virtual device (right corner devices) in graphs up to scale_up_proxy type
        // all loca group devices in different graph would be linked by scale_up_proxy
        // each local group ( in graph) must have at least one scale_up_proxy device
        const ccl::device_index_type& last_in_graph_index = *graph.rbegin();
        auto scale_virt = detail::add_numa_proxy_device<ccl_virtual_gpu_comm, group_id(), class_id>(
            *group_gpu_comms, last_in_graph_index, context, devices_factory);
        if (scale_virt) {
            out << "Added scaleup virtual device:\n"
                << scale_virt->to_string() << "\nby idx: " << last_in_graph_index << std::endl;
        }
        else {
            auto scale_real = detail::add_numa_proxy_device<ccl_gpu_comm, group_id(), class_id>(
                *group_gpu_comms, last_in_graph_index, context, devices_factory);
            if (scale_real) {
                out << "Added scaleup real device:\n"
                    << scale_real->to_string() << "\nby idx: " << last_in_graph_index << std::endl;
            }
            else {
                assert(false && "Unsupported device type in topology creation");
                std::ostringstream ss;
                ss << out.rdbuf();
                throw std::runtime_error(
                    std::string("Unsupported device type in topology creation. Log:\n") + ss.str());
            }
        }

        /* use plain (non-indexed) device wrapper list, which is allocated from device_storage
         * in the following way:
         *
         * Try to iterate over all wrappers in that list for devices allocated for that device_group.
         * Find wrapper by device_id in each 'graph' list.
         * Offset for founded device from graph beginning give us 'rank' for founded device.
         * By 'rank' - means logical rank in local device group ( local for ring kernel execution)
         * Need to remember about previous graphs sizes, when calculate offset... total offset for founded * device in total graphs is a 'user rank' for device in process/cluster
         */
        ccl_tuple_for_each(*group_gpu_comms, rank_builder);

        // just print partial topology progress for current 'graph'
        detail::printer<group_id(), class_id> p;
        ccl_tuple_for_each(device_topology->get_device_storage(), p);
        out << "\nIndexer for graph num: " << graph_num++ << ", result: \n" << p.to_string();

        index_offset += graph.size();
    }

    out << "\nFinished building topology: " << ::to_string(class_id) << std::endl;

    // remember constructed topology
    context.device_topology.get_community<class_id>().set_additiona_topology(device_topology);

    detail::printer<group_id(), class_id> p;
    ccl_tuple_for_each(device_topology->get_device_storage(), p);
    out << "\nFinal topology: \n" << p.to_string();
    return true;
}

bool device_group_ring_topology::build_scale_up_specific(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::device_indices_type& group_device_indices,
    const detail::plain_graph_list& graph_list,
    const detail::adjacency_matrix& matrix) {
    bool result = build_scale_up_specific_topology<ccl::device_topology_type::ring>(
        out, comm_addr, group_device_indices, graph_list);
    /*
    // check a2a possibility
    bool a2a_capable = true;
    for (const auto& graph : graph_list)
    {
        a2a_capable &= detail::check_graph_a2a_capable(graph, matrix, out);
    }

    if (a2a_capable)
    {
        // a2a should starts from real device
        // if do not reset, than it continue creation from existing ring devices
        devices_factory.reset(comm_addr.thread_idx);  <--- AVOID because affects thread_group

        a2a_capable =
            build_scale_up_specific_topology<ccl::group_split_type::a2a_device_group>(
                                                out,
                                                comm_addr,
                                                group_device_indices,
                                                graph_list);
    }

    return result || a2a_capable;
    */
    return result;
}
} // namespace native
