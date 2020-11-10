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
#include "common/comm/l0/topology/ring/thread_group_ring_creator.hpp"

namespace native {

thread_group_ring_topology::thread_group_ring_topology(thread_group_context& ctx,
                                                       device_storage& devs)
        : context(ctx),
          devices_factory(devs) {}

size_t thread_group_ring_topology::default_property_p2p_rating_calculator(const ccl_device& lhs,
                                                                          const ccl_device& rhs) {
    return detail::property_p2p_rating_calculator(lhs, rhs, THREAD_GROUP_WEIGHT);
}

detail::adjacency_matrix thread_group_ring_topology::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::process_aggregated_device_mask_t& per_thread_device_masks,
    detail::p2p_rating_function ping) {
    ccl::process_device_indices_type per_thread_device_indices;
    for (const auto& mask : per_thread_device_masks) {
        per_thread_device_indices.insert(
            { mask.first, ccl_device_driver::get_device_indices(mask.second) });
    }

    return build_p2p_capability_matrix(out, per_thread_device_indices, ping);
}

detail::adjacency_matrix thread_group_ring_topology::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::process_device_indices_type& per_thread_device_indices,
    detail::p2p_rating_function ping) {
    // Build adjacency matrix with P2P capability:
    // Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // element values - is a weight of P2P activity: 0 means - devices are not connected
    // If values is not 0 - than two devies can be combined together

    detail::adjacency_matrix ring_p2p_matrix;
    if (per_thread_device_indices.empty()) {
        out << "No indices - nothing to build" << std::endl;
        return ring_p2p_matrix;
    }

    out << "Build adjacency matrix by: " << thread_group_ring_topology::name()
        << " - threads count: " << per_thread_device_indices.size() << std::endl;

    ccl::device_indices_type aggregated_thread_indices = std::accumulate(
        per_thread_device_indices.begin(),
        per_thread_device_indices.end(),
        ccl::device_indices_type(),
        [](ccl::device_indices_type& partial_mask,
           const std::pair<size_t, ccl::device_indices_type>& thread_mask) {
            partial_mask.insert(thread_mask.second.begin(), thread_mask.second.end());
            return partial_mask;
        });
    out << "Create devices for aggregared thread indices count: "
        << aggregated_thread_indices.size() << std::endl;
    for (const auto& ind : aggregated_thread_indices) {
        out << ind << ", ";
    }
    out << std::endl;

    return get_platform().calculate_device_access_metric(aggregated_thread_indices, ping);
}

bool thread_group_ring_topology::build(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::process_device_indices_type& per_thread_device_indices,
    const detail::adjacency_matrix& matrix,
    detail::p2p_rating_function ping) {
    out << "\n/*************\"" << thread_group_ring_topology::name()
        << "\" for threads: " << context.thread_device_topology.size() << "*************/\n"
        << std::endl;

    out << "Resolve device graph: " << std::endl;
    detail::plain_graph_list id_rings =
        graph_list_resolver(matrix, per_thread_device_indices, ping);

    size_t size = id_rings.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size) {
        out << "Cannot build any ring" << std::endl;
        return false;
    }
    else if (id_rings.size() == 1) // whole ring
    {
        return build_specific(out, comm_addr, per_thread_device_indices, *id_rings.begin());
    }

    //torn-apart ring
    return build_scale_up_specific(out, comm_addr, per_thread_device_indices, id_rings);
}

bool thread_group_ring_topology::build(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::process_aggregated_device_mask_t& per_thread_device_masks,
    const detail::adjacency_matrix& matrix,
    detail::p2p_rating_function ping) {
    ccl::process_device_indices_type per_thread_device_indices;
    for (const auto& mask : per_thread_device_masks) {
        per_thread_device_indices.insert(
            { mask.first, ccl_device_driver::get_device_indices(mask.second) });
    }

    return build(out, comm_addr, per_thread_device_indices, matrix);
}

bool thread_group_ring_topology::build_specific(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::process_device_indices_type& per_thread_device_indices,
    const detail::plain_graph& id_ring) {
    size_t ring_index = 0;
    constexpr ccl::device_topology_type class_id = ccl::device_topology_type::ring;

    out << "Start building topology: " << ::to_string(class_id) << ", for graph:\n";
    out << detail::to_string(id_ring);

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    detail::id_thread_table assigned_ids; //device_id -> thread_id
    auto& ctx_per_thread_data = context.thread_device_topology;
    std::vector<detail::marked_idx> marked_id_ring = detail::create_marked(id_ring); // marked graph

    auto topology_comm_addr = comm_addr;
    topology_comm_addr.comm_size = marked_id_ring.size();

    for (auto per_thread_it = ctx_per_thread_data.begin();
         per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it) {
        size_t thread_id = per_thread_it->first; // first

        //prepared empty topology
        auto out_indexed_devices = std::make_shared<device_community<class_id>>(topology_comm_addr);
        std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
            devices_factory.thread_gpu_comms.find(thread_id)->second;

        // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
        auto rank_builder = create_device_functor<detail::graph_ring_indexer<group_id(), class_id>>(
            marked_id_ring, assigned_ids, thread_id, out_indexed_devices->get_device_storage());
        ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

        detail::printer<group_id(), class_id> p;
        ccl_tuple_for_each(out_indexed_devices->get_device_storage(), p);
        out << "Indexer result for devices in thread idx (" << thread_id << "/"
            << ctx_per_thread_data.size() << "):\n"
            << p.to_string() << std::endl;

        //remember topology
        context.get_thread_topology<class_id>(thread_id).set_topology(out_indexed_devices);
    }

    out << "\nStart ring builder" << std::endl;
    for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
         current_thread_idx++) {
        // find max rank in current thread device list
        auto& indexed_devices_for_current_thread =
            context.get_thread_topology<class_id>(current_thread_idx)
                .get_topology(ring_index)
                ->get_device_storage();
        const auto& curr_real =
            detail::get_device_with_min_rank<ccl_gpu_comm, group_id(), class_id>(
                indexed_devices_for_current_thread, id_ring);
        const auto& curr_virt =
            detail::get_device_with_min_rank<ccl_virtual_gpu_comm, group_id(), class_id>(
                indexed_devices_for_current_thread, id_ring);

        size_t tg_max_rank = std::max({ std::get<0>(curr_real), std::get<0>(curr_virt) });

        // find thread, which will connect to current thread max rank with next_rank
        size_t next_rank = (tg_max_rank + 1) % id_ring.size();

        out << "Current thread: " << current_thread_idx
            << ", max rank candidates: " << std::get<0>(curr_real) << ", " << std::get<0>(curr_virt)
            << ", selected max rank: " << tg_max_rank << ", expected next_rank: " << next_rank
            << std::endl;

        for (size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size();
             next_thread_id++) {
            if (next_thread_id == current_thread_idx) {
                // wrong thread, get next
                continue;
            }

            // search next_rank in that thread
            auto& next_thread_ring_topology = context.get_thread_topology<class_id>(next_thread_id)
                                                  .get_topology(ring_index)
                                                  ->get_device_storage();
            const auto& real = detail::get_device_with_max_rank<ccl_gpu_comm, group_id(), class_id>(
                next_thread_ring_topology, id_ring);
            const auto& virt =
                detail::get_device_with_max_rank<ccl_virtual_gpu_comm, group_id(), class_id>(
                    next_thread_ring_topology, id_ring);

            if (next_rank != std::min({ std::get<0>(real), std::get<0>(virt) })) {
                // wrong thread, get next
                continue;
            }

            out << "next thread: " << next_thread_id
                << ", min rank candidates: " << std::get<0>(real) << ", " << std::get<0>(virt)
                << std::endl;

            out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id
                << ")" << std::endl;
            if (next_rank == std::get<0>(real)) {
                auto locker =
                    detail::add_concurrent_locker_device<ccl_gpu_comm, group_id(), class_id>(
                        next_rank, 0, real, devices_factory, indexed_devices_for_current_thread);
                out << "Added real locker by index: " << next_rank
                    << ", for thread idx: " << current_thread_idx << ":\n"
                    << locker->to_string() << std::endl;
            }
            else if (next_rank == std::get<0>(virt)) {
                auto locker = detail::
                    add_concurrent_locker_device<ccl_virtual_gpu_comm, group_id(), class_id>(
                        next_rank, 0, virt, devices_factory, indexed_devices_for_current_thread);
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
    }
    return true;
}

bool thread_group_ring_topology::build_scale_up_specific(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::process_device_indices_type& per_thread_device_indicess,
    const detail::plain_graph_list& graph_list) {
    size_t ring_index = 0;
    constexpr ccl::device_topology_type class_id = ccl::device_topology_type::ring;

    out << "Start building topology: " << ::to_string(class_id)
        << ", for graphs: " << graph_list.size() << "\n";
    out << detail::to_string(graph_list);

    auto& ctx_per_thread_data = context.thread_device_topology;
    (void)ctx_per_thread_data;

    out << "\nStart gpu comm transformation for graph list count: " << graph_list.size()
        << std::endl;

    std::set<ccl::device_index_type> created_scaleup_indices;

    // let's start scale-up devices search & creation
    for (const auto& id_ring : graph_list) {
        for (const auto& per_thread : per_thread_device_indicess) {
            size_t thread_id = per_thread.first;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices_factory.thread_gpu_comms.find(thread_id)->second;

            // promote real-virtual device (right corner devices) in graphs up to scale_up_proxy type
            // all loca group devices in different graph would be linked by scale_up_proxy
            // each local group ( in graph) must have at least one scale_up_proxy device
            const ccl::device_index_type& last_in_graph_index = *id_ring.rbegin();
            if (per_thread.second.find(last_in_graph_index) != per_thread.second.end()) {
                out << "thread: " << thread_id
                    << " wants to create scale_up device by idx: " << last_in_graph_index
                    << std::endl;
                if (created_scaleup_indices.find(last_in_graph_index) !=
                    created_scaleup_indices.end()) {
                    out << "skip existing scale_up device candidate by: " << last_in_graph_index
                        << std::endl;
                    continue;
                }

                auto scale_virt =
                    detail::add_numa_proxy_device<ccl_virtual_gpu_comm, group_id(), class_id>(
                        *non_indexed_plain_devices, last_in_graph_index, context, devices_factory);
                if (scale_virt) {
                    created_scaleup_indices.insert(last_in_graph_index);
                    out << "added scaleup virtual device: " << scale_virt->to_string()
                        << ", by idx: " << last_in_graph_index << std::endl;
                }
                else {
                    auto scale_real =
                        detail::add_numa_proxy_device<ccl_gpu_comm, group_id(), class_id>(
                            *non_indexed_plain_devices,
                            last_in_graph_index,
                            context,
                            devices_factory);
                    if (scale_real) {
                        created_scaleup_indices.insert(last_in_graph_index);
                        out << "added scaleup real device: " << scale_real->to_string()
                            << ", by idx: " << last_in_graph_index << std::endl;
                    }
                    //                    else
                    //                    {
                    //                        assert(false && "Unsupported device type in torn-apart ring creation");
                    //                        std::ostringstream ss;
                    //                        ss << out.rdbuf();
                    //                        throw std::runtime_error(std::string("Unsupported device type in torn-apart ring creation. Log:\n") +
                    //                                                 ss.str());
                    //                    }
                }
            }
        }
    }

    out << "\nStart indexer for graph list count: " << graph_list.size() << std::endl;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;

    std::map<size_t /*graph_num*/, size_t /*offset*/>
        index_offset_for_graphs; // calculate indexed devices count in each graph

    ccl::device_indices_type total_device_indices;
    for (const auto& graph : graph_list) {
        total_device_indices.insert(graph.begin(), graph.end());
    }

    auto topology_comm_addr = comm_addr;
    topology_comm_addr.comm_size = total_device_indices.size();

    for (const auto& id_ring : graph_list) {
        detail::id_thread_table assigned_ids; //device_id -> thread_id
        auto& ctx_per_thread_data = context.thread_device_topology;
        std::vector<detail::marked_idx> marked_id_ring =
            detail::create_marked(id_ring); // marked graph
        size_t index_offset = accumulated_index_offset_for_graph;

        for (auto per_thread_it = ctx_per_thread_data.begin();
             per_thread_it != ctx_per_thread_data.end();
             ++per_thread_it) {
            size_t thread_id = per_thread_it->first; //first

            // prepare ropology
            if (context.get_thread_topology<class_id>(thread_id).torn_apart_rings.empty()) {
                context.get_thread_topology<class_id>(thread_id).set_additiona_topology(
                    std::make_shared<device_community<class_id>>(topology_comm_addr));
            }
            auto& out_indexed_devices = context.get_thread_topology<class_id>(thread_id)
                                            .get_additiona_topology(ring_index)
                                            ->get_device_storage();

            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id
                << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices_factory.thread_gpu_comms.find(thread_id)->second;

            // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
            auto rank_builder = create_device_functor<
                detail::graph_ring_indexer_unique_index<group_id(), class_id>>(
                marked_id_ring, assigned_ids, thread_id, out_indexed_devices, index_offset, 0, 0);

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            // print partial topology enumeration for 'graph' from 'graph_list'
            detail::printer<group_id(), class_id> p;
            ccl_tuple_for_each(out_indexed_devices, p);
            out << "Indexer result for devices in thread idx (" << thread_id << "/"
                << ctx_per_thread_data.size() << "):\n"
                << p.to_string() << std::endl;

            // remember enumerated (marked) devices fro current thread & current graph
            // to continue right enumeration order for other graphs & threas
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
            auto& indexed_devices_for_current_thread =
                context.get_thread_topology<class_id>(current_thread_idx)
                    .get_additiona_topology(ring_index)
                    ->get_device_storage();

            //find max device rank in current thread devices
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

            for (size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size();
                 next_thread_id++) {
                if (next_thread_id == current_thread_idx) {
                    // wrong thread, get next
                    continue;
                }

                // search next_rank in that thread
                auto& next_thread_ring_topology =
                    context.get_thread_topology<class_id>(next_thread_id)
                        .get_additiona_topology(ring_index)
                        ->get_device_storage();
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
                            next_rank,
                            index_offset_for_graphs[graph_num],
                            real,
                            devices_factory,
                            indexed_devices_for_current_thread);
                    out << "Added real locker by index: "
                        << index_offset_for_graphs[graph_num] + next_rank
                        << ", for thread idx: " << current_thread_idx << ":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(virt)) {
                    auto locker = detail::
                        add_concurrent_locker_device<ccl_virtual_gpu_comm, group_id(), class_id>(
                            next_rank,
                            index_offset_for_graphs[graph_num],
                            virt,
                            devices_factory,
                            indexed_devices_for_current_thread);
                    out << "Added virtual locker by index: "
                        << index_offset_for_graphs[graph_num] + next_rank
                        << ", for thread idx: " << current_thread_idx << ":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(scale_real)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for numa real proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
                else if (next_rank == std::get<0>(scale_virt)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for numa virtual proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
            }
        }
        graph_num++;
    }

    out << "\nFinished building topology: " << ::to_string(class_id) << std::endl;
    for (auto per_thread_it = ctx_per_thread_data.begin();
         per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it) {
        size_t thread_id = per_thread_it->first;

        detail::printer<group_id(), class_id> p;
        ccl_tuple_for_each(context.get_thread_topology<class_id>(thread_id)
                               .get_additiona_topology(ring_index)
                               ->get_device_storage(),
                           p);
        out << "\nFinal topology thread: " << thread_id << "\n" << p.to_string();
    }
    return true;
}
} // namespace native
