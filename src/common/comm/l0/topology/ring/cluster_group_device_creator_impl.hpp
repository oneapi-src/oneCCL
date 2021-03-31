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

#include "cluster_group_device_creator.hpp"
#include "common/comm/l0/topology/ring/ring_construction_utils.hpp"
#include "common/comm/l0/topology/ring/device_group_ring_creator.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

#include "common/comm/l0/topology/cluster_device_utils.hpp"

namespace native {

inline cluster_group_device_creator::cluster_group_device_creator(size_t process_idx,
                                                                  size_t process_nums,
                                                                  process_group_context& ctx,
                                                                  device_storage& devs)
        : process_index(process_idx),
          process_size(process_nums),
          context(ctx),
          devices_factory(devs) {}

inline size_t cluster_group_device_creator::default_property_p2p_rating_calculator(
    const ccl_device& lhs,
    const ccl_device& rhs) {
    return detail::property_p2p_rating_calculator(lhs, rhs, PROCESS_GROUP_WEIGHT);
}

inline detail::adjacency_matrix cluster_group_device_creator::build_p2p_capability_matrix(
    std::ostream& out,
    const ccl::process_device_indices_type& single_node_device_indices,
    detail::p2p_rating_function ping) {
    // Build adjacency matrix with P2P capability:
    // Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // element values - is a weight of P2P activity: 0 means - devices are not connected
    // If values is not 0 - than two devies can be combined together

    detail::adjacency_matrix ring_p2p_matrix;
    if (single_node_device_indices.empty()) {
        out << "No indices nothing to build" << std::endl;
        return ring_p2p_matrix;
    }

    out << "Build adjacency matrix by: " << cluster_group_device_creator::name() << std::endl;
    out << "Processes count: " << single_node_device_indices.size() << "\t";
    out << "Delegate to thread group ring, consider 'process' as 'thread'" << std::endl;
    return thread_group_ring_topology::build_p2p_capability_matrix(
        out, single_node_device_indices, ping);
}

inline bool cluster_group_device_creator::build_all(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::process_device_indices_type& cur_process_per_thread_device_indices,
    const detail::adjacency_matrix& single_node_matrix,
    detail::p2p_rating_function ping) {
    out << "\n/************* \"" << cluster_group_device_creator::name()
        << "\" for threads: " << context.process_device_topology.size() << "*************/\n"
        << std::endl;

    detail::plain_graph_list my_device_graphs = detail::graph_list_resolver(
        single_node_matrix, cur_process_per_thread_device_indices, ping);

    size_t size = my_device_graphs.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size) {
        out << "Cannot build any ring" << std::endl;
        return false;
    }

    out << "Transform graph to colored with process color: " << process_index << "\n";
    detail::colored_plain_graph_list my_colored_graphs =
        detail::create_colored(my_device_graphs, process_index);

    out << "Process graphs:\n" << detail::to_string(my_colored_graphs) << std::endl;

    detail::global_sorted_colored_plain_graphs global_graphs;
    context.collect_cluster_colored_plain_graphs(my_colored_graphs, global_graphs);

    //calculate my devicses offset (rank) from cluster devices
    std::map<size_t, size_t> process_device_rank_offset;
    size_t accumulated_offset = 0;
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

    //check cluster topology on symmetric nodes configurations
    bool symm_test = true;

    // TODO consider situation, when we have symmetric process configuration
    // But unsymmetric IPC devices count

    std::vector<size_t> ipc_devices_on_node;
    std::vector<size_t> processes_on_node;
    ipc_devices_on_node.reserve(context.cluster_gpu_indices.size());
    processes_on_node.reserve(context.cluster_gpu_indices.size());

    ccl::device_indices_type shared_ipc_devices_candidates;
    ccl::device_indices_type ipc_p2p_devices_candidates;
    for (const auto& node_conf : context.cluster_gpu_indices) {
        const ccl::host_id& hostname = node_conf.first;
        const ccl::process_device_indices_type& processes = node_conf.second;

        ccl::device_indices_type node_device_intersection; //shared devics

        // each node should have the same processes count
        if (!processes_on_node.empty()) {
            symm_test &= (*processes_on_node.rbegin() == processes.size());

            node_device_intersection = shared_ipc_devices_candidates;
        }
        else {
            node_device_intersection = processes.begin()->second;
        }
        processes_on_node.push_back(processes.size());

        //find shared devices for processes on node.
        for (auto it = processes.begin(); it != processes.end() && symm_test; ++it) {
            ccl::device_indices_type result_intersection;
            std::set_intersection(it->second.begin(),
                                  it->second.end(),
                                  node_device_intersection.begin(),
                                  node_device_intersection.end(),
                                  std::inserter(result_intersection, result_intersection.end()));

            symm_test &= result_intersection.size();

            node_device_intersection.swap(result_intersection);
        }

        if (hostname == context.get_host_id() && symm_test) {
            // remember ipc device candidates for my node
            shared_ipc_devices_candidates = node_device_intersection;
        }
        //TODO - make smart logic: access each device to each for processes
        // because not necesary to have shared device id for both processes

        //common devices for processes on node
        ipc_devices_on_node.push_back(node_device_intersection.size());
    }

    out << "Cluster Symmetric Capability:\n";
    out << "\nNodes in cluster:\t" << context.cluster_gpu_indices.size();
    out << "\nProcs on nodes:\t";
    std::copy(processes_on_node.begin(),
              processes_on_node.end(),
              std::ostream_iterator<size_t>(out, ","));
    out << "\nIPCs devices on nodes:\t";
    std::copy(ipc_devices_on_node.begin(),
              ipc_devices_on_node.end(),
              std::ostream_iterator<size_t>(out, ","));
    out << std::endl;

    // additional device types to inject in a final topology
    using thread_idx_t = size_t;
    using colored_device_per_thread = detail::colored_indexed_data<thread_idx_t>;

    std::vector<colored_device_per_thread> shared_ipc_devices;
    size_t shared_ipc_links_per_proc = 0;
    std::vector<colored_device_per_thread> scale_up_devices;
    size_t scale_up_links_per_proc = 0;
    std::vector<colored_device_per_thread> scale_out_devices;

    //TODO only single thread supported - thread 0
    thread_idx_t thread_index = 0;

    // calculate scale-out links: by default each process use scale-out
    size_t scale_out_links_per_proc = process_size;

    // choose last device in my graph for scale-out ( why not?)
    scale_out_devices.emplace_back(0 /*use default color as host communicator ( all-to-all)*/,
                                   (my_colored_graphs.begin()->begin()->index),
                                   thread_index);

    // check topology optimization
    if (symm_test && not shared_ipc_devices_candidates.empty()) {
        out << "Symmetric Configuration Detected: ICP for scale-up" << std::endl;

        //TODO schoose the first one
        shared_ipc_devices.emplace_back(shared_ipc_devices_candidates.size() /*color*/,
                                        *shared_ipc_devices_candidates.begin() /*device*/,
                                        thread_index /*thread to insertion*/);

        shared_ipc_links_per_proc = *processes_on_node.begin();
        scale_out_links_per_proc = process_size / shared_ipc_links_per_proc;

        if (scale_out_links_per_proc == 0 or scale_out_links_per_proc == 1) {
            scale_out_links_per_proc = 0;
            scale_out_devices.clear(); //no links, no devices
        }
        else {
            // scale-out links exist, then recalcuate comm color
            size_t scale_out_color = process_size % shared_ipc_links_per_proc;
            std::for_each(scale_out_devices.begin(),
                          scale_out_devices.end(),
                          [scale_out_color](colored_device_per_thread& idx) {
                              idx.color = scale_out_color;
                          });
        }
    }
    else {
        out << "Unsymmetric IPC Configuration Detected" << std::endl;
        size_t procs_on_node = *processes_on_node.begin();
        bool process_symmetric_test =
            (procs_on_node != 1); //nothing to scale-up for 1 process on node
        process_symmetric_test &= std::all_of(
            processes_on_node.begin(), processes_on_node.end(), [procs_on_node](size_t val) {
                return procs_on_node == val;
            });
        if (process_symmetric_test) {
            out << "Symmetric scale-up Configuration Detected. Build scale-up devices" << std::endl;

            //TODO  assign first device in my graph for scale-up
            size_t scale_up_color = std::hash<std::string>{}(context.get_host_id());
            scale_up_devices.emplace_back(
                scale_up_color, my_colored_graphs.begin()->begin()->index, thread_index);

            scale_up_links_per_proc = procs_on_node;
            scale_out_links_per_proc = process_size / scale_up_links_per_proc;

            if (scale_out_links_per_proc == 0 or scale_out_links_per_proc == 1) {
                scale_out_links_per_proc = 0;
                scale_out_devices.clear(); //no links, no devices
            }
            else {
                // change scale-out color to separate it from scale-up processes to use different communicator
                size_t scale_out_color = process_size % scale_up_links_per_proc;
                std::for_each(scale_out_devices.begin(),
                              scale_out_devices.end(),
                              [scale_out_color](colored_device_per_thread& idx) {
                                  idx.color = scale_out_color;
                              });

                if (scale_out_color == scale_up_color) {
                    //TODO
                    out << "UNHANDLED CASE: scale-up & scale-out comm color is the same!\n"
                        << "Reassign one of them: plus '1', re hash and broad-cast it!"
                        << std::endl;
                    abort();
                }
            }
        }
        else {
            out << "Each nodes contains different processes. "
                << "No optimization, use scale-out for all" << std::endl;

            scale_out_links_per_proc = process_size;
        }
    }

    out << "Final configuration info:\n";
    out << "SHARED IPC: ";
    for (const auto& idx : shared_ipc_devices) {
        out << idx << ", ";
    }
    out << "\nshared ipc comm count: " << shared_ipc_links_per_proc;

    out << "\nScaleUp: ";
    for (const auto& idx : scale_up_devices) {
        out << idx << ", ";
    }
    out << "\nscale-up comm count: " << scale_up_links_per_proc;

    out << "\nScaleOut: ";
    for (const auto& idx : scale_out_devices) {
        out << idx << ", ";
    }
    out << "\nscale-out comm count: " << scale_out_links_per_proc << std::endl;

    // enumerate as thread_group_devices, with syntetic device types injection
    return build_impl<ccl::device_topology_type::ring>(
        out,
        comm_addr,
        cur_process_per_thread_device_indices,
        single_node_matrix,
        { shared_ipc_devices, scale_up_devices, scale_out_devices },
        my_colored_graphs,
        process_device_rank_offset,
        accumulated_offset,
        ping);
}

template <ccl::device_topology_type class_id>
inline bool cluster_group_device_creator::build_impl(
    std::ostream& out,
    const ccl::context_comm_addr& comm_addr,
    const ccl::process_device_indices_type& cur_process_per_thread_device_indices,
    const detail::adjacency_matrix& single_node_matrix,
    const std::vector<std::vector<detail::colored_indexed_data<size_t>>>& syntetic_devices,
    detail::colored_plain_graph_list& graph_list,
    std::map<size_t, size_t> process_device_rank_offset,
    size_t cluster_device_total_size,
    detail::p2p_rating_function ping /* = default_property_p2p_rating_calculator*/) {
    size_t ring_index = 0;
    out << "Start building topology: " << ::to_string(class_id)
        << ", for graphs: " << graph_list.size() << "\n"
        << "ring index: " << ring_index << std::endl;
    out << detail::to_string(graph_list);

    auto& ctx_per_thread_data = context.process_device_topology;
    (void)ctx_per_thread_data;

    out << "\nStart indexer:" << std::endl;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t /*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    auto offset_it = process_device_rank_offset.find(process_index);
    if (offset_it == process_device_rank_offset.end()) {
        assert(false && "");
    }

    accumulated_index_offset_for_graph = offset_it->second;
    out << "My global rank offset: " << accumulated_index_offset_for_graph << std::endl;

    std::set<ccl::device_index_type> created_cpu_context_indices;

    // let's start numa-connector devices search & creation/
    for (const auto& id_ring : graph_list) {
        // todo
        if (graph_list.size() == 1) {
            // no NUMA in this case
            break;
        }
        for (const auto& per_thread : cur_process_per_thread_device_indices) {
            size_t thread_id = per_thread.first;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices_factory.thread_gpu_comms.find(thread_id)->second;
            // create device comm wrappers and upgrade last devices in list up to numa type
            detail::color_t process;
            (void)process;
            ccl::device_index_type last_in_graph_index;
            const auto& tmp = *id_ring.rbegin();
            process = tmp.color;
            last_in_graph_index = tmp.index;
            if (per_thread.second.find(last_in_graph_index) != per_thread.second.end()) {
                CCL_ASSERT(process == process_index);
                out << "thread: " << thread_id
                    << " wants to create numa-proxy device by idx: " << last_in_graph_index
                    << std::endl;
                if (created_cpu_context_indices.find(last_in_graph_index) !=
                    created_cpu_context_indices.end()) {
                    out << "skip existing numa-proxy device candidate by: " << last_in_graph_index
                        << std::endl;
                    continue;
                }

                size_t inserted_device_type_index = detail::role_mod::inject_numa_device<
                    group_id(),
                    class_id,
                    process_group_context,
                    ccl_virtual_gpu_comm, /* `virtual` is better candiate*/
                    ccl_gpu_comm>(
                    *non_indexed_plain_devices, last_in_graph_index, context, devices_factory);
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
                created_cpu_context_indices.insert(last_in_graph_index);
            }
        }
    }

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    auto topology_comm_addr = comm_addr;
    topology_comm_addr.comm_size = cluster_device_total_size;
    for (auto& id_ring : graph_list) {
        size_t index_offset = accumulated_index_offset_for_graph;

        for (auto per_thread_it = ctx_per_thread_data.begin();
             per_thread_it != ctx_per_thread_data.end();
             ++per_thread_it) {
            size_t thread_id = per_thread_it->first; //first

            // prepare ropology
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
                << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                devices_factory.thread_gpu_comms.find(thread_id)->second;

            // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
            auto rank_builder =
                create_device_functor<detail::colored_graph_ring_indexer<group_id(), class_id>>(
                    id_ring,
                    thread_id,
                    process_index,
                    out_indexed_devices->get_device_storage(),
                    0,
                    0,
                    index_offset);

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            // print partial topology enumeration for 'graph' from 'graph_list'
            detail::printer<group_id(), class_id> p;
            ccl_tuple_for_each(out_indexed_devices->get_device_storage(), p);
            out << "Indexer result for devices in thread idx (" << thread_id << "/"
                << ctx_per_thread_data.size() << "):\n"
                << p.to_string() << std::endl;

            // remember enumerated (marked) devices fro current thread & current graph
            // to continue right enumeration order for other graphs & threas
            accumulated_index_offset_for_graph +=
                rank_builder.get_functor().get_marked_indices_count();
            out << "\nIndexer for graph num: " << graph_num << ", finished. marked_indices: "
                << rank_builder.get_functor().get_marked_indices_count()
                << ", next rank index: " << accumulated_index_offset_for_graph << "\n";
        }
        index_offset_for_graphs[graph_num] = index_offset;
        graph_num++;
    }

    out << "\nStart devices builder for graphs count: " << graph_list.size() << std::endl;
    graph_num = 0;
    for (const auto& id_ring : graph_list) {
        out << "\nStart ring builder for graph num: " << graph_num << std::endl;
        for (size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size();
             current_thread_idx++) {
            std::shared_ptr<device_community<class_id>> community;
            if (graph_list.size() == 1) {
                community =
                    context.get_process_topology<class_id>(process_index, current_thread_idx)
                        .get_topology(ring_index);
            }
            else {
                community =
                    context.get_process_topology<class_id>(process_index, current_thread_idx)
                        .get_additiona_topology(ring_index);
            }

            auto& indexed_devices_for_current_thread = community->get_device_storage();

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
                std::shared_ptr<device_community<class_id>> community;
                if (graph_list.size() == 1) {
                    community =
                        context.get_process_topology<class_id>(process_index, next_thread_id)
                            .get_topology(ring_index);
                }
                else {
                    community =
                        context.get_process_topology<class_id>(process_index, next_thread_id)
                            .get_additiona_topology(ring_index);
                }
                auto& next_thread_ring_topology = community->get_device_storage();

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
                        << " for scaleup  real proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
                else if (next_rank == std::get<0>(scale_virt)) {
                    out << "No need to add concurrent proxy for next thread: " << next_thread_id
                        << " for scaleup virtual proxy in current thread: " << current_thread_idx
                        << std::endl;
                }
            }
        }
        graph_num++;
    }

    out << "\nStart gpu types injection for graph list count: " << graph_list.size() << std::endl;
    size_t syntetic_device_type_index = 0;
    for (auto& colored_devices : syntetic_devices) {
        switch (syntetic_device_type_index) {
            case 0: //IPC device
            {
                for (const auto& idx : colored_devices) {
                    size_t thread_id = idx.get_payload();

                    std::shared_ptr<device_community<class_id>> community;
                    if (graph_list.size() == 1) {
                        community = context.get_process_topology<class_id>(process_index, thread_id)
                                        .get_topology(ring_index);
                    }
                    else {
                        community = context.get_process_topology<class_id>(process_index, thread_id)
                                        .get_additiona_topology(ring_index);
                    }

                    auto& out_indexed_devices = community->get_device_storage();

                    size_t inserted_device_type_index =
                        detail::role_mod::inject_ipc_src_device<group_id(),
                                                       class_id,
                                                       process_group_context,
                                                       ccl_gpu_comm,
                                                       ccl_virtual_gpu_comm
                                                       /*,
                                                        Too complex to support such topology without generic topology builder
                                                       ccl_numa_proxy<ccl_gpu_comm>,
                                                       ccl_numa_proxy<ccl_virtual_gpu_comm>
                                                       */>(
                            out_indexed_devices, idx.index, context, devices_factory);
                    if (inserted_device_type_index != std::numeric_limits<size_t>::max()) {
                        out << "Inject IPC_src device by order: " << inserted_device_type_index
                            << "\nby idx: " << idx.to_string() << std::endl;
                    }
                    else {
                        abort();
                        assert(false && "Unsupported device type in topology creation");
                        std::ostringstream ss;
                        ss << out.rdbuf();
                        throw std::runtime_error(
                            std::string("Unsupported device type in topology creation. Log:\n") +
                            ss.str());
                    }
                }
                syntetic_device_type_index++;
                break;
            }
            case 1: //scale-up device
            {
                for (const auto& idx : colored_devices) {
                    size_t thread_id = idx.get_payload();

                    std::shared_ptr<device_community<class_id>> community;
                    if (graph_list.size() == 1) {
                        community = context.get_process_topology<class_id>(process_index, thread_id)
                                        .get_topology(ring_index);
                    }
                    else {
                        community = context.get_process_topology<class_id>(process_index, thread_id)
                                        .get_additiona_topology(ring_index);
                    }

                    auto& out_indexed_devices = community->get_device_storage();

                    size_t inserted_device_type_index = detail::role_mod::inject_scaleup_device<
                        group_id(),
                        class_id,
                        process_group_context,
                        ccl_gpu_comm,
                        ccl_virtual_gpu_comm,
                        ccl_numa_proxy<ccl_gpu_comm>,
                        ccl_numa_proxy<ccl_virtual_gpu_comm>>(
                        out_indexed_devices, idx.index, context, devices_factory);
                    if (inserted_device_type_index != std::numeric_limits<size_t>::max()) {
                        out << "Inject scaleUp device by order: " << inserted_device_type_index
                            << "\nby idx: " << idx.to_string() << std::endl;
                    }
                    else {
                        abort();
                        assert(false && "Unsupported device type in topology creation");
                        std::ostringstream ss;
                        ss << out.rdbuf();
                        throw std::runtime_error(
                            std::string("Unsupported device type in topology creation. Log:\n") +
                            ss.str());
                    }
                }
                syntetic_device_type_index++;
                break;
            }
            case 2: //scale-out device
            {
                for (const auto& idx : colored_devices) {
                    size_t thread_id = idx.get_payload();
                    std::shared_ptr<device_community<class_id>> community;
                    if (graph_list.size() == 1) {
                        community = context.get_process_topology<class_id>(process_index, thread_id)
                                        .get_topology(ring_index);
                    }
                    else {
                        community = context.get_process_topology<class_id>(process_index, thread_id)
                                        .get_additiona_topology(ring_index);
                    }

                    auto& out_indexed_devices = community->get_device_storage();

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
                        ccl_gpu_comm,
                        ccl_virtual_gpu_comm>(
                        out_indexed_devices, idx.index, context, devices_factory);
                    if (inserted_device_type_index != std::numeric_limits<size_t>::max()) {
                        out << "Inject scaleout device by order: " << inserted_device_type_index
                            << "\nby idx: " << idx.to_string() << std::endl;
                    }
                    else {
                        abort();
                        assert(false && "Unsupported device type in topology creation");
                        std::ostringstream ss;
                        ss << out.rdbuf();
                        throw std::runtime_error(
                            std::string("Unsupported device type in topology creation. Log:\n") +
                            ss.str());
                    }
                }
                syntetic_device_type_index++;
                break;
            }
            default:
                throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                         "Unexpected injected device index: " +
                                         std::to_string(syntetic_device_type_index));
        }
    }

    out << "\nFinished building topology: " << ::to_string(class_id) << std::endl;
    for (auto per_thread_it = ctx_per_thread_data.begin();
         per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it) {
        size_t thread_id = per_thread_it->first;

        detail::printer<group_id(), class_id> p;

        std::shared_ptr<device_community<class_id>> community;
        if (graph_list.size() == 1) {
            community = context.get_process_topology<class_id>(process_index, thread_id)
                            .get_topology(ring_index);
        }
        else {
            community = context.get_process_topology<class_id>(process_index, thread_id)
                            .get_additiona_topology(ring_index);
        }

        auto& out_indexed_devices = community->get_device_storage();

        ccl_tuple_for_each(out_indexed_devices, p);
        out << "\nFinal topology thread: " << thread_id << "\n" << p.to_string();
    }

    return true;
}

} // namespace native
