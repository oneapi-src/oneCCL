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

namespace native
{

thread_group_ring_topology::thread_group_ring_topology(thread_group_context &ctx,
                                                       device_storage& devs) :
    context(ctx),
    devices_factory(devs)
{
}

size_t
    thread_group_ring_topology::default_property_p2p_rating_calculator(const ccl_device &lhs,
                                                                       const ccl_device &rhs)
{
    return details::property_p2p_rating_calculator(lhs, rhs, THREAD_GROUP_WEIGHT);
}

details::adjacency_matrix
        thread_group_ring_topology::build_p2p_capability_matrix(std::ostream& out,
                                                                const ccl::process_aggregated_device_mask_t &per_thread_device_masks,
                                                                details::p2p_rating_function ping)
{
    ccl::process_device_indices_t per_thread_device_indices;
    for(const auto& mask : per_thread_device_masks)
    {
        per_thread_device_indices.insert({mask.first, ccl_device_driver::get_device_indices(mask.second)});
    }

    return build_p2p_capability_matrix(out, per_thread_device_indices,
                                       ping);
}

details::adjacency_matrix
    thread_group_ring_topology::build_p2p_capability_matrix(std::ostream& out,
                                                            const ccl::process_device_indices_t &per_thread_device_indices,
                                                            details::p2p_rating_function ping)
{
    // Build adjacency matrix with P2P capability:
    // Rows & columnn is a device IDs ( froms 0 to CCL_GPU_DEVICES_AFFINITY_MASK_SIZE)
    // element values - is a weight of P2P activity: 0 means - devices are not connected
    // If values is not 0 - than two devies can be combined together

    details::adjacency_matrix ring_p2p_matrix;
    if (per_thread_device_indices.empty())
    {
        out << "No indices - nothing to build" << std::endl;
        return ring_p2p_matrix;
    }

    out << "Build adjacency matrix by: " << thread_group_ring_topology::name()
        << " - threads count: " << per_thread_device_indices.size() << std::endl;

    ccl::device_indices_t aggregated_thread_indices  =
            std::accumulate(per_thread_device_indices.begin(), per_thread_device_indices.end(), ccl::device_indices_t(),
                    [](ccl::device_indices_t &partial_mask, const std::pair<size_t, ccl::device_indices_t>& thread_mask)
                    {
                        partial_mask.insert(thread_mask.second.begin(), thread_mask.second.end());
                        return partial_mask;
                    });
    out << "Create devices for aggregared thread indices count: "
        << aggregated_thread_indices.size() << std::endl;
    for (const auto& ind : aggregated_thread_indices)
    {
        out << ind << ", ";
    }
    out << std::endl;

    //Request for alldevices in all allied processes on the node
    size_t driver_idx = 0;  //TODO


    std::shared_ptr<ccl_device_platform> platform_ptr = ccl_device_platform::create(aggregated_thread_indices);
    ccl_device_platform::driver_ptr driver = platform_ptr->get_driver(driver_idx);// TODO MUST EXIST
    for(auto device_group_mask_it = per_thread_device_indices.begin();
        device_group_mask_it != per_thread_device_indices.end();
        ++device_group_mask_it)
    {
        //1)
        //build device group at first
        /*
        *         t0_g0 | t1_g1 | t2_g2 |
        *        ------------------------
        * t0_g0 |   x   |       |       |
        * t1_g1 |       |  (x)  |       |
        * t2_g2 |       |       |  (x)  |
        *        ------------------------
        *
        */
        const auto& device_group_ring =
                    device_group_ring_topology::build_p2p_capability_matrix(out,
                                                                device_group_mask_it->second,
                                                                ping);
        for(const auto& lhs_pair : device_group_ring)
        {
            const auto& lhs_index = lhs_pair.first;
            const auto &affinities_with_devices = lhs_pair.second;
            for(const auto& rhs_pair : affinities_with_devices)
            {
                const auto& rhs_index = rhs_pair.first;
                details::cross_device_rating rating = rhs_pair.second;
                ring_p2p_matrix[lhs_index][rhs_index] = rating;
                ring_p2p_matrix[rhs_index][lhs_index] = rating;
            }
        }
        out << "After thread: " << device_group_mask_it->first
            << ", group processing matrix:\n" << ring_p2p_matrix << "\n";

        //2)
        //build compare with device group
        /*
        *         t0_g0 | t1_g1 | t2_g2 |
        *        ------------------------
        * t0_g0 |   x   |   y   |  (y)  |
        * t1_g1 |   y   |       |       |
        * t2_g2 |  (y)  |       |       |
        *        ------------------------
        *
        */
        const auto& prev_group_idx = device_group_mask_it->second;
        for(auto next_group_it = std::next(device_group_mask_it);
            next_group_it != per_thread_device_indices.end();
            ++next_group_it)
        {
            out << "Compare with next thread idx: " << next_group_it->first << std::endl;
            const auto& next_group_idx_container = next_group_it->second;
            for(const ccl::device_index_type& lhs_index : prev_group_idx)
            {
                const auto& lhs_device = driver->get_device(lhs_index); //TODO

                details::fill_adjacency_matrix_for_single_device_in_devices_by_cond(*lhs_device, lhs_index,
                                                                                   driver->get_devices(),
                                                                                   ring_p2p_matrix, ping,
                                                                                   [&next_group_idx_container](const ccl::device_index_type& rhs)
                                                                                   {
                                                                                       return next_group_idx_container.find(rhs) != next_group_idx_container.end();
                                                                                   });
            }
        }

        out << "After thread: " << device_group_mask_it->first
            << ", processing matrix:\n" << ring_p2p_matrix << "\n";
    }

    return ring_p2p_matrix;
}

bool thread_group_ring_topology::build(std::ostream& out,
                                       const ccl::process_device_indices_t& per_thread_device_indices,
                                       const details::adjacency_matrix& matrix,
                                       details::p2p_rating_function ping)
{
    out << "\n/*************\"" << thread_group_ring_topology::name()
        << "\" for threads: " << context.thread_device_topology.size()
        << "*************/\n" << std::endl;

    out << "Build device graph: " << std::endl;
    details::plain_graph_list id_rings = graph_list_resolver(matrix, per_thread_device_indices,
                                                             ping);

    size_t size = id_rings.size();
    out << "Resolved graphs count: " << size << "\n";
    if (!size)
    {
        out << "Cannot build any ring" << std::endl;
        return false;
    }
    else if (id_rings.size() == 1) // whole ring
    {
        return build_specific(out, per_thread_device_indices, *id_rings.begin());
    }

    //torn-apart ring
    return build_specific(out, per_thread_device_indices, id_rings);
}

bool thread_group_ring_topology::build(std::ostream& out,
                                       const ccl::process_aggregated_device_mask_t& per_thread_device_masks,
                                       const details::adjacency_matrix& matrix,
                                       details::p2p_rating_function ping)
{
    ccl::process_device_indices_t per_thread_device_indices;
    for(const auto& mask : per_thread_device_masks)
    {
        per_thread_device_indices.insert({mask.first, ccl_device_driver::get_device_indices(mask.second)});
    }

    return build(out, per_thread_device_indices, matrix);
}

bool thread_group_ring_topology::build_specific(std::ostream& out,
                                                const ccl::process_device_indices_t& per_thread_device_indices,
                                                const details::plain_graph& id_ring)
{
    constexpr ccl::device_topology_type topology_type = ccl::device_topology_type::thread_group_ring;

    out << "Start building topology: " << ::to_string(topology_type) << ", for graph:\n";
    for (const auto& id : id_ring)
    {
        out << id << ", ";
    }

    // id_ring - inter-thread ring
    out << "\nStart indexer:" << std::endl;
    details::id_thread_table assigned_ids;  //device_id -> thread_id
    auto& ctx_per_thread_data = context.thread_device_topology;
    std::vector<details::marked_idx> marked_id_ring = details::create_marked(id_ring);  // marked graph
    for (auto per_thread_it = ctx_per_thread_data.begin(); per_thread_it != ctx_per_thread_data.end();
         ++per_thread_it)
    {
        size_t thread_id = per_thread_it->first;        // first
        auto& out_indexed_devices =
                context.get_thread_topology<topology_type>(thread_id)->get_device_storage_ptr(); // just second

        std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices_factory.thread_gpu_comms.find(thread_id)->second;

        // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
        auto rank_builder =
                    create_device_functor<details::graph_ring_indexer<topology_type>>(marked_id_ring,
                                                                                      assigned_ids,
                                                                                      thread_id,
                                                                                      out_indexed_devices);
        ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

        details::printer<topology_type> p;
        ccl_tuple_for_each(*out_indexed_devices, p);
        out << "Indexer result for devices in thread idx ("
            << thread_id << "/" << ctx_per_thread_data.size() << "):\n"
            << p.to_string() << std::endl;
    }

    out << "\nStart ring builder" << std::endl;
    for(size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size(); current_thread_idx++)
    {
        // find max rank in current thread device list
        auto& indexed_devices_for_current_thread = context.get_thread_topology<topology_type>(current_thread_idx)->get_device_storage_ptr();
        const auto& curr_real = details::get_device_with_min_rank<ccl_gpu_comm, topology_type>(*indexed_devices_for_current_thread, id_ring);
        const auto& curr_virt = details::get_device_with_min_rank<ccl_virtual_gpu_comm, topology_type>(*indexed_devices_for_current_thread, id_ring);

        size_t tg_max_rank = std::max({std::get<0>(curr_real), std::get<0>(curr_virt)});

        // find thread, which will connect to current thread max rank with next_rank
        size_t next_rank = (tg_max_rank + 1 ) % id_ring.size();

        out << "Current thread: " << current_thread_idx << ", max rank candidates: "
            << std::get<0>(curr_real) << ", " << std::get<0>(curr_virt)
            << ", selected max rank: " << tg_max_rank
            << ", expected next_rank: " << next_rank << std::endl;

        for(size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size(); next_thread_id++)
        {
            if( next_thread_id == current_thread_idx)
            {
                // wrong thread, get next
                continue;
            }

            // search next_rank in that thread
            auto& next_thread_ring_topology = context.get_thread_topology<topology_type>(next_thread_id)->get_device_storage_ptr();
            const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(*next_thread_ring_topology, id_ring);
            const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(*next_thread_ring_topology, id_ring);

            if (next_rank != std::min({std::get<0>(real), std::get<0>(virt)}))
            {
                // wrong thread, get next
                continue;
            }

            out << "next thread: " << next_thread_id << ", min rank candidates: "
                << std::get<0>(real) << ", " << std::get<0>(virt) << std::endl;

            out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id << ")" << std::endl;
            if (next_rank == std::get<0>(real))
            {
                auto locker =
                    details::add_concurrent_locker_device<ccl_gpu_comm, topology_type>(next_rank,
                                                                                       0,
                                                                                       real,
                                                                                       devices_factory,
                                                                                       *indexed_devices_for_current_thread);
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
                                                                                               devices_factory,
                                                                                               *indexed_devices_for_current_thread);
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
    }
    return true;
}

bool thread_group_ring_topology::build_specific(std::ostream& out,
                                                const ccl::process_device_indices_t& per_thread_device_indicess,
                                                const details::plain_graph_list& graph_list)
{
    constexpr ccl::device_topology_type topology_type = ccl::device_topology_type::thread_group_torn_apart_ring;

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

    auto& ctx_per_thread_data = context.thread_device_topology;
    (void)ctx_per_thread_data;

    out << "\nStart gpu comm transformation for graph list count: " << graph_list.size() << std::endl;

    std::set<ccl::device_index_type> created_scaleup_indices;

    // let's start scale-up devices search & creation
    for (const auto& id_ring : graph_list)
    {
        for(const auto& per_thread : per_thread_device_indicess)
        {
            size_t thread_id = per_thread.first;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                devices_factory.thread_gpu_comms.find(thread_id)->second;
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

                auto scale_virt = details::add_scaleup_device<ccl_virtual_gpu_comm, topology_type>(
                                                                        *non_indexed_plain_devices,
                                                                        last_in_graph_index,
                                                                        context,
                                                                        devices_factory);
                if (scale_virt)
                {
                    created_scaleup_indices.insert(last_in_graph_index);
                    out << "added scaleup virtual device: " << scale_virt->to_string()
                        << ", by idx: " << last_in_graph_index << std::endl;
                }
                else
                {
                    auto scale_real = details::add_scaleup_device<ccl_gpu_comm, topology_type>(
                                                                        *non_indexed_plain_devices,
                                                                        last_in_graph_index,
                                                                        context,
                                                                        devices_factory);
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

    out << "\nStart indexer for graph list count: " << graph_list.size() << std::endl;
    size_t accumulated_index_offset_for_graph = 0;
    size_t graph_num = 0;
    std::map<size_t/*graph_num*/, size_t /*offset*/> index_offset_for_graphs;
    for (const auto& id_ring : graph_list)
    {
        details::id_thread_table assigned_ids;  //device_id -> thread_id
        auto& ctx_per_thread_data = context.thread_device_topology;
        std::vector<details::marked_idx> marked_id_ring = details::create_marked(id_ring);  // marked graph
        size_t index_offset = accumulated_index_offset_for_graph;
        for (auto per_thread_it = ctx_per_thread_data.begin(); per_thread_it != ctx_per_thread_data.end();
            ++per_thread_it)
        {
            size_t thread_id = per_thread_it->first;        //first
            auto& out_indexed_devices =
                    context.get_thread_topology<topology_type>(thread_id)->get_device_storage_ptr(); //just second

            out << "\nStart indexer for graph num: " << graph_num << ", thread: " << thread_id << std::endl;
            std::shared_ptr<specific_plain_device_storage> non_indexed_plain_devices =
                                                    devices_factory.thread_gpu_comms.find(thread_id)->second;

            // use graph ids to enumerate thread plain list `thread_gpu_comms` into `out_indexed_devices`
            auto rank_builder =
                    create_device_functor<details::graph_ring_indexer_unique_index<topology_type>>(marked_id_ring,
                                                                                      assigned_ids,
                                                                                      thread_id,
                                                                                      out_indexed_devices,
                                                                                      index_offset,
                                                                                      0,
                                                                                      0);

            ccl_tuple_for_each(*non_indexed_plain_devices, rank_builder);

            details::printer<topology_type> p;
            ccl_tuple_for_each(*out_indexed_devices, p);
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

    out << "\nStart ring builder for graphs count: " << graph_list.size() << std::endl;
    graph_num = 0;
    for (const auto& id_ring : graph_list)
    {
        out << "\nStart ring builder for graph num: " << graph_num << std::endl;
        for(size_t current_thread_idx = 0; current_thread_idx < ctx_per_thread_data.size(); current_thread_idx++)
        {
            auto& indexed_devices_for_current_thread = context.get_thread_topology<topology_type>(current_thread_idx)->get_device_storage_ptr();

            //find max device ramk in current thread devices
            const auto& curr_real = details::get_device_with_min_rank<ccl_gpu_comm, topology_type>(*indexed_devices_for_current_thread, id_ring);
            const auto& curr_virt = details::get_device_with_min_rank<ccl_virtual_gpu_comm, topology_type>(*indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_real = details::get_device_with_min_rank<ccl_gpu_scaleup_proxy<ccl_gpu_comm>, topology_type>(*indexed_devices_for_current_thread, id_ring);
            const auto& curr_scale_virt = details::get_device_with_min_rank<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>, topology_type>(*indexed_devices_for_current_thread, id_ring);

            size_t tg_max_rank = std::max({std::get<0>(curr_real), std::get<0>(curr_virt),
                                           std::get<0>(curr_scale_real), std::get<0>(curr_scale_virt)});
            // find thread, which will connect to current thread max rank with next_rank
            size_t next_rank = (tg_max_rank + 1 ) % id_ring.size();

            out << "Current thread: " << current_thread_idx << ", max rank candidates: "
                << std::get<0>(curr_real) << ", " << std::get<0>(curr_virt) << ", "
                << std::get<0>(curr_scale_real) << ", " << std::get<0>(curr_scale_virt)
                << ", selected max rank: " << tg_max_rank
                << ", expected next_rank: " << next_rank << std::endl;

            for(size_t next_thread_id = 0; next_thread_id < ctx_per_thread_data.size(); next_thread_id++)
            {
                if( next_thread_id == current_thread_idx)
                {
                    // wrong thread, get next
                    continue;
                }

                // search next_rank in that thread
                auto& next_thread_ring_topology = context.get_thread_topology<topology_type>(next_thread_id)->get_device_storage_ptr();
                const auto& real = details::get_device_with_max_rank<ccl_gpu_comm, topology_type>(*next_thread_ring_topology, id_ring);
                const auto& virt = details::get_device_with_max_rank<ccl_virtual_gpu_comm, topology_type>(*next_thread_ring_topology, id_ring);
                const auto& scale_real = details::get_device_with_max_rank<ccl_gpu_scaleup_proxy<ccl_gpu_comm>, topology_type>(*next_thread_ring_topology, id_ring);
                const auto& scale_virt = details::get_device_with_max_rank<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>, topology_type>(*next_thread_ring_topology, id_ring);
                if (next_rank != std::min({std::get<0>(real), std::get<0>(virt),
                                           std::get<0>(scale_real), std::get<0>(scale_virt)}))
                {
                    // wrong thread, get next
                    continue;
                }

                out << "next thread: " << next_thread_id << ", min rank candidates: "
                    << std::get<0>(real) << ", " << std::get<0>(virt) << ", "
                    << std::get<0>(scale_real) << ", " << std::get<0>(scale_virt) << std::endl;

                out << "Lock ring for threads (" << current_thread_idx << " <-> " << next_thread_id << ")" << std::endl;
                if (next_rank == std::get<0>(real))
                {
                    auto locker =
                            details::add_concurrent_locker_device<ccl_gpu_comm,
                                                                  topology_type>(next_rank,
                                                                                 index_offset_for_graphs[graph_num],
                                                                                 real,
                                                                                 devices_factory,
                                                                                 *indexed_devices_for_current_thread);
                    out << "Added real locker by index: " << index_offset_for_graphs[graph_num]  + next_rank
                        << ", for thread idx: " << current_thread_idx  <<":\n"
                        << locker->to_string() << std::endl;
                }
                else if (next_rank == std::get<0>(virt))
                {
                    auto locker =
                            details::add_concurrent_locker_device<ccl_virtual_gpu_comm,
                                                                  topology_type>(next_rank,
                                                                                 index_offset_for_graphs[graph_num],
                                                                                 virt,
                                                                                 devices_factory,
                                                                                 *indexed_devices_for_current_thread);
                    out << "Added virtual locker by index: " << index_offset_for_graphs[graph_num] + next_rank
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
            }
        }
        graph_num++;
    }

    out << "\nFinished building topology: " << ::to_string(topology_type) << std::endl;
    return true;
}
}
