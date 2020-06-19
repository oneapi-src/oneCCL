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
#include <map>
#include <memory>

#include "common/comm/l0/topology/topology_construction_utils.hpp"
#include "ccl_config.h"

#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"
#include "common/comm/l0/topology/ring_topology.hpp"
#include "common/comm/l0/device_community.hpp"
#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"

#include "common/comm/l0/context/device_storage.hpp"


namespace native
{

namespace details
{

template<ccl::device_topology_type topology_type>
struct simple_ring_indexer
{
    simple_ring_indexer(size_t& cur_rank_offset,
                        size_t size,
                        std::unique_ptr<specific_indexed_device_storage>& device_topology):
        rank_offset(cur_rank_offset),
        total_ranks(size),
        topology(device_topology)
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& container)
    {
        if(!topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        // fill device group by rank sequencially
        indexed_device_container<device_t>& indexed_container = std::get<device_t::type_idx()>(*topology);
        for(auto& gpu_device : container)
        {
            gpu_device->template reset_rank<topology_type>(rank_offset, total_ranks);
            indexed_container.insert({rank_offset, gpu_device});

            rank_offset++;
        }
    }

private:
    size_t& rank_offset;
    size_t total_ranks;
    std::unique_ptr<specific_indexed_device_storage>& topology;
};

inline std::vector<marked_idx> create_marked(const plain_graph& id_vector)
{
    std::vector<marked_idx> ret;
    ret.reserve(id_vector.size());

    std::transform(id_vector.begin(), id_vector.end(),
                   std::back_inserter(ret),
                   [] (const ccl::device_index_type& idx)
                   {
                        return marked_idx(false, idx);
                   });
    return ret;
}

inline colored_plain_graph create_colored(const plain_graph& id_vector, color_t color)
{
    colored_plain_graph ret;
    ret.reserve(id_vector.size());

    std::transform(id_vector.begin(), id_vector.end(),
                   std::back_inserter(ret),
                   [color] (const ccl::device_index_type& idx)
                   {
                        return colored_idx(color, idx);
                   });
    return ret;
}

inline colored_plain_graph_list create_colored(const plain_graph_list& list, color_t color)
{
    colored_plain_graph_list ret;
    for (const plain_graph& graph : list)
    {
        ret.emplace_back(create_colored(graph, color));
    }
    return ret;
}

using id_thread_table = std::multimap<ccl::device_index_type, size_t /*thread id*/>;

//TODO use inheritance or policy for indexers!!!
template<ccl::device_topology_type topology_type>
struct graph_ring_indexer
{
    graph_ring_indexer(std::vector<marked_idx>& id_ring_vector,
                       id_thread_table &thread_id_storage,
                       size_t thread_id,
                       std::unique_ptr<specific_indexed_device_storage>& device_topology):
        id_array(id_ring_vector),
        topology(device_topology),
        assigned_ids(thread_id_storage),
        thread_idx(thread_id)
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& container)
    {
        if(!topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        // fill device group by rank sequencially
        indexed_device_container<device_t>& indexed_container = std::get<device_t::type_idx()>(*topology);
        for(auto& gpu_device : container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id](marked_idx& val)
                                   {
                                       if(!val.first) //non marked
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                throw std::logic_error(std::string("Unknown device in id ring vector: ") +
                                       ccl::to_string(id));
            }
            size_t rank = std::distance(id_array.begin(), it);

            size_t already_assigned_ids_count = assigned_ids.count(id);
            //rank += already_assigned_ids_count; TODO
            (void)already_assigned_ids_count;

            size_t size = id_array.size();
            gpu_device->template reset_rank<topology_type>(rank, size);
            indexed_container.insert({rank, gpu_device});

            assigned_ids.insert({id, thread_idx});

            it->first = true;//marked
        }
    }

protected:
    std::vector<marked_idx>& id_array;
    std::unique_ptr<specific_indexed_device_storage>& topology;

    id_thread_table& assigned_ids;
    size_t thread_idx;
    size_t ring_id_offset;
};

template<ccl::device_topology_type topology_type>
struct colored_graph_ring_indexer
{
    static constexpr color_t marked_color = std::numeric_limits<color_t>::max();

   colored_graph_ring_indexer(colored_plain_graph& id_ring_vector,
                              size_t thread_id,
                              color_t process_id,
                              std::unique_ptr<specific_indexed_device_storage>& device_topology,
                              size_t r_offset = 0,
                              size_t s_offset = 0,
                              size_t i_offset = 0):
        id_array(id_ring_vector),
        topology(device_topology),
        thread_idx(thread_id),
        process_idx(process_id),
        rank_offset(r_offset),
        size_offset(s_offset),
        index_offset(i_offset),
        marked_indices_count()
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& in_container)
    {
        if(!topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        indexed_device_container<device_t>& out_container = std::get<device_t::type_idx()>(*topology);

        // fill device group by rank sequencially
        for(auto& gpu_device : in_container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id, this](colored_idx& val)
                                   {
                                       if(val.first == process_idx) // find in my process
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                throw std::logic_error(std::string("Unknown device in id ring vector: ") +
                                       ccl::to_string(id) + ". ring vector:\n" +
                                       to_string(id_array));
            }

            //rank in local graph_ring
            size_t rank = std::distance(id_array.begin(), it);
            size_t size = id_array.size();

            //apply offsets
            gpu_device->template reset_rank<topology_type>(rank + rank_offset,
                                                           size + size_offset);
            out_container.insert({rank + index_offset, gpu_device});

            it->first = marked_color;//marked
            marked_indices_count++;
        }

    }

    size_t get_marked_indices_count() const
    {
        return marked_indices_count;;
    }

protected:
    colored_plain_graph& id_array;
    std::unique_ptr<specific_indexed_device_storage>& topology;
    size_t thread_idx;
    size_t ring_id_offset;
    color_t process_idx;
    size_t rank_offset;
    size_t size_offset;
    size_t index_offset;
    size_t marked_indices_count;
};



static constexpr color_t marked_color = std::numeric_limits<color_t>::max();
inline void separate_ipc_devices(const ccl::process_device_indices_t& ipc_indices,
                                 size_t process_idx,
                                 size_t process_num,
                                 const colored_plain_graph& id_array,
                                 ccl::process_device_indices_t& ipc_src_indices,
                                 ccl::process_device_indices_t& ipc_dst_indices,
                                 color_t exclude_color = marked_color)
{
    // find right ipcs
    do
    {
        auto graph_it = std::find_if(id_array.begin(), id_array.end(),
                                     [process_idx](const colored_idx& val)
                                     {
                                         return val.first == process_idx;
                                     });
        if (graph_it == id_array.end())
        {
            assert(false && "Invalide configuration: not my graph");
            throw std::runtime_error(std::string(__FUNCTION__) + " - unexpected graph for process: " +
                                     std::to_string(process_idx));
        }


        // calc IPC process Index
        size_t ipc_process_index_to_find = process_idx + 1;
        size_t actual_ipc_process_index = ipc_process_index_to_find;
        if (process_idx == process_num - 1)
        {
            //replace terminator as index for right
            actual_ipc_process_index = 0;
        }

        // find  first IPC device
        graph_it = std::find_if(graph_it, id_array.end(),
                                    [ipc_process_index_to_find](const colored_idx& val)
                                    {
                                        return val.first == ipc_process_index_to_find;
                                    });

        if (graph_it == id_array.end())
        {
            break;
        }

        //test on ipc filter
        auto candidate_it = ipc_indices.find(actual_ipc_process_index);
        if (candidate_it == ipc_indices.end()
            or (candidate_it->second.find(graph_it->second) == candidate_it->second.end()))
        {
            break;
        }

        //remember
        ipc_dst_indices.insert({graph_it->first, {graph_it->second}});
    } while (false);


    //find left ipc
     // find right ipcs
    do
    {

        auto graph_rit = std::find_if(id_array.rbegin(), id_array.rend(),
                                     [process_idx](const colored_idx& val)
                                     {
                                         return val.first == process_idx;
                                     });
        if (graph_rit == id_array.rend())
        {
            assert(false && "Invalide configuration: not my graph from left");
            throw std::runtime_error(std::string(__FUNCTION__) + " - unexpected graph (left) for process: " +
                                     std::to_string(process_idx));
        }


        // calc IPC process Index
        size_t ipc_process_index_to_find = process_idx - 1;
        size_t actual_ipc_process_index = ipc_process_index_to_find;
        if (process_idx == 0)
        {
            //replace terminator as index for left
            ipc_process_index_to_find = process_num;
            actual_ipc_process_index = process_num - 1;
        }

        // find  first IPC device
        graph_rit = std::find_if(graph_rit, id_array.rend(),
                                 [ipc_process_index_to_find](const colored_idx& val)
                                 {
                                      return val.first == ipc_process_index_to_find;
                                 });

        if (graph_rit == id_array.rend())
        {
            break;
        }

        //test on ipc filter
        auto candidate_it = ipc_indices.find(actual_ipc_process_index);
        if (candidate_it == ipc_indices.end()
            or (candidate_it->second.find(graph_rit->second) == candidate_it->second.end()))
        {
            break;
        }

        graph_rit = std::prev(graph_rit); // use my device to upgrade as IPC source

        //remember
        ipc_src_indices.insert({graph_rit->first, {graph_rit->second}});
    } while (false);
}

template<ccl::device_topology_type topology_type>
struct smart_ring_indexer
{
    static constexpr color_t marked_color = std::numeric_limits<color_t>::max();

    smart_ring_indexer(colored_plain_graph& id_ring_vector,
                       color_t process_id,
                       size_t process_count,
                       device_storage& device_factory,
                       std::unique_ptr<specific_indexed_device_storage>& device_topology,
                       const ccl::process_device_indices_t& ipc_device,
                       const ccl::process_device_indices_t& scaleout_device_indices):
        id_array(id_ring_vector),
        process_idx(process_id),
        process_num(process_count),
        factory(device_factory),
        topology(device_topology),
        ipc_src_indices(),
        ipc_dst_indices(),
        scaleout_indices(scaleout_device_indices),
        marked_indices_count()
    {

        separate_ipc_devices(ipc_device, process_idx, process_num, id_array,
                             ipc_src_indices, ipc_dst_indices);
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& in_container)
    {
        if(!topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        // fill device group by rank sequencially
        for(auto& gpu_device : in_container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id, this](colored_idx& val)
                                   {
                                       if(val.first == process_idx) // find in my process
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                throw std::logic_error(std::string("Unknown device in id ring vector: ") +
                                       ccl::to_string(id) + ". ring vector:\n" +
                                       to_string(id_array));
            }

            //rank in local graph_ring
            size_t rank = std::distance(id_array.begin(), it);
            size_t size = id_array.size();


            //Check on IPC source candidate at first
            /*
            auto process_set = ipc_src_indices.find(process_idx);
            if (process_set != ipc_src_indices.end()
                and
                process_set->second.find(it->second) != process_set->second.end())
            {
                // ipc device
                using ipc_device_t = ccl_ipc_source_gpu_comm<device_t>;
                device_t_ptr<ipc_device_t> new_ipc_source_comm =
                            factory.create_gpu_device<ipc_device_t>(gpu_device->get_device(),
                                                                    rank,
                                                                    *gpu_device,
                                                                    topology_type);
                new_ipc_source_comm->template reset_rank<topology_type>(rank, size);
                indexed_device_container<ipc_device_t>& out_ipc_container =
                        std::get<ipc_device_t::type_idx()>(*topology);
                out_ipc_container.insert({rank, new_ipc_source_comm});
                */
            if (!try_as_ipc_source(gpu_device, rank, size))
            {
                // regular device
                gpu_device->template reset_rank<topology_type>(rank, size);

                indexed_device_container<device_t>& out_container =
                        std::get<device_t::type_idx()>(*topology);
                out_container.insert({rank, gpu_device});

                it->first = marked_color;//marked
                marked_indices_count++;
            }
        }
    }

    void operator() (plain_device_container<ccl_ipc_gpu_comm>& in_container)
    {
        //Insert IPC destination device
        if(!topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        indexed_device_container<ccl_ipc_gpu_comm>& out_container = std::get<ccl_ipc_gpu_comm::type_idx()>(*topology);
        for(auto& gpu_device : in_container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            size_t foreign_process_idx = (process_idx + 1) % process_num;
            if (process_idx == process_num - 1)
            {
                foreign_process_idx = process_num;  //use terminator
            }
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id, foreign_process_idx](colored_idx& val)
                                   {
                                       if(val.first == foreign_process_idx) // find in my process
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                throw std::logic_error(std::string("Unknown device in id ring vector: ") +
                                       ccl::to_string(id) + ". ring vector:\n" +
                                       to_string(id_array));
            }

            //rank in local graph_ring
            size_t rank = std::distance(id_array.begin(), it);
            size_t size = id_array.size();

            //apply offsets
            gpu_device->template reset_rank<topology_type>(rank, size);
            out_container.insert({rank, gpu_device});
        }
    }

    size_t get_marked_indices_count() const
    {
        return marked_indices_count;
    }

protected:
    colored_plain_graph& id_array;
    color_t process_idx;
    size_t process_num;
    device_storage& factory;
    std::unique_ptr<specific_indexed_device_storage>& topology;
    ccl::process_device_indices_t ipc_src_indices;
    ccl::process_device_indices_t ipc_dst_indices;
    const ccl::process_device_indices_t& scaleout_indices;
    size_t marked_indices_count;
private:

    template<class device_t>
    bool try_as_ipc_source(std::shared_ptr<device_t> gpu_device,
                           size_t rank, size_t size)
    {
        return false;
    }

    bool try_as_ipc_source(std::shared_ptr<ccl_gpu_comm> gpu_device,
                           size_t rank, size_t size)
    {
        return try_as_ipc_source_impl(gpu_device, rank, size);
    }

    bool try_as_ipc_source(std::shared_ptr<ccl_virtual_gpu_comm> gpu_device,
                           size_t rank, size_t size)
    {
        return try_as_ipc_source_impl(gpu_device, rank, size);
    }

    template<class device_t>
    bool try_as_ipc_source_impl(std::shared_ptr<device_t> gpu_device,
                                size_t rank, size_t size)
    {
        //Check on IPC source candidate at first
        const ccl::device_index_type& id = gpu_device->get_device().get_device_path();
        auto process_set = ipc_src_indices.find(process_idx);
        if (process_set == ipc_src_indices.end()
              or
            process_set->second.find(id) == process_set->second.end())
        {
            return false;
        }

        // ipc device
        using ipc_device_t = ccl_ipc_source_gpu_comm<device_t>;
        device_t_ptr<ipc_device_t> new_ipc_source_comm =
                            factory.create_gpu_device<ipc_device_t>(gpu_device->get_device(),
                                                                    rank,
                                                                    *gpu_device,
                                                                    topology_type);

        new_ipc_source_comm->template reset_rank<topology_type>(rank, size);
        indexed_device_container<ipc_device_t>& out_ipc_container =
                        std::get<ipc_device_t::type_idx()>(*topology);
        out_ipc_container.insert({rank, new_ipc_source_comm});
        return true;
    }
};

template<ccl::device_topology_type topology_type>
struct graph_ring_indexer_ext : public graph_ring_indexer<topology_type>
{
    using base = graph_ring_indexer<topology_type>;
    using base::topology;
    using base::thread_idx;
    using base::id_array;
    using base::assigned_ids;
    graph_ring_indexer_ext(std::vector<marked_idx>& id_ring_vector,
                       id_thread_table &thread_id_storage,
                       size_t thread_id,
                       std::unique_ptr<specific_indexed_device_storage>& device_topology,
                       size_t index_offset_val,
                       size_t rank_offset_val,
                       size_t size_offset_val):
        graph_ring_indexer<topology_type>(id_ring_vector, thread_id_storage, thread_id, device_topology),
        index_offset(index_offset_val),
        rank_offset(rank_offset_val),
        size_offset(size_offset_val)
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& container)
    {
        if(!this->topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        // fill device group by rank sequencially
        indexed_device_container<device_t>& indexed_container = std::get<device_t::type_idx()>(*topology);
        for(auto& gpu_device : container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id](marked_idx& val)
                                   {
                                       if(!val.first) //non marked
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                throw std::logic_error(std::string("Unknown device in id ring vector: ") +
                                       ccl::to_string(id));
            }
            size_t rank = std::distance(id_array.begin(), it);
            rank += rank_offset;

            size_t already_assigned_ids_count = assigned_ids.count(id);
            //rank += already_assigned_ids_count; TODO
            (void)already_assigned_ids_count;

            size_t size = id_array.size();
            size = size_offset;

            gpu_device->template reset_rank<topology_type>(rank, size);
            indexed_container.insert({rank + index_offset, gpu_device});

            assigned_ids.insert({id, thread_idx});

            it->first = true;//marked
            marked_indices_count++;
        }
    }

    size_t get_marked_indices_count() const
    {
        return marked_indices_count;;
    }

private:
    size_t index_offset;
    size_t rank_offset;
    size_t size_offset;

    size_t marked_indices_count;
};

template<ccl::device_topology_type topology_type>
struct graph_ring_indexer_unique_index : public graph_ring_indexer<topology_type>
{
    using base = graph_ring_indexer<topology_type>;
    using base::topology;
    using base::thread_idx;
    using base::id_array;
    using base::assigned_ids;
    graph_ring_indexer_unique_index(std::vector<marked_idx>& id_ring_vector,
                       id_thread_table &thread_id_storage,
                       size_t thread_id,
                       std::unique_ptr<specific_indexed_device_storage>& device_topology,
                       size_t index_offset_val,
                       size_t rank_offset_val,
                       size_t size_offset_val):
        graph_ring_indexer<topology_type>(id_ring_vector, thread_id_storage, thread_id, device_topology),
        index_offset(index_offset_val),
        rank_offset(rank_offset_val),
        size_offset(size_offset_val),
        marked_indices_count()
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& container)
    {
        if(!this->topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        // fill device group by rank sequencially
        indexed_device_container<device_t>& indexed_container = std::get<device_t::type_idx()>(*topology);
        for(auto& gpu_device : container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id](marked_idx& val)
                                   {
                                       if(!val.first) //non marked
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                continue;
            }

            size_t rank = std::distance(id_array.begin(), it);
            size_t already_assigned_ids_count = assigned_ids.count(id);
            //rank += already_assigned_ids_count; TODO
            (void)already_assigned_ids_count;

            size_t size = id_array.size();
            size += size_offset;

            gpu_device->template reset_rank<topology_type>(rank, size);
            indexed_container.insert({rank + index_offset, gpu_device});

            assigned_ids.insert({id, thread_idx});

            it->first = true;//marked
            marked_indices_count++;
        }

    }

    size_t get_marked_indices_count() const
    {
        return marked_indices_count;;
    }
private:
    size_t index_offset;
    size_t rank_offset;
    size_t size_offset;

    size_t marked_indices_count;
};


template<ccl::device_topology_type topology_type>
struct graph_ring_indexer_unique_index_ext : public graph_ring_indexer<topology_type>
{
    using base = graph_ring_indexer<topology_type>;
    using base::topology;
    using base::thread_idx;
    using base::id_array;
    using base::assigned_ids;
    graph_ring_indexer_unique_index_ext(std::vector<marked_idx>& id_ring_vector,
                       id_thread_table &thread_id_storage,
                       size_t thread_id,
                       std::unique_ptr<specific_indexed_device_storage>& device_topology,
                       size_t index_offset_val,
                       size_t rank_offset_val,
                       size_t size_offset_val):
        graph_ring_indexer<topology_type>(id_ring_vector, thread_id_storage, thread_id, device_topology),
        index_offset(index_offset_val),
        rank_offset(rank_offset_val),
        size_offset(size_offset_val),
        marked_indices_count()
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& container)
    {
        if(!this->topology)
        {
            topology.reset(new specific_indexed_device_storage());
        }

        // fill device group by rank sequencially
        indexed_device_container<device_t>& indexed_container = std::get<device_t::type_idx()>(*topology);
        for(auto& gpu_device : container)
        {
            //get device id from group
            const ccl::device_index_type& id = gpu_device->get_device().get_device_path();

            //find rank for device id in our ring
            auto it = std::find_if(id_array.begin(), id_array.end(),
                                   [id](marked_idx& val)
                                   {
                                       if(!val.first) //non marked
                                       {
                                           return val.second == id;
                                       }
                                       return false;
                                    });
            if(it == id_array.end())
            {
                continue;
            }

            size_t rank = std::distance(id_array.begin(), it);
            rank += rank_offset;

            size_t already_assigned_ids_count = assigned_ids.count(id);
            //rank += already_assigned_ids_count; TODO
            (void)already_assigned_ids_count;

            size_t size = id_array.size();
            size += size_offset;

            gpu_device->template reset_rank<topology_type>(rank, size);
            indexed_container.insert({rank + index_offset, gpu_device});

            assigned_ids.insert({id, thread_idx});

            it->first = true;//marked
            marked_indices_count++;
        }

    }

    size_t get_marked_indices_count() const
    {
        return marked_indices_count;;
    }
private:
    size_t index_offset;
    size_t rank_offset;
    size_t size_offset;

    size_t marked_indices_count;
};

template<class device_t, ccl::device_topology_type topology>
std::tuple<size_t, device_t_ptr<device_t>>
        get_device_with_min_rank(const specific_indexed_device_storage& indexed_devices,
                                 const plain_graph& id_ring)
{
    const indexed_device_container<device_t>& container = std::get<device_t::type_idx()>(indexed_devices);

    //search in map from end (max element)
    size_t idx = std::numeric_limits<size_t>::min();
    device_t_ptr<device_t> dev;
    for (auto it = container.rbegin(); it != container.rend(); ++it)
    {
        device_t_ptr<device_t> tmp_dev = it->second;
        const auto& path = tmp_dev->get_device().get_device_path();
        if (std::find(id_ring.begin(), id_ring.end(), path) != id_ring.end())
        {
            idx = tmp_dev->template get_comm_data<topology>().rank;
            dev = tmp_dev;
            break;
        }
    }
    return std::tuple<size_t, device_t_ptr<device_t>> {idx, dev };
}

template<class device_t, ccl::device_topology_type topology>
std::tuple<size_t, device_t_ptr<device_t>>
        get_device_with_min_rank(const specific_indexed_device_storage& indexed_devices,
                                 const colored_plain_graph& id_ring)
{
    const indexed_device_container<device_t>& container =
                            std::get<device_t::type_idx()>(indexed_devices);

    //search in map from end (max element)
    size_t idx = std::numeric_limits<size_t>::min();
    device_t_ptr<device_t> dev;
    for (auto it = container.rbegin(); it != container.rend(); ++it)
    {
        device_t_ptr<device_t> tmp_dev = it->second;
        const auto& path = tmp_dev->get_device().get_device_path();
        if (std::find_if(id_ring.begin(), id_ring.end(),
            [path] (const typename colored_plain_graph::value_type &val)
            {
                return val.second == path;
            }) != id_ring.end())
        {
            idx = tmp_dev->template get_comm_data<topology>().rank;
            dev = tmp_dev;
            break;
        }
    }
    return std::tuple<size_t, device_t_ptr<device_t>>{ idx, dev };
}


template<class device_t, ccl::device_topology_type topology>
std::tuple<size_t, device_t_ptr<device_t>>
        get_device_with_max_rank(const specific_indexed_device_storage& indexed_devices,
                                 const plain_graph& id_ring)
{
    const indexed_device_container<device_t>& container = std::get<device_t::type_idx()>(indexed_devices);

    //search in map from begin (min element)
    size_t idx = std::numeric_limits<size_t>::max();
    device_t_ptr<device_t> dev;
    for (auto it = container.begin(); it != container.end(); ++it)
    {
        device_t_ptr<device_t> tmp_dev = it->second;
        const auto& path = tmp_dev->get_device().get_device_path();
        if (std::find(id_ring.begin(), id_ring.end(), path) != id_ring.end())
        {
            idx = tmp_dev->template get_comm_data<topology>().rank;
            dev = tmp_dev;
            break;
        }
    }
    return std::tuple<size_t, device_t_ptr<device_t>>{ idx, dev };
}

template<class device_t, ccl::device_topology_type topology>
std::tuple<size_t, device_t_ptr<device_t>>
        get_device_with_max_rank(const specific_indexed_device_storage& indexed_devices,
                                 const colored_plain_graph& id_ring)
{
    const indexed_device_container<device_t>& container = std::get<device_t::type_idx()>(indexed_devices);

    //search in map from begin (min element)
    size_t idx = std::numeric_limits<size_t>::max();
    device_t_ptr<device_t> dev;
    for (auto it = container.begin(); it != container.end(); ++it)
    {
        device_t_ptr<device_t> tmp_dev = it->second;
        const auto& path = tmp_dev->get_device().get_device_path();
        if (std::find_if(id_ring.begin(), id_ring.end(),
            [path] (const typename colored_plain_graph::value_type &val)
            {
                return val.second == path;
            }) != id_ring.end())
        {
            idx = tmp_dev->template get_comm_data<topology>().rank;
            dev = tmp_dev;
            break;
        }
    }
    return std::tuple<size_t, device_t_ptr<device_t>>{ idx, dev };
}

template<class device_t, ccl::device_topology_type topology>
device_t_ptr<ccl_thread_comm<device_t>>
        add_concurrent_locker_device(size_t next_rank,
                                     size_t index_offset,
                                     const std::tuple<size_t, device_t_ptr<device_t>>& dev_to_lock,
                                     device_storage& device_factory,
                                     specific_indexed_device_storage& storage_to_lock)
{
    device_t_ptr<device_t> dev = std::get<1>(dev_to_lock);
    device_t_ptr<ccl_thread_comm<device_t>> new_concurrent_comm =
                                    device_factory.create_gpu_device<ccl_thread_comm<device_t>>(
                                                                dev->get_device(),
                                                                next_rank,
                                                                *dev);

    const auto& comm_addr = new_concurrent_comm->template get_comm_data<topology>();

    indexed_device_container<ccl_thread_comm<device_t>>& current_locker_map =
            std::get<ccl_thread_comm<device_t>::type_idx()>(storage_to_lock);
    current_locker_map.insert({comm_addr.rank + index_offset, new_concurrent_comm});
    return new_concurrent_comm;
}

template<class device_t, ccl::device_topology_type topology>
device_t_ptr<ccl_ipc_source_gpu_comm<device_t>>
        add_ipc_source_locker_device(size_t next_rank,
                                     size_t index_offset,
                                     const std::tuple<size_t, device_t_ptr<device_t>>& dev_to_lock,
                                     device_storage& device_factory,
                                     specific_indexed_device_storage& storage_to_lock)
{
    using ipc_device_t = ccl_ipc_source_gpu_comm<device_t>;

    device_t_ptr<device_t> dev = std::get<1>(dev_to_lock);
    device_t_ptr<ipc_device_t> new_ipc_source_comm =
                            device_factory.create_gpu_device<ipc_device_t>(dev->get_device(),
                                                                           next_rank,
                                                                           *dev,
                                                                           topology);

    const auto& comm_addr = new_ipc_source_comm->template get_comm_data<topology>();

    indexed_device_container<ipc_device_t>& current_locker_map =
                                std::get<ipc_device_t::type_idx()>(storage_to_lock);

    //Exchange old device_t with new ipc_device_t
    indexed_device_container<device_t>& original_dev_container =
                                std::get<device_t::type_idx()>(storage_to_lock);
    auto original_dev_it = original_dev_container.find(std::get<0>(dev_to_lock) + index_offset);
    if (original_dev_it == original_dev_container.end()
        or original_dev_it->second.get() != dev.get())
    {
        assert(false && "unexpected device");
    }

    current_locker_map.insert({comm_addr.rank + index_offset, new_ipc_source_comm});
    original_dev_container.erase(original_dev_it);

    return new_ipc_source_comm;
}

template<class device_t, ccl::device_topology_type topology, class context>
device_t_ptr<ccl_gpu_scaleup_proxy<device_t>>
        add_scaleup_device(specific_plain_device_storage &plain_storage,
                           const ccl::device_index_type& index,
                           context& context_to_register,
                           device_storage& device_factory)
{
    device_t_ptr<ccl_gpu_scaleup_proxy<device_t>> ret;
    plain_device_container<device_t>& container = std::get<device_t::type_idx()>(plain_storage);
    for (auto it = container.begin(); it != container.end(); ++it)
    {
        if ((*it)->get_device().get_device_path() == index)
        {
            device_t_ptr<device_t> device = *it;
            container.erase(it);

            ret = device_factory.create_gpu_device<ccl_gpu_scaleup_proxy<device_t>>(device->get_device(),
                                                                                    container.size(),
                                                                                    *device);
            ret->template assign<topology>(context_to_register);
            std::get<ccl_gpu_scaleup_proxy<device_t>::type_idx()>(plain_storage).push_back(ret);
            break;
        }
    }
    return ret;
}

using ipc_devices_pool = std::map<size_t/*rank*/, device_t_ptr<ccl_ipc_gpu_comm>>;

template<ccl::device_topology_type topology>
inline ipc_devices_pool create_ipc_gpu_comms(id_thread_table assigned_ids_copy,
                                             const plain_graph& id_ring,
                                             device_storage& device_factory,
                                             size_t size_override_value,
                                             size_t rank_offset_value)
{
    // allocate IPC devices pool with rank from unassigned IDs
    // need to find symmetric_difference between graph ids and assigned ids
    // unassigned ids is a ipc device candidate

    ipc_devices_pool ret;
    for (auto graph_it = id_ring.begin(); graph_it != id_ring.end(); )
    {
        auto assigned_id_it = assigned_ids_copy.find(*graph_it);
        if(assigned_id_it != assigned_ids_copy.end())
        {
            assigned_ids_copy.erase(assigned_id_it);
            ++graph_it;
            continue;
        }

        //find unassigned_device
        size_t rank = std::distance(id_ring.begin(), graph_it);
        size_t size = size_override_value;

        //recalculate rank to apply offset for other processes count
        rank = (rank + rank_offset_value ) % size;

        ccl_device_driver::device_ptr ipc_device = get_runtime_device(*graph_it);
        device_t_ptr<ccl_ipc_gpu_comm> locker =
                                device_factory.create_gpu_device<ccl_ipc_gpu_comm>(*ipc_device,
                                                                                   rank,
                                                                                   size,
                                                                                   topology);
        ret.insert({rank, std::move(locker)});
        ++graph_it;
    }
    return ret;
}

using cluster_ipc_devices_pool = std::map<size_t/*process_id*/, ipc_devices_pool>;

template<ccl::device_topology_type topology>
inline cluster_ipc_devices_pool create_filtered_ipc_gpu_comms(const colored_plain_graph& id_ring,
                                                     const ccl::process_device_indices_t& ipc_indices,
                                                     size_t process_idx,
                                                     size_t process_size,
                                                     device_storage& device_factory)
{
    cluster_ipc_devices_pool ret;
    for (auto graph_it = id_ring.begin(); graph_it != id_ring.end(); ++graph_it)
    {
        if (graph_it->first != colored_graph_ring_indexer<topology>::marked_color and
            graph_it->first != process_idx)
        {
            size_t ipc_process_index = graph_it->first;
            if (process_idx == 0 and ipc_process_index > process_size)
            {
                //replace terminator as index
                ipc_process_index = process_size;
            }

            if (process_idx == process_size - 1 and ipc_process_index > process_size)
            {
                //replace terminator as index
                ipc_process_index = 0;
            }
            //find ipc_device in candidates list
            auto candidate_it = ipc_indices.find(ipc_process_index);
            if (candidate_it == ipc_indices.end()
                or (candidate_it->second.find(graph_it->second) == candidate_it->second.end()))
            {
                continue;
            }

            //device is IPC
            size_t rank = std::distance(id_ring.begin(), graph_it);
            size_t size = id_ring.size();

            ccl_device_driver::device_ptr ipc_device = get_runtime_device(graph_it->second);
            device_t_ptr<ccl_ipc_gpu_comm> locker =
                                device_factory.create_gpu_device<ccl_ipc_gpu_comm>(*ipc_device,
                                                                                   rank,
                                                                                   size,
                                                                                   topology);
            ret[graph_it->first].insert({rank, std::move(locker)});
        }
    }
    return ret;
}




template<ccl::device_topology_type topology>
inline cluster_ipc_devices_pool create_filtered_ipc_destination_gpu_comms(
                                            const colored_plain_graph& id_ring,
                                            const ccl::process_device_indices_t& ipc_indices,
                                            size_t process_idx,
                                            size_t process_size,
                                            device_storage& device_factory,
                                            specific_plain_device_storage& out_container)
{
    //destination is right device
    cluster_ipc_devices_pool ret;
    for (auto graph_it = id_ring.begin(); graph_it != id_ring.end(); ++graph_it)
    {
        if (graph_it->first != colored_graph_ring_indexer<topology>::marked_color and
            graph_it->first > process_idx)
        {
            size_t ipc_process_index = graph_it->first;
            if ((process_idx == process_size - 1) and ipc_process_index > process_size)
            {
                //replace terminator as index
                ipc_process_index = 0;
            }

            //find ipc_device in candidates list
            auto candidate_it = ipc_indices.find(ipc_process_index);
            if (candidate_it == ipc_indices.end()
                or (candidate_it->second.find(graph_it->second) == candidate_it->second.end()))
            {
                continue;
            }

            //device is IPC
            size_t rank = std::distance(id_ring.begin(), graph_it);
            size_t size = id_ring.size();

            ccl_device_driver::device_ptr ipc_device = get_runtime_device(graph_it->second);
            device_t_ptr<ccl_ipc_gpu_comm> locker =
                                device_factory.create_gpu_device<ccl_ipc_gpu_comm>(*ipc_device,
                                                                                   rank,
                                                                                   size,
                                                                                   topology);

            std::get<ccl_ipc_gpu_comm::type_idx()>(out_container).push_back(locker);
            ret[graph_it->first].insert({rank, std::move(locker)});
        }
    }
    return ret;
}

template<ccl::device_topology_type topology>
inline cluster_ipc_devices_pool create_ipc_gpu_comms(const colored_plain_graph& id_ring,
                                                     size_t process_idx,
                                                     device_storage& device_factory,
                                                     size_t size_override_value,
                                                     size_t rank_offset_value)
{
    cluster_ipc_devices_pool ret;
    for (auto graph_it = id_ring.begin(); graph_it != id_ring.end(); ++graph_it)
    {
        if (graph_it->first != colored_graph_ring_indexer<topology>::marked_color and
            graph_it->first != process_idx)
        {
            size_t rank = std::distance(id_ring.begin(), graph_it);
            size_t size = size_override_value;

            //recalculate rank to apply offset for other processes count
            rank = (rank + rank_offset_value ) % size;

            ccl_device_driver::device_ptr ipc_device = get_runtime_device(graph_it->second);
            device_t_ptr<ccl_ipc_gpu_comm> locker =
                                device_factory.create_gpu_device<ccl_ipc_gpu_comm>(*ipc_device,
                                                                                   rank,
                                                                                   size,
                                                                                   topology);
            ret[graph_it->first].insert({rank, std::move(locker)});
        }
    }
    return ret;
}


template<ccl::device_topology_type topology>
inline cluster_ipc_devices_pool create_ipc_gpu_comms(const colored_plain_graph_list& list,
                                                     size_t process_idx,
                                                     device_storage& device_factory,
                                                     size_t size_override_value,
                                                     size_t rank_offset_value)
{
    cluster_ipc_devices_pool ret;
    for (const auto& graph : list)
    {
        auto graph_ret = create_ipc_gpu_comms<topology>(graph, process_idx, device_factory,
                                                        size_override_value, rank_offset_value);
        ret.insert(graph_ret.begin(), graph_ret.end());
    }
    return ret;
}

inline std::vector<size_t> get_ipc_proceses(const cluster_ipc_devices_pool& ipc_comms,
                                     size_t process_index,
                                     size_t process_count)
{
    std::vector<size_t> ipc_processes_id;
    ipc_processes_id.reserve(ipc_comms.size());
    for(auto it = ipc_comms.begin(); it != ipc_comms.end(); ++it)
    {
        if (it->first != process_index)
        {
            ipc_processes_id.push_back(it->first);
        }
    }
    return ipc_processes_id;
}
}
}
