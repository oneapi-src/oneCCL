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
#include "common/comm/l0/topology/topology_construction_utils.hpp"

namespace native
{
std::ostream& operator<< (std::ostream& out, const details::adjacency_matrix& matrix)
{
    if(matrix.empty())
    {
        return out;
    }

    for (auto device_it : matrix)
    {
        const ccl::device_index_type& left_index = device_it.first;
        const auto &device_adjacencies = device_it.second;

        out << left_index << "\t:\t{";
        for (const auto& device_cross_rating_value : device_adjacencies)
        {
            const ccl::device_index_type& right_index = device_cross_rating_value.first;
            details::cross_device_rating rating = device_cross_rating_value.second;
            out << right_index << "/ " << rating << ", ";
        }
        out << "},\n";
    }
    out << std::endl;
    return out;
}

namespace details
{
std::ostream& operator<< (std::ostream& out, const adjacency_matrix& matrix)
{
    if(matrix.empty())
    {
        return out;
    }

    for (auto device_it : matrix)
    {
        const ccl::device_index_type& left_index = device_it.first;
        const auto &device_adjacencies = device_it.second;

        out << ccl::to_string(left_index) << "\t:\t{";
        for (const auto& device_cross_rating_value : device_adjacencies)
        {
            const ccl::device_index_type& right_index = device_cross_rating_value.first;
            details::cross_device_rating rating = device_cross_rating_value.second;
            out <<  ccl::to_string(right_index) << "/ " << rating << ", ";
        }
        out << "},\n";
    }
    out << std::endl;
    return out;
}

std::ostream& operator<<(std::ostream& out, const colored_idx& idx)
{
    out << ccl::to_string(idx.second) << "/" << idx.first;
    return out;
}

size_t property_p2p_rating_calculator(const native::ccl_device& lhs,
                                      const native::ccl_device& rhs,
                                      size_t weight)
{
    ze_device_p2p_properties_t p2p = lhs.get_p2p_properties(rhs);
    return p2p.accessSupported * weight;
}


std::string to_string(const plain_graph& cont)
{
    std::stringstream ss;
    for (const auto& id : cont)
    {
        ss << ccl::to_string(id) << ", ";
    }
    return ss.str();
}

std::string to_string(const plain_graph_list& lists, const std::string& prefix)
{
    std::stringstream ss;
    ss << "Graphs counts: " << lists.size();
    size_t graph_num = 0;
    for (const plain_graph& graph : lists)
    {
        ss << "\n\t" << prefix << graph_num++ << "\t" << to_string(graph);
    }
    return ss.str();
}

std::string to_string(const colored_plain_graph& cont)
{
    std::stringstream ss;
    for (const auto& id : cont)
    {
        ss << id << ", ";
    }
    return ss.str();
}

std::string to_string(const colored_plain_graph_list& lists, const std::string& prefix)
{
    std::stringstream ss;
    ss << "Graphs counts: " << lists.size();
    size_t graph_num = 0;
    for (const colored_plain_graph& graph : lists)
    {
        ss << "\n\t" << prefix << graph_num++ << "\t" << to_string(graph);
    }
    return ss.str();
}

template<class composite_container>
std::string to_string_impl(const composite_container& cont)
{
    std::stringstream ss;
    ss << "Cluster size: " << cont.size();
    for (const auto& process_graphs : cont)
    {
        ss << "\nprx: " << process_graphs.first << "\n{\n"
           << to_string(process_graphs.second, "\t") << "\n},";
    }
    return ss.str();
}

std::string to_string(const global_sorted_plain_graphs& cluster)
{
    return std::string("Sorted - ") + to_string_impl(cluster);
}

std::string to_string(const global_plain_graphs& cluster)
{
    return std::string("Plain - ") + to_string_impl(cluster);
}

std::string to_string(const global_sorted_colored_plain_graphs& cluster)
{
    return std::string("Sorted Colored - ") + to_string_impl(cluster);
}

std::string to_string(const global_plain_colored_graphs& cluster)
{
    return std::string("Plain Colored- ") + to_string_impl(cluster);
}

void fill_adjacency_matrix_for_single_device_in_devices_by_cond(const native::ccl_device& left_device,
                                                                const ccl::device_index_type& lhs_index,
                                                                const ccl_device_driver::devices_storage_type &devices,
                                                                adjacency_matrix& matrix,
                                                                p2p_rating_function ping,
                                                                std::function<bool(const ccl::device_index_type&)> rhs_filter)
{
    //TODO - more elegant way is needed
    //TODO measure latency as additional weight argument???
    const auto& l_subdevices = left_device.get_subdevices();
    if (!l_subdevices.empty())
    {
        for(const auto &lhs_sub_pair : l_subdevices)
        {
            const auto& left_subdevice = *lhs_sub_pair.second;
            const auto& lhs_sub_index = left_subdevice.get_device_path();

            for(const auto& rhs_pair : devices)
            {
                const auto &right_device = *rhs_pair.second;
                const auto& rhs_index = right_device.get_device_path();

                if (!rhs_filter or rhs_filter(rhs_index))                //check cond on right
                {
                    const auto& right_subdevices = right_device.get_subdevices();
                    for(const auto &rhs_sub_pair : right_subdevices)
                    {
                        const auto& right_subdevice = *rhs_sub_pair.second;
                        const auto& rhs_sub_index = right_subdevice.get_device_path();

                        if (!rhs_filter or rhs_filter(rhs_sub_index))    //check cond on right
                        {
                            // across subdevices only
                            matrix[lhs_sub_index][rhs_sub_index] = ping(left_subdevice, right_subdevice);
                            matrix[rhs_sub_index][lhs_sub_index] = ping(right_subdevice, left_subdevice);
                            // across left device & right subdevices only
                            matrix[lhs_index][rhs_sub_index] = ping(left_device, right_subdevice);
                            matrix[rhs_sub_index][lhs_index] = ping(right_subdevice, left_device);
                        }
                    }

                    // across left sub devices & right device only
                    matrix[lhs_sub_index][rhs_index] = ping(left_subdevice, right_device);
                    matrix[rhs_index][lhs_sub_index] = ping(right_device, left_subdevice);

                    // across left device & right device only
                    matrix[lhs_index][rhs_index] = ping(left_device, right_device);
                    matrix[rhs_index][lhs_index] = ping(right_device, left_device);
                }
            }
        }
    }
    else
    {
        for(const auto& rhs_pair : devices)
        {
            const auto &right_device = *rhs_pair.second;
            const auto& rhs_index = right_device.get_device_path();

            if (!rhs_filter or rhs_filter(rhs_index))                //check cond on right
            {
                const auto& right_subdevices = right_device.get_subdevices();
                for(const auto &rhs_sub_pair : right_subdevices)
                {
                    const auto& right_subdevice = *rhs_sub_pair.second;
                    const auto& rhs_sub_index = right_subdevice.get_device_path();

                    if (!rhs_filter or rhs_filter(rhs_index))        //check cond on right
                    {
                        // across left device & right subdevices only
                        matrix[lhs_index][rhs_sub_index] = ping(left_device, right_subdevice);
                        matrix[rhs_sub_index][lhs_index] = ping(right_subdevice, left_device);

                        // across left device & right subdevices only
                        matrix[rhs_sub_index][lhs_index] = ping(right_subdevice, left_device);
                        matrix[lhs_index][rhs_sub_index] = ping(left_device, right_subdevice);
                    }
                }

                // across left device & right device only
                matrix[lhs_index][rhs_index] = ping(left_device, right_device);
                matrix[rhs_index][lhs_index] = ping(right_device, left_device);
                // across left device & right device only
                matrix[rhs_index][lhs_index] = ping(right_device, left_device);
                matrix[lhs_index][rhs_index] = ping(left_device, right_device);
            }
        }
    }
}

void fill_adjacency_matrix_for_single_device_in_devices(const native::ccl_device& left_device,
                                                        const ccl::device_index_type& lhs_index,
                                                        const ccl_device_driver::devices_storage_type &devices,
                                                        adjacency_matrix& matrix,
                                                        p2p_rating_function ping)
{
    //TODO - more elegant way is needed
    //TODO measure latency as additional weight argument???
    const auto& l_subdevices = left_device.get_subdevices();
    if (!l_subdevices.empty())
    {
        for(const auto &lhs_sub_pair : l_subdevices)
        {
            const auto& left_subdevice = *lhs_sub_pair.second;
            const auto& lhs_sub_index = left_subdevice.get_device_path();

            for(const auto& rhs_pair : devices)
            {
                const auto &right_device = *rhs_pair.second;
                const auto& rhs_index = right_device.get_device_path();

                const auto& right_subdevices = right_device.get_subdevices();
                for(const auto &rhs_sub_pair : right_subdevices)
                {
                    const auto& right_subdevice = *rhs_sub_pair.second;
                    const auto& rhs_sub_index = right_subdevice.get_device_path();

                    // across subdevices only
                    matrix[lhs_sub_index][rhs_sub_index] = ping(left_subdevice, right_subdevice);

                    // across left device & right subdevices only
                    matrix[lhs_index][rhs_sub_index] = ping(left_device, right_subdevice);
                }

                // across left sub devices & right device only
                matrix[lhs_sub_index][rhs_index] = ping(left_subdevice, right_device);

                // across left device & right device only
                matrix[lhs_index][rhs_index] = ping(left_device, right_device);
            }
        }
    }
    else
    {
        for(const auto& rhs_pair : devices)
        {
            const auto &right_device = *rhs_pair.second;
            const auto& rhs_index = right_device.get_device_path();

            const auto& right_subdevices = right_device.get_subdevices();
            for(const auto &rhs_sub_pair : right_subdevices)
            {
                const auto& right_subdevice = *rhs_sub_pair.second;
                const auto& rhs_sub_index = right_subdevice.get_device_path();

                // across left device & right subdevices only
                matrix[lhs_index][rhs_sub_index] = ping(left_device, right_subdevice);

            }

            // across left device & right device only
            matrix[lhs_index][rhs_index] = ping(left_device, right_device);
        }
    }
}

adjacency_matrix create_adjacency_matrix_for_devices(const ccl_device_driver::devices_storage_type &devices,
                                                     p2p_rating_function ping)
{
    adjacency_matrix matrix;
    for(const auto& lhs_pair : devices)
    {
        const auto& left_device = *lhs_pair.second;
        const auto& lhs_index = left_device.get_device_path();

        fill_adjacency_matrix_for_single_device_in_devices_by_cond(left_device, lhs_index, devices,
                                                                   matrix, ping);
    }
    return matrix;
}

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::device_indices_t& device_indexes)
{
    plain_graph ids_ring;

    std::multimap<ccl::device_index_type, bool> marked_indices;
    std::transform(device_indexes.begin(), device_indexes.end(),
                   std::inserter(marked_indices, marked_indices.end()),
                   [] (const ccl::device_index_type& idx)
    {
        return std::pair<ccl::device_index_type, bool> {idx, false};
    });

    ids_ring.push_back(marked_indices.begin()->first);
    marked_indices.erase(marked_indices.begin());
    try
    {
        while(!marked_indices.empty())
        {
            auto it = marked_indices.begin();

            //find next idx from elapsed
            bool find = false;
            for( ; it != marked_indices.end(); ++it)
            {
                if(it->second == true) continue; //skip dirty index

                auto adjacencies_list_it = matrix.find(ids_ring.back());

                //sanity check
                if (adjacencies_list_it == matrix.end())
                {
                    throw std::runtime_error(std::string("Requested invalid device index: ") +
                                             ccl::to_string(ids_ring.back()) +
                                             ". Check adjacency_matrix construction");
                }

                const adjacency_list& device_adjacencies = adjacencies_list_it->second;

                auto rating_it = device_adjacencies.find(it->first);
                if (rating_it == device_adjacencies.end())
                {
                    throw std::runtime_error(std::string("Requested invalid adjacency index: ") +
                                             ccl::to_string(it->first) + ", for parent device: " +
                                             ccl::to_string(ids_ring.back()) +
                                             ". Check adjacency_matrix construction");
                }

                details::cross_device_rating rating = rating_it->second;
                if(rating != 0)
                {
                    //find next
                    ids_ring.push_back(it->first);
                    marked_indices.erase(it);
                    find = true;
                    break;
                }
            }

            if(!find)    //cannot find next node
            {
                /*if(ids_ring.empty())
                {
                    throw std::logic_error("qqq");
                }*/
                //the current device cannot communicate with any other
                ccl::device_index_type idx = ids_ring.back();
                ids_ring.pop_back();
                if(ids_ring.empty())
                {
                    throw std::logic_error("No one device has no access to others");
                }

                //mark it as dirty
                auto inserted_it = marked_indices.emplace(idx, true);
                //get next device
                std::for_each(inserted_it, marked_indices.end(),
                              [] (typename std::multimap<ccl::device_index_type, bool>::value_type& idx)
                {
                    idx.second = false;
                });
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << __PRETTY_FUNCTION__ << " - exception: " << ex.what() << std::endl;
        std::cerr << __PRETTY_FUNCTION__ << "Adjacencies matrix:\n" << matrix << std::endl;

        abort();
        return {};
    }
    return ids_ring;
}

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::process_device_indices_t &per_process_device_indexes)
{
    plain_graph ids_ring;

    for(const auto& thread_group_val : per_process_device_indexes)
    {
        const auto& indices =  thread_group_val;
        auto group_devices = graph_resolver(matrix, indices.second);
        ids_ring.insert(ids_ring.end(), group_devices.begin(), group_devices.end());
    }
    return ids_ring;
}

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::process_aggregated_device_mask_t &per_process_device_masks)
{
    plain_graph ids_ring;

    for(const auto& thread_group_val : per_process_device_masks)
    {
        const auto& indices =  ccl_device_driver::get_device_indices(thread_group_val.second);
        auto group_devices = graph_resolver(matrix, indices);
        ids_ring.insert(ids_ring.end(), group_devices.begin(), group_devices.end());
    }
    return ids_ring;
}



/* graph list creation utils */
plain_graph_list graph_list_resolver(const adjacency_matrix& matrix,
                                     const ccl::device_indices_t& device_indexes)
{
    plain_graph_list isles;

    using marked_storage = std::multimap<ccl::device_index_type, bool>;
    marked_storage marked_indices;
    std::transform(device_indexes.begin(), device_indexes.end(),
                   std::inserter(marked_indices, marked_indices.end()),
                   [] (const ccl::device_index_type& idx)
    {
        return std::pair<ccl::device_index_type, bool> {idx, false};
    });

    plain_graph cur_graph;
    cur_graph.push_back(marked_indices.begin()->first);
    marked_indices.erase(marked_indices.begin());


    // maximization problem
    using maximization_solution_data_slice =
            std::tuple<plain_graph, marked_storage>;
    maximization_solution_data_slice max_slice(cur_graph, marked_indices);
    enum
    {
        MAX_GRAPH,
        MAX_MARKED
    };

    try
    {
        while(!marked_indices.empty())
        {
            auto it = marked_indices.begin();

            //find next idx from elapsed
            bool find = false;
            for( ; it != marked_indices.end(); ++it)
            {
                ccl::device_index_type index{};
                bool marked{};

                std::tie(index, marked) = *it;
                if (marked)
                {
                    continue; //skip dirty index
                }

                auto adjacencies_list_it = matrix.find(cur_graph.back());

                //sanity check
                if (adjacencies_list_it == matrix.end())
                {
                    throw std::runtime_error(std::string("Requested invalid device index: ") +
                                             ccl::to_string(cur_graph.back()) +
                                             ". Check adjacency_matrix construction");
                }

                const adjacency_list& device_adjacencies = adjacencies_list_it->second;

                auto rating_it = device_adjacencies.find(index);
                if (rating_it == device_adjacencies.end())
                {
                    throw std::runtime_error(std::string("Requested invalid adjacency index: ") +
                                             ccl::to_string(index) + ", for parent device: " +
                                             ccl::to_string(cur_graph.back()) +
                                             ". Check adjacency_matrix construction");
                }

                details::cross_device_rating rating = rating_it->second;
                if(rating != 0)
                {
                    //find next
                    cur_graph.push_back(index);
                    marked_indices.erase(it);
                    find = true;

                    //update maximization data
                    if(cur_graph.size() > std::get<MAX_GRAPH>(max_slice).size())
                    {
                        std::get<MAX_GRAPH>(max_slice) = cur_graph;
                        std::get<MAX_MARKED>(max_slice) = marked_indices;
                    }
                    break;
                }
            }

            if(!find)    //cannot find next node
            {
                //the current device cannot communicate with any other
                ccl::device_index_type idx = cur_graph.back();
                cur_graph.pop_back();
                if(cur_graph.empty())
                {
                    //push the longest graph path into isles
                    isles.push_back(std::get<MAX_GRAPH>(max_slice));

                    // get current marked slice
                    marked_indices = std::get<MAX_MARKED>(max_slice);

                    // check end
                    if (marked_indices.empty())
                    {
                        return isles;
                    }

                    //reboot searching parameters
                    cur_graph.push_back(marked_indices.begin()->first);
                    marked_indices.erase(marked_indices.begin());
                    std::get<MAX_GRAPH>(max_slice) = cur_graph;
                    std::get<MAX_MARKED>(max_slice) = marked_indices;
                }
                else
                {
                    //mark it as dirty
                    auto inserted_it = marked_indices.emplace(idx, true);
                    //get next device
                    if (inserted_it != marked_indices.end())
                    {
                        ++inserted_it;
                        std::for_each(inserted_it, marked_indices.end(),
                                    [] (typename std::multimap<ccl::device_index_type, bool>::value_type& idx)
                        {
                            idx.second = false;
                        });
                    }
                }
            }
        }

        //process last
        if (!std::get<MAX_GRAPH>(max_slice).empty())
        {
            isles.push_back(std::get<MAX_GRAPH>(max_slice));
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << __PRETTY_FUNCTION__ << " - exception: " << ex.what() << std::endl;
        std::cerr << __PRETTY_FUNCTION__ << "Adjacencies matrix:\n" << matrix << std::endl;

        abort();
        return {};
    }
    return isles;
}

template <class device_idx_container>
struct index_extractor
{using T = device_idx_container;};

template <>
struct index_extractor<ccl::device_index_type>
{
    static const ccl::device_index_type& index(const ccl::device_index_type& in)
    {
        return in;
    }
};

template <>
struct index_extractor<typename colored_plain_graph::value_type>
{
    static const ccl::device_index_type& index(const typename colored_plain_graph::value_type& in)
    {
        return in.second;
    }
};

template<template<class...> class container, class graph_list, class index_getter>
graph_list merge_graphs_stable(const container<graph_list>& lists,
                               details::p2p_rating_function ping,
                               index_getter get, bool brake_on_incompatible,
                               bool to_right,
                               size_t &merged_process_count)
{
    merged_process_count = 0;
    graph_list isles;
    for (const auto& group_graph_list : lists)
    {
        // merge into single list
        // first graph list becomes first
        if (isles.empty())
        {
            isles = group_graph_list;
            merged_process_count++;
            continue;
        }

        graph_list list_to_merge;
        for(auto graph_it = group_graph_list.begin(); graph_it != group_graph_list.end(); ++graph_it)
        {
            const auto& graph = *graph_it;
            if (graph.empty())
            {
                continue;
            }

            // find accessible pairs
            bool merged = false;
            const auto& graph_first_device = get_platform().get_device(index_getter::index(*graph.begin()));
            const auto& graph_last_device = get_platform().get_device(index_getter::index(*graph.rbegin()));
            for(auto total_graph_it = isles.begin(); total_graph_it != isles.end(); ++total_graph_it)
            {
                auto& total_graph = *total_graph_it;
                if (total_graph.empty())
                {
                     total_graph.insert(total_graph.end(), graph.begin(), graph.end());
                     merged = true;
                     break;
                }

                const auto& total_graph_first_device = get_platform().get_device(index_getter::index(*total_graph.begin()));
                const auto& total_graph_last_device = get_platform().get_device(index_getter::index(*total_graph.rbegin()));
                if(to_right)
                {
                    if (ping(*total_graph_last_device, *graph_first_device))
                    {
                        total_graph.insert(total_graph.end(), graph.begin(), graph.end());
                        merged = true;
                        break;
                    }
                    else if(ping(*graph_last_device, *total_graph_first_device))
                    {
                        auto tmp_graph = graph;
                        tmp_graph.insert(tmp_graph.end(), total_graph.begin(), total_graph.end());
                        total_graph.swap(tmp_graph);
                        merged = true;
                        break;
                    }
                }
                else
                {
                    if(ping(*graph_last_device, *total_graph_first_device))
                    {
                        auto tmp_graph = graph;
                        tmp_graph.insert(tmp_graph.end(), total_graph.begin(), total_graph.end());
                        total_graph.swap(tmp_graph);
                        merged = true;
                        break;
                    }
                }
            }

            if (!merged)
            {
                if(brake_on_incompatible)
                {
                    return isles;
                }
                list_to_merge.push_back(graph);
            }

            merged_process_count++;
        }
        std::copy(list_to_merge.begin(), list_to_merge.end(), std::back_inserter(isles));
    }
    return isles;
}

plain_graph_list merge_graph_lists_stable(const std::list<plain_graph_list>& lists,
                                          details::p2p_rating_function ping,
                                          bool brake_on_incompatible)
{
    size_t  merged_process_count = 0;
    return merge_graphs_stable(lists, ping,
                               index_extractor<ccl::device_index_type> {},
                               brake_on_incompatible,
                               true,
                               merged_process_count);
}

colored_plain_graph_list merge_graph_lists_stable(const std::list<colored_plain_graph_list>& lists,
                                                  details::p2p_rating_function ping,
                                                  bool brake_on_incompatible)
{
    size_t  merged_process_count = 0;
    return merge_graphs_stable(lists, ping,
                               index_extractor<typename colored_plain_graph::value_type> {},
                               brake_on_incompatible,
                               true,
                               merged_process_count);
}

colored_plain_graph_list merge_graph_lists_stable_for_process(const std::list<colored_plain_graph_list>& lists,
                                          details::p2p_rating_function ping,
                                          bool to_right,
                                          size_t& merged_process_count)
{
    return merge_graphs_stable(lists, ping,
                               index_extractor<typename colored_plain_graph::value_type> {},
                               true,
                               to_right,
                               merged_process_count);
}

plain_graph_list graph_list_resolver(const adjacency_matrix& matrix,
                                     const ccl::process_device_indices_t &per_process_device_indexes,
                                     details::p2p_rating_function ping)
{
    std::list<plain_graph_list> lists;
    for(const auto& thread_group_val : per_process_device_indexes)
    {
        lists.emplace_back(graph_list_resolver(matrix, thread_group_val.second));
    }
    return merge_graph_lists_stable(lists, ping);
}
/*
        // merge into single list
        if (isles.empty())
        {
            isles.swap(group_graph_list);
            continue;
        }
        plain_graph_list list_to_merge;
        for(auto graph_it = group_graph_list.begin(); graph_it != group_graph_list.end(); ++graph_it)
        {
            plain_graph& graph = *graph_it;
            if (graph.empty())
            {
                continue;
            }
            // find accessible pairs
            bool merged = false;
            const auto& graph_first_device = get_platform().get_device(*graph.begin());
            const auto& graph_last_device = get_platform().get_device(*graph.rbegin());
            for(auto total_graph_it = isles.begin(); total_graph_it != isles.end(); ++total_graph_it)
            {
                plain_graph& total_graph = *total_graph_it;
                if (total_graph.empty())
                {
                     total_graph.insert(total_graph.end(), graph.begin(), graph.end());
                     merged = true;
                     break;
                }
                const auto& total_graph_first_device = get_platform().get_device(*total_graph.begin());
                const auto& total_graph_last_device = get_platform().get_device(*total_graph.rbegin());
                if (ping(*graph_first_device, *total_graph_last_device))
                {
                    total_graph.insert(total_graph.end(), graph.begin(), graph.end());
                    merged = true;
                    break;
                }
                else if(ping(*graph_last_device, *total_graph_first_device))
                {
                    graph.insert(graph.end(), total_graph.begin(), total_graph.end());
                    total_graph.swap(graph);
                    merged = true;
                    break;
                }
            }
            if (!merged)
            {
                list_to_merge.push_back(graph);
            }
        }
        std::copy(list_to_merge.begin(), list_to_merge.end(), std::back_inserter(isles));
    }
    return isles;
}
*/
plain_graph_list graph_list_resolver(const adjacency_matrix& matrix,
                                     const ccl::process_aggregated_device_mask_t &per_process_device_masks)
{
    plain_graph_list isles;
    return isles;
}

void reset_color(colored_plain_graph_list& list, color_t new_color)
{
    for(auto &graph : list)
    {
        for (colored_idx& idx : graph)
        {
            idx.first = new_color;
        }
    }
}

namespace serialize
{

device_path_serializable::raw_data_t device_path_serializable::result()
{
    return data;
}

device_path_serializer::device_path_serializer(size_t expected_devices,
                                               size_t data_offset,
                                               size_t stride) :
    base(),
    expected_capacity(expected_devices * (device_index_size() + stride) + data_offset),
    stride_bytes(stride)

{
    data.reserve(expected_capacity);

    //fill preambule by zeros
    for(size_t i = 0; i < data_offset; i++)
    {
        data.push_back(0);
    }
}

template<class T>
device_path_serializer::raw_data_t
        device_path_serializer::serialize_indices_impl(const std::list<T>& list, size_t data_offset)
{
    std::list<raw_data_t> serialized_list;
    size_t list_size = list.size();

    size_t total_size = sizeof(list_size) + data_offset;
    for (const auto& graph : list)
    {
        size_t graph_count = graph.size();
        raw_data_t serialized_graph = device_path_serializer::serialize_indices(graph,
                                                                                sizeof(graph_count));
        //copy graph count into preambule to recover multiple graphs
        memcpy(serialized_graph.data(), &graph_count, sizeof(graph_count));

        total_size += serialized_graph.size();
        serialized_list.push_back(std::move(serialized_graph));
    }

    raw_data_t total_data;
    total_data.reserve(total_size); //graphs with preambules + list size;

    //fill global preambule: list size
    for(size_t i = 0; i < data_offset + sizeof(list_size); i++)
    {
        total_data.push_back(0);
    }
    memcpy(total_data.data() + data_offset, &list_size, sizeof(list_size));

    //use std::accumulate in c++20
    for (const raw_data_t& data: serialized_list)
    {
        std::copy(data.begin(), data.end(), std::back_inserter(total_data));
    }

    /* [data_offset] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... */
    return total_data;
}

template<class T>
device_path_serializer::raw_data_t
        device_path_serializer::serialize_indices_impl(const std::map<size_t, T>& list)
{
    std::list<raw_data_t> serialized_list;
    size_t cluster_size = list.size();

    size_t total_size = sizeof(cluster_size); //preambule size
    for (const auto& process_graph_list : list)
    {
        raw_data_t serialized_graph =
                device_path_serializer::serialize_indices(process_graph_list.second,
                                                          sizeof(process_graph_list.first));
        /* [process_id] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... */
        memcpy(serialized_graph.data(), &process_graph_list.first, sizeof(process_graph_list.first));

        total_size += serialized_graph.size();
        serialized_list.push_back(std::move(serialized_graph));
    }

    raw_data_t total_data;
    total_data.reserve(total_size); //process graphs with preambules + cluster size;

    //fill global preambule: list size
    for(size_t i = 0; i < sizeof(cluster_size); i++)
    {
        total_data.push_back(0);
    }
    memcpy(total_data.data(), &cluster_size, sizeof(cluster_size));

    //use std::accumulate in c++20
    for (const raw_data_t& data: serialized_list)
    {
        std::copy(data.begin(), data.end(), std::back_inserter(total_data));
    }

    /* [cluster_size] [process_id] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... */
    return total_data;
}

device_path_serializable::raw_data_t
        device_path_serializer::serialize_indices(const details::plain_graph_list& list,
                                                  size_t data_offset)
{
    /*
    std::list<raw_data_t> serialized_list;
    size_t list_size = list.size();
    size_t total_size = sizeof(list_size) + data_offset;
    for (const details::plain_graph& graph : list)
    {
        size_t graph_count = graph.size();
        raw_data_t serialized_graph = device_path_serializer::serialize_indices(graph,
                                                                                sizeof(graph_count));
        //copy graph count into preambule to recover multiple graphs
        memcpy(serialized_graph.data(), &graph_count, sizeof(graph_count));
        total_size += serialized_graph.size();
        serialized_list.push_back(std::move(serialized_graph));
    }
    raw_data_t total_data;
    total_data.reserve(total_size); //graphs with preambules + list size;
    //fill global preambule: list size
    for(size_t i = 0; i < data_offset + sizeof(list_size); i++)
    {
        total_data.push_back(0);
    }
    memcpy(total_data.data() + data_offset, &list_size, sizeof(list_size));
    //use std::accumulate in c++20
    for (const raw_data_t& data: serialized_list)
    {
        std::copy(data.begin(), data.end(), std::back_inserter(total_data));
    }
    / * [data_offset] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... * /
    return total_data;*/
    return device_path_serializer::serialize_indices_impl(list, data_offset);
}

device_path_serializable::raw_data_t
        device_path_serializer::serialize_indices(const details::global_sorted_plain_graphs& list)
{
    /*std::list<raw_data_t> serialized_list;
    size_t cluster_size = list.size();
    size_t total_size = sizeof(cluster_size); //preambule size
    for (const auto& process_graph_list : list)
    {
        raw_data_t serialized_graph =
                device_path_serializer::serialize_indices(process_graph_list.second,
                                                          sizeof(process_graph_list.first));
        / * [process_id] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... * /
        memcpy(serialized_graph.data(), &process_graph_list.first, sizeof(process_graph_list.first));
        total_size += serialized_graph.size();
        serialized_list.push_back(std::move(serialized_graph));
    }
    raw_data_t total_data;
    total_data.reserve(total_size); //process graphs with preambules + cluster size;
    //fill global preambule: list size
    for(size_t i = 0; i < sizeof(cluster_size); i++)
    {
        total_data.push_back(0);
    }
    memcpy(total_data.data(), &cluster_size, sizeof(cluster_size));
    //use std::accumulate in c++20
    for (const raw_data_t& data: serialized_list)
    {
        std::copy(data.begin(), data.end(), std::back_inserter(total_data));
    }
    / * [cluster_size] [process_id] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... * /
    return total_data;
    */
    return device_path_serializer::serialize_indices_impl(list);
}

device_path_serializable::raw_data_t
        device_path_serializer::serialize_indices(const details::colored_plain_graph_list& list,
                                                    size_t offset)
{
    return device_path_serializer::serialize_indices_impl(list, offset);
}

device_path_serializable::raw_data_t
        device_path_serializer::serialize_indices(const details::global_sorted_colored_plain_graphs& list)
{
    return device_path_serializer::serialize_indices_impl(list);
}

/* Deserializer */
template<class T>
std::list<T> device_path_deserializer::deserialize_generic_indices_list_impl(const raw_data_t& data,
                                               size_t &deserialized_bytes_count,
                                               size_t offset, size_t stride)
{
    std::list<T> list;
    size_t list_size = 0;

    // preconditions
    if (data.size() < sizeof(list_size) + offset)
    {
        throw std::runtime_error(std::string(__FUNCTION__) + " - too short data size: " +
                                 std::to_string(data.size()) +
                                 ", expected: " + std::to_string(sizeof(list_size)) +
                                 ", with offset: " + std::to_string(offset));
    }
    memcpy(&list_size, data.data() + offset, sizeof(list_size));

    auto data_it = data.begin();
    std::advance(data_it, offset + sizeof(list_size));
    deserialized_bytes_count += sizeof(list_size);

    size_t deserialized_graphs_count = 0;
    for ( ; data_it != data.end() and deserialized_graphs_count < list_size; )
    {
        //get graph_size
        size_t graph_size = 0;
        size_t elapsed_byte_count = std::distance(data_it, data.end());
        size_t expected_count = sizeof(graph_size);
        if (elapsed_byte_count < expected_count)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot extract graph_size, too short data elapsed: " +
                                     std::to_string(elapsed_byte_count) +
                                     ", expected: " + std::to_string(expected_count) +
                                     ". initial data size: " + std::to_string(data.size()) +
                                     ", with offset: " + std::to_string(offset));
        }
        memcpy(&graph_size, &(*data_it), expected_count);
        std::advance(data_it, expected_count);
        deserialized_bytes_count += expected_count;

        //get graph_data
        elapsed_byte_count = std::distance(data_it, data.end());
        expected_count = (device_path_serializable::device_index_size() + stride) * graph_size;
        if (elapsed_byte_count < expected_count)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot extract graph_data, too short data elapsed: " +
                                     std::to_string(elapsed_byte_count) +
                                     ", expected: " + std::to_string(expected_count) +
                                     ". initial data size: " + std::to_string(data.size()) +
                                     ", with offset: " + std::to_string(offset));
        }

        //deserialize graph portion
        auto data_end_it = data_it;
        std::advance(data_end_it, expected_count);

        T graph = device_path_deserializer::deserialize_indices<std::vector, typename T::value_type>(data_it, data_end_it, stride);

        data_it = data_end_it;
        deserialized_bytes_count += expected_count;

        list.push_back(std::move(graph));
        deserialized_graphs_count ++;
    }

    // postconditions
    if (list.size() != list_size)
    {
         throw std::runtime_error(std::string(__FUNCTION__) + " - unexpected deserilized graphs count: " +
                                  std::to_string(list.size()) + ", expected: " + std::to_string(list_size));

    }
    return list;
}

template<class T>
std::map<size_t, T> device_path_deserializer::deserialize_generic_indices_map_impl(const raw_data_t& data,
                                                                            size_t stride)
{
    std::map<size_t, T> global;
    size_t global_size = 0;
    size_t deserialized_bytes_count = 0;

    // preconditions
    if (data.size() < sizeof(global_size))
    {
        throw std::runtime_error(std::string(__FUNCTION__) + " - too short data size: " +
                                 std::to_string(data.size()) +
                                 ", expected: " + std::to_string(sizeof(global_size)));
    }
    memcpy(&global_size, data.data(), sizeof(global_size));

    auto data_it = data.begin();
    std::advance(data_it, sizeof(global_size));
    deserialized_bytes_count += sizeof(global_size);

    size_t deserialized_processes_count = 0;
    for ( ; data_it != data.end(); )
    {
        //get process
        size_t process_id = 0;
        size_t elapsed_byte_count = std::distance(data_it, data.end());
        size_t expected_count = sizeof(process_id);
        if (elapsed_byte_count < expected_count)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot extract process_id, too short data elapsed: " +
                                     std::to_string(elapsed_byte_count) +
                                     ", expected: " + std::to_string(expected_count) +
                                     ". initial data size: " + std::to_string(data.size()));
        }
        memcpy(&process_id, &(*data_it), expected_count);
        std::advance(data_it, expected_count);
        deserialized_bytes_count += expected_count;

        //get graph_data for process
        size_t process_deserialized_count = 0;
        T process_list =
                device_path_deserializer::template deserialize_generic_indices_list_impl<typename T::value_type>(raw_data_t(data_it,
                                                                        data.end()),
                                                                        process_deserialized_count,
                                                                        0,
                                                                        stride);
        std::advance(data_it, process_deserialized_count);
        deserialized_bytes_count += process_deserialized_count;

        if (!global.emplace(process_id, std::move(process_list)).second)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot insert deserialized graphs list for process indx: " +
                                     std::to_string(process_id));
        }

        deserialized_processes_count++;
    }

    // postconditions
    if (global.size() != global_size)
    {
         throw std::runtime_error(std::string(__FUNCTION__) +
                                  " - unexpected deserialized cluster graphs count: " +
                                  std::to_string(global.size()) +
                                  ", expected: " + std::to_string(global_size));
    }

    return global;
}

details::plain_graph_list
        device_path_deserializer::deserialize_graph_list_indices(const raw_data_t& data,
                                                                 size_t &deserialized_bytes_count,
                                                                 size_t offset)
{
    return device_path_deserializer::deserialize_generic_indices_list_impl<typename details::plain_graph_list::value_type>
                            (data, deserialized_bytes_count, offset, 0);
    /*details::plain_graph_list list;
    size_t list_size = 0;
    // preconditions
    if (data.size() < sizeof(list_size) + offset)
    {
        throw std::runtime_error(std::string(__FUNCTION__) + " - too short data size: " +
                                 std::to_string(data.size()) +
                                 ", expected: " + std::to_string(sizeof(list_size)) +
                                 ", with offset: " + std::to_string(offset));
    }
    memcpy(&list_size, data.data() + offset, sizeof(list_size));
    auto data_it = data.begin();
    std::advance(data_it, offset + sizeof(list_size));
    deserialized_bytes_count += sizeof(list_size);
    size_t deserialized_graphs_count = 0;
    for ( ; data_it != data.end() and deserialized_graphs_count < list_size; )
    {
        //get graph_size
        size_t graph_size = 0;
        size_t elapsed_byte_count = std::distance(data_it, data.end());
        size_t expected_count = sizeof(graph_size);
        if (elapsed_byte_count < expected_count)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot extract graph_size, too short data elapsed: " +
                                     std::to_string(elapsed_byte_count) +
                                     ", expected: " + std::to_string(expected_count) +
                                     ". initial data size: " + std::to_string(data.size()) +
                                     ", with offset: " + std::to_string(offset));
        }
        memcpy(&graph_size, &(*data_it), expected_count);
        std::advance(data_it, expected_count);
        deserialized_bytes_count += expected_count;
        //get graph_data
        elapsed_byte_count = std::distance(data_it, data.end());
        expected_count = device_path_serializable::device_index_size() * graph_size;
        if (elapsed_byte_count < expected_count)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot extract graph_data, too short data elapsed: " +
                                     std::to_string(elapsed_byte_count) +
                                     ", expected: " + std::to_string(expected_count) +
                                     ". initial data size: " + std::to_string(data.size()) +
                                     ", with offset: " + std::to_string(offset));
        }
        //deserialize graph_data
        details::plain_graph graph;
        graph.reserve(graph_size);
        //deserialize graph portion
        auto data_end_it = data_it;
        std::advance(data_end_it, expected_count);
        for (size_t elem_index = 0; elem_index < graph_size; elem_index++)
        {
            graph.insert(graph.end(), device_path_deserializer::extract_index(data_it,
                                                                              data_it + device_path_serializable::device_index_size()));
            std::advance(data_it, device_path_serializable::device_index_size());
            deserialized_bytes_count += device_path_serializable::device_index_size();
        }
        list.push_back(std::move(graph));
        deserialized_graphs_count ++;
    }
    // postconditions
    if (list.size() != list_size)
    {
         throw std::runtime_error(std::string(__FUNCTION__) + " - unexpected deserilized graphs count: " +
                                  std::to_string(list.size()) + ", expected: " + std::to_string(list_size));
    }
    return list;*/
}

details::global_sorted_plain_graphs
        device_path_deserializer::deserialize_global_graph_list_indices(const raw_data_t& data)
{
    return device_path_deserializer::
            template deserialize_generic_indices_map_impl<typename details::global_sorted_plain_graphs::mapped_type>
                    (data, 0);
    /*
    details::global_sorted_plain_graphs global;
    size_t global_size = 0;
    size_t deserialized_bytes_count = 0;
    // preconditions
    if (data.size() < sizeof(global_size))
    {
        throw std::runtime_error(std::string(__FUNCTION__) + " - too short data size: " +
                                 std::to_string(data.size()) +
                                 ", expected: " + std::to_string(sizeof(global_size)));
    }
    memcpy(&global_size, data.data(), sizeof(global_size));
    auto data_it = data.begin();
    std::advance(data_it, sizeof(global_size));
    deserialized_bytes_count += sizeof(global_size);
    size_t deserialized_processes_count = 0;
    for ( ; data_it != data.end(); )
    {
        //get process
        size_t process_id = 0;
        size_t elapsed_byte_count = std::distance(data_it, data.end());
        size_t expected_count = sizeof(process_id);
        if (elapsed_byte_count < expected_count)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot extract process_id, too short data elapsed: " +
                                     std::to_string(elapsed_byte_count) +
                                     ", expected: " + std::to_string(expected_count) +
                                     ". initial data size: " + std::to_string(data.size()));
        }
        memcpy(&process_id, &(*data_it), expected_count);
        std::advance(data_it, expected_count);
        deserialized_bytes_count += expected_count;
        //get graph_data for process
        size_t process_deserialized_count = 0;
        details::plain_graph_list process_list =
                device_path_deserializer::deserialize_graph_list_indices(raw_data_t(data_it,
                                                                                    data.end()),
                                                                         process_deserialized_count);
        std::advance(data_it, process_deserialized_count);
        deserialized_bytes_count += process_deserialized_count;
        if (!global.emplace(process_id, std::move(process_list)).second)
        {
            throw std::runtime_error(std::string(__FUNCTION__) +
                                     " - Cannot insert deserialized graphs list for process indx: " +
                                     std::to_string(process_id));
        }
        deserialized_processes_count++;
    }
    // postconditions
    if (global.size() != global_size)
    {
         throw std::runtime_error(std::string(__FUNCTION__) +
                                  " - unexpected deserialized cluster graphs count: " +
                                  std::to_string(global.size()) +
                                  ", expected: " + std::to_string(global_size));
    }
    return global;
    */
}


details::colored_plain_graph_list
        device_path_deserializer::deserialize_colored_graph_list_indices(
                                                                    const raw_data_t& list,
                                                                    size_t &deserialized_bytes_count,
                                                                    size_t offset)
{
    return device_path_deserializer::
            deserialize_generic_indices_list_impl<typename details::colored_plain_graph_list::value_type>
                    (list, deserialized_bytes_count, offset,
                    sizeof(details::colored_idx) - sizeof(ccl::device_index_type));
}

details::global_sorted_colored_plain_graphs
        device_path_deserializer::deserialize_global_colored_graph_list_indices(const raw_data_t& list)
{
    return device_path_deserializer::deserialize_generic_indices_map_impl<typename details::global_sorted_colored_plain_graphs::mapped_type>
                    (list, sizeof(details::colored_idx) - sizeof(ccl::device_index_type));
}

details::colored_idx device_path_deserializer::extract_index(raw_data_t::const_iterator it_begin,
                                                             raw_data_t::const_iterator it_end,
                                                             std::false_type raw_index)
{
    constexpr size_t color_size = sizeof(details::color_t);
    constexpr size_t stride = sizeof(details::colored_idx) - sizeof(ccl::device_index_type);
    if (std::distance(it_begin, it_end) %
        (device_path_serializable::device_index_size() + stride) != 0)
    {
        assert(false && "Unexpected data bytes count!");
        throw std::runtime_error(std::string("Unexpected deserializing data bytes count: ") +
                                     std::to_string(std::distance(it_begin, it_end)) + ", is not divided by:" +
                                     std::to_string(device_path_serializable::device_index_size() + stride));
    }

    details::color_t color = 0;
    memcpy(&color, &(*it_begin), color_size);
    std::advance(it_begin, stride);

    return details::colored_idx(color, device_path_deserializer::extract_index(it_begin, it_end, std::true_type{}));
}

ccl::device_index_type device_path_deserializer::extract_index(raw_data_t::const_iterator it_begin,
                                                               raw_data_t::const_iterator it_end,
                                                               std::true_type raw_index)
{
    if ( (std::distance(it_begin, it_end) % device_path_serializable::device_index_size()) != 0)
    {
        assert(false && "Unexpected data bytes count!");
        throw std::runtime_error(std::string("Unexpected deserializing data bytes count: ") +
                                     std::to_string(std::distance(it_begin, it_end)) + ", is not divided by:" +
                                     std::to_string(device_path_serializable::device_index_size()));
    }

    ccl::device_index_type path;
    for(auto raw_data_it = it_begin; raw_data_it != it_end; )
    {
        ccl::index_type index;
        std::copy(raw_data_it, raw_data_it + device_path_serializable::index_size(),
                  reinterpret_cast<unsigned char*>(&index));
        raw_data_it += device_path_serializable::index_size();
        std::get<ccl::device_index_enum::driver_index_id>(path) = index;

        std::copy(raw_data_it, raw_data_it + device_path_serializable::index_size(),
                  reinterpret_cast<unsigned char*>(&index));
        raw_data_it += device_path_serializable::index_size();
        std::get<ccl::device_index_enum::device_index_id>(path) = index;

        std::copy(raw_data_it, raw_data_it + device_path_serializable::index_size(),
                  reinterpret_cast<unsigned char*>(&index));
        raw_data_it += device_path_serializable::index_size();
        std::get<ccl::device_index_enum::subdevice_index_id>(path) = index;
    }
    return path;
}
/*
ccl::device_indices_t device_path_deserializer::operator()(const std::vector<unsigned char>& raw_data)
{
    size_t elem_count = base::get_indices_count(raw_data.size());
    ccl::device_indices_t data;
    constexpr auto offset = sizeof(ccl::index_type) / sizeof(unsigned char);
    for(auto raw_data_it = raw_data.begin(); raw_data_it != raw_data.end(); )
    {
        ccl::device_index_type path;
        ccl::index_type index;
        std::copy(raw_data_it, raw_data_it + offset, reinterpret_cast<unsigned char*>(&index));
        raw_data_it += offset;
        std::get<ccl::device_index_enum::driver_index_id>(path) = index;
        std::copy(raw_data_it, raw_data_it + offset, reinterpret_cast<unsigned char*>(&index));
        raw_data_it += offset;
        std::get<ccl::device_index_enum::device_index_id>(path) = index;
        std::copy(raw_data_it, raw_data_it + offset, reinterpret_cast<unsigned char*>(&index));
        raw_data_it += offset;
        std::get<ccl::device_index_enum::subdevice_index_id>(path) = index;
        data.insert(std::move(path));
    }
    return data;
}
*/
}
}
}
