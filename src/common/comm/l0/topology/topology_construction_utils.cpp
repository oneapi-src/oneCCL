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

namespace native {
std::ostream& operator<<(std::ostream& out, const details::adjacency_matrix& matrix) {
    if (matrix.empty()) {
        return out;
    }

    for (auto device_it : matrix) {
        const ccl::device_index_type& left_index = device_it.first;
        const auto& device_adjacencies = device_it.second;

        out << left_index << "\t:\t{";
        for (const auto& device_cross_rating_value : device_adjacencies) {
            const ccl::device_index_type& right_index = device_cross_rating_value.first;
            details::cross_device_rating rating = device_cross_rating_value.second;
            out << right_index << "/ " << rating << ", ";
        }
        out << "},\n";
    }
    out << std::endl;
    return out;
}

namespace details {
std::ostream& operator<<(std::ostream& out, const adjacency_matrix& matrix) {
    if (matrix.empty()) {
        return out;
    }

    for (auto device_it : matrix) {
        const ccl::device_index_type& left_index = device_it.first;
        const auto& device_adjacencies = device_it.second;

        out << ccl::to_string(left_index) << "\t:\t{";
        for (const auto& device_cross_rating_value : device_adjacencies) {
            const ccl::device_index_type& right_index = device_cross_rating_value.first;
            details::cross_device_rating rating = device_cross_rating_value.second;
            out << ccl::to_string(right_index) << "/ " << rating << ", ";
        }
        out << "},\n";
    }
    out << std::endl;
    return out;
}

std::ostream& operator<<(std::ostream& out, const colored_idx& idx) {
    out << ccl::to_string(idx.index) << "/" << idx.color;
    return out;
}

size_t property_p2p_rating_calculator(const native::ccl_device& lhs,
                                      const native::ccl_device& rhs,
                                      size_t weight) {
    ze_device_p2p_properties_t p2p = lhs.get_p2p_properties(rhs);
    if (p2p.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS)
        return weight;
    else
        return 0;
}

std::string to_string(const plain_graph& cont) {
    std::stringstream ss;
    for (const auto& id : cont) {
        ss << ccl::to_string(id) << ", ";
    }
    return ss.str();
}

std::string to_string(const plain_graph_list& lists, const std::string& prefix) {
    std::stringstream ss;
    ss << "Graphs counts: " << lists.size();
    size_t graph_num = 0;
    for (const plain_graph& graph : lists) {
        ss << "\n\t" << prefix << graph_num++ << "\t" << to_string(graph);
    }
    return ss.str();
}

std::string to_string(const colored_plain_graph& cont) {
    std::stringstream ss;
    for (const auto& id : cont) {
        ss << id << ", ";
    }
    return ss.str();
}

std::string to_string(const colored_plain_graph_list& lists, const std::string& prefix) {
    std::stringstream ss;
    ss << "Graphs counts: " << lists.size();
    size_t graph_num = 0;
    for (const colored_plain_graph& graph : lists) {
        ss << "\n\t" << prefix << graph_num++ << "\t" << to_string(graph);
    }
    return ss.str();
}

template <class composite_container>
std::string to_string_impl(const composite_container& cont) {
    std::stringstream ss;
    ss << "Cluster size: " << cont.size();
    for (const auto& process_graphs : cont) {
        ss << "\nprx: " << process_graphs.first << "\n{\n"
           << to_string(process_graphs.second, "\t") << "\n},";
    }
    return ss.str();
}

std::string to_string(const global_sorted_plain_graphs& cluster) {
    return std::string("Sorted - ") + to_string_impl(cluster);
}

std::string to_string(const global_plain_graphs& cluster) {
    return std::string("Plain - ") + to_string_impl(cluster);
}

std::string to_string(const global_sorted_colored_plain_graphs& cluster) {
    return std::string("Sorted Colored - ") + to_string_impl(cluster);
}

std::string to_string(const global_plain_colored_graphs& cluster) {
    return std::string("Plain Colored- ") + to_string_impl(cluster);
}

void fill_adjacency_matrix_for_single_device_in_devices_by_cond(
    const native::ccl_device& left_device,
    const ccl::device_index_type& lhs_index,
    const ccl_device_driver::devices_storage_type& devices,
    adjacency_matrix& matrix,
    p2p_rating_function ping,
    std::function<bool(const ccl::device_index_type&)> rhs_filter) {
    //TODO - more elegant way is needed
    //TODO measure latency as additional weight argument???
    const auto& l_subdevices = left_device.get_subdevices();
    if (!l_subdevices.empty()) {
        for (const auto& lhs_sub_pair : l_subdevices) {
            const auto& left_subdevice = *lhs_sub_pair.second;
            const auto& lhs_sub_index = left_subdevice.get_device_path();

            for (const auto& rhs_pair : devices) {
                const auto& right_device = *rhs_pair.second;
                const auto& rhs_index = right_device.get_device_path();

                if (!rhs_filter or rhs_filter(rhs_index)) //check cond on right
                {
                    const auto& right_subdevices = right_device.get_subdevices();
                    for (const auto& rhs_sub_pair : right_subdevices) {
                        const auto& right_subdevice = *rhs_sub_pair.second;
                        const auto& rhs_sub_index = right_subdevice.get_device_path();

                        if (!rhs_filter or rhs_filter(rhs_sub_index)) //check cond on right
                        {
                            // across subdevices only
                            matrix[lhs_sub_index][rhs_sub_index] =
                                ping(left_subdevice, right_subdevice);
                            matrix[rhs_sub_index][lhs_sub_index] =
                                ping(right_subdevice, left_subdevice);
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
    else {
        for (const auto& rhs_pair : devices) {
            const auto& right_device = *rhs_pair.second;
            const auto& rhs_index = right_device.get_device_path();

            if (!rhs_filter or rhs_filter(rhs_index)) //check cond on right
            {
                const auto& right_subdevices = right_device.get_subdevices();
                for (const auto& rhs_sub_pair : right_subdevices) {
                    const auto& right_subdevice = *rhs_sub_pair.second;
                    const auto& rhs_sub_index = right_subdevice.get_device_path();

                    if (!rhs_filter or rhs_filter(rhs_index)) //check cond on right
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

void fill_adjacency_matrix_for_single_device_in_devices(
    const native::ccl_device& left_device,
    const ccl::device_index_type& lhs_index,
    const ccl_device_driver::devices_storage_type& devices,
    adjacency_matrix& matrix,
    p2p_rating_function ping) {
    //TODO - more elegant way is needed
    //TODO measure latency as additional weight argument???
    const auto& l_subdevices = left_device.get_subdevices();
    if (!l_subdevices.empty()) {
        for (const auto& lhs_sub_pair : l_subdevices) {
            const auto& left_subdevice = *lhs_sub_pair.second;
            const auto& lhs_sub_index = left_subdevice.get_device_path();

            for (const auto& rhs_pair : devices) {
                const auto& right_device = *rhs_pair.second;
                const auto& rhs_index = right_device.get_device_path();

                const auto& right_subdevices = right_device.get_subdevices();
                for (const auto& rhs_sub_pair : right_subdevices) {
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
    else {
        for (const auto& rhs_pair : devices) {
            const auto& right_device = *rhs_pair.second;
            const auto& rhs_index = right_device.get_device_path();

            const auto& right_subdevices = right_device.get_subdevices();
            for (const auto& rhs_sub_pair : right_subdevices) {
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

adjacency_matrix create_adjacency_matrix_for_devices(
    const ccl_device_driver::devices_storage_type& devices,
    p2p_rating_function ping) {
    adjacency_matrix matrix;
    for (const auto& lhs_pair : devices) {
        const auto& left_device = *lhs_pair.second;
        const auto& lhs_index = left_device.get_device_path();

        fill_adjacency_matrix_for_single_device_in_devices_by_cond(
            left_device, lhs_index, devices, matrix, ping);
    }
    return matrix;
}

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::device_indices_t& device_indexes) {
    plain_graph ids_ring;

    std::multimap<ccl::device_index_type, bool> marked_indices;
    std::transform(device_indexes.begin(),
                   device_indexes.end(),
                   std::inserter(marked_indices, marked_indices.end()),
                   [](const ccl::device_index_type& idx) {
                       return std::pair<ccl::device_index_type, bool>{ idx, false };
                   });

    ids_ring.push_back(marked_indices.begin()->first);
    marked_indices.erase(marked_indices.begin());
    try {
        while (!marked_indices.empty()) {
            auto it = marked_indices.begin();

            //find next idx from elapsed
            bool find = false;
            for (; it != marked_indices.end(); ++it) {
                if (it->second == true)
                    continue; //skip dirty index

                auto adjacencies_list_it = matrix.find(ids_ring.back());

                //sanity check
                if (adjacencies_list_it == matrix.end()) {
                    throw std::runtime_error(std::string("Requested invalid device index: ") +
                                             ccl::to_string(ids_ring.back()) +
                                             ". Check adjacency_matrix construction");
                }

                const adjacency_list& device_adjacencies = adjacencies_list_it->second;

                auto rating_it = device_adjacencies.find(it->first);
                if (rating_it == device_adjacencies.end()) {
                    throw std::runtime_error(std::string("Requested invalid adjacency index: ") +
                                             ccl::to_string(it->first) + ", for parent device: " +
                                             ccl::to_string(ids_ring.back()) +
                                             ". Check adjacency_matrix construction");
                }

                details::cross_device_rating rating = rating_it->second;
                if (rating != 0) {
                    //find next
                    ids_ring.push_back(it->first);
                    marked_indices.erase(it);
                    find = true;
                    break;
                }
            }

            if (!find) //cannot find next node
            {
                /*if(ids_ring.empty())
                {
                    throw std::logic_error("qqq");
                }*/
                //the current device cannot communicate with any other
                ccl::device_index_type idx = ids_ring.back();
                ids_ring.pop_back();
                if (ids_ring.empty()) {
                    throw std::logic_error("No one device has no access to others");
                }

                //mark it as dirty
                auto inserted_it = marked_indices.emplace(idx, true);
                //get next device
                std::for_each(
                    inserted_it,
                    marked_indices.end(),
                    [](typename std::multimap<ccl::device_index_type, bool>::value_type& idx) {
                        idx.second = false;
                    });
            }
        }
    }
    catch (const std::exception& ex) {
        std::cerr << __PRETTY_FUNCTION__ << " - exception: " << ex.what() << std::endl;
        std::cerr << __PRETTY_FUNCTION__ << "Adjacencies matrix:\n" << matrix << std::endl;

        abort();
        return {};
    }
    return ids_ring;
}

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::process_device_indices_t& per_process_device_indexes) {
    plain_graph ids_ring;

    for (const auto& thread_group_val : per_process_device_indexes) {
        const auto& indices = thread_group_val;
        auto group_devices = graph_resolver(matrix, indices.second);
        ids_ring.insert(ids_ring.end(), group_devices.begin(), group_devices.end());
    }
    return ids_ring;
}

plain_graph graph_resolver(const adjacency_matrix& matrix,
                           const ccl::process_aggregated_device_mask_t& per_process_device_masks) {
    plain_graph ids_ring;

    for (const auto& thread_group_val : per_process_device_masks) {
        const auto& indices = ccl_device_driver::get_device_indices(thread_group_val.second);
        auto group_devices = graph_resolver(matrix, indices);
        ids_ring.insert(ids_ring.end(), group_devices.begin(), group_devices.end());
    }
    return ids_ring;
}

/* graph list creation utils */
plain_graph_list graph_list_resolver(const adjacency_matrix& matrix,
                                     const ccl::device_indices_t& device_indexes) {
    plain_graph_list isles;

    using marked_storage = std::multimap<ccl::device_index_type, bool>;
    marked_storage marked_indices;
    std::transform(device_indexes.begin(),
                   device_indexes.end(),
                   std::inserter(marked_indices, marked_indices.end()),
                   [](const ccl::device_index_type& idx) {
                       return std::pair<ccl::device_index_type, bool>{ idx, false };
                   });

    plain_graph cur_graph;
    cur_graph.push_back(marked_indices.begin()->first);
    marked_indices.erase(marked_indices.begin());

    // maximization problem
    using maximization_solution_data_slice = std::tuple<plain_graph, marked_storage>;
    maximization_solution_data_slice max_slice(cur_graph, marked_indices);
    enum { MAX_GRAPH, MAX_MARKED };

    try {
        while (!marked_indices.empty()) {
            auto it = marked_indices.begin();

            //find next idx from elapsed
            bool find = false;
            for (; it != marked_indices.end(); ++it) {
                ccl::device_index_type index{};
                bool marked{};

                std::tie(index, marked) = *it;
                if (marked) {
                    continue; //skip dirty index
                }

                auto adjacencies_list_it = matrix.find(cur_graph.back());

                //sanity check
                if (adjacencies_list_it == matrix.end()) {
                    throw std::runtime_error(std::string("Requested invalid device index: ") +
                                             ccl::to_string(cur_graph.back()) +
                                             ". Check adjacency_matrix construction");
                }

                const adjacency_list& device_adjacencies = adjacencies_list_it->second;

                auto rating_it = device_adjacencies.find(index);
                if (rating_it == device_adjacencies.end()) {
                    throw std::runtime_error(
                        std::string("Requested invalid adjacency index: ") + ccl::to_string(index) +
                        ", for parent device: " + ccl::to_string(cur_graph.back()) +
                        ". Check adjacency_matrix construction");
                }

                details::cross_device_rating rating = rating_it->second;
                if (rating != 0) {
                    //find next
                    cur_graph.push_back(index);
                    marked_indices.erase(it);
                    find = true;

                    //update maximization data
                    if (cur_graph.size() > std::get<MAX_GRAPH>(max_slice).size()) {
                        std::get<MAX_GRAPH>(max_slice) = cur_graph;
                        std::get<MAX_MARKED>(max_slice) = marked_indices;
                    }
                    break;
                }
            }

            if (!find) //cannot find next node
            {
                //the current device cannot communicate with any other
                ccl::device_index_type idx = cur_graph.back();
                cur_graph.pop_back();
                if (cur_graph.empty()) {
                    //push the longest graph path into isles
                    isles.push_back(std::get<MAX_GRAPH>(max_slice));

                    // get current marked slice
                    marked_indices = std::get<MAX_MARKED>(max_slice);

                    // check end
                    if (marked_indices.empty()) {
                        return isles;
                    }

                    //reboot searching parameters
                    cur_graph.push_back(marked_indices.begin()->first);
                    marked_indices.erase(marked_indices.begin());
                    std::get<MAX_GRAPH>(max_slice) = cur_graph;
                    std::get<MAX_MARKED>(max_slice) = marked_indices;
                }
                else {
                    //mark it as dirty
                    auto inserted_it = marked_indices.emplace(idx, true);
                    //get next device
                    if (inserted_it != marked_indices.end()) {
                        ++inserted_it;
                        std::for_each(inserted_it,
                                      marked_indices.end(),
                                      [](typename std::multimap<ccl::device_index_type,
                                                                bool>::value_type& idx) {
                                          idx.second = false;
                                      });
                    }
                }
            }
        }

        //process last
        if (!std::get<MAX_GRAPH>(max_slice).empty()) {
            isles.push_back(std::get<MAX_GRAPH>(max_slice));
        }
    }
    catch (const std::exception& ex) {
        std::cerr << __PRETTY_FUNCTION__ << " - exception: " << ex.what() << std::endl;
        std::cerr << __PRETTY_FUNCTION__ << "Adjacencies matrix:\n" << matrix << std::endl;

        abort();
        return {};
    }
    return isles;
}

template <class device_idx_container>
struct index_extractor {
    using T = device_idx_container;
};

template <>
struct index_extractor<ccl::device_index_type> {
    static const ccl::device_index_type& index(const ccl::device_index_type& in) {
        return in;
    }
};

template <>
struct index_extractor<typename colored_plain_graph::value_type> {
    static const ccl::device_index_type& index(const typename colored_plain_graph::value_type& in) {
        return in.index;
    }
};

template <template <class...> class container, class graph_list, class index_getter>
graph_list merge_graphs_stable(const container<graph_list>& lists,
                               details::p2p_rating_function ping,
                               index_getter get,
                               bool brake_on_incompatible,
                               bool to_right,
                               size_t& merged_process_count) {
    merged_process_count = 0;
    graph_list isles;
    for (const auto& group_graph_list : lists) {
        // merge into single list
        // first graph list becomes first
        if (isles.empty()) {
            isles = group_graph_list;
            merged_process_count++;
            continue;
        }

        graph_list list_to_merge;
        for (auto graph_it = group_graph_list.begin(); graph_it != group_graph_list.end();
             ++graph_it) {
            const auto& graph = *graph_it;
            if (graph.empty()) {
                continue;
            }

            // find accessible pairs
            bool merged = false;
            const auto& graph_first_device =
                get_platform().get_device(index_getter::index(*graph.begin()));
            const auto& graph_last_device =
                get_platform().get_device(index_getter::index(*graph.rbegin()));
            for (auto total_graph_it = isles.begin(); total_graph_it != isles.end();
                 ++total_graph_it) {
                auto& total_graph = *total_graph_it;
                if (total_graph.empty()) {
                    total_graph.insert(total_graph.end(), graph.begin(), graph.end());
                    merged = true;
                    break;
                }

                const auto& total_graph_first_device =
                    get_platform().get_device(index_getter::index(*total_graph.begin()));
                const auto& total_graph_last_device =
                    get_platform().get_device(index_getter::index(*total_graph.rbegin()));
                if (to_right) {
                    if (ping(*total_graph_last_device, *graph_first_device)) {
                        total_graph.insert(total_graph.end(), graph.begin(), graph.end());
                        merged = true;
                        break;
                    }
                    else if (ping(*graph_last_device, *total_graph_first_device)) {
                        auto tmp_graph = graph;
                        tmp_graph.insert(tmp_graph.end(), total_graph.begin(), total_graph.end());
                        total_graph.swap(tmp_graph);
                        merged = true;
                        break;
                    }
                }
                else {
                    if (ping(*graph_last_device, *total_graph_first_device)) {
                        auto tmp_graph = graph;
                        tmp_graph.insert(tmp_graph.end(), total_graph.begin(), total_graph.end());
                        total_graph.swap(tmp_graph);
                        merged = true;
                        break;
                    }
                }
            }

            if (!merged) {
                if (brake_on_incompatible) {
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

bool check_graph_a2a_capable(const plain_graph& graph,
                             const adjacency_matrix& matrix,
                             std::ostream& out) {
    bool a2a_capable = true;
    size_t graph_power = graph.size();
    out << __FUNCTION__ << " - graph power: " << graph_power << std::endl;
    for (const ccl::device_index_type& lhs_index : graph) {
        size_t device_power = 0;
        auto m_it = matrix.find(lhs_index);
        if (m_it == matrix.end()) {
            std::stringstream ss;
            ss << __FUNCTION__ << " - invalid control matrix: no device by "
               << ccl::to_string(lhs_index);
            out << ss.str();
            throw std::runtime_error(ss.str());
        }

        const details::adjacency_list& control_list = m_it->second;
        for (const ccl::device_index_type& rhs_index : graph) {
            auto c_it = control_list.find(rhs_index);
            if (c_it != control_list.end() and c_it->second != 0) {
                device_power++;
            }
        }

        out << "device " << ccl::to_string(lhs_index)
            << ", has connection point count: " << device_power << std::endl;
        if (device_power != graph_power) {
            a2a_capable = false;
            break;
        }
    }
    return a2a_capable;
}

plain_graph_list merge_graph_lists_stable(const std::list<plain_graph_list>& lists,
                                          details::p2p_rating_function ping,
                                          bool brake_on_incompatible) {
    size_t merged_process_count = 0;
    return merge_graphs_stable(lists,
                               ping,
                               index_extractor<ccl::device_index_type>{},
                               brake_on_incompatible,
                               true,
                               merged_process_count);
}

colored_plain_graph_list merge_graph_lists_stable(const std::list<colored_plain_graph_list>& lists,
                                                  details::p2p_rating_function ping,
                                                  bool brake_on_incompatible) {
    size_t merged_process_count = 0;
    return merge_graphs_stable(lists,
                               ping,
                               index_extractor<typename colored_plain_graph::value_type>{},
                               brake_on_incompatible,
                               true,
                               merged_process_count);
}

colored_plain_graph_list merge_graph_lists_stable_for_process(
    const std::list<colored_plain_graph_list>& lists,
    details::p2p_rating_function ping,
    bool to_right,
    size_t& merged_process_count) {
    return merge_graphs_stable(lists,
                               ping,
                               index_extractor<typename colored_plain_graph::value_type>{},
                               true,
                               to_right,
                               merged_process_count);
}

plain_graph_list graph_list_resolver(
    const adjacency_matrix& matrix,
    const ccl::process_device_indices_t& per_process_device_indexes,
    details::p2p_rating_function ping) {
    std::list<plain_graph_list> lists;
    for (const auto& thread_group_val : per_process_device_indexes) {
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
plain_graph_list graph_list_resolver(
    const adjacency_matrix& matrix,
    const ccl::process_aggregated_device_mask_t& per_process_device_masks) {
    plain_graph_list isles;
    return isles;
}

void reset_color(colored_plain_graph_list& list, color_t new_color) {
    for (auto& graph : list) {
        for (colored_idx& idx : graph) {
            idx.color = new_color;
        }
    }
}
} // namespace details
} // namespace native
