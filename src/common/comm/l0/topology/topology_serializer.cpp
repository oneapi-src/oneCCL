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
#include "common/comm/l0/topology/topology_serializer.hpp"

namespace native {
namespace details {
namespace serialize {
device_path_serializable::raw_data_t device_path_serializable::result() {
    return data;
}

device_path_serializer::device_path_serializer(size_t expected_devices,
                                               size_t data_offset,
                                               size_t stride)
        : base(),
          expected_capacity(expected_devices * (device_index_size() + stride) + data_offset),
          stride_bytes(stride)

{
    data.reserve(expected_capacity);

    //fill preambule by zeros
    for (size_t i = 0; i < data_offset; i++) {
        data.push_back(0);
    }
}

template <class T>
device_path_serializer::raw_data_t device_path_serializer::serialize_indices_impl(
    const std::list<T>& list,
    size_t data_offset) {
    std::list<raw_data_t> serialized_list;
    size_t list_size = list.size();

    size_t total_size = sizeof(list_size) + data_offset;
    for (const auto& graph : list) {
        size_t graph_count = graph.size();
        raw_data_t serialized_graph =
            device_path_serializer::serialize_indices(graph, sizeof(graph_count));
        //copy graph count into preambule to recover multiple graphs
        memcpy(serialized_graph.data(), &graph_count, sizeof(graph_count));

        total_size += serialized_graph.size();
        serialized_list.push_back(std::move(serialized_graph));
    }

    raw_data_t total_data;
    total_data.reserve(total_size); //graphs with preambules + list size;

    //fill global preambule: list size
    for (size_t i = 0; i < data_offset + sizeof(list_size); i++) {
        total_data.push_back(0);
    }
    memcpy(total_data.data() + data_offset, &list_size, sizeof(list_size));

    //use std::accumulate in c++20
    for (const raw_data_t& data : serialized_list) {
        std::copy(data.begin(), data.end(), std::back_inserter(total_data));
    }

    /* [data_offset] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... */
    return total_data;
}

template <class T>
device_path_serializer::raw_data_t device_path_serializer::serialize_indices_impl(
    const std::map<size_t, T>& list) {
    std::list<raw_data_t> serialized_list;
    size_t cluster_size = list.size();

    size_t total_size = sizeof(cluster_size); //preambule size
    for (const auto& process_graph_list : list) {
        raw_data_t serialized_graph = device_path_serializer::serialize_indices(
            process_graph_list.second, sizeof(process_graph_list.first));
        /* [process_id] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... */
        memcpy(
            serialized_graph.data(), &process_graph_list.first, sizeof(process_graph_list.first));

        total_size += serialized_graph.size();
        serialized_list.push_back(std::move(serialized_graph));
    }

    raw_data_t total_data;
    total_data.reserve(total_size); //process graphs with preambules + cluster size;

    //fill global preambule: list size
    for (size_t i = 0; i < sizeof(cluster_size); i++) {
        total_data.push_back(0);
    }
    memcpy(total_data.data(), &cluster_size, sizeof(cluster_size));

    //use std::accumulate in c++20
    for (const raw_data_t& data : serialized_list) {
        std::copy(data.begin(), data.end(), std::back_inserter(total_data));
    }

    /* [cluster_size] [process_id] [graphs_count] [graph_size_0] [graph_data_0] [graph_size_1] [graph_data_1] ... */
    return total_data;
}

device_path_serializable::raw_data_t device_path_serializer::serialize_indices(
    const details::plain_graph_list& list,
    size_t data_offset) {
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

device_path_serializable::raw_data_t device_path_serializer::serialize_indices(
    const details::global_sorted_plain_graphs& list) {
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

device_path_serializable::raw_data_t device_path_serializer::serialize_indices(
    const details::colored_plain_graph_list& list,
    size_t offset) {
    return device_path_serializer::serialize_indices_impl(list, offset);
}

device_path_serializable::raw_data_t device_path_serializer::serialize_indices(
    const details::global_sorted_colored_plain_graphs& list) {
    return device_path_serializer::serialize_indices_impl(list);
}

/* Deserializer */
template <class T>
std::list<T> device_path_deserializer::deserialize_generic_indices_list_impl(
    const raw_data_t& data,
    size_t& deserialized_bytes_count,
    size_t offset,
    size_t stride) {
    std::list<T> list;
    size_t list_size = 0;

    // preconditions
    if (data.size() < sizeof(list_size) + offset) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - too short data size: " + std::to_string(data.size()) +
                                 ", expected: " + std::to_string(sizeof(list_size)) +
                                 ", with offset: " + std::to_string(offset));
    }
    memcpy(&list_size, data.data() + offset, sizeof(list_size));

    auto data_it = data.begin();
    std::advance(data_it, offset + sizeof(list_size));
    deserialized_bytes_count += sizeof(list_size);

    size_t deserialized_graphs_count = 0;
    for (; data_it != data.end() and deserialized_graphs_count < list_size;) {
        //get graph_size
        size_t graph_size = 0;
        size_t elapsed_byte_count = std::distance(data_it, data.end());
        size_t expected_count = sizeof(graph_size);
        if (elapsed_byte_count < expected_count) {
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
        if (elapsed_byte_count < expected_count) {
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

        T graph =
            device_path_deserializer::deserialize_indices<std::vector, typename T::value_type>(
                data_it, data_end_it, stride);

        data_it = data_end_it;
        deserialized_bytes_count += expected_count;

        list.push_back(std::move(graph));
        deserialized_graphs_count++;
    }

    // postconditions
    if (list.size() != list_size) {
        throw std::runtime_error(
            std::string(__FUNCTION__) + " - unexpected deserilized graphs count: " +
            std::to_string(list.size()) + ", expected: " + std::to_string(list_size));
    }
    return list;
}

template <class T>
std::map<size_t, T> device_path_deserializer::deserialize_generic_indices_map_impl(
    const raw_data_t& data,
    size_t stride) {
    std::map<size_t, T> global;
    size_t global_size = 0;
    size_t deserialized_bytes_count = 0;

    // preconditions
    if (data.size() < sizeof(global_size)) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - too short data size: " + std::to_string(data.size()) +
                                 ", expected: " + std::to_string(sizeof(global_size)));
    }
    memcpy(&global_size, data.data(), sizeof(global_size));

    auto data_it = data.begin();
    std::advance(data_it, sizeof(global_size));
    deserialized_bytes_count += sizeof(global_size);

    size_t deserialized_processes_count = 0;
    for (; data_it != data.end();) {
        //get process
        size_t process_id = 0;
        size_t elapsed_byte_count = std::distance(data_it, data.end());
        size_t expected_count = sizeof(process_id);
        if (elapsed_byte_count < expected_count) {
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
        T process_list = device_path_deserializer::template deserialize_generic_indices_list_impl<
            typename T::value_type>(
            raw_data_t(data_it, data.end()), process_deserialized_count, 0, stride);
        std::advance(data_it, process_deserialized_count);
        deserialized_bytes_count += process_deserialized_count;

        if (!global.emplace(process_id, std::move(process_list)).second) {
            throw std::runtime_error(
                std::string(__FUNCTION__) +
                " - Cannot insert deserialized graphs list for process indx: " +
                std::to_string(process_id));
        }

        deserialized_processes_count++;
    }

    // postconditions
    if (global.size() != global_size) {
        throw std::runtime_error(
            std::string(__FUNCTION__) + " - unexpected deserialized cluster graphs count: " +
            std::to_string(global.size()) + ", expected: " + std::to_string(global_size));
    }

    return global;
}

details::plain_graph_list device_path_deserializer::deserialize_graph_list_indices(
    const raw_data_t& data,
    size_t& deserialized_bytes_count,
    size_t offset) {
    return device_path_deserializer::deserialize_generic_indices_list_impl<
        typename details::plain_graph_list::value_type>(data, deserialized_bytes_count, offset, 0);
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

details::global_sorted_plain_graphs device_path_deserializer::deserialize_global_graph_list_indices(
    const raw_data_t& data) {
    return device_path_deserializer::template deserialize_generic_indices_map_impl<
        typename details::global_sorted_plain_graphs::mapped_type>(data, 0);
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

details::colored_plain_graph_list device_path_deserializer::deserialize_colored_graph_list_indices(
    const raw_data_t& list,
    size_t& deserialized_bytes_count,
    size_t offset) {
    return device_path_deserializer::deserialize_generic_indices_list_impl<
        typename details::colored_plain_graph_list::value_type>(
        list,
        deserialized_bytes_count,
        offset,
        sizeof(details::colored_idx) - sizeof(ccl::device_index_type));
}

details::global_sorted_colored_plain_graphs
device_path_deserializer::deserialize_global_colored_graph_list_indices(const raw_data_t& list) {
    return device_path_deserializer::deserialize_generic_indices_map_impl<
        typename details::global_sorted_colored_plain_graphs::mapped_type>(
        list, sizeof(details::colored_idx) - sizeof(ccl::device_index_type));
}

details::colored_idx device_path_deserializer::extract_index(raw_data_t::const_iterator it_begin,
                                                             raw_data_t::const_iterator it_end,
                                                             std::false_type raw_index) {
    constexpr size_t color_size = sizeof(details::color_t);
    constexpr size_t stride = sizeof(details::colored_idx) - sizeof(ccl::device_index_type);
    if (std::distance(it_begin, it_end) %
            (device_path_serializable::device_index_size() + stride) !=
        0) {
        assert(false && "Unexpected data bytes count!");
        throw std::runtime_error(
            std::string("Unexpected deserializing data bytes count: ") +
            std::to_string(std::distance(it_begin, it_end)) + ", is not divided by:" +
            std::to_string(device_path_serializable::device_index_size() + stride));
    }

    details::color_t color = 0;
    memcpy(&color, &(*it_begin), color_size);
    std::advance(it_begin, stride);

    return details::colored_idx(
        color, device_path_deserializer::extract_index(it_begin, it_end, std::true_type{}));
}

ccl::device_index_type device_path_deserializer::extract_index(raw_data_t::const_iterator it_begin,
                                                               raw_data_t::const_iterator it_end,
                                                               std::true_type raw_index) {
    if ((std::distance(it_begin, it_end) % device_path_serializable::device_index_size()) != 0) {
        assert(false && "Unexpected data bytes count!");
        throw std::runtime_error(
            std::string("Unexpected deserializing data bytes count: ") +
            std::to_string(std::distance(it_begin, it_end)) +
            ", is not divided by:" + std::to_string(device_path_serializable::device_index_size()));
    }

    ccl::device_index_type path;
    for (auto raw_data_it = it_begin; raw_data_it != it_end;) {
        ccl::index_type index;
        std::copy(raw_data_it,
                  raw_data_it + device_path_serializable::index_size(),
                  reinterpret_cast<unsigned char*>(&index));
        raw_data_it += device_path_serializable::index_size();
        std::get<ccl::device_index_enum::driver_index_id>(path) = index;

        std::copy(raw_data_it,
                  raw_data_it + device_path_serializable::index_size(),
                  reinterpret_cast<unsigned char*>(&index));
        raw_data_it += device_path_serializable::index_size();
        std::get<ccl::device_index_enum::device_index_id>(path) = index;

        std::copy(raw_data_it,
                  raw_data_it + device_path_serializable::index_size(),
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
} // namespace serialize
} // namespace details
} // namespace native
