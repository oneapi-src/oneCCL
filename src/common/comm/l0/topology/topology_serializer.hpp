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
#include "common/comm/l0/topology/topology_construction_utils.hpp"
namespace native {
namespace details {
namespace serialize {
struct device_path_serializable {
    using raw_data_t = std::vector<unsigned char>;

    static constexpr size_t index_size() {
        return sizeof(ccl::index_type) / sizeof(unsigned char);
    }

    static constexpr size_t device_index_size() {
        return std::tuple_size<ccl::device_index_type>::value * index_size();
    }

    static size_t get_indices_count(size_t raw_data_size, size_t stride = 0) {
        if (raw_data_size % (device_index_size() + stride)) {
            assert(false && "Unexpected deserializing bytes count!");
            throw std::runtime_error(
                std::string("Unexpected deserializing bytes count: ") +
                std::to_string(raw_data_size) + ", extra bytes :" +
                std::to_string((raw_data_size % (device_index_size() + stride))) +
                ", stride: " + std::to_string(stride));
        }
        return raw_data_size / (device_index_size() + stride);
    }

    raw_data_t result();

protected:
    raw_data_t data;
};

struct device_path_serializer : device_path_serializable {
    using base = device_path_serializable;
    device_path_serializer(size_t expected_devices, size_t data_offset, size_t stride = 0);

    template <template <class...> class container>
    static raw_data_t serialize_indices(const container<ccl::device_index_type>& indices,
                                        size_t additional_reserved_bytes = 0) {
        device_path_serializer consumer(indices.size(), additional_reserved_bytes);
        for (const auto& path : indices) {
            ccl_tuple_for_each(path, consumer);
        }
        return consumer.result();
    }

    template <template <class...> class container>
    static raw_data_t serialize_indices(const container<details::colored_idx>& indices,
                                        size_t additional_reserved_bytes = 0) {
        static_assert(sizeof(details::colored_idx) >= sizeof(ccl::device_index_type),
                      "'stride' must be positive or zero");
        constexpr size_t stride = sizeof(details::colored_idx) - sizeof(ccl::device_index_type);
        device_path_serializer consumer(indices.size(), additional_reserved_bytes, stride);
        for (const auto& path : indices) {
            //serialize color
            size_t offset = consumer.data.size();
            for (size_t skip_bytes = 0; skip_bytes < consumer.stride_bytes; skip_bytes++) {
                consumer.data.push_back(0);
            }
            memcpy(consumer.data.data() + offset, &path.color, sizeof(path.color));

            //serialize index
            ccl_tuple_for_each(path.index, consumer);
        }
        return consumer.result();
    }

    static raw_data_t serialize_indices(const details::plain_graph_list& list, size_t offset = 0);
    static raw_data_t serialize_indices(const details::global_sorted_plain_graphs& list);
    static raw_data_t serialize_indices(const details::colored_plain_graph_list& list,
                                        size_t offset = 0);
    static raw_data_t serialize_indices(const details::global_sorted_colored_plain_graphs& list);

    template <class index_type>
    void operator()(const index_type& value) {
        static_assert(std::is_same<index_type, ccl::index_type>::value,
                      "Only ccl::index_type is supported");

        data.insert(data.end(),
                    reinterpret_cast<const unsigned char*>(&value),
                    reinterpret_cast<const unsigned char*>(&value) + sizeof(index_type));
    }

private:
    template <class T>
    static raw_data_t serialize_indices_impl(const std::list<T>& list, size_t offset = 0);
    template <class T>
    static raw_data_t serialize_indices_impl(const std::map<size_t, T>& list);

    size_t expected_capacity;
    size_t stride_bytes;
};

struct device_path_deserializer : device_path_serializable {
    using base = device_path_serializable;
    /*
    template<template<class...> class container>
    static container<ccl::device_index_type>
            deserialize_indices(const device_path_serializable::raw_data_t& data)
    {
        size_t elem_count = base::get_indices_count(data.size());
        container<ccl::device_index_type> ret;
        for (size_t elem_index = 0; elem_index < elem_count; elem_index++)
        {
            auto start_it = data.begin();
            std::advance(start_it, elem_index * device_path_serializable::device_index_size());
            ret.insert(ret.end(),
                       device_path_deserializer::extract_index(start_it,
                                                               start_it + device_index_size()));
        }
        return ret;
    }
   */
    template <template <class...> class container, class index_type, class iterator>
    static container<index_type> deserialize_indices(
        iterator it_begin,
        iterator it_end,
        size_t stride = sizeof(index_type) - sizeof(ccl::device_index_type)) {
        static_assert(sizeof(index_type) >= sizeof(ccl::device_index_type),
                      "'stride' must be positive or zero");
        size_t elem_count = base::get_indices_count(std::distance(it_begin, it_end), stride);
        container<index_type> ret;

        for (size_t elem_index = 0; elem_index < elem_count; elem_index++) {
            auto start_it = it_begin;
            std::advance(start_it,
                         elem_index * (device_path_serializable::device_index_size() + stride));
            ret.insert(ret.end(),
                       device_path_deserializer::extract_index(
                           start_it,
                           start_it + device_index_size() + stride,
                           std::integral_constant<
                               bool,
                               std::is_same<index_type, ccl::device_index_type>::value>{}));
        }
        return ret;
    }

    static details::plain_graph_list deserialize_graph_list_indices(
        const raw_data_t& list,
        size_t& deserialized_bytes_count,
        size_t offset = 0);
    static details::global_sorted_plain_graphs deserialize_global_graph_list_indices(
        const raw_data_t& list);

    static details::colored_plain_graph_list deserialize_colored_graph_list_indices(
        const raw_data_t& list,
        size_t& deserialized_bytes_count,
        size_t offset = 0);
    static details::global_sorted_colored_plain_graphs
    deserialize_global_colored_graph_list_indices(const raw_data_t& list);

    static ccl::device_index_type extract_index(raw_data_t::const_iterator it_begin,
                                                raw_data_t::const_iterator it_end,
                                                std::true_type raw_index);
    static details::colored_idx extract_index(raw_data_t::const_iterator it_begin,
                                              raw_data_t::const_iterator it_end,
                                              std::false_type colored_index);

private:
    template <class T>
    static std::list<T> deserialize_generic_indices_list_impl(const raw_data_t& list,
                                                              size_t& deserialized_bytes_count,
                                                              size_t offset = 0,
                                                              size_t stride = 0);
    template <class T>
    static std::map<size_t, T> deserialize_generic_indices_map_impl(const raw_data_t& list,
                                                                    size_t stride = 0);
};
} // namespace serialize
} // namespace details
} // namespace native
