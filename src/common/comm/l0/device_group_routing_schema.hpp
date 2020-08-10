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
#include <cassert>
#include <memory>
#include <sstream>
#include "ccl_types.hpp"
#include "common/utils/enums.hpp"
#include "common/utils/tuple.hpp"

using device_group_split_type_names = utils::enum_to_str<ccl::device_group_split_type::last_value>;
inline std::string to_string(ccl::device_group_split_type type) {
    return device_group_split_type_names({
                                             "TG",
                                             "PG",
                                             "CG",
                                         })
        .choose(type, "INVALID_VALUE");
}

using device_topology_type_names = utils::enum_to_str<ccl::device_topology_type::last_class_value>;
inline std::string to_string(ccl::device_topology_type class_value) {
    return device_topology_type_names({ "RING_CLASS", "A2A_CLASS" })
        .choose(class_value, "INVALID_VALUE");
}

template <ccl::device_group_split_type schema_id, ccl::device_topology_type class_id>
struct topology_addr {
    using comm_value_t = size_t;
    /*using type_idx_t = typename std::underlying_type<ccl::device_group_split_type>::type;
    using type_idx_t = typename std::underlying_type<ccl::device_topology_type>::type;
    static constexpr type_idx_t type_idx()
    {
        return static_cast<type_idx_t>(schema_id);
    }*/

    topology_addr(comm_value_t new_rank, comm_value_t new_size) : rank(new_rank), size(new_size) {}

    std::string to_string() const {
        std::stringstream ss;
        ss << ::to_string(class_id) << ": " << rank << "/" << size;
        return ss.str();
    }

    comm_value_t rank;
    comm_value_t size;
};

template <ccl::device_group_split_type schema_id, ccl::device_topology_type class_id>
using topology_addr_ptr = std::unique_ptr<topology_addr<schema_id, class_id>>;

template <ccl::device_group_split_type group_id, ccl::device_topology_type... class_ids>
using topology_addr_pointers_tuple_t = std::tuple<topology_addr_ptr<group_id, class_ids>...>;

namespace details {
struct topology_printer {
    template <ccl::device_group_split_type type, ccl::device_topology_type... class_ids>
    void operator()(const topology_addr_pointers_tuple_t<type, class_ids...>& topology) {
        details::topology_printer p;
        ccl_tuple_for_each(topology, p);
        result << ::to_string(type) << "\n\t{ ";
        result << p.result.str() << " }";
        result << std::endl;
    }

    template <ccl::device_group_split_type type, ccl::device_topology_type class_id>
    void operator()(const topology_addr_ptr<type, class_id>& topology) {
        if (topology) {
            result << topology->to_string();
        }
        else {
            result << to_string(class_id) << ": EMPTY";
        }
        result << ", ";
    }

    std::stringstream result;
};
} // namespace details

struct aggregated_topology_addr {
    template <ccl::device_group_split_type schema_id,
              ccl::device_topology_type class_id,
              class... SchemaArgs>
    bool insert(SchemaArgs&&... args) {
        if (std::get<class_id>(std::get<schema_id>(web))) {
            assert(false && "Topology is registered already");
            return false;
        }
        auto& schema_ptr = std::get<class_id>(std::get<schema_id>(web));
        schema_ptr.reset(new topology_addr<schema_id, class_id>(std::forward<SchemaArgs>(args)...));
        return true;
    }

    template <ccl::device_group_split_type schema_id, ccl::device_topology_type class_id>
    const topology_addr<schema_id, class_id>& get() const {
        const auto& schema_ptr = std::get<class_id>(std::get<schema_id>(web));
        if (!schema_ptr) {
            assert(false && "Topology is not registered");
            throw std::runtime_error("Invalid communication topology");
        }
        return *schema_ptr;
    }

    template <ccl::device_group_split_type schema_id, ccl::device_topology_type class_id>
    std::string to_string() const {
        details::topology_printer p;
        p(std::get<schema_id>(web));
        return p.result.str();
    }

    template <ccl::device_group_split_type schema_id, ccl::device_topology_type class_id>
    bool is_registered() const {
        return std::get<class_id>(std::get<schema_id>(web));
    }

    std::string to_string() const {
        details::topology_printer p;
        ccl_tuple_for_each(web, p);
        return p.result.str();
    }

    template <ccl::device_group_split_type... types>
    using topology_addr_storage_t =
        std::tuple<topology_addr_pointers_tuple_t<types, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>...>;

    using aggregated_topology_addr_storage_t =
        topology_addr_storage_t<SUPPORTED_HW_TOPOLOGIES_DECL_LIST>;

    aggregated_topology_addr_storage_t web;
};
