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
#include <list>
#include <set>
#include <vector>

#include "coll/algorithms/algorithms_enum.hpp"

#include "common/comm/l0/device_group_routing_schema.hpp"
#include "common/comm/l0/gpu_device_types.hpp"

#include "common/comm/l0/modules/ring/allgatherv_entry_module.hpp"
#include "common/comm/l0/modules/ring/allreduce_entry_module.hpp"
#include "common/comm/l0/modules/ring/alltoallv_entry_module.hpp"
#include "common/comm/l0/modules/ring/bcast_entry_module.hpp"
#include "common/comm/l0/modules/ring/reduce_entry_module.hpp"

#include "common/comm/l0/modules/a2a/allreduce_module.hpp"
#include "common/comm/l0/modules/supported_modules.hpp"

#include "common/comm/l0/modules/modules_source_data.hpp"
#include "common/comm/l0/gpu_comm_utils.hpp"

namespace native {

template <class gpu_impl, gpu_types type>
class ccl_gpu_base_comm {
public:
    using comm_rank_t = size_t;
    using type_idx_t = typename std::underlying_type<gpu_types>::type;
    ccl_gpu_base_comm(ccl_device& assigned_device, comm_rank_t idx)
            : index_in_group(idx),
              device(assigned_device)

    {}

    ~ccl_gpu_base_comm() = default;

    gpu_impl* get_this() {
        return static_cast<gpu_impl*>(this);
    }

    const gpu_impl* get_this() const {
        return static_cast<const gpu_impl*>(this);
    }

    static constexpr const char* name() {
        return gpu_impl::name_impl();
    }

    std::string to_string() const {
        return get_this()->to_string_impl();
    }

    static constexpr type_idx_t type_idx() {
        return static_cast<type_idx_t>(type);
    }

    ccl_device& get_device() {
        return device;
    }
    [[deprecated]] comm_rank_t get_index_in_group() const {
        return index_in_group;
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    bool reset_rank(comm_rank_t new_rank, comm_rank_t new_size) {
        rank = new_rank;
        size = new_size;
        return device_routing_web.insert<group_id, class_id>(new_rank,
                                                             new_size); //consider inheritance
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    const topology_addr<group_id, class_id>& get_comm_data() const {
        return device_routing_web.get<group_id, class_id>();
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    bool is_registered() const {
        return device_routing_web.is_registered<group_id, class_id>();
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    std::string comm_to_str() const {
        return device_routing_web.to_string<group_id, class_id>();
    }

    std::string comm_to_str() const {
        return device_routing_web.to_string();
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              template <ccl_coll_type, ccl::group_split_type, ccl::device_topology_type>
              class module_impl>
    static std::shared_ptr<module_impl<module_type, group_id, class_id>>& get_gpu_module_unsafe(
        supported_device_modules<module_impl>& modules) {
        return std::get<utils::enum_to_underlying(class_id)>(
            std::get<utils::enum_to_underlying(group_id)>(std::get<module_type>(modules)));
    }

protected:
    size_t index_in_group;

    aggregated_topology_addr device_routing_web;
    ccl_device& device;

    mutable size_t rank; //TODO
    mutable size_t size; //TODO
};

} // namespace native
