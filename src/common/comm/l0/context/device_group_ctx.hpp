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
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>

#include "oneapi/ccl/types.hpp"
#include "supported_topologies.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/context/scale/numa/numa_ctx.hpp"
#include "common/comm/l0/device_community_holder_impl.hpp"

class device_group_router;
namespace native {
struct device_storage;
/*
template<ccl::group_split_type>
struct device_community;

template<ccl::group_split_type type>
using device_community_ptr = std::shared_ptr<device_community<type>>;

template<ccl::group_split_type ...types>
using device_community_tuple_t = std::tuple<device_community_ptr<types>...>;
*/
struct device_group_scheduler;
/*
template<ccl::group_split_type,
         ccl::device_topology_type...>
struct device_group_community_holder;
*/
struct device_group_context : numa_ctx<device_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST> {
    using scaling_context_base =
        numa_ctx<device_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;

    friend class device_group_ring_topology;

    static constexpr ccl::group_split_type group_id() {
        return ccl::group_split_type::thread;
    }

    using topologies = device_group_community_holder<ccl::group_split_type::thread,
                                                     SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;

    ccl::device_indices_type device_indices;
    topologies device_topology;

    template <ccl::device_topology_type class_id>
    typename std::tuple_element<class_id, typename topologies::device_topologies_t>::type&
    get_group_topology() {
        return device_topology.get_community<class_id>();
    }

    ~device_group_context();

    static std::shared_ptr<device_group_context> create(
        const ccl::context_comm_addr& comm_addr,
        const ccl::device_indices_type& group_device_ids,
        device_storage& devices);
    const ccl::device_indices_type& get_group_device_indices() const;

    ccl::context_comm_addr context_addr;
    std::unique_ptr<device_group_scheduler> scheduler_impl;

    scaling_context_base& get_numa_ctx();
    const scaling_context_base& get_numa_ctx() const;

private:
    device_group_context(const ccl::context_comm_addr& comm_addr,
                         const ccl::device_indices_type& device_mask);
};
} // namespace native
