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

#include "ccl.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "scaleup_ctx_types.hpp"

class device_group_router;
namespace native
{
struct device_storage;

template<ccl::device_topology_type>
struct device_community;

template<ccl::device_topology_type type>
using device_community_ptr = std::shared_ptr<device_community<type>>;

template<ccl::device_topology_type ...types>
using device_community_tuple_t = std::tuple<device_community_ptr<types>...>;

struct device_group_scheduler;


struct device_group_context :
        scale_up_ctx_specific<device_group_context>
{
    using scaleup_context_base = scale_up_ctx_specific<device_group_context>;

    friend class device_group_ring_topology;

    //TODO - quick fix
    static constexpr int top_to_index(ccl::device_topology_type top)
    {
        return top == ccl::device_topology_type::device_group_ring ? 0 :
                        top == ccl::device_topology_type::device_group_torn_apart_ring ? 1 :
                            top == ccl::device_topology_type::a2a_device_group ? 2 : -1;
    }
    using topologies = device_community_tuple_t<DEVICE_GROUP_TOPOLOGIES_DECL_LIST>;
    using observable_scale_up_topologies = typename scaleup_context_base::observable_topologies<DEVICE_GROUP_TOPOLOGIES_DECL_LIST>;

    ccl::device_indices_t                                             device_indices;
    topologies                                                        device_topology;
    observable_scale_up_topologies                                    observables;

    template<ccl::device_topology_type topology_type>
    typename std::tuple_element<top_to_index(topology_type), topologies>::type
             get_group_topology()
    {
        return std::get<top_to_index(topology_type)>(device_topology);
    }

    ~device_group_context();

    static std::shared_ptr<device_group_context> create(const ccl::context_comm_addr& comm_addr,
                                                        const ccl::device_indices_t& group_device_ids,
                                                        device_storage& devices);
    const ccl::device_indices_t& get_group_device_indices() const;

    ccl::context_comm_addr context_addr;
    std::unique_ptr<device_group_scheduler> scheduler_impl;

    //observer subject interface implementations
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_ring> val);
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_ring> val);

    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_torn_apart_ring> val);
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_torn_apart_ring> val);


    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_torn_apart_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::device_group_torn_apart_ring> val);
private:
    device_group_context(const ccl::context_comm_addr& comm_addr,
                         const ccl::device_indices_t& device_mask);

    template<ccl::device_topology_type topology_type,
             class device_t>
    void register_observer_impl(proxy_observer<ccl_gpu_scaleup_proxy<device_t>>* observer)
    {
        auto &topologu_specific_observers = std::get<top_to_index(topology_type)>(observables);
        observers_container_t<device_t>& container = std::get<device_t::type_idx()>(topologu_specific_observers);
        container.insert(observer);
    }
};
}
