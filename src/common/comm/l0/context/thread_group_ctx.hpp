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
#include "common/comm/l0/context/device_group_ctx.hpp"
#include "common/log/log.hpp"

namespace native
{
struct device_storage;

template<ccl::device_topology_type>
struct device_community;

struct thread_group_scheduler;
struct thread_group_context :
        scale_up_ctx_specific<thread_group_context>
{
    using scaleup_context_base = scale_up_ctx_specific<thread_group_context>;

    friend class device_group_ring_topology;
    friend class thread_group_ring_topology;
    friend class allied_process_group_ring_topology;

    //TODO - quick fix
    static constexpr int top_to_index(ccl::device_topology_type top)
    {
        return top == ccl::device_topology_type::thread_group_ring ? 0 :
                        top == ccl::device_topology_type::thread_group_torn_apart_ring ? 1 :
                            top == ccl::device_topology_type::a2a_thread_group ? 2 : -1;
    }

    using topologies = device_community_tuple_t<THREAD_GROUP_TOPOLOGIES_DECL_LIST>;
    using topologies_storage = std::map<size_t, topologies>;

    using device_group_ctx_ptr = std::shared_ptr<device_group_context>;
    using device_group_ctx_storage = std::map<size_t, device_group_ctx_ptr>;

    using observable_scale_up_topologies = typename scaleup_context_base::observable_topologies<THREAD_GROUP_TOPOLOGIES_DECL_LIST>;

    ~thread_group_context();
    bool sync_barrier(const ccl::device_indices_t& thread_device_mask,
                      ccl::context_comm_addr& comm_addr,
                      device_storage& devices);

    const ccl::process_device_indices_t& get_thread_group_device_indices() const;
    const ccl::device_indices_t& get_device_group_indices(size_t thread_id) const;

    template<ccl::device_topology_type topology_type>
    typename std::tuple_element<top_to_index(topology_type), topologies>::type
             get_thread_topology(size_t thread_id)
    {
         auto it = thread_device_topology.find(thread_id);
        if(it == thread_device_topology.end())
        {
            LOG_ERROR("Cannot find device group for thread: ", thread_id, ". Empty topology");
            return {};
        }
        return std::get<top_to_index(topology_type)>(it->second);
    }

    device_group_ctx_ptr get_device_group_ctx(size_t thread_id);

    std::unique_ptr<thread_group_scheduler> scheduler_impl;

    void dump_thread_topologies(std::ostream& out) const;

    //observer subject interface implementations
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val);
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val);

    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val);
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val);


    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val);
private:

    ccl::process_device_indices_t                                       per_thread_indices;
    device_group_ctx_storage                                            thread_device_group_ctx;
    topologies_storage                                                  thread_device_topology;
    observable_scale_up_topologies                                      observables;
    void aggregate_device_indices(size_t thread_id, const ccl::device_indices_t& new_indices);

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
