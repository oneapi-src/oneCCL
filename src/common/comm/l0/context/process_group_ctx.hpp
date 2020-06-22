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
#include "common/comm/l0/context/thread_group_ctx.hpp"


namespace native
{
struct device_storage;

template<ccl::device_topology_type>
struct device_community;

struct allied_process_group_scheduler;
//TODO separate class on two: context & process device requestor
struct process_group_context :
        scale_up_ctx_specific<process_group_context>
{
    using scaleup_context_base = scale_up_ctx_specific<process_group_context>;

    friend class device_group_ring_topology;
    friend class thread_group_ring_topology;
    friend class allied_process_group_ring_topology;

    //TODO - quick fix
    static constexpr int top_to_index(ccl::device_topology_type top)
    {
        return top == ccl::device_topology_type::allied_process_group_ring ? 0 :
                        top == ccl::device_topology_type::process_group_torn_apart_ring ? 1 :
                            top == ccl::device_topology_type::a2a_allied_process_group ? 2 : -1;
    }
    using topologies = device_community_tuple_t<PROCESS_GROUP_TOPOLOGIES_DECL_LIST>;
    using topologies_storage = std::map<size_t, topologies>;
    using observable_scale_up_topologies = typename scaleup_context_base::observable_topologies<PROCESS_GROUP_TOPOLOGIES_DECL_LIST>;

    process_group_context(std::shared_ptr<ccl::communicator> communicator);
    ~process_group_context();

    bool sync_barrier(const ccl::device_indices_t& thread_device_indices,
                      ccl::context_comm_addr& comm_addr);
    bool sync_barrier(const ccl::device_mask_t& thread_device_mask,
                      ccl::context_comm_addr& comm_addr);


    std::shared_ptr<thread_group_context> get_thread_context(size_t process_id);

    template<ccl::device_topology_type topology_type>
    typename std::tuple_element<top_to_index(topology_type), topologies>::type
             get_process_topology(size_t process_id, size_t thread_id)
    {
        //return get_process_topology<ccl::topology_to_class<topology_type>()>(process_id, thread_id);
        auto it = process_device_topology.find(thread_id);
        if(it == process_device_topology.end())
        {
            LOG_ERROR("Cannot find device group for process: ", process_id, ", thread: ", thread_id, ". Empty topology");
            return {};
        }
        return std::get<top_to_index(topology_type)>(it->second);
    }

    const ccl::cluster_aggregated_device_mask_t& get_afinity_mask() const;
    const ccl::cluster_device_indices_t& get_affinity_indices() const;

    const ccl::process_aggregated_device_mask_t& get_node_afinity_mask(const ccl::host_id& host) const;
    const ccl::process_device_indices_t& get_node_afinity_indices(const ccl::host_id& host) const;

    void set_node_afinity_indices(const ccl::host_id& host, size_t rank_id,
                                  const ccl::device_indices_t& indices);

    const ccl::host_id get_host_id() const;

    std::string to_string() const;
    device_storage& get_device_storage();
    std::vector<ccl::device_indices_t> get_ipc_device_indices() const;
    static std::vector<ccl::device_indices_t> get_ipc_device_indices_for_id(size_t process_idx, ccl::process_device_indices_t node_indices);

    static void dump_cluster_affinity_mask(const ccl::cluster_aggregated_device_mask_t& mask, std::ostream& out);
    static void dump_node_aggregated_mask(const std::string& node_name, const ccl::process_aggregated_device_mask_t& mask, std::ostream& out);
    static void dump_process_mask(size_t process_id, const ccl::device_mask_t& mask, std::ostream& out);

    static void dump_cluster_affinity_indices(const ccl::cluster_device_indices_t& mask, std::ostream& out);
    static void dump_node_aggregated_indices(const std::string& node_name, const ccl::process_device_indices_t& mask, std::ostream& out);
    static void dump_process_indices(size_t process_id, const ccl::device_indices_t& mask, std::ostream& out);

    void dump_process_topologies(std::ostream& out) const;
    std::unique_ptr<allied_process_group_scheduler> scheduler_impl;


//observer subject interface implementations
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::allied_process_group_ring> val);
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::allied_process_group_ring> val);

    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::process_group_torn_apart_ring> val);
    void attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::process_group_torn_apart_ring> val);


    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::allied_process_group_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::allied_process_group_ring> val);

    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::process_group_torn_apart_ring> val);
    void invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::process_group_torn_apart_ring> val);

private:
    bool delegate_sync(const ccl::device_indices_t& thread_device_indices,
                       ccl::context_comm_addr& comm_addr);
    bool build_cluster_affinity_table(const ccl::device_indices_t& process_aggregated_device_indices);

    std::shared_ptr<ccl::communicator> get_communicator();

    std::shared_ptr<ccl::communicator> ccl_communicator;
    std::shared_ptr<thread_group_context> thread_group_ctx;
    ccl::host_id my_host_name;
    ccl::cluster_aggregated_device_mask_t global_mask;
    ccl::cluster_device_indices_t cluster_gpu_indices;
    size_t cluster_device_rank_offset;
    size_t cluster_device_size;

    std::unique_ptr<device_storage>                                     gpu_device_storage;
    topologies_storage                                                  process_device_topology;
    observable_scale_up_topologies                                      observables;

    template<ccl::device_topology_type topology_type,
             class device_t>
    void register_observer_impl(proxy_observer<ccl_gpu_scaleup_proxy<device_t>>* observer)
    {
        auto &topologu_specific_observers = std::get<top_to_index(topology_type)>(observables);
        observers_container_t<device_t>& container = std::get<device_t::type_idx()>(topologu_specific_observers);
        container.insert(observer);
    }
    size_t process_idx;         //cached
    size_t process_count;     //cached
};
}
