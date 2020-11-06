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
#include "common/comm/l0/context/scaling_ctx/numa_ctx.hpp"
#include "common/comm/l0/context/scaling_ctx/scale_up_ctx.hpp"
#include "common/comm/l0/context/scaling_ctx/scale_out_ctx.hpp"

#include "common/comm/l0/topology/topology_declarations.hpp"
namespace ccl {
class host_communicator;
}

namespace native {
struct device_storage;

struct allied_process_group_scheduler;

//TODO separate class on two: context & process device requestor
struct process_group_context
        : numa_ctx<process_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>,
          scale_up_ctx<process_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>,
          scale_out_ctx<process_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST> {
    using numa_context_base = numa_ctx<process_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;
    using scaleup_context_base =
        scale_up_ctx<process_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;
    using scaleout_context_base =
        scale_out_ctx<process_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;

    friend class device_group_ring_topology;
    friend class thread_group_ring_topology;
    friend class cluster_group_device_creator;

    static constexpr ccl::group_split_type group_id() {
        return ccl::group_split_type::cluster;
    }

    using topologies = device_group_community_holder<ccl::group_split_type::cluster,
                                                     SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;
    using topologies_storage = std::map<size_t, topologies>;

    process_group_context(std::shared_ptr<ccl::host_communicator> communicator);
    virtual //TODO use stub
        ~process_group_context();

    bool sync_barrier(const ccl::device_indices_t& thread_device_indices,
                      ccl::context_comm_addr& comm_addr);
    bool sync_barrier(const ccl::device_mask_t& thread_device_mask,
                      ccl::context_comm_addr& comm_addr);

    std::shared_ptr<thread_group_context> get_thread_context(size_t process_id);

    template <ccl::device_topology_type class_id>
    typename std::tuple_element<class_id, typename topologies::device_topologies_t>::type&
    get_process_topology(size_t process_id, size_t thread_id) {
        auto it = process_device_topology.find(thread_id);
        if (it == process_device_topology.end()) {
            LOG_ERROR("Cannot find device group for process: ",
                      process_id,
                      ", thread: ",
                      thread_id,
                      ". Empty topology");
            static
                typename std::tuple_element<class_id,
                                            typename topologies::device_topologies_t>::type empty;
            return empty;
        }
        return it->second.get_community<class_id>();
    }

    const ccl::cluster_aggregated_device_mask_t& get_afinity_mask() const;
    const ccl::cluster_device_indices_t& get_affinity_indices() const;

    const ccl::process_aggregated_device_mask_t& get_node_afinity_mask(
        const ccl::host_id& host) const;
    const ccl::process_device_indices_t& get_node_afinity_indices(const ccl::host_id& host) const;

    void set_node_afinity_indices(const ccl::host_id& host,
                                  size_t rank_id,
                                  const ccl::device_indices_t& indices);

    const ccl::host_id get_host_id() const;

    std::string to_string() const;
    device_storage& get_device_storage();
    std::vector<ccl::device_indices_t> get_ipc_device_indices() const;
    static std::vector<ccl::device_indices_t> get_ipc_device_indices_for_id(
        size_t process_idx,
        ccl::process_device_indices_t node_indices);

    static void dump_cluster_affinity_mask(const ccl::cluster_aggregated_device_mask_t& mask,
                                           std::ostream& out);
    static void dump_node_aggregated_mask(const std::string& node_name,
                                          const ccl::process_aggregated_device_mask_t& mask,
                                          std::ostream& out);
    static void dump_process_mask(size_t process_id,
                                  const ccl::device_mask_t& mask,
                                  std::ostream& out);

    static void dump_cluster_affinity_indices(const ccl::cluster_device_indices_t& mask,
                                              std::ostream& out);
    static void dump_node_aggregated_indices(const std::string& node_name,
                                             const ccl::process_device_indices_t& mask,
                                             std::ostream& out);
    static void dump_process_indices(size_t process_id,
                                     const ccl::device_indices_t& mask,
                                     std::ostream& out);

    void dump_process_topologies(std::ostream& out) const;
    std::unique_ptr<allied_process_group_scheduler> scheduler_impl;

    numa_context_base& get_numa_ctx();
    const numa_context_base& get_numa_ctx() const;
    scaleup_context_base& get_scaleup_ctx();
    const scaleup_context_base& get_scaleup_ctx() const;
    scaleout_context_base& get_scaleout_ctx();
    const scaleout_context_base& get_scaleout_ctx() const;

    virtual /*TODO use stub*/
        void
        collect_cluster_colored_plain_graphs(
            const details::colored_plain_graph_list& send_graph,
            details::global_sorted_colored_plain_graphs& received_graphs);

private:
    bool delegate_sync(const ccl::device_indices_t& thread_device_indices,
                       ccl::context_comm_addr& comm_addr);
    bool build_cluster_affinity_table(
        const ccl::device_indices_t& process_aggregated_device_indices);

    std::shared_ptr<ccl::host_communicator> get_communicator();

    std::shared_ptr<ccl::host_communicator> ccl_communicator;
    std::shared_ptr<thread_group_context> thread_group_ctx;
    ccl::host_id my_host_name;
    ccl::cluster_aggregated_device_mask_t global_mask;
    ccl::cluster_device_indices_t cluster_gpu_indices;

    std::unique_ptr<device_storage> gpu_device_storage;
    topologies_storage process_device_topology;

    size_t process_idx; //cached
    size_t process_count; //cached
};
} // namespace native
