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

#include "common/comm/l0/context/scaling_ctx/numa_ctx.hpp"

namespace native {
struct device_storage;
struct thread_group_scheduler;

struct thread_group_context : numa_ctx<thread_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST> {
    using scaling_context_base =
        numa_ctx<thread_group_context, SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;

    friend class device_group_ring_topology;
    friend class thread_group_ring_topology;
    friend class allied_process_group_ring_topology;

    static constexpr ccl::device_group_split_type group_id() {
        return ccl::device_group_split_type::process;
    }

    using topologies = device_group_community_holder<ccl::device_group_split_type::process,
                                                     SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>;
    using topologies_storage = std::map<size_t, topologies>;
    using device_group_ctx_ptr = std::shared_ptr<device_group_context>;
    using device_group_ctx_storage = std::map<size_t, device_group_ctx_ptr>;

    ~thread_group_context();
    bool sync_barrier(const ccl::device_indices_t& thread_device_mask,
                      ccl::context_comm_addr& comm_addr,
                      device_storage& devices);

    const ccl::process_device_indices_t& get_thread_group_device_indices() const;
    const ccl::device_indices_t& get_device_group_indices(size_t thread_id) const;

    template <ccl::device_topology_type class_id>
    typename std::tuple_element<class_id, typename topologies::device_topologies_t>::type&
    get_thread_topology(size_t thread_id) {
        auto it = thread_device_topology.find(thread_id);
        if (it == thread_device_topology.end()) {
            LOG_ERROR("Cannot find device group for thread: ", thread_id, ". Empty topology");
            static
                typename std::tuple_element<class_id,
                                            typename topologies::device_topologies_t>::type empty;
            return empty;
        }
        return it->second.get_community<class_id>();
    }

    device_group_ctx_ptr get_device_group_ctx(size_t thread_id);

    std::unique_ptr<thread_group_scheduler> scheduler_impl;

    void dump_thread_topologies(std::ostream& out) const;

    scaling_context_base& get_numa_ctx();
    const scaling_context_base& get_numa_ctx() const;

private:
    ccl::process_device_indices_t per_thread_indices;
    device_group_ctx_storage thread_device_group_ctx;
    topologies_storage thread_device_topology;

    void aggregate_device_indices(size_t thread_id, const ccl::device_indices_t& new_indices);
};
} // namespace native
