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

#include "common/comm/l0/devices/ccl_gpu_base_comm.hpp"
#include "common/comm/l0/devices/proxy_observer_types.hpp"
#include "common/comm/l0/context/scaling_ctx/observer_session_key.hpp"

namespace native {

//Adapter for different thread devices
template <class device_t>
class ccl_numa_proxy
        : public ccl_gpu_base_comm<ccl_numa_proxy<device_t>,
                                   gpu_types::NUMA_PROXY_GPU_TYPES + device_t::type_idx()>,
          public proxy_observer_specific<ccl_numa_proxy<device_t>> {
public:
    using base = ccl_gpu_base_comm<ccl_numa_proxy<device_t>,
                                   gpu_types::NUMA_PROXY_GPU_TYPES + device_t::type_idx()>;
    using typename base::comm_rank_t;
    using impl_t = device_t;
    using proxy_base = proxy_observer_specific<ccl_numa_proxy<device_t>>;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_module_t =
        typename device_t::template gpu_module_t<algo_type, group, mode>; //same as in-process GPU

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using kernel_class_t = typename gpu_module_t<algo_type, group, mode>::numa_class;

    template <ccl_coll_type algo_type,
              ccl::group_split_type group,
              ccl::device_topology_type mode,
              class kernel_params>
    using gpu_kernel_t =
        typename kernel_class_t<algo_type, group, mode>::template kernel_t<kernel_params>;

    static constexpr const char* name_impl() {
        return "NUMA_PROXY";
    }

    ccl_numa_proxy(ccl_device& assigned_device,
                   typename base::comm_rank_t idx,
                   device_t& process_device)
            : base(assigned_device, idx),
              wrapped_gpu_comm(process_device) {}

    ~ccl_numa_proxy() = default;

    std::string to_string_impl() const {
        std::string ret(name_impl());
        ret = ret + "(" + wrapped_gpu_comm.to_string_impl() + ")";
        return ret;
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class kernel_params>
    gpu_kernel_t<module_type, group_id, class_id, kernel_params>& get_gpu_kernel() {
        auto& ptr = wrapped_gpu_comm.template get_gpu_module<module_type, group_id, class_id>();

        using requested_class = kernel_class_t<module_type, group_id, class_id>;
        return ptr.template get_class<requested_class>().template get<kernel_params>();
    }

    template <class kernel_params,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id, kernel_params>& register_entry(
        gpu_entry& entry) {
        static_assert(group_id == ccl::group_split_type::cluster,
                      "ccl_numa_proxy available for ccl::group_split_type::cluster only");

        const topology_addr<group_id, class_id>& comm_addr =
            base::template get_comm_data<group_id, class_id>();
        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());

        using kernel_func_type = gpu_kernel_t<gpu_entry::type(), group_id, class_id, kernel_params>;
        kernel_func_type& main_func =
            get_gpu_kernel<gpu_entry::type(), group_id, class_id, kernel_params>();
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size);

        // alloc shared data structure to notify host side with device parital result
        observer::invoke_params<gpu_entry::type(), kernel_params> params = entry.get_numa_data();

        // invoke host-side context creation
        this->template invoke<group_id, class_id>(entry.get_numa_session_key(), params);

        // bind shared data to kernel
        const auto& out_ctx_params = params.get_ctx_params();
        main_func.template set_arg<typename kernel_func_type::event_prod_chunk_mem_arg>(
            out_ctx_params.numa_staged_memory->get());
        main_func.template set_arg<typename kernel_func_type::event_prod_bytes_arg>(
            out_ctx_params.staged_memory_size_counter->get());

        main_func.template set_arg<typename kernel_func_type::event_consumed_bytes_offset_arg>(
            out_ctx_params.producer_aggregated_memory_offset->get());
        main_func.template set_arg<typename kernel_func_type::event_consumed_chunk_mem_arg>(
            out_ctx_params.total_producers_aggregated_memory->get());
        main_func.template set_arg<typename kernel_func_type::event_consumed_bytes_arg>(
            out_ctx_params.total_producers_aggregated_size_counter->get());

        return main_func;
    }

private:
    device_t& wrapped_gpu_comm;
};
} // namespace native
