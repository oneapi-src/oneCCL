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

namespace native {

// Base scale-up adapter for different thread devices
template <class device_t>
class ccl_gpu_scaleup_proxy
        : public ccl_gpu_base_comm<ccl_gpu_scaleup_proxy<device_t>,
                                   gpu_types::SCALE_UP_GPU_TYPES + device_t::type_idx()>,
          public proxy_observer_specific<ccl_gpu_scaleup_proxy<device_t>> {
public:
    using base = ccl_gpu_base_comm<ccl_gpu_scaleup_proxy<device_t>,
                                   gpu_types::SCALE_UP_GPU_TYPES + device_t::type_idx()>;
    using typename base::comm_rank_t;

    using impl_t = device_t;

    using proxy_base = proxy_observer_specific<ccl_gpu_scaleup_proxy<device_t>>;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_module_t =
        typename device_t::template gpu_module_t<algo_type, group, mode>; //same as in-process GPU

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using kernel_class_t = typename gpu_module_t<algo_type, group, mode>::main_class;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_kernel_t = typename kernel_class_t<algo_type, group, mode>::kernel_t;

    //using ctx_ptr = std::weak_ptr<scale_up_ctx_t>;

    static constexpr const char* name_impl() {
        return "SCALE_UP_GPU_PROXY";
    }

    ccl_gpu_scaleup_proxy(ccl_device& assigned_device,
                          typename base::comm_rank_t idx,
                          device_t& process_device)
            : base(assigned_device, idx),
              wrapped_gpu_comm(process_device) {}

    ~ccl_gpu_scaleup_proxy() = default;

    std::string to_string_impl() const {
        std::string ret(name_impl());
        ret = ret + "(" + wrapped_gpu_comm.to_string_impl() + ")";
        return ret;
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id>
    gpu_kernel_t<module_type, group_id, class_id>& get_gpu_kernel(const coll_param_gpu& params) {
        this->template invoke<group_id, class_id>();

        return wrapped_gpu_comm.template get_gpu_kernel<module_type, group_id, class_id>(params);
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    topology_addr<group_id, class_id> get_comm_data() const {
        return wrapped_gpu_comm.template get_comm_data<group_id, class_id>();
    }

    template <ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry,
              class = typename std::enable_if<group_id == ccl::group_split_type::cluster>::type>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id>& register_entry(gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr = get_comm_data<group_id, class_id>();
        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());

        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id>(entry.get_params());
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size);
        return main_func;
    }

private:
    device_t& wrapped_gpu_comm;
};

//specialization for mix class NUMA
template <class device_t>
class ccl_numa_proxy;

template <class device_t>
class ccl_gpu_scaleup_proxy<ccl_numa_proxy<device_t>>
        : public ccl_gpu_base_comm<ccl_gpu_scaleup_proxy<ccl_numa_proxy<device_t>>,
                                   gpu_types::MIX_SCALE_UP_NUMA_TYPES + device_t::type_idx()>,
          public proxy_observer_specific<ccl_gpu_scaleup_proxy<ccl_numa_proxy<device_t>>> {
public:
    using base = ccl_gpu_base_comm<ccl_gpu_scaleup_proxy<ccl_numa_proxy<device_t>>,
                                   gpu_types::MIX_SCALE_UP_NUMA_TYPES + device_t::type_idx()>;
    using typename base::comm_rank_t;

    using impl_t = device_t;

    using proxy_base = proxy_observer_specific<ccl_gpu_scaleup_proxy<ccl_numa_proxy<device_t>>>;

    template <ccl_coll_type algo_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id>
    using gpu_module_t =
        typename device_t::template gpu_module_t<algo_type,
                                                 group_id,
                                                 class_id>; //same as in-process GPU

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using kernel_class_t = typename gpu_module_t<algo_type, group, mode>::main_class;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_kernel_t = typename kernel_class_t<algo_type, group, mode>::kernel_t;

    //using ctx_ptr = std::weak_ptr<scale_up_ctx_t>;
    using device_impl_t = ccl_numa_proxy<device_t>;

    static constexpr const char* name_impl() {
        return "MIX_SCALE_UP_NUMA";
    }

    ccl_gpu_scaleup_proxy(ccl_device& assigned_device,
                          typename base::comm_rank_t idx,
                          device_impl_t& process_device)
            : base(assigned_device, idx),
              wrapped_gpu_comm(process_device) {}

    ~ccl_gpu_scaleup_proxy() = default;

    std::string to_string_impl() const {
        std::string ret(name_impl());
        ret = ret + "(" + wrapped_gpu_comm.to_string_impl() + ")";
        return ret;
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    topology_addr<group_id, class_id> get_comm_data() const {
        return wrapped_gpu_comm.template get_comm_data<group_id, class_id>();
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id>
    gpu_kernel_t<module_type, group_id, class_id>& get_gpu_kernel(const coll_param_gpu& params) {
        this->template invoke<group_id>();
        return wrapped_gpu_comm.template get_gpu_kernel<module_type, group_id, class_id>(params);
    }

    template <ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry,
              class = typename std::enable_if<group_id == ccl::group_split_type::cluster>::type>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id>& register_entry(gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr = get_comm_data<group_id, class_id>();
        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());

        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id>(entry.get_params());
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size);
        return main_func;
    }

private:
    device_impl_t& wrapped_gpu_comm;
};
} // namespace native
