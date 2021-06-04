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

#include "common/comm/l0/devices/ccl_gpu_comm.hpp"

namespace native {

class ccl_virtual_gpu_comm : public ccl_gpu_base_comm<ccl_virtual_gpu_comm, gpu_types::VIRTUAL_GPU>,
                             public module_loader<ccl_virtual_gpu_comm> {
public:
    using base = ccl_gpu_base_comm<ccl_virtual_gpu_comm, gpu_types::VIRTUAL_GPU>;
    using base::comm_rank_t;

    using impl_t = ccl_virtual_gpu_comm;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_module_t = virtual_device_coll_module<algo_type, group, mode>;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using kernel_class_t = typename gpu_module_t<algo_type, group, mode>::main_class;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_kernel_t = typename kernel_class_t<algo_type, group, mode>::kernel_t;

    using supported_modules = supported_device_modules<gpu_module_t>;

    static constexpr const char* name_impl() {
        return "VIRTUAL_GPU";
    }

    std::string to_string_impl() const;

    ccl_virtual_gpu_comm(ccl_device& device, comm_rank_t idx, ccl_gpu_comm& real_gpu);
    ~ccl_virtual_gpu_comm() = default;

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id>
    topology_addr<group_id, class_id> get_real_comm_data() const {
        return real_gpu_comm.get_comm_data<group_id, class_id>();
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id>
    gpu_module_t<module_type, group_id, class_id>& get_gpu_module() {
        auto& ptr =
            base::template get_gpu_module_unsafe<module_type, group_id, class_id, gpu_module_t>(
                registered_modules);
        assert(ptr);
        return *ptr;
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id>
    gpu_kernel_t<module_type, group_id, class_id>& get_gpu_kernel(const coll_param_gpu& params) {
        auto& ptr = get_gpu_module<module_type, group_id, class_id>();

        using requested_class = kernel_class_t<module_type, group_id, class_id>;
        return ptr.template get_class<requested_class>().get(params);
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id, class gpu_entry>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id>& register_entry(gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr = get_comm_data<group_id, class_id>();
        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());

        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id>(entry.get_params());
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size);
        return main_func;
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id>
    std::string create_module_impl(const ze_module_desc_t& module_data) {
        //virtual based on real
        auto real_kernel = real_gpu_comm.get_gpu_module_ptr<module_type, group_id, class_id>();

        std::get<::utils::enum_to_underlying(class_id)>(
            std::get<::utils::enum_to_underlying(group_id)>(
                std::get<module_type>(registered_modules)))
            .reset(new gpu_module_t<module_type, group_id, class_id>(real_kernel));
        return { "virtual module" };
    }

    cmd_list_proxy get_cmd_list(std::shared_ptr<ccl_context> ctx,
                                const ze_command_list_desc_t& properties);

    fence_proxy get_fence(const ccl_device::device_queue& cmd_queue,
                          std::shared_ptr<ccl_context> ctx);

private:
    ccl_gpu_comm& real_gpu_comm;
    supported_modules registered_modules;
};
} // namespace native
