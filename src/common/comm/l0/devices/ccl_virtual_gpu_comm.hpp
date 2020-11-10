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

    template <ccl_coll_type algo_type,
              ccl::group_split_type group,
              ccl::device_topology_type mode,
              class native_data_type>
    using gpu_kernel_t =
        typename gpu_module_t<algo_type, group, mode>::template kernel<native_data_type>;

    using supported_modules = supported_device_modules<virtual_device_coll_module>;

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
              ccl::device_topology_type class_id,
              class native_data_type>
    gpu_kernel_t<module_type, group_id, class_id, native_data_type>& get_gpu_kernel() {
        auto& ptr =
            base::template get_gpu_module_unsafe<module_type, group_id, class_id, gpu_module_t>(
                registered_modules);
        assert(ptr);
        return ptr->template get_main_function<native_data_type>();
    }

    template <class native_data_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id, native_data_type>& register_entry(
        gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr = get_comm_data<group_id, class_id>();
        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());

        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id, native_data_type>();
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

        std::get<utils::enum_to_underlying(class_id)>(
            std::get<utils::enum_to_underlying(group_id)>(
                std::get<module_type>(registered_modules)))
            .reset(new gpu_module_t<module_type, group_id, class_id>(real_kernel));
        return { "virtual module" };
    }

private:
    ccl_gpu_comm& real_gpu_comm;
    supported_modules registered_modules;
};
} // namespace native
