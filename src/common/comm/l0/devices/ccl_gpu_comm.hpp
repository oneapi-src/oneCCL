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

#include "common/comm/l0/devices/ccl_gpu_base_comm.hpp"

namespace native {
class ccl_virtual_gpu_comm;
class ccl_gpu_comm : public ccl_gpu_base_comm<ccl_gpu_comm, gpu_types::REAL_GPU>,
                     public module_loader<ccl_gpu_comm> {
public:
    using base = ccl_gpu_base_comm<ccl_gpu_comm, gpu_types::REAL_GPU>;
    using base::comm_rank_t;
    using impl_t = ccl_gpu_comm;

    template <ccl_coll_type algo_type,
              ccl::device_group_split_type group,
              ccl::device_topology_type mode>
    using gpu_module_t = device_coll_module<algo_type, group, mode>;

    template <ccl_coll_type algo_type,
              ccl::device_group_split_type group,
              ccl::device_topology_type mode,
              class native_data_type>
    using gpu_kernel_t =
        typename gpu_module_t<algo_type, group, mode>::template kernel<native_data_type>;

    using supported_modules = supported_device_modules<device_coll_module>;

    static constexpr const char* name_impl() {
        return "REAL_GPU";
    }

    ccl_gpu_comm(ccl_device& assigned_device, comm_rank_t group_rank_idx);
    ~ccl_gpu_comm() = default;

    template <ccl_coll_type module_type,
              ccl::device_group_split_type group_id,
              ccl::device_topology_type class_id>
    gpu_module_t<module_type, group_id, class_id>& get_gpu_module() {
        auto& ptr =
            base::template get_gpu_module_unsafe<module_type, group_id, class_id, gpu_module_t>(
                registered_modules);
        assert(ptr);
        return *ptr;
    }

    template <ccl_coll_type module_type,
              ccl::device_group_split_type group_id,
              ccl::device_topology_type class_id>
    std::shared_ptr<gpu_module_t<module_type, group_id, class_id>> get_gpu_module_ptr() {
        return base::template get_gpu_module_unsafe<module_type, group_id, class_id, gpu_module_t>(
            registered_modules);
    }

    std::string to_string_impl() const;

    template <ccl_coll_type module_type,
              ccl::device_group_split_type group_id,
              ccl::device_topology_type class_id,
              class native_data_type>
    gpu_kernel_t<module_type, group_id, class_id, native_data_type>& get_gpu_kernel() {
        auto& ptr = get_gpu_module<module_type, group_id, class_id>();
        if (not std::is_same<native_data_type, float>::value) {
            throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + "Only float is supported");
        }
        return ptr.template get_main_function<native_data_type>();
    }

    template <class native_data_type,
              ccl::device_group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id, native_data_type>& register_entry(
        gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr = get_comm_data<group_id, class_id>();

        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());
        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id, native_data_type>();
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size); //threads count!!!
        return main_func;
    }

    template <ccl_coll_type module_type,
              ccl::device_group_split_type group_id,
              ccl::device_topology_type class_id>
    std::string create_module_impl(const ze_module_desc_t& module_data) {
        bool ret;
        ze_module_handle_t handle;
        std::string descr;

        size_t module_hash_val = module_hash(module_type, group_id, class_id);
        LOG_DEBUG("Module hash for \"",
                  ccl_coll_type_to_str(module_type),
                  "\", \"",
                  ::to_string(group_id),
                  "\", \"",
                  ::to_string(class_id),
                  "\", is: ",
                  module_hash_val);
        std::tie(ret, handle, descr) = create_module_handle(module_data, module_hash_val);
        if (!ret) {
            std::string err_str;
            {
                std::stringstream str;
                ccl_logger::format(str,
                                   "Cannot create module for:",
                                   name_impl(),
                                   " on device: ",
                                   device.get_device_properties().deviceId,
                                   ", error: ",
                                   descr);
                err_str = str.str();
            }
            LOG_ERROR(err_str);
            throw ccl::ccl_error(err_str);
        }
        std::get<class_id>(std::get<group_id>(std::get<module_type>(registered_modules)))
            .reset(new gpu_module_t<module_type, group_id, class_id>(handle));
        return descr;
    }

    void register_virtual_gpu(ccl_virtual_gpu_comm* gpu);
    size_t get_virtual_gpu_count() const {
        return registered_virtual_gpu_count;
    }

protected:
    supported_modules registered_modules;
    size_t registered_virtual_gpu_count = 0;

private:
    std::tuple<bool, ze_module_handle_t, std::string> create_module_handle(
        const ze_module_desc_t& descr,
        size_t hash);
};

} // namespace native
