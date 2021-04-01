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

// Proxy classes to encapsulate command list and fence handles. The extend the base classes to adding
// thread-safety and refernece-counting semantic for L0 calls. The main purpose of this is to enable
// multiple device emulation which is done by using "real" and "virtual" devices while using the same
// underlying command list and command queue for the same hardware device
// (this is sort of a workaround for a L0 limitation since submitting kernels into separate multiple
// queues on the same device doesn't provide forward progress guarantee required by our kernels)
// So reference-counting semantic allows to provide a simple interface and enforce the correct
// order of L0 calls(i.e. some calls should be called only once while others - multiple times per logical device)
//
// They require an external state stored in ccl_gpu_comm(mutex and several atomic variables for reference
// counting). The initial value of these ref counts is equal to the number of attached "virtual" devices + 1 for
// "real" one(e.g. every time we register a "virtual" device the value is incremented by 1). On each call(e.g.
// reset) we decrement the value and once it's 0, execute it and them restore to the same initial value, so
// the state can be reused for further collective launches.
class cmd_list_proxy : public cmd_list_proxy_base {
private:
    using base = cmd_list_proxy_base;
    ccl_gpu_comm& comm;

public:
    cmd_list_proxy(ccl_device& device, ccl_device::device_cmd_list& cmd_list, ccl_gpu_comm& comm)
            : base(device, cmd_list),
              comm{ comm } {}

    cmd_list_proxy(const cmd_list_proxy& other)
            : base(other.device, other.cmd_list),
              comm{ other.comm } {}

    cmd_list_proxy& operator=(const cmd_list_proxy& other) = delete;

    void append_kernel(ze_kernel_handle_t handle, ze_group_count_t* launch_args);
    bool close_and_execute(std::shared_ptr<ccl_context> ctx, ze_fence_handle_t fence);
    void reset();

private:
    int get_init_count() const;
};

class fence_proxy : public fence_proxy_base {
private:
    using base = fence_proxy_base;

    ccl_gpu_comm& comm;

public:
    fence_proxy(ccl_device& device, ccl_device::device_queue_fence& fence, ccl_gpu_comm& comm)
            : base{ device, fence },
              comm{ comm } {}

    fence_proxy(const fence_proxy& other) : base{ other.device, other.fence }, comm{ other.comm } {}
    fence_proxy& operator=(const fence_proxy& other) = delete;

    void reset();

private:
    int get_init_count() const;
};

class ccl_virtual_gpu_comm;
class ccl_gpu_comm : public ccl_gpu_base_comm<ccl_gpu_comm, gpu_types::REAL_GPU>,
                     public module_loader<ccl_gpu_comm> {
public:
    using base = ccl_gpu_base_comm<ccl_gpu_comm, gpu_types::REAL_GPU>;
    using base::comm_rank_t;
    using impl_t = ccl_gpu_comm;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using gpu_module_t = device_coll_module<algo_type, group, mode>;

    template <ccl_coll_type algo_type, ccl::group_split_type group, ccl::device_topology_type mode>
    using kernel_class_t = typename gpu_module_t<algo_type, group, mode>::main_class;

    template <ccl_coll_type algo_type,
              ccl::group_split_type group,
              ccl::device_topology_type mode,
              class kernel_params>
    using gpu_kernel_t =
        typename kernel_class_t<algo_type, group, mode>::template kernel_t<kernel_params>;

    using supported_modules = supported_device_modules<gpu_module_t>;

    static constexpr const char* name_impl() {
        return "REAL_GPU";
    }

    ccl_gpu_comm(ccl_device& assigned_device, comm_rank_t group_rank_idx);
    ~ccl_gpu_comm() = default;

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
    std::shared_ptr<gpu_module_t<module_type, group_id, class_id>> get_gpu_module_ptr() {
        return base::template get_gpu_module_unsafe<module_type, group_id, class_id, gpu_module_t>(
            registered_modules);
    }

    std::string to_string_impl() const;

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class kernel_params>
    gpu_kernel_t<module_type, group_id, class_id, kernel_params>& get_gpu_kernel() {
        auto& ptr = get_gpu_module<module_type, group_id, class_id>();

        using requested_class = kernel_class_t<module_type, group_id, class_id>;
        return ptr.template get_class<requested_class>().template get<kernel_params>();
    }

    template <class kernel_params,
              ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class gpu_entry>
    gpu_kernel_t<gpu_entry::type(), group_id, class_id, kernel_params>& register_entry(
        gpu_entry& entry) {
        const topology_addr<group_id, class_id>& comm_addr = get_comm_data<group_id, class_id>();

        LOG_DEBUG("entry: ", gpu_entry::class_name(), " registered on: ", comm_addr.to_string());
        auto& main_func = get_gpu_kernel<gpu_entry::type(), group_id, class_id, kernel_params>();
        main_func.set_rank(comm_addr.rank);
        main_func.set_size(comm_addr.size); //threads count!!!
        return main_func;
    }

    template <ccl_coll_type module_type,
              ccl::group_split_type group_id,
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
            throw ccl::exception(err_str);
        }
        std::get<::utils::enum_to_underlying(class_id)>(
            std::get<::utils::enum_to_underlying(group_id)>(
                std::get<module_type>(registered_modules)))
            .reset(new gpu_module_t<module_type, group_id, class_id>(handle));
        return descr;
    }

    void register_virtual_gpu(ccl_virtual_gpu_comm* gpu);
    size_t get_virtual_gpu_count() const {
        return registered_virtual_gpu_count;
    }

    cmd_list_proxy get_cmd_list(std::shared_ptr<ccl_context> ctx,
                                const ze_command_list_desc_t& properties);

    fence_proxy get_fence(const ccl_device::device_queue& cmd_queue,
                          std::shared_ptr<ccl_context> ctx);

    friend class cmd_list_proxy;
    friend class fence_proxy;

protected:
    supported_modules registered_modules;
    size_t registered_virtual_gpu_count = 0;

    std::atomic<int> cmd_list_reset_ref_count;
    std::atomic<int> cmd_list_close_ref_count;
    std::atomic<int> fence_reset_ref_count;
    std::mutex cmd_list_mutex;

private:
    std::tuple<bool, ze_module_handle_t, std::string> create_module_handle(
        const ze_module_desc_t& descr,
        size_t hash);
};

} // namespace native
