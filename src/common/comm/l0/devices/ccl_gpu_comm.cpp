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
#include <iostream>
#include <vector>
#include <set>
#include "common/comm/l0/devices/ccl_gpu_comm.hpp"
#include "sched/sched.hpp"
#include "sched/entry/l0/l0_entry.hpp"
#include "common/comm/l0/modules/specific_modules_source_data.hpp"

namespace native {

void cmd_list_proxy::append_kernel(ze_kernel_handle_t handle, ze_group_count_t* launch_args) {
    std::lock_guard<std::mutex> lg(comm.cmd_list_mutex);
    base::append_kernel(handle, launch_args);
}

bool cmd_list_proxy::close_and_execute(std::shared_ptr<ccl_context> ctx, ze_fence_handle_t fence) {
    auto& ref = comm.cmd_list_close_ref_count;

    if (--ref == 0) {
        LOG_DEBUG("Closing list and executing on the queue");
        // Technically close operation requires a synchronization, but due to ref count semantic we
        // do it only once even if multiple threads are executing.
        base::close_and_execute(ctx, fence);
        ref = get_init_count();
        return true;
    }
    else {
        LOG_DEBUG("Skip list close, ref count: ", ref);
        return false;
    }
}

void cmd_list_proxy::reset() {
    auto& ref = comm.cmd_list_reset_ref_count;
    if (--ref == 0) {
        base::reset();
        // Once we reset the list, set the initial value as a ref counter
        // so it can be re-used later
        ref = get_init_count();
    }
}

int cmd_list_proxy::get_init_count() const {
    return static_cast<int>(comm.registered_virtual_gpu_count + 1);
}

void fence_proxy::reset() {
    auto& ref = comm.fence_reset_ref_count;

    if (--ref == 0) {
        base::reset();
        ref = get_init_count();
    }
}

int fence_proxy::get_init_count() const {
    return static_cast<int>(comm.registered_virtual_gpu_count + 1);
}

ccl_gpu_comm::ccl_gpu_comm(ccl_device& assigned_device, comm_rank_t idx)
        : base(assigned_device, idx),
          // Count this "real" device
          cmd_list_reset_ref_count{ 1 },
          cmd_list_close_ref_count{ 1 },
          fence_reset_ref_count{ 1 } {
    auto queue_prop = ccl_device::get_default_queue_desc();
    queue_prop.ordinal = 0;
    std::shared_ptr<ccl_context> ctx;
    (void)device.get_cmd_queue(queue_prop, ctx); //default -for execution

    //compile and load modules from all sources
    load_modules(specific_modules_source_data_storage::instance());
}

std::string ccl_gpu_comm::to_string_impl() const {
    std::string ret(name());
    ret = ret + ", comm:\n" + comm_to_str() +
          ", virtual count: " + std::to_string(get_virtual_gpu_count());
    return ret;
}

void ccl_gpu_comm::register_virtual_gpu(ccl_virtual_gpu_comm* gpu) {
    registered_virtual_gpu_count++;
    // Increment ref counters each time we register a virtual device, them must be equal to the total number of
    // virtual devices + 1 for real one
    cmd_list_reset_ref_count++;
    fence_reset_ref_count++;
    cmd_list_close_ref_count++;
}

std::tuple<bool, ze_module_handle_t, std::string> ccl_gpu_comm::create_module_handle(
    const ze_module_desc_t& descr,
    size_t hash) {
    std::tuple<bool, ze_module_handle_t, std::string> ret{ true, nullptr, "" };
    std::shared_ptr<ccl_context> ctx;

    native::ccl_device::device_module_ptr mod;
    try {
        mod = device.create_module(descr, hash, ctx);
        std::get<1>(ret) = mod->get();
    }
    catch (const std::exception& ex) {
        std::get<0>(ret) = false;
        std::get<2>(ret) = ex.what();
    }

    return ret;
}

cmd_list_proxy ccl_gpu_comm::get_cmd_list(std::shared_ptr<ccl_context> ctx,
                                          const ze_command_list_desc_t& properties) {
    auto& cmd_list = device.get_cmd_list(ctx, properties);
    // TODO: add dynamic dispatch in case we don't have any registered virtual devices?
    return cmd_list_proxy(device, cmd_list, *this);
}

fence_proxy ccl_gpu_comm::get_fence(const ccl_device::device_queue& cmd_queue,
                                    std::shared_ptr<ccl_context> ctx) {
    auto& fence = device.get_fence(cmd_queue, ctx);
    return fence_proxy(device, fence, *this);
}

} // namespace native
