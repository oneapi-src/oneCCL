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
#include "common/comm/l0/devices/ccl_gpu_base_comm.hpp"

namespace native {

ze_command_list_handle_t cmd_list_proxy_base::get() {
    return cmd_list.get();
}

ze_command_list_handle_t* cmd_list_proxy_base::get_ptr() {
    return cmd_list.get_ptr();
}

// TODO: try to move these level zero calls to a common platform code(e.g. device.cpp file)
void cmd_list_proxy_base::append_kernel(ze_kernel_handle_t handle, ze_group_count_t* launch_args) {
    auto res = zeCommandListAppendLaunchKernel(get(), handle, launch_args, nullptr, 0, nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        LOG_ERROR("zeCommandListAppendLaunchKernel failed, error: ", native::to_string(res));
        throw std::runtime_error("zeCommandListAppendLaunchKernel failed");
    }
}

void cmd_list_proxy_base::close_and_execute(std::shared_ptr<ccl_context> ctx,
                                            ze_fence_handle_t fence) {
    auto res = zeCommandListClose(get());
    if (res != ZE_RESULT_SUCCESS) {
        LOG_ERROR("zeCommandListClose failed, error: ", native::to_string(res));
        throw std::runtime_error("zeCommandListClose failed");
    }

    auto& cmd_queue = device.get_cmd_queue(ccl_device::get_default_queue_desc(), ctx);
    LOG_INFO("Execute list:", cmd_list.get(), ", queue: ", cmd_queue.get(), ", go to submit entry");
    res = zeCommandQueueExecuteCommandLists(cmd_queue.get(), 1, get_ptr(), fence);
    if (res != ZE_RESULT_SUCCESS) {
        throw ccl::exception(std::string("cannot execute command list, error: ") +
                             std::to_string(res));
    }
}

void cmd_list_proxy_base::reset() {
    auto res = zeCommandListReset(get());
    if (res != ZE_RESULT_SUCCESS) {
        LOG_ERROR("zeCommandListReset failed, error: ", native::to_string(res));
        throw std::runtime_error("zeCommandListReset failed");
    }
}

ze_fence_handle_t fence_proxy_base::get() const {
    return fence.get();
}

ze_result_t fence_proxy_base::query_status() const {
    auto res = zeFenceQueryStatus(get());
    // TODO: Should we return some other codes?
    if (res != ZE_RESULT_SUCCESS && res != ZE_RESULT_NOT_READY) {
        LOG_ERROR("zeFenceQueryStatus failed, error: ", native::to_string(res));
        throw std::runtime_error("zeFenceQueryStatus failed");
    }

    return res;
}

void fence_proxy_base::reset() {
    auto res = zeFenceReset(get());
    if (res != ZE_RESULT_SUCCESS) {
        LOG_ERROR("zeFenceReset failed, error: ", native::to_string(res));
        throw std::runtime_error("zeFenceReset failed");
    }
}

} // namespace native
