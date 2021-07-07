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
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <cstring>
#include <sstream>

#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"
#include "oneapi/ccl/native_device_api/l0/platform.hpp"
#include "common/log/log.hpp"

namespace native {

bool event::wait(uint64_t nanosec) const {
    (void)nanosec;
    ze_result_t ret = zeEventHostSynchronize(handle, nanosec);
    if (ret != ZE_RESULT_SUCCESS && ret != ZE_RESULT_NOT_READY) {
        CCL_THROW("zeEventHostSynchronize, error: " + native::to_string(ret));
    }
    return ret == ZE_RESULT_SUCCESS;
}

ze_result_t event::status() const {
    return zeEventQueryStatus(handle);
}

void event::signal() {
    ze_result_t ret = zeEventHostSignal(handle);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("zeEventHostSignal, error: " + native::to_string(ret));
    }
}

namespace detail {
CCL_BE_API void copy_memory_sync_unsafe(void* dst,
                                        const void* src,
                                        size_t size,
                                        std::weak_ptr<ccl_device> device_weak,
                                        std::shared_ptr<ccl_context> ctx) {
    //TODO: NOT THREAD-SAFE!!!
    std::shared_ptr<ccl_device> device = device_weak.lock();
    if (!device) {
        CCL_THROW("device is null");
    }

    // create queue
    ze_command_queue_desc_t queue_description = device->get_default_queue_desc();
    //queue_description.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;   TODO may be &= for flags???

    auto& queue = device->get_cmd_queue(queue_description, ctx);

    //create list
    ze_command_list_desc_t list_description = device->get_default_list_desc();
    //list_description.flags = ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT;  TODO may be &= for flags???
    auto cmd_list = device->create_cmd_list(ctx, list_description);

    ze_result_t ret =
        zeCommandListAppendMemoryCopy(cmd_list.get(), dst, src, size, nullptr, 0, nullptr);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot append memory copy to cmd list, error: " + std::to_string(ret));
    }
    ret = zeCommandListClose(cmd_list.get());
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot close cmd list, error: " + std::to_string(ret));
    }

    //TODO: need to reinmplement
    {
        static std::mutex memory_mutex;

        std::lock_guard<std::mutex> lock(memory_mutex);
        // Execute command list in command queue
        ret = zeCommandQueueExecuteCommandLists(queue.get(), 1, cmd_list.get_ptr(), nullptr);
        if (ret != ZE_RESULT_SUCCESS) {
            CCL_THROW("cannot execute command list, error: " + std::to_string(ret));
        }

        // synchronize host and device
        ret = zeCommandQueueSynchronize(queue.get(), std::numeric_limits<uint32_t>::max());
        if (ret != ZE_RESULT_SUCCESS) {
            CCL_THROW("cannot sync queue, error: " + std::to_string(ret));
        }
    }
    // Reset (recycle) command list for new commands
    ret = zeCommandListReset(cmd_list.get());
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot reset queue, error: " + std::to_string(ret));
    }
}

CCL_BE_API void copy_memory_sync_unsafe(void* dst,
                                        const void* src,
                                        size_t size,
                                        std::weak_ptr<ccl_context> ctx_weak,
                                        std::shared_ptr<ccl_context> ctx) {
    // host copy
    memcpy(dst, src, size);
}

/*

event<ccl_device, ccl_context>
copy_memory_async_unsafe(void* dst,
                             const void* src,
                             size_t size,
                             std::weak_ptr<ccl_device> device_weak,
                             std::shared_ptr<ccl_context> ctx,
                             queue<ccl_device, ccl_context>& q);
event<ccl_device, ccl_context>
copy_memory_async_unsafe(void* dst,
                             const void* src,
                             size_t size,
                             std::weak_ptr<ccl_context> ctx_weak,
                             std::shared_ptr<ccl_context> ctx,
                             queue<ccl_device, ccl_context>& q)
{
    (void)q;
    copy_memory_sync_unsafe(dst, src, size, ctx_weak, ctx);
    event<ccl_device, ccl_context> e(h, get_ptr(), ctx);
}
*/
} // namespace detail

std::string get_build_log_string(const ze_module_build_log_handle_t& build_log) {
    ze_result_t result = ZE_RESULT_SUCCESS;

    size_t build_log_size = 0;
    result = zeModuleBuildLogGetString(build_log, &build_log_size, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        CCL_THROW("xeModuleBuildLogGetString failed: " + to_string(result));
    }

    std::vector<char> build_log_c_string(build_log_size);
    result = zeModuleBuildLogGetString(build_log, &build_log_size, build_log_c_string.data());
    if (result != ZE_RESULT_SUCCESS) {
        CCL_THROW("xeModuleBuildLogGetString failed: " + to_string(result));
    }
    return std::string(build_log_c_string.begin(), build_log_c_string.end());
}

bool command_queue_desc_comparator::operator()(const ze_command_queue_desc_t& lhs,
                                               const ze_command_queue_desc_t& rhs) const {
    return memcmp(&lhs, &rhs, sizeof(rhs)) < 0;
}

bool command_list_desc_comparator::operator()(const ze_command_list_desc_t& lhs,
                                              const ze_command_list_desc_t& rhs) const {
    return memcmp(&lhs, &rhs, sizeof(rhs)) < 0;
}

} // namespace native
