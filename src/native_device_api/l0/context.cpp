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
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"
#include "oneapi/ccl/native_device_api/l0/base_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/driver.hpp"
#include "oneapi/ccl/native_device_api/l0/platform.hpp"
#include "common/log/log.hpp"

namespace native {

ccl_context::ccl_context(handle_t h, owner_ptr_t&& platform)
        : base(h, std::move(platform), std::weak_ptr<ccl_context>{}) {}

CCL_BE_API const ze_host_mem_alloc_desc_t& ccl_context::get_default_host_alloc_desc() {
    static const ze_host_mem_alloc_desc_t common{
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
    };
    return common;
}

CCL_BE_API void* ccl_context::host_alloc_memory(size_t bytes_count,
                                                size_t alignment,
                                                const ze_host_mem_alloc_desc_t& host_descr) {
    void* out_ptr = nullptr;
    ze_result_t ret = zeMemAllocHost(handle, &host_descr, bytes_count, alignment, &out_ptr);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot allocate host memory, error: " + std::to_string(ret));
    }
    return out_ptr;
}

CCL_BE_API void ccl_context::host_free_memory(void* mem_handle) {
    if (!mem_handle) {
        return;
    }

    if (zeMemFree(handle, mem_handle) != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot release host memory");
    }
}

CCL_BE_API std::string ccl_context::to_string() const {
    std::stringstream ss;
    ss << handle;
    return ss.str();
}

// Thread safe array
CCL_BE_API context_array_t::context_array_accessor context_array_t::access() {
    return context_array_accessor(m, contexts);
}

CCL_BE_API context_array_t::const_context_array_accessor context_array_t::access() const {
    return const_context_array_accessor(m, contexts);
}

// Thread safe context storage holder
ze_context_handle_t ccl_context_holder::get() {
    return nullptr;
}

std::shared_ptr<ccl_context> ccl_context_holder::emplace(ccl_device_driver* key,
                                                         std::shared_ptr<ccl_context>&& ctx) {
    std::unique_lock<std::mutex> lock(m); //TODO use shared lock

    context_array_t& cont = drivers_context[key];
    auto acc = cont.access();
    acc.get().push_back(std::move(ctx));
    return acc.get().back();
}

CCL_BE_API context_array_t& ccl_context_holder::get_context_storage(ccl_device_driver* driver) {
    std::unique_lock<std::mutex> lock(m); //TODO use shared lock with upgrade concept
    context_array_t& cont = drivers_context[driver];
    return cont;
}

CCL_BE_API const context_array_t& ccl_context_holder::get_context_storage(
    const ccl_device_driver* driver) const {
    std::unique_lock<std::mutex> lock(m); //TODO use simple shared lock
    auto it = drivers_context.find(driver);
    if (it == drivers_context.end()) {
        CCL_THROW("cannot find context for driver: " + driver->to_string() +
                  "\nTotal driver_context count: " + std::to_string(drivers_context.size()));
    }
    return it->second;
}
} // namespace native
