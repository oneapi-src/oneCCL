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
#include "oneapi/ccl/native_device_api/l0/context.hpp"
#include "oneapi/ccl/native_device_api/l0/base_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/driver.hpp"
#include "oneapi/ccl/native_device_api/l0/platform.hpp"

namespace native {

ccl_context::ccl_context(handle_t h, owner_ptr_t&& platform)
        : base(h, std::move(platform), std::weak_ptr<ccl_context>{}) {}

// Thread safe array
context_array_t::context_array_accessor context_array_t::access() {
    return context_array_accessor(m, contexts);
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

context_array_t& ccl_context_holder::get_context_storage(ccl_device_driver* driver) {
    std::unique_lock<std::mutex> lock(m); //TODO use shared lock
    context_array_t& cont = drivers_context[driver];
    return cont;
}
} // namespace native
