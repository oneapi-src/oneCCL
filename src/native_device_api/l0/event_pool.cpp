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
#include "oneapi/ccl/native_device_api/l0/event_pool.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/driver.hpp"
#include "oneapi/ccl/native_device_api/l0/platform.hpp"
#include "common/log/log.hpp"

namespace native {

// event pool
ccl_event_pool::ccl_event_pool(const ze_event_pool_desc_t& descr,
                               handle_t h,
                               owner_ptr_t&& holder,
                               context_ptr_t&& ctx)
        : base(h, std::move(holder), std::move(ctx)),
          pool_description(descr),
          allocated_event_count(0) {}

ccl_event_pool::~ccl_event_pool() {
    CCL_ASSERT(allocated_event_count.load() == 0,
               "there are in-use event objects during ccl_event_pool destruction");
}

const ze_event_pool_desc_t& ccl_event_pool::get_pool_description() const {
    return pool_description;
}

size_t ccl_event_pool::get_allocated_events() const {
    return allocated_event_count.load();
}

ccl_event_pool::event_ptr ccl_event_pool::create_event(const ze_event_desc_t& descr) {
    ze_event_handle_t event_handle;
    ze_result_t ret = zeEventCreate(get(), &descr, &event_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot execute zeEventCreate, error: " + native::to_string(ret));
    }
    event_ptr event_ret(new event(event_handle, get_ptr(), get_ctx()));
    allocated_event_count.fetch_add(1);
    return event_ret;
}

void ccl_event_pool::on_delete(ze_event_handle_t event_handle, ze_context_handle_t& ctx) {
    (void)ctx;
    ze_result_t ret = zeEventDestroy(event_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot execute zeEventDestroy, error: " + native::to_string(ret));
    }

    allocated_event_count.fetch_sub(1);
}

// Thread safe array
CCL_BE_API event_pool_array_t::context_array_accessor event_pool_array_t::access() {
    return context_array_accessor(m, event_pools);
}

CCL_BE_API event_pool_array_t::const_context_array_accessor event_pool_array_t::access() const {
    return const_context_array_accessor(m, event_pools);
}

// Thread safe context storage holder
ze_event_pool_handle_t ccl_event_pool_holder::get() {
    return nullptr;
}

std::shared_ptr<ccl_event_pool> ccl_event_pool_holder::emplace(
    const std::initializer_list<ccl_device*>& devices,
    std::shared_ptr<ccl_event_pool> pool) {
    std::unique_lock<std::mutex> lock(m); //TODO use shared lock

    if (devices.size() != 0) {
        for (ccl_device* d : devices) {
            event_pool_array_t& cont = contexts_pool[d];
            auto acc = cont.access();
            acc.get().push_back(pool);
        }
    }
    else {
        event_pool_array_t& cont = contexts_pool[nullptr];
        auto acc = cont.access();
        acc.get().push_back(pool);
    }
    return pool;
}

CCL_BE_API std::vector<std::shared_ptr<ccl_event_pool>>
ccl_event_pool_holder::get_event_pool_storage(std::initializer_list<ccl_device*> devices) {
    return static_cast<const ccl_event_pool_holder*>(this)->get_event_pool_storage(devices);
}

CCL_BE_API std::vector<std::shared_ptr<ccl_event_pool>>
ccl_event_pool_holder::get_event_pool_storage(std::initializer_list<ccl_device*> devices) const {
    using pool_array = std::vector<std::shared_ptr<ccl_event_pool>>;
    pool_array shared_pool;

    std::unique_lock<std::mutex> lock(m); //TODO use simple shared lock

    if (devices.size() == 0) {
        auto it = contexts_pool.find(nullptr);
        if (it != contexts_pool.end()) {
            shared_pool = it->second.access().get();
        }
    }
    else {
        for (ccl_device* d : devices) {
            auto it = contexts_pool.find(d);
            if (it == contexts_pool.end()) {
                CCL_THROW("cannot find event_pool for device: " + d->to_string() +
                          "\nTotal contexts_pool count: " + std::to_string(contexts_pool.size()));
            }

            auto acc = it->second.access();
            auto& event_pools = acc.get();

            //find common pools for devices
            if (shared_pool.empty()) {
                // copy
                shared_pool = event_pools;
                continue;
            }

            //find intersection
            pool_array intersected;
            std::set_intersection(event_pools.begin(),
                                  event_pools.end(),
                                  shared_pool.begin(),
                                  shared_pool.end(),
                                  std::back_inserter(intersected));
            shared_pool.swap(intersected);

            // nothing to do
            if (shared_pool.empty()) {
                break;
            }
        }
    }
    return shared_pool;
}

void ccl_event_pool_holder::on_delete(ze_event_pool_handle_t pool_handle,
                                      ze_context_handle_t& ctx) {
    (void)ctx;
    ze_result_t ret = zeEventPoolDestroy(pool_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot execute zeEventPoolDestroy, error: " + native::to_string(ret));
    }
}
} // namespace native
