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
#include "common/global/global.hpp"
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/ze/ze_event_manager.hpp"

using namespace ccl;
using namespace ze;

event_pool::event_pool(ze_context_handle_t context)
        : context{ context },
          pool_desc{ event_manager::get_default_event_pool_desc() },
          event_desc{ event_manager::get_default_event_desc() } {}

event_pool::event_pool(ze_context_handle_t context, const ze_event_pool_desc_t& pool_desc)
        : context{ context },
          pool_desc{ pool_desc },
          event_desc{ event_manager::get_default_event_desc() } {}

event_pool::event_pool(ze_context_handle_t context,
                       const ze_event_pool_desc_t& pool_desc,
                       const ze_event_desc_t& event_desc)
        : context{ context },
          pool_desc{ pool_desc },
          event_desc{ event_desc } {}

event_pool::~event_pool() {
    clear();
}

event_pool::operator ze_event_pool_handle_t() const {
    return pool;
}

ze_event_handle_t event_pool::create_event() {
    ze_event_desc_t desc = event_desc;
    return create_event(desc);
}

ze_event_handle_t event_pool::create_event(ze_event_desc_t& desc) {
    create_pool();
    ze_event_handle_t event{};
    CCL_THROW_IF_NOT(size() < capacity());
    desc.index = events.size();
    ZE_CALL(zeEventCreate, (pool, &desc, &event));
    events.push_back(event);

    return event;
}

size_t event_pool::size() const {
    return events.size();
}

size_t event_pool::capacity() const {
    return pool_desc.count;
}

void event_pool::reset() {
    for (auto& event : events) {
        ZE_CALL(zeEventHostReset, (event));
    }
}

void event_pool::clear() {
    if (pool) {
        for (auto& event : events) {
            ZE_CALL(zeEventDestroy, (event));
        }
        events.clear();
        ccl::global_data::get().ze_cache->push(worker_idx, context, pool_desc, pool);
        pool = {};
    }
}

void event_pool::create_pool() {
    if (!pool) {
        ccl::global_data::get().ze_cache->get(worker_idx, context, pool_desc, &pool);
    }
}

event_manager::event_manager(ze_context_handle_t context) : context{ context } {
    CCL_THROW_IF_NOT(context, "no context");
}

event_manager::event_manager(const ccl_stream* stream) {
    CCL_THROW_IF_NOT(stream, "no stream");
    CCL_THROW_IF_NOT(stream->get_backend() == utils::get_level_zero_backend(), "no ze backend");
    context = stream->get_ze_context();
}

event_manager::~event_manager() {
    clear();
}

ze_event_handle_t event_manager::create(ze_event_desc_t desc) {
    return create(1, desc).front();
}

std::vector<ze_event_handle_t> event_manager::create(size_t count, ze_event_desc_t desc) {
    std::vector<ze_event_handle_t> events(count);
    if (count <= 0) {
        return events;
    }

    if (pools.empty() || (pools.back().size() + count) > pools.back().capacity()) {
        add_pool();
    }

    for (auto& event : events) {
        // TODO: place add_pool
        event = pools.back().create_event(desc);
    }

    return events;
}

void event_manager::reset() {
    for (auto& pool : pools) {
        pool.reset();
    }
}

void event_manager::clear() {
    for (auto& pool : pools) {
        pool.clear();
    }
    pools.clear();
}

ze_event_desc_t event_manager::get_default_event_desc() {
    ze_event_desc_t desc{ default_event_desc };
    desc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
    desc.wait = ZE_EVENT_SCOPE_FLAG_DEVICE;
    return desc;
}

ze_event_pool_desc_t event_manager::get_default_event_pool_desc() {
    ze_event_pool_desc_t desc{ default_event_pool_desc };
    desc.count = default_pool_size;
    if (global_data::env().enable_kernel_profile) {
        desc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    }
    return desc;
}

event_pool* event_manager::add_pool(ze_event_pool_desc_t pool_desc, ze_event_desc_t event_desc) {
    pools.emplace_back(context, pool_desc, event_desc);
    return &pools.back();
}
