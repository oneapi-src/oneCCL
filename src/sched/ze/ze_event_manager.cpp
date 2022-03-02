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
#include <algorithm>
#include <mutex>
#include "common/global/global.hpp"
#include "common/log/log.hpp"
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
        ccl::global_data::get().ze_data->cache->push(worker_idx, context, pool_desc, pool);
        pool = {};
    }
}

void event_pool::create_pool() {
    if (!pool) {
        ccl::global_data::get().ze_data->cache->get(worker_idx, context, pool_desc, &pool);
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

    if (pools.empty()) {
        add_pool();
    }

    for (auto& event : events) {
        if (pools.back().size() >= pools.back().capacity()) {
            add_pool();
        }
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
    return desc;
}

event_pool* event_manager::add_pool(ze_event_pool_desc_t pool_desc, ze_event_desc_t event_desc) {
    pools.emplace_back(context, pool_desc, event_desc);
    return &pools.back();
}

dynamic_event_pool::dynamic_event_pool(const ccl_stream* stream) {
    CCL_THROW_IF_NOT(stream, "no stream");
    CCL_THROW_IF_NOT(stream->get_backend() == utils::get_level_zero_backend(), "no ze backend");
    context = stream->get_ze_context();
}

dynamic_event_pool::~dynamic_event_pool() {
    // we expect that all events are released by the callee, at this point there
    // must be no allocated events, otherwise this indicates an error in the event handling
    if (!event_alloc_info.empty())
        LOG_ERROR("all events are expected to be released");
    if (!event_pools.empty())
        LOG_ERROR("all event pools are expected to be released");
}

ze_event_pool_desc_t dynamic_event_pool::get_default_event_pool_desc() {
    ze_event_pool_desc_t desc{ default_event_pool_desc };
    desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    desc.count = event_pool_size;
    return desc;
}

const ze_event_pool_desc_t dynamic_event_pool::common_pool_desc = get_default_event_pool_desc();

ze_event_handle_t dynamic_event_pool::get_event() {
    std::lock_guard<std::mutex> lg(lock);

    event_info slot = {};

    if (find_free_slot(slot)) {
        return create_event(slot);
    }

    event_pool_info pool_info = {};

    // no free slot, need to allocate a new pool and create event from it
    // TODO: move to a separate function
    ccl::global_data::get().ze_data->cache->get(
        worker_idx, context, common_pool_desc, &pool_info.pool);
    pool_info.event_alloc_status.resize(event_pool_size, false);
    pool_info.num_alloc_events = 0;

    slot.pool = event_pools.insert(event_pools.end(), pool_info);
    slot.pool_idx = 0;

    return create_event(slot);
}

void dynamic_event_pool::put_event(ze_event_handle_t event) {
    std::lock_guard<std::mutex> lg(lock);

    auto it = event_alloc_info.find(event);
    CCL_THROW_IF_NOT(it != event_alloc_info.end(), "event is not from the pool");

    event_info slot = it->second;
    event_alloc_info.erase(it);

    // make sure we always release the completed event
    CCL_ASSERT(zeEventQueryStatus(event) == ZE_RESULT_SUCCESS);

    // TODO: can we just reset event instead?(e.g. keep a list of available events)
    // need to measure the possible improvement
    ZE_CALL(zeEventDestroy, (event));

    for (auto it = event_pools.begin(); it != event_pools.end(); ++it) {
        if (it == slot.pool) {
            CCL_THROW_IF_NOT(it->num_alloc_events > 0, "pool must be non-empty");
            it->num_alloc_events -= 1;
            it->event_alloc_status[slot.pool_idx] = false;

            if (it->num_alloc_events == 0) {
                auto pool_handle = it->pool;
                ccl::global_data::get().ze_data->cache->push(
                    worker_idx, context, common_pool_desc, pool_handle);
                event_pools.erase(it);
            }
            return;
        }
    }

    CCL_THROW("pool is not found");
}

bool dynamic_event_pool::find_free_slot(event_info& slot) {
    for (auto it = event_pools.begin(); it != event_pools.end(); ++it) {
        if (it->num_alloc_events < event_pool_size) {
            slot.pool = it;
            auto status_it =
                // TODO: we can potentially improve this by introducing a circular queue on
                // top of buffer of event_pool_size elements and store available indexes there
                // check whether this make sense to implement
                std::find(it->event_alloc_status.begin(), it->event_alloc_status.end(), false);
            CCL_THROW_IF_NOT(status_it != it->event_alloc_status.end(),
                             "status vector must have free slots");
            slot.pool_idx = (status_it - it->event_alloc_status.begin());

            return true;
        }
    }

    return false;
}

ze_event_handle_t dynamic_event_pool::create_event(const event_info& slot) {
    ze_event_desc_t desc{ default_event_desc };
    ze_event_handle_t event{};

    auto& pool_info = *slot.pool;

    desc.index = slot.pool_idx;
    ZE_CALL(zeEventCreate, (pool_info.pool, &desc, &event));

    pool_info.num_alloc_events++;
    CCL_THROW_IF_NOT(pool_info.num_alloc_events <= event_pool_size);

    pool_info.event_alloc_status[slot.pool_idx] = true;

    event_alloc_info.insert({ event, slot });

    return event;
}
