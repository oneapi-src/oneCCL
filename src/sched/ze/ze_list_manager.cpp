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
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/ze/ze_list_manager.hpp"

using namespace ccl;
using namespace ccl::ze;

list_manager::list_manager(ze_device_handle_t device, ze_context_handle_t context)
        : device(device),
          context(context) {
    LOG_DEBUG("create list manager");
    CCL_THROW_IF_NOT(device, "no device");
    CCL_THROW_IF_NOT(context, "no context");
    get_queues_properties(device, &queue_props);

    // Even if ze_copy_engine != ccl_ze_copy_engine_none,
    // copy queue can be created with ordinal equal comp queue ordinal,
    // it can cause deadlock for events between queues on card without blitter engine
    use_copy_queue =
        (global_data::env().ze_copy_engine != ccl_ze_copy_engine_none) && (queue_props.size() > 1);
}

list_manager::list_manager(const ccl_stream* stream)
        : list_manager(stream->get_ze_device(), stream->get_ze_context()) {}

list_manager::~list_manager() {
    clear();
}

ze_command_list_handle_t list_manager::get_comp_list(size_t worker_idx) {
    if (!comp_list) {
        comp_queue = create_queue(init_mode::compute, worker_idx);
        comp_list = create_list(comp_queue, worker_idx);
    }
    return comp_list.list;
}

ze_command_list_handle_t list_manager::get_copy_list(size_t worker_idx) {
    if (use_copy_queue) {
        if (!copy_list) {
            copy_queue = create_queue(init_mode::copy, worker_idx);
            copy_list = create_list(copy_queue, worker_idx);
        }
        return copy_list.list;
    }
    return get_comp_list(worker_idx);
}

queue_info list_manager::create_queue(init_mode mode, size_t worker_idx) {
    queue_info info{};
    uint32_t ordinal{}, queue_index{};
    if (mode == init_mode::copy) {
        LOG_DEBUG("create copy queue");
        get_copy_queue_ordinal(device, queue_props, &ordinal);
    }
    else {
        LOG_DEBUG("create comp queue");
        get_comp_queue_ordinal(device, queue_props, &ordinal);
    }
    get_queue_index(queue_props, ordinal, 0, &queue_index);

    info.desc = default_cmd_queue_desc;
    info.desc.index = queue_index;
    info.desc.ordinal = ordinal;
    info.worker_idx = worker_idx;

    global_data::get().ze_cache->get(worker_idx, context, device, info.desc, &info.queue);
    return info;
}

void list_manager::free_queue(queue_info& info) {
    if (!info)
        return;
    global_data::get().ze_cache->push(info.worker_idx, context, device, info.desc, info.queue);
    info.queue = nullptr;
}

list_info list_manager::create_list(const queue_info& queue, size_t worker_idx) {
    LOG_DEBUG("create list");
    list_info info{};
    info.desc = default_cmd_list_desc;
    info.desc.commandQueueGroupOrdinal = queue.desc.ordinal;
    info.worker_idx = worker_idx;
    global_data::get().ze_cache->get(worker_idx, context, device, info.desc, &info.list);
    return info;
}

void list_manager::free_list(list_info& info) {
    if (!info)
        return;
    global_data::get().ze_cache->push(info.worker_idx, context, device, info.desc, info.list);
    info.list = nullptr;
}

void list_manager::clear() {
    LOG_DEBUG("destroy lists and queues");
    free_list(comp_list);
    free_list(copy_list);
    free_queue(comp_queue);
    free_queue(copy_queue);
}

bool list_manager::can_use_copy_queue() const {
    return use_copy_queue;
}

void list_manager::execute_list(queue_info& queue, list_info& list) {
    if (list.list) {
        if (!list.is_closed) {
            ZE_CALL(zeCommandListClose, (list.list));
            list.is_closed = true;
        }
        ZE_CALL(zeCommandQueueExecuteCommandLists, (queue.queue, 1, &list.list, nullptr));
    }
}

void list_manager::execute() {
    if (use_copy_queue) {
        LOG_DEBUG("execute copy list");
        execute_list(copy_queue, copy_list);
    }
    LOG_DEBUG("execute comp list");
    execute_list(comp_queue, comp_list);
}
