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
#include "common/stream/stream.hpp"
#include "sched/queue/queue.hpp"

#include "sched/entry/gpu/ze_base_entry.hpp"
#include "sched/entry/gpu/ze_cache.hpp"
#include "sched/entry/gpu/ze_call.hpp"
#include "ze_primitives.hpp"

#include <CL/sycl/backend/level_zero.hpp>

using namespace ccl;
using namespace ccl::ze;

ze_base_entry::ze_base_entry(ccl_sched *sched, ccl_comm *comm, uint32_t add_event_count)
        : sched_entry(sched),
          sched(sched),
          comm(comm),
          add_event_count(add_event_count) {
    CCL_THROW_IF_NOT(sched, "no sched");
    if (!comm) {
        comm = sched->coll_param.comm;
    }
    CCL_THROW_IF_NOT(comm, "no comm");
    comm_rank = comm->rank();
    comm_size = comm->size();
}

void ze_base_entry::init(init_mode ze_init_mode) {
    if (is_initialized) {
        return;
    }
    worker_idx = sched->queue->get_idx();

    CCL_THROW_IF_NOT(sched->coll_param.stream, "null stream");

    LOG_DEBUG("getting a native stream");
    auto native_stream = sched->coll_param.stream->get_native_stream(worker_idx);
    if (native_stream->get_backend() != sycl::backend::level_zero) {
        CCL_THROW("unsupported sycl backend");
    }

    auto sycl_device = native_stream->get_device();
    device = sycl_device.template get_native<sycl::backend::level_zero>();

    auto sycl_context = native_stream->get_context();
    context = sycl_context.template get_native<sycl::backend::level_zero>();

    /* get queue properties */
    uint32_t num_queue_groups;
    get_num_queue_groups(device, &num_queue_groups);

    ze_queue_properties_t queue_props;
    get_queues_properties(device, num_queue_groups, &queue_props);

    /* init compute queue, list */
    if (init_mode::compute & ze_init_mode) {
        LOG_DEBUG("compute init mode is enabled");
        get_comp_primitives(queue_props, comp_primitives);
        init_primitives(comp_primitives);
    }

    /* init copy queue, list */
    if (init_mode::copy & ze_init_mode) {
        LOG_DEBUG("copy init mode is enabled");
        get_copy_primitives(queue_props, copy_primitives, ze_init_mode);
        init_primitives(copy_primitives);
    }

    /* create event pool */
    event_pool_desc = default_event_pool_desc;
    event_pool_desc.count = 1 + add_event_count; // at least one event to track progress
    global_data::get().ze_cache->get(worker_idx, context, event_pool_desc, &event_pool);
    LOG_DEBUG("get event pool: { max event count: ", event_pool_desc.count, " }");

    /* create event */
    ze_event_desc_t event_desc = default_event_desc;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_SUBDEVICE;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_SUBDEVICE;
    event_desc.index = 0;
    ZE_CALL(zeEventCreate, (event_pool, &event_desc, &entry_event));

    is_initialized = true;
}

void ze_base_entry::finalize() {
    if (!is_initialized) {
        return;
    }
    ZE_CALL(zeEventDestroy, (entry_event));

    /* event pool */
    global_data::get().ze_cache->push(worker_idx, context, event_pool_desc, event_pool);

    if (comp_primitives.list && comp_primitives.queue) {
        LOG_DEBUG("push from cache for compute list and queue");
        /* list */
        global_data::get().ze_cache->push(
            worker_idx, context, device, comp_primitives.list_desc, comp_primitives.list);

        /* queue */
        global_data::get().ze_cache->push(
            worker_idx, context, device, comp_primitives.queue_desc, comp_primitives.queue);
    }

    if (copy_primitives.list && copy_primitives.queue) {
        LOG_DEBUG("push from cache for copy list and queue");
        /* copy list */
        global_data::get().ze_cache->push(
            worker_idx, context, device, copy_primitives.list_desc, copy_primitives.list);

        /* copy queue */
        global_data::get().ze_cache->push(
            worker_idx, context, device, copy_primitives.queue_desc, copy_primitives.queue);
    }

    is_initialized = false;
}

void ze_base_entry::start() {
    CCL_THROW_IF_NOT(entry_event, "no entry event");
    ZE_CALL(zeEventHostReset, (entry_event));

    if (comp_primitives.list && comp_primitives.queue) {
        LOG_DEBUG("execute compute command list");
        ZE_CALL(zeCommandQueueExecuteCommandLists,
                (comp_primitives.queue, 1, &comp_primitives.list, nullptr));
    }

    if (copy_primitives.list && copy_primitives.queue) {
        LOG_DEBUG("execute copy command list");
        ZE_CALL(zeCommandQueueExecuteCommandLists,
                (copy_primitives.queue, 1, &copy_primitives.list, nullptr));
    }

    if (((global_data::env().ze_serialize_mode & ze_call::serialize_mode::block)) != 0) {
        LOG_DEBUG("wait until command lists are executed");
        if (copy_primitives.queue)
            ZE_CALL(zeHostSynchronize, (copy_primitives.queue));
        if (comp_primitives.queue)
            ZE_CALL(zeHostSynchronize, (comp_primitives.queue));
    }
}

void ze_base_entry::update() {
    ze_result_t query_status;

    if (global_data::env().kernel_debug == 0) {
        query_status = zeEventQueryStatus(entry_event);
    }
    else {
        if (copy_primitives.queue)
            query_status = zeHostSynchronize(copy_primitives.queue);
        if (comp_primitives.queue)
            query_status = zeHostSynchronize(comp_primitives.queue);
    }

    if (query_status == ZE_RESULT_SUCCESS) {
        LOG_DEBUG("command list complete");
        status = ccl_sched_entry_status_complete;
    }
    else if (query_status == ZE_RESULT_NOT_READY) {
        // just return in case if the kernel is not ready yet, will check again on the next iteration
        return;
    }
    else {
        CCL_THROW("error at zeEventQueryStatus");
    }
}

ze_command_list_handle_t ze_base_entry::get_copy_list() {
    ze_command_list_handle_t list = nullptr;
    if (copy_primitives.list) {
        list = copy_primitives.list;
        LOG_DEBUG("copy list is returned");
    }
    else {
        list = comp_primitives.list;
        LOG_DEBUG("compute list is returned");
    }
    CCL_THROW_IF_NOT(list, "command list is invalid");
    return list;
}

void ze_base_entry::get_comp_primitives(const ze_queue_properties_t &queue_props,
                                        cmd_primitives &comp_primitives) {
    uint32_t ordinal, queue_index;
    get_comp_queue_ordinal(device, queue_props, &ordinal);
    get_queue_index(queue_props, ordinal, comm_rank, &queue_index);

    comp_primitives.queue_desc.ordinal = ordinal;
    comp_primitives.queue_desc.index = queue_index;
    comp_primitives.list_desc.commandQueueGroupOrdinal = ordinal;
}

void ze_base_entry::get_copy_primitives(const ze_queue_properties_t &queue_props,
                                        cmd_primitives &copy_primitives,
                                        init_mode ze_init_mode) {
    uint32_t ordinal, queue_index;
    get_copy_queue_ordinal(device, queue_props, &ordinal);

    // TODO: index depends on rank's changing, when > 1 queues are created,
    // the index is still the same for different queues, that's the issue.
    // WA is adding optional counter, which says the order number of a queue.
    // Need to think, how we'd calculate the index for every queue.
    // Hang in case of CCL_KERNEL_1S_USE_COPY_OPS=1 CCL_ZE_COPY_ENGINE=none
    if (ze_init_mode == (init_mode::copy | init_mode::compute)) {
        get_queue_index(queue_props, ordinal, comm_rank + 1, &queue_index);
    }
    else {
        get_queue_index(queue_props, ordinal, comm_rank, &queue_index);
    }

    copy_primitives.queue_desc.ordinal = ordinal;
    copy_primitives.queue_desc.index = queue_index;
    copy_primitives.list_desc.commandQueueGroupOrdinal = ordinal;
}

void ze_base_entry::init_primitives(cmd_primitives &cmd_primitives) {
    global_data::get().ze_cache->get(
        worker_idx, context, device, cmd_primitives.queue_desc, &cmd_primitives.queue);
    LOG_DEBUG("get queue: { ordinal: ",
              cmd_primitives.queue_desc.ordinal,
              ", index: ",
              cmd_primitives.queue_desc.index,
              " }");

    global_data::get().ze_cache->get(
        worker_idx, context, device, cmd_primitives.list_desc, &cmd_primitives.list);
    LOG_DEBUG("get list: { ordinal: ", cmd_primitives.list_desc.commandQueueGroupOrdinal, " }");
}
