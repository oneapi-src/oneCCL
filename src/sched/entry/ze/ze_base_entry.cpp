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

#include "sched/entry/ze/ze_base_entry.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_call.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include "common/utils/sycl_utils.hpp"

using namespace ccl;
using namespace ccl::ze;

ze_base_entry::ze_base_entry(ccl_sched *sched,
                             init_mode mode,
                             ccl_comm *comm,
                             uint32_t add_event_count,
                             std::vector<ze_event_handle_t> wait_events)
        : sched_entry(sched),
          mode(mode),
          comm(comm),
          use_single_list(sched->get_memory().use_single_list),
          wait_events(wait_events) {
    if (!comm) {
        comm = sched->coll_param.comm;
    }
    CCL_THROW_IF_NOT(comm, "no comm");
    comm_rank = comm->rank();
    comm_size = comm->size();

    // we can be here in case of copy_entry which may not have ze backend here, so check it
    if (sched->coll_param.stream &&
        sched->coll_param.stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        entry_event = sched->get_memory().event_manager->create();
        sched->get_memory().ze_entries.push_back(this);
    }
    events.resize(add_event_count, nullptr);
}

ze_base_entry::~ze_base_entry() {
    finalize();
}

void ze_base_entry::init() {
    if (is_initialized) {
        return;
    }

    LOG_DEBUG("init");

    worker_idx = sched->queue->get_idx();

    CCL_THROW_IF_NOT(sched->coll_param.stream, "no stream");
    device = sched->coll_param.stream->get_ze_device();
    context = sched->coll_param.stream->get_ze_context();

    if (!use_single_list) {
        /* get queue properties */
        ze_queue_properties_t queue_props;
        get_queues_properties(device, &queue_props);

        if ((queue_props.size() == 1) && (queue_props[0].numQueues == 1)) { // magic index?
            CCL_THROW_IF_NOT(sched->coll_param.stream->get_device_family() ==
                             ccl::device_family::unknown);
            LOG_DEBUG("numQueues = 1, switch to compute init mode");
            mode = init_mode::compute;
        }

        /* init compute queue and list */
        if (init_mode::compute & mode) {
            LOG_DEBUG("compute init mode is enabled");
            get_comp_primitives(queue_props, comp_primitives);
            init_primitives(comp_primitives);
        }

        /* init copy queue and list */
        if (init_mode::copy & mode) {
            LOG_DEBUG("copy init mode is enabled");
            get_copy_primitives(queue_props, copy_primitives, mode);
            init_primitives(copy_primitives);
        }
    }

    /* create event pool */
    if (events.size() > 0) {
        event_pool_desc = default_event_pool_desc;
        event_pool_desc.count = events.size();
        if (ccl::global_data::env().enable_kernel_profile) {
            event_pool_desc.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        }
        global_data::get().ze_cache->get(worker_idx, context, event_pool_desc, &event_pool);
        LOG_DEBUG("get event pool: { max event count: ", event_pool_desc.count, " }");
    }

    append_wait_on_events();

    init_ze_hook();
    if (!use_single_list) {
        close_lists();
    }

    is_initialized = true;

    LOG_DEBUG("init completed");
}

void ze_base_entry::finalize() {
    if (!is_initialized) {
        return;
    }

    LOG_DEBUG("finalize");

    finalize_ze_hook();
    destroy_events();

    /* event pool */
    if (event_pool) {
        global_data::get().ze_cache->push(worker_idx, context, event_pool_desc, event_pool);
    }

    if (!use_single_list) {
        if (comp_primitives.list && comp_primitives.queue) {
            LOG_DEBUG("push to cache compute list and queue");
            /* list */
            global_data::get().ze_cache->push(
                worker_idx, context, device, comp_primitives.list_desc, comp_primitives.list);

            /* queue */
            global_data::get().ze_cache->push(
                worker_idx, context, device, comp_primitives.queue_desc, comp_primitives.queue);
        }

        if (copy_primitives.list && copy_primitives.queue) {
            LOG_DEBUG("push to cache copy list and queue");
            /* copy list */
            global_data::get().ze_cache->push(
                worker_idx, context, device, copy_primitives.list_desc, copy_primitives.list);

            /* copy queue */
            global_data::get().ze_cache->push(
                worker_idx, context, device, copy_primitives.queue_desc, copy_primitives.queue);
        }
    }

    is_initialized = false;

    LOG_DEBUG("finalize completed");
}

void ze_base_entry::init_entries() {
    auto &entries = sched->get_memory().ze_entries;
    if (entries.front() == this) {
        LOG_DEBUG("init ", entries.size(), " entries");
        for (auto &entry : entries) {
            entry->init();
        }
    }
}

void ze_base_entry::finalize_entries() {
    auto &entries = sched->get_memory().ze_entries;
    if (entries.back() == this) {
        LOG_DEBUG("finalize ", entries.size(), " entries");
        for (auto &entry : entries) {
            entry->finalize();
        }
    }
}

void ze_base_entry::start() {
    if (global_data::env().enable_kernel_profile) {
        sched->master_sched->get_kernel_timer().set_kernel_submit_time(
            calculate_global_time(sched->coll_param.stream->get_ze_device()));
    }

    if (use_single_list) {
        init_entries();

        if (sched->get_memory().ze_entries.front() == this) {
            sched->get_memory().list_manager->execute();
        }
    }
    else {
        init();
        reset_events();

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

    status = ccl_sched_entry_status_started;
}

bool ze_base_entry::is_event_completed(ze_event_handle_t event) {
    ze_result_t res = zeEventQueryStatus(event);
    CCL_THROW_IF_NOT(res == ZE_RESULT_SUCCESS || res == ZE_RESULT_NOT_READY,
                     "unexpected result from zeEventQueryStatus: ",
                     to_string(res));
    return (res == ZE_RESULT_SUCCESS);
}

bool ze_base_entry::is_queue_completed(ze_command_queue_handle_t queue) {
    ze_result_t res = zeHostSynchronize(queue);
    CCL_THROW_IF_NOT(res == ZE_RESULT_SUCCESS || res == ZE_RESULT_NOT_READY,
                     "unexpected result from zeHostSynchronize: ",
                     to_string(res));
    return (res == ZE_RESULT_SUCCESS);
}

void ze_base_entry::update() {
    bool complete{};

    if (global_data::env().kernel_debug == 0) {
        complete = is_event_completed(entry_event);
    }
    else {
        bool copy_q_complete{ true };
        bool comp_q_complete{ true };
        if (copy_primitives.queue)
            copy_q_complete = is_queue_completed(copy_primitives.queue);
        if (comp_primitives.queue)
            comp_q_complete = complete && is_queue_completed(comp_primitives.queue);
        complete = copy_q_complete && comp_q_complete;
    }

    if (complete) {
        LOG_DEBUG(name(), " entry complete");
        status = ccl_sched_entry_status_complete;

        if (ccl::global_data::env().enable_kernel_profile) {
            auto kernel_time = calculate_event_time(entry_event, device);

            // if we run this code, this sched must be a sub-sched of some master sched
            // so the field must be non null
            CCL_THROW_IF_NOT(sched->master_sched, "field must be set");
            sched->master_sched->get_kernel_timer().set_name(name_ext());
            sched->master_sched->get_kernel_timer().set_kernel_time(kernel_time);
        }

        if (use_single_list) {
            reset_events();
        }

        if (sched->get_memory().ze_entries.back() == this) {
            LOG_DEBUG("reset sched events\n");
            sched->get_memory().event_manager->reset();
        }

        // Finalize must go after all operation with the event because it's destroyed there.
        if (!sched->coll_attr.to_cache) {
            if (use_single_list) {
                finalize_entries();
            }
            else {
                finalize();
            }
        }
    }
    else {
        // just return in case if the kernel is not ready yet
        // will check again on the next iteration
        return;
    }
}

ze_command_list_handle_t ze_base_entry::get_comp_list() {
    if (use_single_list) {
        return sched->get_memory().list_manager->get_comp_list();
    }
    return comp_primitives.list;
}

ze_command_list_handle_t ze_base_entry::get_copy_list() {
    if (use_single_list) {
        return sched->get_memory().list_manager->get_copy_list();
    }

    ze_command_list_handle_t list{};
    if (copy_primitives.list) {
        list = copy_primitives.list;
        LOG_DEBUG("copy list is returned");
    }
    else if (comp_primitives.list) {
        list = comp_primitives.list;
        LOG_DEBUG("compute list is returned");
    }
    CCL_THROW_IF_NOT(list, "command list is invalid");
    return list;
}

void ze_base_entry::append_wait_on_events() {
    if (use_single_list && !wait_events.empty()) {
        if (sched->get_memory().list_manager->can_use_copy_queue() &&
            (init_mode::copy & mode)) { // to prevent double append
            ZE_CALL(zeCommandListAppendWaitOnEvents,
                    (get_copy_list(), wait_events.size(), wait_events.data()));
        }
        ZE_CALL(zeCommandListAppendWaitOnEvents,
                (get_comp_list(), wait_events.size(), wait_events.data()));
    }
}

std::string ze_base_entry::name_ext() const {
    return "[empty]";
}

void ze_base_entry::get_comp_primitives(const ze_queue_properties_t &queue_props,
                                        cmd_primitives &comp_primitives) {
    uint32_t ordinal{}, queue_index{};
    get_comp_queue_ordinal(device, queue_props, &ordinal);
    get_queue_index(queue_props, ordinal, 0, &queue_index);

    comp_primitives.queue_desc.ordinal = ordinal;
    comp_primitives.queue_desc.index = queue_index;
    comp_primitives.list_desc.commandQueueGroupOrdinal = ordinal;
}

void ze_base_entry::get_copy_primitives(const ze_queue_properties_t &queue_props,
                                        cmd_primitives &copy_primitives,
                                        init_mode mode) {
    uint32_t ordinal{}, queue_index{};
    get_copy_queue_ordinal(device, queue_props, &ordinal);

    // TODO: index depends on rank's changing, when > 1 queues are created,
    // the index is still the same for different queues, that's the issue.
    // WA is adding optional counter, which says the order number of a queue.
    // Need to think, how we'd calculate the index for every queue.
    // Hang in case of CCL_KERNEL_1S_USE_COPY_OPS=1 CCL_ZE_COPY_ENGINE=none
    if (mode == (init_mode::compute | init_mode::copy)) {
        get_queue_index(queue_props, ordinal, 1, &queue_index);
    }
    else {
        get_queue_index(queue_props, ordinal, 0, &queue_index);
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

ze_event_handle_t ze_base_entry::create_event(ze_event_pool_handle_t event_pool,
                                              ze_event_desc_t event_desc) {
    ze_event_handle_t event;
    ZE_CALL(zeEventCreate, (event_pool, &event_desc, &event));
    return event;
}

ze_event_handle_t ze_base_entry::create_event() {
    ze_event_desc_t event_desc{ default_event_desc };
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_DEVICE;
    event_desc.index = event_counter++;
    LOG_DEBUG("create event with index ", event_desc.index);
    CCL_THROW_IF_NOT(event_desc.index < event_pool_desc.count,
                     ", event creation limit exceeded: ",
                     event_desc.index,
                     ", event_pool_desc.count: ",
                     event_pool_desc.count);
    CCL_THROW_IF_NOT(event_desc.index < events.size());

    ze_event_handle_t event = create_event(event_pool, event_desc);
    events[event_desc.index] = event;

    return event;
}

void ze_base_entry::reset_events() {
    for (size_t idx = 0; idx < events.size(); idx++) {
        if (events[idx])
            ZE_CALL(zeEventHostReset, (events[idx]));
    }
}

void ze_base_entry::destroy_events() {
    for (size_t idx = 0; idx < events.size(); idx++) {
        if (events[idx])
            ZE_CALL(zeEventDestroy, (events[idx]));
    }
    events.clear();
}

void ze_base_entry::close_lists() {
    if (comp_primitives.list)
        ZE_CALL(zeCommandListClose, (comp_primitives.list));
    if (copy_primitives.list)
        ZE_CALL(zeCommandListClose, (copy_primitives.list));
}

ze_kernel::ze_kernel(ze_module_handle_t module, const std::string &kernel_name, size_t worker_idx)
        : module(module),
          kernel_name(kernel_name),
          worker_idx(worker_idx) {
    global_data::get().ze_cache->get(worker_idx, module, kernel_name, &kernel);
    CCL_THROW_IF_NOT(kernel);
    LOG_DEBUG("get kernel: name: ", kernel_name);
}

ze_kernel::ze_kernel(ze_kernel &&other) noexcept
        : module(std::move(other.module)),
          kernel_name(std::move(other.kernel_name)),
          worker_idx(std::move(other.worker_idx)),
          group_count(std::move(other.group_count)),
          group_size(std::move(other.group_size)),
          kernel(std::move(other.kernel)) {
    other.module = nullptr;
    other.kernel_name.clear();
    other.worker_idx = 0;
    other.group_count = { 0, 0, 0 };
    other.group_size = { 0, 0, 0 };
    other.kernel = nullptr;
};

ze_kernel::~ze_kernel() {
    if (kernel) {
        global_data::get().ze_cache->push(worker_idx, module, kernel_name, kernel);
    }
}

void ze_kernel::set_args(ze_kernel_args_t kernel_args) {
    LOG_DEBUG("kernel ", kernel, " args:\n", to_string(kernel_args));
    set_kernel_args(kernel, kernel_args);
}

void ze_kernel::calculate_group_size(size_t count) {
    get_suggested_group_size(kernel, count, &group_size);
    LOG_DEBUG("suggested group size: ", to_string(group_size));

    ZE_CALL(zeKernelSetGroupSize,
            (kernel, group_size.groupSizeX, group_size.groupSizeY, group_size.groupSizeZ));

    get_suggested_group_count(group_size, count, &group_count);
    LOG_DEBUG("suggested group count: ", to_string(group_count));
}

ze_kernel_handle_t ze_kernel::get_kernel() const {
    return kernel;
}

const ze_group_count_t *ze_kernel::get_group_count() const {
    return &group_count;
}
