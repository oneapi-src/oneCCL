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

ze_command_list_handle_t list_info::get_native() const {
    return list;
}

ze_command_list_handle_t* list_info::get_native_ptr() {
    return &list;
}

const ze_command_list_desc_t& list_info::get_desc() const {
    return desc;
}

bool list_info::is_valid() const {
    return list != nullptr;
}

bool list_info::is_copy() const {
    return is_copy_list;
}

uint32_t list_info::get_queue_index() const {
    return queue_index;
}

ze_command_queue_handle_t queue_info::get_native() const {
    return queue;
}

const ze_command_queue_desc_t& queue_info::get_desc() const {
    return desc;
}

queue_group_type queue_info::get_type() const {
    return type;
}

bool queue_info::is_valid() const {
    return queue != nullptr;
}

bool queue_info::is_copy() const {
    return is_copy_queue;
}

queue_factory::queue_factory(ze_device_handle_t device,
                             ze_context_handle_t context,
                             queue_group_type type)
        : device(device),
          context(context),
          is_copy_queue(type == queue_group_type::main || type == queue_group_type::link),
          type(type) {
    ze_queue_properties_t queue_props;
    get_queues_properties(device, &queue_props);

    if (!global_data::env().disable_ze_family_check) {
        if (queue_props.size() == 1 && queue_props.front().numQueues == 1 &&
            (get_device_family(device) == ccl::device_family::unknown)) {
            ze_device_properties_t device_props;
            ZE_CALL(zeDeviceGetProperties, (device, &device_props));
            bool is_integrated = device_props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

            CCL_THROW_IF_NOT(is_integrated,
                             "unexpected device properties flags: ",
                             flags_to_string<ze_device_property_flag_t>(device_props.flags));
        }
    }

    queue_ordinal = get_queue_group_ordinal(queue_props, type);
    LOG_DEBUG(get_type_str(),
              " queue factory: use ",
              to_string(type),
              " queue group with ordinal: ",
              queue_ordinal,
              ", total group count: ",
              queue_props.size());
    CCL_THROW_IF_NOT(queue_ordinal < queue_props.size(),
                     "wrong queue group ordinal or properties size: { ordinal: ",
                     queue_ordinal,
                     ", size: ",
                     queue_props.size(),
                     ", type: ",
                     to_string(type),
                     " }");
    CCL_THROW_IF_NOT(queue_ordinal > 0 || !is_copy_queue,
                     "selected comp ordinal for copy queue: { ordinal: ",
                     queue_ordinal,
                     ", size: ",
                     queue_props.size(),
                     ", type: ",
                     to_string(type),
                     " }");

    uint32_t queue_count = queue_props.at(queue_ordinal).numQueues;
    CCL_THROW_IF_NOT(queue_count > 0, "no hw queues");
    queues.resize(queue_count);
}

queue_factory::~queue_factory() {
    clear();
}

queue_info_t queue_factory::get(uint32_t index) {
    CCL_THROW_IF_NOT(!queues.empty(), "no queues");

    uint32_t queue_index = get_queue_index(index);
    CCL_THROW_IF_NOT(queue_index < queues.size(), "wrong queue index");
    auto& queue = queues.at(queue_index);
    if (!queue || !queue->is_valid()) {
        queue = std::make_shared<queue_info>();
        queue->desc = default_cmd_queue_desc;
        queue->desc.ordinal = queue_ordinal;
        queue->desc.index = queue_index;
        queue->is_copy_queue = is_copy_queue;
        queue->type = type;
        global_data::get().ze_data->cache->get(
            worker_idx, context, device, queue->desc, &queue->queue);
        LOG_DEBUG("created new ",
                  get_type_str(),
                  " queue: { ordinal: ",
                  queue_ordinal,
                  ", index: ",
                  queue_index,
                  " }");
    }
    return queue;
}

void queue_factory::destroy(queue_info_t& queue) {
    if (!queue || !queue->is_valid()) {
        return;
    }

    global_data::get().ze_data->cache->push(worker_idx, context, device, queue->desc, queue->queue);
    queue.reset();
}

const char* queue_factory::get_type_str() const {
    return (is_copy_queue) ? "copy" : "comp";
}

uint32_t queue_factory::get_max_available_queue_count() const {
    ssize_t user_max_queues = CCL_ENV_SIZET_NOT_SPECIFIED;
    if (is_copy_queue) {
        user_max_queues = global_data::env().ze_max_copy_queues;
    }
    else {
        user_max_queues = global_data::env().ze_max_compute_queues;
    }
    if (user_max_queues != CCL_ENV_SIZET_NOT_SPECIFIED) {
        return std::min(static_cast<uint32_t>(user_max_queues),
                        static_cast<uint32_t>(queues.size()));
    }
    else {
        return queues.size();
    }
}

uint32_t queue_factory::get_queue_index(uint32_t requested_index) const {
    uint32_t max_queues = get_max_available_queue_count();
    CCL_THROW_IF_NOT(max_queues > 0, "wrong max queues count");
    uint32_t queue_index = requested_index % max_queues;
    queue_index = (queue_index + global_data::env().ze_queue_index_offset) % queues.size();
    return queue_index;
}

void queue_factory::clear() {
    for (auto& queue : queues) {
        destroy(queue);
    }
    queues.clear();
}

bool queue_factory::is_copy() const {
    return is_copy_queue;
}

uint32_t queue_factory::get_ordinal() const {
    return queue_ordinal;
}

bool queue_factory::can_use_queue_group(ze_device_handle_t device, queue_group_type type) {
    auto user_engine = global_data::env().ze_copy_engine;
    switch (type) {
        case queue_group_type::compute: break;

        case queue_group_type::main:
            if (user_engine != ccl_ze_copy_engine_auto && user_engine != ccl_ze_copy_engine_main) {
                return false;
            }
            break;

        case queue_group_type::link:
            if (user_engine != ccl_ze_copy_engine_auto && user_engine != ccl_ze_copy_engine_link) {
                return false;
            }
            break;

        default: CCL_THROW("unknown queue group type"); break;
    }

    if (type != queue_group_type::compute && user_engine == ccl_ze_copy_engine_none) {
        return false;
    }

    ze_queue_properties_t queue_props;
    get_queues_properties(device, &queue_props);
    uint32_t ordinal = get_queue_group_ordinal(queue_props, type);
    if (ordinal >= queue_props.size()) {
        return false;
    }

    return true;
}

list_factory::list_factory(ze_device_handle_t device, ze_context_handle_t context, bool is_copy)
        : device(device),
          context(context),
          is_copy_list(is_copy) {}

list_info_t list_factory::get(const queue_info_t& queue) {
    CCL_THROW_IF_NOT(queue && queue->is_valid(), "no queue");

    list_info_t list = std::make_shared<list_info>();
    list->desc = default_cmd_list_desc;
    list->desc.commandQueueGroupOrdinal = queue->get_desc().ordinal;
    list->is_copy_list = queue->is_copy();
    list->queue_index = queue->get_desc().index;
    global_data::get().ze_data->cache->get(worker_idx, context, device, list->desc, &list->list);
    LOG_DEBUG("created new ",
              get_type_str(),
              " list: { ordinal: ",
              list->desc.commandQueueGroupOrdinal,
              " } for queue: { ordinal: ",
              queue->get_desc().ordinal,
              ", index: ",
              list->queue_index,
              " }");
    return list;
}

void list_factory::destroy(list_info_t& list) {
    if (!list || !list->is_valid()) {
        return;
    }

    global_data::get().ze_data->cache->push(worker_idx, context, device, list->desc, list->list);
    list.reset();
}

const char* list_factory::get_type_str() const {
    return (is_copy_list) ? "copy" : "comp";
}

bool list_factory::is_copy() const {
    return is_copy_list;
}

list_manager::list_manager(const ccl_sched_base* sched, const ccl_stream* stream)
        : sched(sched),
          device(stream->get_ze_device()),
          context(stream->get_ze_context()) {
    LOG_DEBUG("create list manager");
    CCL_THROW_IF_NOT(device, "no device");
    CCL_THROW_IF_NOT(context, "no context");

    comp_queue_factory =
        std::make_unique<queue_factory>(device, context, queue_group_type::compute);
    comp_list_factory = std::make_unique<list_factory>(device, context, false);

    can_use_main_queue = queue_factory::can_use_queue_group(device, queue_group_type::main);
    if (can_use_main_queue) {
        main_queue_factory =
            std::make_unique<queue_factory>(device, context, queue_group_type::main);
    }

    can_use_link_queue = queue_factory::can_use_queue_group(device, queue_group_type::link);
    if (can_use_link_queue) {
        link_queue_factory =
            std::make_unique<queue_factory>(device, context, queue_group_type::link);
    }

    use_copy_queue = can_use_main_queue || can_use_link_queue;
    if (use_copy_queue) {
        copy_list_factory = std::make_unique<list_factory>(device, context, true);
    }
}

list_manager::~list_manager() {
    clear();
}

std::pair<queue_factory*, list_manager::queue_map_t*> list_manager::get_factory_and_map(
    bool is_copy,
    bool peer_card_copy) const {
    queue_factory* factory = nullptr;
    queue_map_t* queue_map = nullptr;
    if (is_copy) {
        if (can_use_link_queue && peer_card_copy) {
            factory = link_queue_factory.get();
            queue_map = const_cast<queue_map_t*>(&link_queue_map);
        }
        else {
            factory = main_queue_factory.get();
            queue_map = const_cast<queue_map_t*>(&main_queue_map);
        }
    }
    else {
        factory = comp_queue_factory.get();
        queue_map = const_cast<queue_map_t*>(&comp_queue_map);
    }
    CCL_THROW_IF_NOT(factory, "no factory");
    CCL_THROW_IF_NOT(queue_map, "no map");
    return std::make_pair(factory, queue_map);
}

list_info_t list_manager::get_list(const sched_entry* entry,
                                   uint32_t index,
                                   bool is_copy,
                                   const std::vector<ze_event_handle_t>& wait_events,
                                   bool peer_card_copy) {
    // get comp or copy primitives
    auto factory_map_pair = get_factory_and_map(is_copy, peer_card_copy);
    queue_factory* factory = factory_map_pair.first;
    queue_map_t* queue_map = factory_map_pair.second;
    auto queue = factory->get(index);
    auto& map = *queue_map;
    uint32_t queue_index = queue->get_desc().index;

    // get queue from map. if no list for this queue, then create new one
    bool new_list_for_queue = false;
    auto& list_pair = map[queue_index];
    auto& list = list_pair.first;
    if ((!list || !list->is_valid()) && sched->use_single_list) {
        new_list_for_queue = true;
    }
    else if (!sched->use_single_list) {
        new_list_for_queue = true;
    }

    // check if current entry is used list at the first time
    // it is needed to append wait events from this entry to the list if single_List mode is active
    bool new_entry_for_list = false;
    auto& list_entry_map = list_pair.second;
    auto found = list_entry_map.find(entry);
    if (found == list_entry_map.end()) {
        // remember entry. value in the map is not used
        list_entry_map.insert({ entry, true });
        new_entry_for_list = true;
    }

    // if we dont have any lists for current queue
    if (new_list_for_queue && new_entry_for_list) {
        // creaete new list
        list = (is_copy) ? copy_list_factory->get(queue) : comp_list_factory->get(queue);
        access_list.push_back({ queue, list });
        // remember list for current entry
        entry_map[entry].push_back(std::make_pair(queue, list));
        LOG_DEBUG("[entry ",
                  entry->name(),
                  "] created new ",
                  list->is_copy() ? "copy" : "comp",
                  " list with queue index ",
                  list->get_queue_index(),
                  ". total list count ",
                  access_list.size());
    }

    // if single_list mode is active and current entry never use this list before,
    // then append wait events from that entry,
    // because we must wait commands from the previous entries (which can be in other lists too)
    if (new_entry_for_list && sched->use_single_list && !wait_events.empty()) {
        LOG_DEBUG("[entry ",
                  entry->name(),
                  "] append wait events to ",
                  list->is_copy() ? "copy" : "comp",
                  " list with queue index ",
                  list->get_queue_index());
        ZE_CALL(zeCommandListAppendWaitOnEvents,
                (list->get_native(),
                 wait_events.size(),
                 const_cast<ze_event_handle_t*>(wait_events.data())));
    }

    return list;
}

ze_command_list_handle_t list_manager::get_comp_list(
    const sched_entry* entry,
    const std::vector<ze_event_handle_t>& wait_events,
    uint32_t index) {
    auto list = get_list(entry, index, false, wait_events, false);
    return list->get_native();
}

ze_command_list_handle_t list_manager::get_copy_list(
    const sched_entry* entry,
    const std::vector<ze_event_handle_t>& wait_events,
    uint32_t index,
    bool peer_card_copy) {
    // use main for intra copy or link for inter copy
    if ((!peer_card_copy && can_use_main_queue) || (peer_card_copy && can_use_link_queue)) {
        auto list = get_list(entry, index, true, wait_events, peer_card_copy);
        return list->get_native();
    }
    else {
        return get_comp_list(entry, wait_events, index);
    }
}

void list_manager::clear() {
    LOG_DEBUG("destroy lists and queues");
    reset_execution_state();
    comp_queue_map.clear();
    link_queue_map.clear();
    main_queue_map.clear();
    for (auto& queue_list_pair : access_list) {
        auto& list = queue_list_pair.second;
        if (list->is_copy()) {
            copy_list_factory->destroy(list);
        }
        else {
            comp_list_factory->destroy(list);
        }
    }
    access_list.clear();
    entry_map.clear();

    if (comp_queue_factory) {
        comp_queue_factory->clear();
    }
    if (link_queue_factory) {
        link_queue_factory->clear();
    }
    if (main_queue_factory) {
        main_queue_factory->clear();
    }
}

void list_manager::reset_execution_state() {
    LOG_DEBUG("reset list manager execution state");
    executed = false;
    for (auto& queue_list_pair : access_list) {
        auto& list = queue_list_pair.second;
        CCL_THROW_IF_NOT(list->is_closed, "detected list that has not been closed");
        list->is_executed = false;
    }
}

bool list_manager::can_use_copy_queue() const {
    return use_copy_queue;
}

bool list_manager::is_executed() const {
    return executed;
}

void list_manager::execute_list(queue_info_t& queue, list_info_t& list) {
    CCL_THROW_IF_NOT(list && list->is_valid(), "trying to execute uninitialized list");
    CCL_THROW_IF_NOT(queue && queue->is_valid(), "trying to execute list on uninitialized queue");
    CCL_THROW_IF_NOT(!list->is_executed, "trying to execute list that already has been executed");
    CCL_THROW_IF_NOT((queue->is_copy() && list->is_copy()) || !queue->is_copy(),
                     "trying to execute comp list on copy queue");

    if (!list->is_closed) {
        ZE_CALL(zeCommandListClose, (list->get_native()));
        list->is_closed = true;
    }
    LOG_DEBUG("execute ",
              list->is_copy() ? "copy" : "comp",
              " list with queue index ",
              queue->get_desc().index);

    ZE_CALL(zeCommandQueueExecuteCommandLists,
            (queue->get_native(), 1, list->get_native_ptr(), nullptr));
    list->is_executed = true;
}

void list_manager::execute(const sched_entry* entry) {
    CCL_THROW_IF_NOT(!sched->use_single_list || !executed, "lists are executed already");
    LOG_DEBUG("execute ", entry->name(), " entry");

    if (global_data::env().enable_ze_list_dump) {
        print_dump();
    }

    auto& container = (sched->use_single_list) ? access_list : entry_map[entry];
    for (auto& queue_list_pair : container) {
        auto& queue = queue_list_pair.first;
        auto& list = queue_list_pair.second;
        execute_list(queue, list);
    }

    executed = true;
}

void list_manager::print_dump() const {
    // queue_group_type, <queue index, counter>
    std::unordered_map<queue_group_type, std::unordered_map<uint32_t, size_t>> queue_index_count;

    // collect counters
    for (auto& queue_list_pair : access_list) {
        auto& queue = queue_list_pair.first;
        uint32_t queue_index = queue->get_desc().index;
        queue_group_type queue_type = queue->get_type();
        queue_index_count[queue_type][queue_index]++;
    }

    // sort
    std::vector<std::pair<queue_group_type, std::unordered_map<uint32_t, size_t>>> groups(
        queue_index_count.begin(), queue_index_count.end());
    std::sort(groups.begin(), groups.end(), [](auto& left, auto& right) {
        return left.first < right.first;
    });

    // print
    std::stringstream ss;
    ss << "\nsched: " << sched << "\n";
    ss << "single list mode: " << sched->use_single_list << "\n";
    for (const auto& group : groups) {
        queue_map_t* map = nullptr;
        queue_factory* factory = nullptr;
        if (group.first == queue_group_type::main) {
            map = const_cast<queue_map_t*>(&main_queue_map);
            factory = main_queue_factory.get();
        }
        else if (group.first == queue_group_type::link) {
            map = const_cast<queue_map_t*>(&link_queue_map);
            factory = link_queue_factory.get();
            ;
        }
        else {
            map = const_cast<queue_map_t*>(&comp_queue_map);
            factory = comp_queue_factory.get();
        }

        ss << to_string(group.first) << " (ordinal " << factory->get_ordinal()
           << ") queues usage (unsorted): {\n";
        for (const auto& index_count_pair : group.second) {
            uint32_t queue_index = index_count_pair.first;
            ss << "  queue index: " << queue_index << " list count: " << index_count_pair.second;
            ss << " entries:";
            for (const auto& list_entries_pair : map->at(queue_index).second) {
                auto entry = list_entries_pair.first;
                ss << " " << entry->name();
            }
            ss << "\n";
        }
        ss << "}\n";
    }

    if (sched->use_single_list && !access_list.empty()) {
        ss << "execution order: {\n";
        for (auto& queue_list_pair : access_list) {
            auto& queue = queue_list_pair.first;
            auto& list = queue_list_pair.second;
            ss << "  " << to_string(queue->get_type())
               << " queue index: " << list->get_queue_index()
               << ", list type: " << ((list->is_copy()) ? "copy" : "comp") << "\n";
        }
        ss << "}\n";
    }

    logger.info(ss.str());
}
