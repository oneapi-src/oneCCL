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
#pragma once

#include "sched/entry/ze/ze_call.hpp"

#include <initializer_list>
#include <string>
#include <vector>
#include <ze_api.h>

namespace ccl {

namespace ze {

#define ZE_CALL(ze_name, ze_args) ccl::ze::ze_call().do_call(ze_name ze_args, #ze_name)

enum class init_mode : int {
    compute = 1,
    copy = 2,
};

enum class device_id : uint32_t { unknown = 0x0, id1 = 0x200, id2 = 0xbd0 };

constexpr ze_context_desc_t default_context_desc = { .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                                                     .pNext = nullptr,
                                                     .flags = 0 };

constexpr ze_fence_desc_t default_fence_desc = { .stype = ZE_STRUCTURE_TYPE_FENCE_DESC,
                                                 .pNext = nullptr,
                                                 .flags = 0 };

constexpr ze_kernel_desc_t default_kernel_desc = { .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
                                                   .pNext = nullptr,
                                                   .flags = 0,
                                                   .pKernelName = nullptr };

constexpr ze_command_list_desc_t default_cmd_list_desc = {
    .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
    .pNext = nullptr,
    .commandQueueGroupOrdinal = 0,
    .flags = 0,
};

constexpr ze_command_queue_desc_t default_cmd_queue_desc = {
    .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
    .pNext = nullptr,
    .ordinal = 0,
    .index = 0,
    .flags = 0,
    .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
    .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
};

constexpr ze_device_mem_alloc_desc_t default_device_mem_alloc_desc = {
    .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
    .pNext = nullptr,
    .flags = 0,
    .ordinal = 0
};

constexpr ze_memory_allocation_properties_t default_alloc_props = {
    .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
    .pNext = nullptr,
    .type = ZE_MEMORY_TYPE_UNKNOWN
};

constexpr ze_device_properties_t default_device_props = { .stype =
                                                              ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                                                          .pNext = nullptr };

constexpr ze_event_pool_desc_t default_event_pool_desc = { .stype =
                                                               ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
                                                           .pNext = nullptr,
                                                           .flags = 0,
                                                           .count = 0 };

constexpr ze_event_desc_t default_event_desc = { .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
                                                 .pNext = nullptr,
                                                 .index = 0,
                                                 .signal = 0,
                                                 .wait = 0 };

inline init_mode operator|(init_mode mode1, init_mode mode2) {
    return static_cast<init_mode>(static_cast<int>(mode1) | static_cast<int>(mode2));
}

inline bool operator&(init_mode mode1, init_mode mode2) {
    return static_cast<int>(mode1) & static_cast<int>(mode2);
}

void load_module(const std::string& file_path,
                 ze_device_handle_t device,
                 ze_context_handle_t context,
                 ze_module_handle_t* module);
void create_kernel(ze_module_handle_t module, std::string kernel_name, ze_kernel_handle_t* kernel);

// this structure is just to align with ze_group_count_t
// L0 doesn't have ze_group_size_t
struct ze_group_size_t {
    uint32_t groupSizeX = 0;
    uint32_t groupSizeY = 0;
    uint32_t groupSizeZ = 0;
};

void get_suggested_group_size(ze_kernel_handle_t kernel,
                              size_t elem_count,
                              ze_group_size_t* group_size);
void get_suggested_group_count(const ze_group_size_t& group_size,
                               size_t elem_count,
                               ze_group_count_t* group_count);

struct ze_kernel_arg_t {
    template <class T>
    constexpr ze_kernel_arg_t(const T* arg) noexcept
            : size{ sizeof(T) },
              ptr{ static_cast<const void*>(arg) } {}
    const size_t size;
    const void* ptr;
};

using ze_kernel_args_t = typename std::initializer_list<ze_kernel_arg_t>;
void set_kernel_args(ze_kernel_handle_t kernel, const ze_kernel_args_t& kernel_args);

using ze_queue_properties_t = typename std::vector<ze_command_queue_group_properties_t>;

void get_queues_properties(ze_device_handle_t device, ze_queue_properties_t* props);
void get_comp_queue_ordinal(ze_device_handle_t device,
                            const ze_queue_properties_t& props,
                            uint32_t* ordinal);
void get_copy_queue_ordinal(ze_device_handle_t device,
                            const ze_queue_properties_t& props,
                            uint32_t* ordinal);
void get_queue_index(const ze_queue_properties_t& props,
                     uint32_t ordinal,
                     int idx,
                     uint32_t* index);

device_family get_device_family(ze_device_handle_t device);
std::pair<uint64_t, uint64_t> calculate_event_time(ze_event_handle_t event,
                                                   ze_device_handle_t device);
uint64_t calculate_global_time(ze_device_handle_t device);

std::string to_string(ze_result_t result);
std::string to_string(const ze_group_size_t& group_size);
std::string to_string(const ze_group_count_t& group_count);
std::string to_string(const ze_kernel_args_t& kernel_args);
std::string to_string(const ze_command_queue_group_property_flag_t& flag);
std::string to_string(const ze_command_queue_group_properties_t& queue_property);

std::string join_strings(const std::vector<std::string>& tokens, const std::string& delimeter);

} // namespace ze
} // namespace ccl
