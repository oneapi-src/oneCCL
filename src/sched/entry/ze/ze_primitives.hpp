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

#include <initializer_list>
#include <string>
#include <vector>

#include "common/ze/ze_api_wrapper.hpp"
#include "sched/entry/ze/ze_call.hpp"

namespace ccl {

namespace ze {

#define ZE_CALL(ze_name, ze_args) ccl::ze::ze_call().do_call(ze_name ze_args, #ze_name)

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

enum class queue_group_type : uint8_t { unknown, compute, main, link };

using ze_queue_properties_t = typename std::vector<ze_command_queue_group_properties_t>;

queue_group_type get_queue_group_type(const ze_queue_properties_t& props, uint32_t ordinal);
uint32_t get_queue_group_ordinal(const ze_queue_properties_t& props, queue_group_type type);

void get_queues_properties(ze_device_handle_t device, ze_queue_properties_t* props);

bool get_buffer_context_and_device(const void* buf,
                                   ze_context_handle_t* context,
                                   ze_device_handle_t* device,
                                   ze_memory_allocation_properties_t* props = nullptr);
bool get_context_global_id(ze_context_handle_t context, ssize_t* id);
bool get_device_global_id(ze_device_handle_t device, ssize_t* id);

int get_fd_from_handle(const ze_ipc_mem_handle_t& handle);
void close_handle_fd(const ze_ipc_mem_handle_t& handle);
ze_ipc_mem_handle_t get_handle_from_fd(int fd);

device_family get_device_family(ze_device_handle_t device);

std::string to_string(ze_result_t result);
std::string to_string(const ze_group_size_t& group_size);
std::string to_string(const ze_group_count_t& group_count);
std::string to_string(const ze_kernel_args_t& kernel_args);
std::string to_string(const ze_device_property_flag_t& flag);
std::string to_string(const ze_command_queue_group_property_flag_t& flag);
std::string to_string(const ze_command_queue_group_properties_t& queue_property);
std::string to_string(const ze_device_uuid_t& uuid);
std::string to_string(const zes_pci_address_t& addr);
std::string to_string(queue_group_type type);
std::string to_string(const zes_fabric_port_id_t& port);

std::string join_strings(const std::vector<std::string>& tokens, const std::string& delimeter);

template <typename T>
std::string flags_to_string(uint32_t flags) {
    constexpr size_t bits = 8;
    std::vector<std::string> output;
    for (size_t i = 0; i < sizeof(flags) * bits; ++i) {
        const size_t mask = 1UL << i;
        const auto flag = flags & mask;
        if (flag != 0) {
            output.emplace_back(to_string(static_cast<T>(flag)));
        }
    }
    return join_strings(output, " | ");
}

} // namespace ze
} // namespace ccl
