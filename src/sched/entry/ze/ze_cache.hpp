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

#include "common/log/log.hpp"
#include "common/utils/hash.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <unordered_map>

namespace ccl {
namespace ze {

class kernel_cache {
public:
    kernel_cache() = default;
    ~kernel_cache();

    void clear();

    void get(ze_module_handle_t module, const std::string& kernel_name, ze_kernel_handle_t* kernel);
    void push(ze_module_handle_t module, const std::string& kernel_name, ze_kernel_handle_t kernel);

private:
    using key_t = typename std::tuple<ze_module_handle_t, std::string>;
    using value_t = ze_kernel_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

// TODO: need to improve with ability to save list with commands for specific algo
class list_cache {
public:
    list_cache() = default;
    ~list_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_list_desc_t& list_desc,
             ze_command_list_handle_t* list);
    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_list_desc_t& list_desc,
              ze_command_list_handle_t list);

private:
    using key_t = typename std::
        tuple<ze_context_handle_t, ze_device_handle_t, uint32_t, ze_command_list_flags_t>;
    using value_t = ze_command_list_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

class queue_cache {
public:
    queue_cache() = default;
    ~queue_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_queue_desc_t& queue_desc,
             ze_command_queue_handle_t* queue);
    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_queue_desc_t& queue_desc,
              ze_command_queue_handle_t queue);

private:
    using key_t = typename std::tuple<ze_context_handle_t,
                                      ze_device_handle_t,
                                      uint32_t,
                                      uint32_t,
                                      ze_command_queue_flags_t,
                                      ze_command_queue_mode_t,
                                      ze_command_queue_priority_t>;
    using value_t = ze_command_queue_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

class event_pool_cache {
public:
    event_pool_cache() = default;
    ~event_pool_cache();

    void clear();

    void get(ze_context_handle_t context,
             const ze_event_pool_desc_t& pool_desc,
             ze_event_pool_handle_t* event_pool);

    void push(ze_context_handle_t context,
              const ze_event_pool_desc_t& pool_desc,
              ze_event_pool_handle_t event_pool);

private:
    using key_t = typename std::tuple<ze_context_handle_t, ze_event_pool_flags_t, uint32_t>;
    using value_t = ze_event_pool_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

class device_mem_cache {
public:
    device_mem_cache() = default;
    ~device_mem_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
             size_t bytes,
             size_t alignment,
             void** pptr);

    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
              size_t bytes,
              size_t alignment,
              void* ptr);

private:
    using key_t = typename std::tuple<ze_context_handle_t,
                                      ze_device_handle_t,
                                      size_t,
                                      ze_device_mem_alloc_flags_t,
                                      uint32_t>;
    using value_t = void*;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

class module_cache {
public:
    module_cache() = default;
    ~module_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const std::string& spv_name,
             ze_module_handle_t* module);

private:
    using key_t = typename std::tuple<ze_device_handle_t, std::string>;
    using value_t = ze_module_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;

    void load(ze_context_handle_t context,
              ze_device_handle_t device,
              const std::string& spv_name,
              ze_module_handle_t* module);
};

class cache {
public:
    cache(size_t instance_count)
            : instance_count(instance_count),
              kernels(instance_count),
              lists(instance_count),
              queues(instance_count),
              event_pools(instance_count),
              device_mems(instance_count) {
        LOG_DEBUG("create cache with ", instance_count, " instances");
    }
    cache(const cache&) = delete;
    cache& operator=(const cache&) = delete;
    ~cache();

    /* get */
    void get(size_t instance_idx,
             ze_module_handle_t module,
             const std::string& kernel_name,
             ze_kernel_handle_t* kernel) {
        kernels.at(instance_idx).get(module, kernel_name, kernel);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_list_desc_t& list_desc,
             ze_command_list_handle_t* list) {
        lists.at(instance_idx).get(context, device, list_desc, list);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_queue_desc_t& queue_desc,
             ze_command_queue_handle_t* queue) {
        queues.at(instance_idx).get(context, device, queue_desc, queue);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             const ze_event_pool_desc_t& pool_desc,
             ze_event_pool_handle_t* event_pool) {
        event_pools.at(instance_idx).get(context, pool_desc, event_pool);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
             size_t bytes,
             size_t alignment,
             void** pptr) {
        device_mems.at(instance_idx % device_mems.size())
            .get(context, device, device_mem_alloc_desc, bytes, alignment, pptr);
    }

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const std::string& spv_name,
             ze_module_handle_t* module) {
        modules.get(context, device, spv_name, module);
    }

    /* push */
    void push(size_t instance_idx,
              ze_module_handle_t module,
              const std::string& kernel_name,
              ze_kernel_handle_t kernel) {
        kernels.at(instance_idx).push(module, kernel_name, kernel);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_list_desc_t& list_desc,
              ze_command_list_handle_t list) {
        lists.at(instance_idx).push(context, device, list_desc, list);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_queue_desc_t& queue_desc,
              ze_command_queue_handle_t queue) {
        queues.at(instance_idx).push(context, device, queue_desc, queue);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              const ze_event_pool_desc_t& pool_desc,
              ze_event_pool_handle_t event_pool) {
        event_pools.at(instance_idx).push(context, pool_desc, event_pool);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
              size_t bytes,
              size_t alignment,
              void* ptr) {
        device_mems.at(instance_idx % device_mems.size())
            .push(context, device, device_mem_alloc_desc, bytes, alignment, ptr);
    }

private:
    const size_t instance_count;
    std::vector<kernel_cache> kernels;
    std::vector<list_cache> lists;
    std::vector<queue_cache> queues;
    std::vector<event_pool_cache> event_pools;
    std::vector<device_mem_cache> device_mems;
    module_cache modules{};
};

} // namespace ze
} // namespace ccl
