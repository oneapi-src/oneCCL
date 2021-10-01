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
#include "sched/entry/gpu/ze_cache.hpp"

#include <iterator>

namespace ccl {
namespace ze {

template <class map_t, class... keys_t>
bool get_from_cache(map_t& cache, typename map_t::mapped_type& object, keys_t... keys) {
    bool success{};

    if (!global_data::env().enable_kernel_cache)
        return success;

    typename map_t::key_type key(keys...);
    auto key_value = cache.find(key);
    if (key_value != cache.end()) {
        object = key_value->second;
        cache.erase(key_value);
        LOG_DEBUG("loaded from cache: object: ", object);
        success = true;
    }
    return success;
}

template <class map_t, class... keys_t>
bool push_to_cache(map_t& cache, const typename map_t::mapped_type& object, keys_t... keys) {
    bool success{};

    if (!global_data::env().enable_kernel_cache)
        return success;

    typename map_t::key_type key(keys...);
    auto range = cache.equal_range(key);
    auto range_len = std::distance(range.first, range.second);
    if (range_len > 0) {
        LOG_DEBUG("cache already contain ", range_len, " objects with the same key");
        for (auto i = range.first; i != range.second; ++i) {
            CCL_THROW_IF_NOT(i->second != object, "trying to push object that already exists");
        }
    }
    cache.insert({ std::move(key), object });
    LOG_DEBUG("inserted to cache: object: ", object);
    success = true;
    return success;
}

// fence_cache
fence_cache::~fence_cache() {
    if (!cache.empty()) {
        LOG_WARN("fence cache is not empty, size: ", cache.size());
        clear();
    }
}

void fence_cache::clear() {
    LOG_DEBUG("clear fence cache: size: ", cache.size());
    for (auto& key_value : cache) {
        ZE_CALL(zeFenceDestroy, (key_value.second));
    }
    cache.clear();
}

void fence_cache::get(ze_command_queue_handle_t queue,
                      const ze_fence_desc_t& fence_desc,
                      ze_fence_handle_t* fence) {
    CCL_THROW_IF_NOT(queue);
    CCL_THROW_IF_NOT(fence);
    if (get_from_cache(cache, *fence, queue)) {
        ZE_CALL(zeFenceReset, (*fence));
    }
    else {
        ZE_CALL(zeFenceCreate, (queue, &fence_desc, fence));
    }
}

void fence_cache::push(ze_command_queue_handle_t queue,
                       const ze_fence_desc_t& fence_desc,
                       ze_fence_handle_t fence) {
    CCL_THROW_IF_NOT(queue);
    CCL_THROW_IF_NOT(fence);
    if (!push_to_cache(cache, fence, queue)) {
        zeFenceDestroy(fence);
    }
}

// kernel_cache
kernel_cache::~kernel_cache() {
    if (!cache.empty()) {
        LOG_WARN("kernel cache is not empty, size: ", cache.size());
        clear();
    }
}

void kernel_cache::clear() {
    LOG_DEBUG("clear kernel cache: size: ", cache.size());
    for (auto& key_value : cache) {
        ZE_CALL(zeKernelDestroy, (key_value.second));
    }
    cache.clear();
}

void kernel_cache::get(ze_module_handle_t module,
                       const std::string& kernel_name,
                       ze_kernel_handle_t* kernel) {
    CCL_THROW_IF_NOT(module);
    CCL_THROW_IF_NOT(!kernel_name.empty());
    CCL_THROW_IF_NOT(kernel);
    if (!get_from_cache(cache, *kernel, module, kernel_name)) {
        create_kernel(module, kernel_name, kernel);
    }
}

void kernel_cache::push(ze_module_handle_t module,
                        const std::string& kernel_name,
                        ze_kernel_handle_t kernel) {
    CCL_THROW_IF_NOT(module);
    CCL_THROW_IF_NOT(!kernel_name.empty());
    CCL_THROW_IF_NOT(kernel);
    if (!push_to_cache(cache, kernel, module, kernel_name)) {
        ZE_CALL(zeKernelDestroy, (kernel));
    }
}

// list_cache
list_cache::~list_cache() {
    if (!cache.empty()) {
        LOG_WARN("list cache is not empty, size: ", cache.size());
        clear();
    }
}

void list_cache::clear() {
    LOG_DEBUG("clear list cache: size: ", cache.size());
    for (auto& key_value : cache) {
        ZE_CALL(zeCommandListDestroy, (key_value.second));
    }
    cache.clear();
}

void list_cache::get(ze_context_handle_t context,
                     ze_device_handle_t device,
                     const ze_command_list_desc_t& list_desc,
                     ze_command_list_handle_t* list) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(list);
    if (get_from_cache(
            cache, *list, context, device, list_desc.commandQueueGroupOrdinal, list_desc.flags)) {
        ZE_CALL(zeCommandListReset, (*list));
    }
    else {
        ZE_CALL(zeCommandListCreate, (context, device, &list_desc, list));
    }
}

void list_cache::push(ze_context_handle_t context,
                      ze_device_handle_t device,
                      const ze_command_list_desc_t& list_desc,
                      ze_command_list_handle_t list) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(list);
    if (!push_to_cache(
            cache, list, context, device, list_desc.commandQueueGroupOrdinal, list_desc.flags)) {
        ZE_CALL(zeCommandListDestroy, (list));
    }
}

// queue_cache
queue_cache::~queue_cache() {
    if (!cache.empty()) {
        LOG_WARN("queue cache is not empty, size: ", cache.size());
        clear();
    }
}

void queue_cache::clear() {
    LOG_DEBUG("clear queue cache: size: ", cache.size());
    for (auto& key_value : cache) {
        ZE_CALL(zeCommandQueueDestroy, (key_value.second));
    }
    cache.clear();
}

void queue_cache::get(ze_context_handle_t context,
                      ze_device_handle_t device,
                      const ze_command_queue_desc_t& queue_desc,
                      ze_command_queue_handle_t* queue) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(queue);
    if (!get_from_cache(cache,
                        *queue,
                        context,
                        device,
                        queue_desc.index,
                        queue_desc.ordinal,
                        queue_desc.flags,
                        queue_desc.mode,
                        queue_desc.priority)) {
        ZE_CALL(zeCommandQueueCreate, (context, device, &queue_desc, queue));
    }
}

void queue_cache::push(ze_context_handle_t context,
                       ze_device_handle_t device,
                       const ze_command_queue_desc_t& queue_desc,
                       ze_command_queue_handle_t queue) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(queue);
    if (!push_to_cache(cache,
                       queue,
                       context,
                       device,
                       queue_desc.index,
                       queue_desc.ordinal,
                       queue_desc.flags,
                       queue_desc.mode,
                       queue_desc.priority)) {
        ZE_CALL(zeCommandQueueDestroy, (queue));
    }
}

// event_pool_cache
event_pool_cache::~event_pool_cache() {
    if (!cache.empty()) {
        LOG_WARN("event pool cache is not empty, size: ", cache.size());
        clear();
    }
}

void event_pool_cache::clear() {
    LOG_DEBUG("clear event pool cache: size: ", cache.size());
    for (auto& key_value : cache) {
        ZE_CALL(zeEventPoolDestroy, (key_value.second));
    }
    cache.clear();
}

void event_pool_cache::get(ze_context_handle_t context,
                           const ze_event_pool_desc_t& pool_desc,
                           ze_event_pool_handle_t* event_pool) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(event_pool);
    // TODO: we can potentially use pool with count >= pool_desc.count
    if (!get_from_cache(cache, *event_pool, context, pool_desc.flags, pool_desc.count)) {
        ZE_CALL(zeEventPoolCreate, (context, &pool_desc, 0, nullptr, event_pool));
    }
}

void event_pool_cache::push(ze_context_handle_t context,
                            const ze_event_pool_desc_t& pool_desc,
                            ze_event_pool_handle_t event_pool) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(event_pool);
    if (!push_to_cache(cache, event_pool, context, pool_desc.flags, pool_desc.count)) {
        ZE_CALL(zeEventPoolDestroy, (event_pool));
    }
}

// device_mem_cache
device_mem_cache::~device_mem_cache() {
    if (!cache.empty()) {
        LOG_WARN("device memory cache is not empty, size: ", cache.size());
        clear();
    }
}

void device_mem_cache::clear() {
    LOG_DEBUG("clear device memory cache: size: ", cache.size());
    //for (auto& key_value : cache) {
    // TODO: there is a segfault on this call, when ~cache is invoked w/ or w/0 api cache.
    // But it passes, when CCL_KERNEL_CACHE=0 (calls of zeMemAllocDevice and ZeMemFree happen on every iteration).
    // We don't control destroying phase and may be key_value.second (mem_ptr) is already away to free?
    // ZE_CALL(zeMemFree, (std::get<0>(key_value.first), key_value.second))
    //}
    cache.clear();
}

void device_mem_cache::get(ze_context_handle_t context,
                           ze_device_handle_t device,
                           const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                           size_t bytes,
                           size_t alignment,
                           void** pptr) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(pptr);
    if (!get_from_cache(cache,
                        *pptr,
                        context,
                        device,
                        bytes,
                        device_mem_alloc_desc.flags,
                        device_mem_alloc_desc.ordinal)) {
        ZE_CALL(zeMemAllocDevice,
                (context, &device_mem_alloc_desc, bytes, alignment, device, pptr));
    }
}

void device_mem_cache::push(ze_context_handle_t context,
                            ze_device_handle_t device,
                            const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                            size_t bytes,
                            size_t alignment,
                            void* ptr) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(ptr);
    if (!push_to_cache(cache,
                       ptr,
                       context,
                       device,
                       bytes,
                       device_mem_alloc_desc.flags,
                       device_mem_alloc_desc.ordinal)) {
        ZE_CALL(zeMemFree, (context, ptr));
    }
}

// module_cache
module_cache::~module_cache() {
    if (!cache.empty()) {
        LOG_WARN("module cache is not empty, size: ", cache.size());
        clear();
    }
}

void module_cache::clear() {
    LOG_DEBUG("clear module cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        ZE_CALL(zeModuleDestroy, (key_value.second));
    }
    cache.clear();
}

void module_cache::get(ze_context_handle_t context,
                       ze_device_handle_t device,
                       const std::string& spv_name,
                       ze_module_handle_t* module) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(!spv_name.empty());
    CCL_THROW_IF_NOT(module);
    std::lock_guard<std::mutex> lock(mutex);
    key_t key(device, spv_name);
    auto key_value = cache.find(key);
    if (key_value != cache.end()) {
        *module = key_value->second;
        LOG_DEBUG("loaded from cache: module: ", *module);
    }
    else {
        load(context, device, spv_name, module);
        cache.insert({ std::move(key), *module });
        LOG_DEBUG("inserted to cache: module: ", *module);
    }
}

void module_cache::load(ze_context_handle_t context,
                        ze_device_handle_t device,
                        const std::string& spv_name,
                        ze_module_handle_t* module) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(!spv_name.empty());
    CCL_THROW_IF_NOT(module);
    std::string modules_dir = global_data::env().kernel_path;
    // TODO: remove
    if (modules_dir.empty()) {
        std::string ccl_root = getenv("CCL_ROOT");
        CCL_THROW_IF_NOT(!ccl_root.empty(), "incorrect comm kernels path, CCL_ROOT not found!");
        modules_dir = ccl_root + "/lib/kernels/";
    }
    load_module(modules_dir, spv_name, device, context, module);
}

// cache
cache::~cache() {
    for (size_t i = 0; i < instance_count; ++i) {
        fences[i].clear();
        kernels[i].clear();
        lists[i].clear();
        queues[i].clear();
        event_pools[i].clear();
        device_mems[i].clear();
    }

    modules.clear();
}

} // namespace ze
} // namespace ccl
