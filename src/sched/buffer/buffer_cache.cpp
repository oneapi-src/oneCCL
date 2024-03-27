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
#include "sched/buffer/buffer_cache.hpp"

namespace ccl {

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
typedef ze_result_t (*pFnzexDriverImportExternalPointer)(ze_driver_handle_t, void*, size_t);
typedef ze_result_t (*pFnzexDriverReleaseImportedPointer)(ze_driver_handle_t, void*);
pFnzexDriverImportExternalPointer zexDriverImportExternalPointer = nullptr;
pFnzexDriverReleaseImportedPointer zexDriverReleaseImportedPointer = nullptr;
ze_driver_handle_t ze_driver_handle;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

buffer_cache::buffer_cache(size_t instance_count)
        : reg_buffers(instance_count)
#ifdef CCL_ENABLE_SYCL
          ,
          sycl_buffers(instance_count)
#endif // CCL_ENABLE_SYCL
{
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // Instruct the UMD to create the internal graphics allocation for each system memory allocation
    // against a driver handle, instead of a command list handle.
    // By doing this, the UMD is able to reuse the internal graphics allocation for any new or reset list,
    // until the application decides to release the imported pointer. Any GPU driver handle fits.
    // This API is a part of exported extensions, therefore have to check for availability first.
    // Note: ze_data may be not initialized in some cases like stub backend mode or CCL_ZE_ENABLE=0
    if (global_data::env().enable_buffer_cache && global_data::get().ze_data &&
        !global_data::get().ze_data->drivers.empty()) {
        ze_driver_handle = global_data::get().ze_data->drivers.front();
        ze_result_t res_import = zeDriverGetExtensionFunctionAddress(
            ze_driver_handle,
            "zexDriverImportExternalPointer",
            reinterpret_cast<void**>(&zexDriverImportExternalPointer));
        ze_result_t res_release = zeDriverGetExtensionFunctionAddress(
            ze_driver_handle,
            "zexDriverReleaseImportedPointer",
            reinterpret_cast<void**>(&zexDriverReleaseImportedPointer));
        if ((res_import != ZE_RESULT_SUCCESS) || (res_release != ZE_RESULT_SUCCESS)) {
            // Reset function pointers for safety
            zexDriverImportExternalPointer = nullptr;
            zexDriverReleaseImportedPointer = nullptr;
            LOG_INFO("Can not initialize Import Extension API ",
                     "(zexDriverReleaseImportedPointer/zexDriverImportExternalPointer: ",
                     std::to_string(res_import),
                     " ",
                     std::to_string(res_release));
        }
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

buffer_cache::~buffer_cache() {
    for (auto& instance : reg_buffers) {
        instance.clear();
    }

#ifdef CCL_ENABLE_SYCL
    for (auto& instance : sycl_buffers) {
        instance.clear();
    }
#endif // CCL_ENABLE_SYCL
}

void buffer_cache::get(size_t idx, size_t bytes, void** pptr) {
    reg_buffers.at(idx % reg_buffers.size()).get(bytes, pptr);
}

void buffer_cache::push(size_t idx, size_t bytes, void* ptr) {
    reg_buffers.at(idx % reg_buffers.size()).push(bytes, ptr);
}

#ifdef CCL_ENABLE_SYCL
void buffer_cache::get(size_t idx, size_t bytes, const sycl::context& ctx, void** pptr) {
    sycl_buffers.at(idx % sycl_buffers.size()).get(bytes, ctx, pptr);
}

void buffer_cache::push(size_t idx, size_t bytes, const sycl::context& ctx, void* ptr) {
    sycl_buffers.at(idx % sycl_buffers.size()).push(bytes, ctx, ptr);
}
#endif // CCL_ENABLE_SYCL

regular_buffer_cache::~regular_buffer_cache() {
    if (!cache.empty()) {
        LOG_WARN("buffer cache is not empty, size: ", cache.size());
        clear();
    }
}

void regular_buffer_cache::clear() {
    std::lock_guard<buffer_cache::lock_t> lock{ guard };
    LOG_DEBUG("clear buffer cache: size: ", cache.size());
    for (auto& key_value : cache) {
        CCL_FREE(key_value.second);
    }
    cache.clear();
}

void regular_buffer_cache::get(size_t bytes, void** pptr) {
    if (global_data::env().enable_buffer_cache) {
        std::lock_guard<buffer_cache::lock_t> lock{ guard };
        key_t key(bytes);
        auto key_value = cache.find(key);
        if (key_value != cache.end()) {
            *pptr = key_value->second;
            cache.erase(key_value);
            LOG_DEBUG("loaded from buffer cache: bytes: ", bytes, ", ptr: ", *pptr);
            return;
        }
    }
    *pptr = CCL_MALLOC(bytes, "buffer");
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (zexDriverImportExternalPointer) {
        ze_result_t res = zexDriverImportExternalPointer(ze_driver_handle, *pptr, bytes);
        if (res != ZE_RESULT_SUCCESS) {
            LOG_INFO("zexDriverImportExternalPointer can not register the pointer with error: ",
                     std::to_string(res));
        }
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

void regular_buffer_cache::push(size_t bytes, void* ptr) {
    if (global_data::env().enable_buffer_cache) {
        std::lock_guard<buffer_cache::lock_t> lock{ guard };
        key_t key(bytes);
        cache.insert({ std::move(key), ptr });
        LOG_DEBUG("inserted to buffer cache: bytes: ", bytes, ", ptr: ", ptr);
        return;
    }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (zexDriverReleaseImportedPointer) {
        ze_result_t res = zexDriverReleaseImportedPointer(ze_driver_handle, ptr);
        if (res != ZE_RESULT_SUCCESS) {
            LOG_INFO("zexDriverReleaseImportPointer can not release the pointer with error: ",
                     std::to_string(res));
        }
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    CCL_FREE(ptr);
}

#ifdef CCL_ENABLE_SYCL
sycl_buffer_cache::~sycl_buffer_cache() {
    if (!cache.empty()) {
        LOG_WARN("sycl buffer cache is not empty, size: ", cache.size());
        clear();
    }
}

void sycl_buffer_cache::clear() {
    std::lock_guard<buffer_cache::lock_t> lock{ guard };
    LOG_DEBUG("clear sycl buffer cache: size: ", cache.size());
    for (auto& key_value : cache) {
        const sycl::context& ctx = std::get<1>(key_value.first);
        if (ctx.get_backend() == sycl::backend::opencl) {
            continue;
        }
        try {
            sycl::free(key_value.second, ctx);
        }
        catch (sycl::exception& e) {
            LOG_INFO("clear: got exception during sycl::free, ptr: ", key_value.second);
        }
    }
    cache.clear();
}

void sycl_buffer_cache::get(size_t bytes, const sycl::context& ctx, void** pptr) {
    if (global_data::env().enable_buffer_cache) {
        std::lock_guard<buffer_cache::lock_t> lock{ guard };
        key_t key(bytes, ctx);
        auto key_value = cache.find(key);
        if (key_value != cache.end()) {
            *pptr = key_value->second;
            cache.erase(key_value);
            LOG_DEBUG("loaded from sycl buffer cache: bytes: ", bytes, ", ptr: ", *pptr);
            return;
        }
    }
    *pptr = sycl::aligned_alloc_host(64, bytes, ctx);
}

void sycl_buffer_cache::push(size_t bytes, const sycl::context& ctx, void* ptr) {
    if (global_data::env().enable_buffer_cache) {
        std::lock_guard<buffer_cache::lock_t> lock{ guard };
        key_t key(bytes, ctx);
        cache.insert({ std::move(key), ptr });
        LOG_DEBUG("inserted to sycl buffer cache: bytes: ", bytes, ", ptr: ", ptr);
        return;
    }
    try {
        sycl::free(ptr, ctx);
    }
    catch (sycl::exception& e) {
        LOG_INFO("push: got exception during sycl::free, ptr: ", ptr);
    }
}
#endif // CCL_ENABLE_SYCL

} // namespace ccl
