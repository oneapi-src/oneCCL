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
#include "common/comm/comm.hpp"
#include "sched/entry/gpu/ze_call.hpp"
#include "sched/ze_handle_manager.hpp"

#include <CL/sycl/backend/level_zero.hpp>

namespace ccl {

namespace ze {

static void cast_mem_to_pool_handle(ze_ipc_event_pool_handle_t* pool,
                                    const ze_ipc_mem_handle_t* mem) {
    static_assert(sizeof(ze_ipc_event_pool_handle_t) == sizeof(ze_ipc_mem_handle_t));
    memcpy(pool, mem, sizeof(*mem));
}

std::string to_string(ipc_mem_type type) {
    switch (type) {
        case ipc_mem_type::memory: return "buffer";
        case ipc_mem_type::pool: return "pool";
        default: return "unknown";
    }
}

ipc_handle_info::ipc_handle_info() {
    memset(&handle, 0, sizeof(handle));
}

ipc_handle_info::ipc_handle_info(const ze_ipc_mem_handle_t& handle,
                                 size_t offset,
                                 ipc_mem_type type)
        : handle(handle),
          offset(offset),
          type(type) {}

ipc_handle_manager::~ipc_handle_manager() {
    clear();
}

void ipc_handle_manager::init(const ccl_comm* init_comm, const ccl_stream* init_stream) {
    LOG_DEBUG("initialization");
    CCL_THROW_IF_NOT(init_comm, "no comm");
    CCL_THROW_IF_NOT(init_stream, "no stream");

    comm = const_cast<ccl_comm*>(init_comm);

    for (int idx = 0; idx < comm->size(); idx++) {
        rank_map.insert({ comm->get_global_rank(idx), idx });
    }

    auto sycl_device = init_stream->get_native_stream().get_device();
    auto sycl_context = init_stream->get_native_stream().get_context();

    device = sycl_device.template get_native<sycl::backend::level_zero>();
    context = sycl_context.template get_native<sycl::backend::level_zero>();

    CCL_THROW_IF_NOT(device, "device is not valid");
    CCL_THROW_IF_NOT(context, "context is not valid");

    LOG_DEBUG("initialization completed");
}

void ipc_handle_manager::clear() {
    for (int rank = 0; rank < static_cast<int>(handles.size()); rank++) {
        for (size_t buf_idx = 0; buf_idx < handles[rank].size(); buf_idx++) {
            const auto& handle_info = handles[rank][buf_idx];
            ze_ipc_mem_handle_t handle = handle_info.handle;
            auto mem_ptr = handle_info.ptr;
            auto mem_type = handle_info.type;
            size_t mem_offset = handle_info.offset;

            LOG_DEBUG("close handle: { base_ptr: ",
                      mem_ptr,
                      ", offset: ",
                      mem_offset,
                      ", fd: ",
                      *(int*)handle.data,
                      ", rank: ",
                      rank,
                      ", buf_idx: ",
                      buf_idx,
                      " }");

            // when closing the handle we need to take care of pointers that points to the
            // same level zero allocation. They're simply offsetted from some base pointer
            // although represented by different FDs. If we close this base pointer,
            // all the derived pointers are closed(unmapped) as well. To handle this case
            // we ignore the result of close function which would fail if we close a pointer
            // which is already closed. The function has semantic of free() call, so the result
            // is not much useful anyway.
            if (mem_ptr) {
                ze_result_t res{};
                if (mem_type == ipc_mem_type::memory) {
                    res = zeMemCloseIpcHandle(context, mem_ptr);
                }
                else if (mem_type == ipc_mem_type::pool) {
                    res = zeEventPoolCloseIpcHandle((ze_event_pool_handle_t)mem_ptr);
                }
                else {
                    CCL_THROW("unknown memory type");
                }

                if (res != ZE_RESULT_SUCCESS) {
                    LOG_TRACE("unable to close memory handle: ",
                              "level-zero res: ",
                              to_string(res),
                              ", rank: ",
                              rank,
                              ", buf_idx: ",
                              buf_idx,
                              ", ptr: ",
                              mem_ptr);
                }
            }

            // TODO: remove, when the fix arrives from L0 side: XDEPS-2302
            int fd;
            memcpy(&fd, handle.data, sizeof(fd));
            close(fd);
        }
    }

    if (!handles.empty()) {
        LOG_DEBUG("handles are cleared successfully");
    }

    handles.clear();
}

void ipc_handle_manager::set(const mem_handle_map_t& handles_arg) {
    CCL_THROW_IF_NOT(!handles_arg.empty(), "handles_arg argument is empty");
    CCL_THROW_IF_NOT(handles_arg.size() == static_cast<size_t>(comm->size()),
                     "handles_arg and comm sizes should be equal");
    CCL_THROW_IF_NOT(handles.empty(), "handles should be empty before set");

    handles = handles_arg;
    LOG_DEBUG("handles are set successfully, size of handles: ", handles.size());
}

void ipc_handle_manager::get(int rank, size_t buf_idx, ccl_buffer& buf, ccl_comm* map_comm) {
    check_rank(rank, (map_comm) ? map_comm : comm);
    if (map_comm && (map_comm->id() != comm->id())) {
        int old_rank = rank;
        rank = map_comm->get_global_rank(rank);
        auto rank_it = rank_map.find(rank);
        if (rank_it == rank_map.end()) {
            CCL_THROW("handle manager can not handle global rank ", rank);
        }
        rank = rank_it->second;
        LOG_DEBUG("convert rank: old_rank: ",
                  old_rank,
                  " old_comm: id: ",
                  map_comm->id(),
                  ", size: ",
                  map_comm->size(),
                  ", new_rank: ",
                  rank,
                  " new_comm: id: ",
                  comm->id(),
                  ", size: ",
                  comm->size());
        check_rank(rank, comm);
    }
    CCL_THROW_IF_NOT(buf_idx < handles[rank].size(), "buf_idx is not valid value: ", buf_idx);

    const auto& handle_info = handles[rank][buf_idx];
    auto handle = handle_info.handle;
    auto mem_ptr = handle_info.ptr;
    auto mem_type = handle_info.type;

    LOG_DEBUG("context: ", context, ", device: ", device, ", rank: ", rank, ", buf_idx: ", buf_idx);
    if (mem_ptr == nullptr) {
        if (mem_type == ccl::ze::ipc_mem_type::memory) {
            open_handle(handle, &mem_ptr);
        }
        else if (mem_type == ccl::ze::ipc_mem_type::pool) {
            ze_ipc_event_pool_handle_t pool_handle;
            cast_mem_to_pool_handle(&pool_handle, &handle);
            open_handle(pool_handle, (ze_event_pool_handle_t*)&mem_ptr);
        }
        else {
            CCL_THROW("unknown memory type");
        }
    }

    LOG_DEBUG("get handle: { mem_ptr: ",
              mem_ptr,
              ", fd: ",
              *(int*)handle.data,
              ", rank: ",
              rank,
              ", buf_idx: ",
              buf_idx,
              " }");

    // add offset that we received along with the handle
    size_t mem_offset = handle_info.offset;
    void* final_ptr = static_cast<void*>(static_cast<char*>(mem_ptr) + mem_offset);
    buf.set(final_ptr);
}

void ipc_handle_manager::get_handle(const void* ptr, ze_ipc_mem_handle_t* handle) {
    CCL_THROW_IF_NOT(ptr, "no mem pointer");
    ZE_CALL(zeMemGetIpcHandle, (context, ptr, handle));
}

void ipc_handle_manager::get_handle(ze_event_pool_handle_t pool,
                                    ze_ipc_event_pool_handle_t* handle) {
    CCL_THROW_IF_NOT(pool, "no pool");
    ZE_CALL(zeEventPoolGetIpcHandle, (pool, handle));
}

void ipc_handle_manager::open_handle(const ze_ipc_mem_handle_t& handle, void** ptr) {
    ZE_CALL(zeMemOpenIpcHandle, (context, device, handle, 0 /* cache allocation */, ptr));
}

void ipc_handle_manager::open_handle(const ze_ipc_event_pool_handle_t& handle,
                                     ze_event_pool_handle_t* pool) {
    ZE_CALL(zeEventPoolOpenIpcHandle, (context, handle, pool));
}

void ipc_handle_manager::get_address_range(const void* ptr, void** base_ptr, size_t* size) {
    ZE_CALL(zeMemGetAddressRange, (context, ptr, base_ptr, size));
    LOG_DEBUG("zeMemGetAddressRange: ptr: ",
              ptr,
              ", base ptr: ",
              *base_ptr,
              ", offset: ",
              ccl_get_ptr_diff(*base_ptr, ptr),
              ", size: ",
              *size);
}

void ipc_handle_manager::check_rank(int rank, ccl_comm* check_comm) {
    CCL_THROW_IF_NOT(
        (rank >= 0) && (rank < static_cast<int>(handles.size())) && (rank < check_comm->size()),
        "rank is not valid value: ",
        rank);
    CCL_THROW_IF_NOT(
        rank != check_comm->rank(), "don't expect to open handle for own rank: ", rank);
}

} // namespace ze
} // namespace ccl
