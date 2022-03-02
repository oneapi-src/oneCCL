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
#include "common/stream/stream.hpp"
#include "common/utils/buffer.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <unordered_map>
#include "common/ze/ze_api_wrapper.hpp"

class ccl_comm;

namespace ccl {

namespace ze {

enum class ipc_mem_type : int { unknown = 0, memory, pool };

std::string to_string(ipc_mem_type type);

struct ipc_handle_desc {
    ze_ipc_mem_handle_t handle{};
    size_t mem_offset{};
    void* mem_ptr{};
    ipc_mem_type mem_type{};
    pid_t remote_pid{};
    ssize_t remote_context_id{ -1 };
    uint64_t remote_mem_alloc_id{};
    ssize_t remote_device_id{ -1 };

    bool is_cached = false;

    ipc_handle_desc();
    ipc_handle_desc(const ze_ipc_mem_handle_t& handle, size_t offset, ipc_mem_type type);
    ipc_handle_desc(const ipc_handle_desc&) = default;
    ipc_handle_desc& operator=(const ipc_handle_desc&) = default;
};

class ipc_handle_manager {
public:
    using mem_handle_map_t = typename std::vector<std::vector<ipc_handle_desc>>;

    ipc_handle_manager() = default;
    ipc_handle_manager(const ipc_handle_manager&) = delete;
    ipc_handle_manager& operator=(const ipc_handle_manager&) = delete;
    ~ipc_handle_manager();

    void init(const ccl_comm* comm, const ccl_stream* stream);
    void clear();

    void set(const mem_handle_map_t& handles_arg);

    void* get_ptr(int rank, size_t buf_idx, ccl_comm* map_comm);
    void get(int rank, size_t buf_idx, ccl_buffer& buf, ccl_comm* map_comm = nullptr);
    void get(int rank, size_t buf_idx, ze_event_pool_handle_t& buf, ccl_comm* map_comm);

    void get_handle(const void* buffer, ze_ipc_mem_handle_t* handle);
    void get_handle(ze_event_pool_handle_t pool, ze_ipc_event_pool_handle_t* handle);
    void open_handle(ipc_handle_desc& info, void** ptr);
    void open_handle(const ze_ipc_event_pool_handle_t& handle, ze_event_pool_handle_t* pool);

    void get_address_range(const void* ptr, void** base_ptr, size_t* size);

private:
    ze_context_handle_t context{};
    ze_device_handle_t device{};
    ccl_comm* comm{};
    std::unordered_map<int, int> rank_map{};
    mem_handle_map_t handles;

    /**
     * The value can be destroyed in the cache if the cache reaches its limit.
     * This can happen at a time when the handle is really needed.
     * We can run a lot of ranks and get fail here.
     * Instead, the value will be popped from the cache, but only destroyed when not needed.
     * We rely on the smart pointer to work.
     * So, in cached_handles we just save handles to increase ref counter
     */
    std::list<mem_handle_cache::value_t> cached_handles;

    void check_rank(int rank, ccl_comm* check_comm);
};

} // namespace ze
} // namespace ccl
