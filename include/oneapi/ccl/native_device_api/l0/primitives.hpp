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
#include <array>

#include <ze_api.h>

#include "oneapi/ccl/native_device_api/l0/base.hpp"

namespace native {

struct ccl_device_platform;
class ccl_event_pool;
struct ccl_context;

std::string to_string(const ze_result_t result);
std::string to_string(ze_memory_type_t type);
std::string to_string(ze_memory_access_cap_flags_t cap);
std::string to_string(const ze_device_properties_t& device_properties,
                      const std::string& prefix = std::string("\n"));
std::string to_string(const ze_device_memory_properties_t& device_mem_properties,
                      const std::string& prefix = std::string("\n"));
std::string to_string(const ze_device_memory_access_properties_t& mem_access_prop,
                      const std::string& prefix = std::string("\n"));
std::string to_string(const ze_device_compute_properties_t& compute_properties,
                      const std::string& prefix = std::string("\n"));
std::string to_string(const ze_memory_allocation_properties_t& prop);
std::string to_string(const ze_device_p2p_properties_t& properties);
std::string to_string(const ze_device_mem_alloc_desc_t& mem_descr);
std::string to_string(const ze_ipc_mem_handle_t& handle);
std::string to_string(const ze_command_queue_desc_t& queue_descr);
std::string to_string(const ze_command_list_desc_t& list_descr);

/**
 * Specific L0 primitives declaration
 */
template <class resource_owner, class cl_context>
using queue = cl_base<ze_command_queue_handle_t, resource_owner, cl_context>;

template <class resource_owner, class cl_context>
using cmd_list = cl_base<ze_command_list_handle_t, resource_owner, cl_context>;

template <class resource_owner, class cl_context>
using module = cl_base<ze_module_handle_t, resource_owner, cl_context>;

template <class resource_owner, class cl_context>
using ipc_memory_handle = cl_base<ze_ipc_mem_handle_t, resource_owner, cl_context>;

template <class resource_owner, class cl_context>
using queue_fence = cl_base<ze_fence_handle_t, resource_owner, cl_context>;

/*
template <class elem_t,
          class resource_owner,
          class cl_context,
          class = typename std::enable_if<ccl::is_supported<elem_t>()>::type>
struct memory;
*/

struct event : private cl_base<ze_event_handle_t, ccl_event_pool, ccl_context> {
    using base = cl_base<ze_event_handle_t, ccl_event_pool, ccl_context>;
    using base::get_owner;
    using base::get_ctx;
    using base::handle;

    using base::base;

    bool wait(uint64_t nanosec = std::numeric_limits<uint64_t>::max()) const;
    ze_result_t status() const;
    void signal();
};

template <class elem_t, class resource_owner, class cl_context>
struct memory /*<elem_t, resource_owner, cl_context>*/ : private cl_base<elem_t*,
                                                                         resource_owner,
                                                                         cl_context> {
    using base = cl_base<elem_t*, resource_owner, cl_context>;
    using base::get_owner;
    using base::get_ctx;
    using base::handle;

    using event_t = event;

    memory(elem_t* h,
           size_t count,
           std::weak_ptr<resource_owner>&& owner,
           std::weak_ptr<cl_context>&& context);

    /**
     *  Memory operations
     */
    // sync memory-copy write
    void enqueue_write_sync(const std::vector<elem_t>& src);
    void enqueue_write_sync(typename std::vector<elem_t>::const_iterator first,
                            typename std::vector<elem_t>::const_iterator last);
    template <int N>
    void enqueue_write_sync(const std::array<elem_t, N>& src);
    void enqueue_write_sync(const elem_t* src, size_t n);
    void enqueue_write_sync(const elem_t* src, int n);

    // async
    event_t enqueue_write_async(const std::vector<elem_t>& src,
                                queue<resource_owner, cl_context>& queue);
    event_t enqueue_write_async(typename std::vector<elem_t>::const_iterator first,
                                typename std::vector<elem_t>::const_iterator last);
    template <int N>
    event_t enqueue_write_async(const std::array<elem_t, N>& src,
                                queue<resource_owner, cl_context>& queue);
    event_t enqueue_write_async(const elem_t* src,
                                size_t n,
                                queue<resource_owner, cl_context>& queue);

    // sync memory-copy read
    std::vector<elem_t> enqueue_read_sync(size_t requested_size = 0) const;

    // async memory-copy
    //TODO
    elem_t* get() noexcept;
    const elem_t* get() const noexcept;

    size_t count() const noexcept;
    size_t size() const noexcept;

private:
    size_t elem_count;
};

struct ip_memory_elem_t {
    void* pointer = nullptr;
};

template <class resource_owner, class cl_context>
using ipc_memory = cl_base<ip_memory_elem_t, resource_owner, cl_context>;

std::string get_build_log_string(const ze_module_build_log_handle_t& build_log);
struct command_queue_desc_comparator {
    bool operator()(const ze_command_queue_desc_t& lhs, const ze_command_queue_desc_t& rhs) const;
};

struct command_list_desc_comparator {
    bool operator()(const ze_command_list_desc_t& lhs, const ze_command_list_desc_t& rhs) const;
};

} // namespace native
