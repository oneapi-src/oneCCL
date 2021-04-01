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
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "oneapi/ccl/types.hpp"
#include "common/comm/l0/device_group_routing_schema.hpp"
#include "common/comm/l0/context/context_barrier.hpp"

#include "common/comm/l0/comm_context_id.hpp"
#include "common/comm/l0/context_comm_addr.hpp"

namespace native {
struct process_group_context;
struct thread_group_context;
} // namespace native

namespace ccl {
namespace v1 {
class communicator;
}
class host_communicator;
struct communicator_interface;

struct gpu_comm_attr {
public:
    friend class device_group_ring_communicator;
    friend class device_group_a2a_communicator;
    friend class thread_device_group_ring_communicator;
    friend class thread_device_group_a2a_communicator;
    friend class process_ring_communicator;
    friend class process_a2a_communicator;
    friend class comm_group;

    using thread_comm_storage = std::multimap<size_t, std::shared_ptr<communicator_interface>>;

    gpu_comm_attr(std::shared_ptr<host_communicator> parent_comm,
                  size_t thread_count,
                  size_t process_device_size,
                  group_unique_key id);
    ~gpu_comm_attr();

    std::shared_ptr<::native::process_group_context> get_process_context();
    bool sync_group_size(size_t device_group_size);
    bool sync_register_communicator(std::shared_ptr<communicator_interface> comm);

    std::shared_ptr<host_communicator> get_host_communicator();

    const group_unique_key& get_unique_id() const;
    size_t get_expected_process_device_size() const noexcept;

private:
    bool delegate_sync_register_communicator(std::shared_ptr<communicator_interface> comm);

    std::shared_ptr<host_communicator> ccl_communicator;
    size_t expected_threads_count;
    size_t expected_process_device_size;
    group_unique_key unique_id;
    std::shared_ptr<::native::process_group_context> ctx;

    std::mutex thread_group_size_mutex;
    std::condition_variable thread_group_size_cond;
    std::vector<size_t> thread_device_group_sizes;
    thread_comm_storage thread_communicators;
    bool ready = false;

    ::native::signal_context barrier;

    //context_comm_addr bind_thread_addr;

    static thread_local size_t thread_id;
};
} // namespace ccl
