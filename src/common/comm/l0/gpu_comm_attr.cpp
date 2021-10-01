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
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/device_community.hpp"
#include "common/comm/comm_interface.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"

#include "common/comm/host_communicator/host_communicator.hpp"

namespace ccl {

thread_local size_t gpu_comm_attr::thread_id = 0;

gpu_comm_attr::gpu_comm_attr(std::shared_ptr<host_communicator> parent_comm,
                             size_t thread_count,
                             size_t process_device_size,
                             group_unique_key id)
        : ccl_communicator(parent_comm),
          expected_threads_count(thread_count),
          expected_process_device_size(process_device_size),
          unique_id(id) {
    ctx = std::make_shared<native::process_group_context>(ccl_communicator);
}

const group_unique_key& gpu_comm_attr::get_unique_id() const {
    return unique_id;
}

std::shared_ptr<host_communicator> gpu_comm_attr::get_host_communicator() {
    return ccl_communicator;
}

bool gpu_comm_attr::sync_group_size(size_t device_group_size) {
    std::unique_lock<std::mutex> lock(thread_group_size_mutex);

    thread_id = thread_device_group_sizes.size();

    thread_device_group_sizes.push_back(device_group_size);
    if (thread_device_group_sizes.size() != expected_threads_count) {
        // slave threads
        thread_group_size_cond.wait(lock, [this]() {
            return ready;
        });
        return false;
    }

    // master thread
    ccl::stream::impl_value_t empty_stream{};
    ccl_communicator->barrier_impl(empty_stream, ccl::default_barrier_attr, {});

    ready = true;
    thread_group_size_cond.notify_all();
    return true;
}

gpu_comm_attr::~gpu_comm_attr() {}

bool gpu_comm_attr::sync_register_communicator(std::shared_ptr<communicator_interface> comm) {
    if (!delegate_sync_register_communicator(comm)) {
        //SlAVE threads or non-completed ccommunicators count
        if (barrier.communicator_ready) {
            // SLAVE thread
            LOG_DEBUG("Process Group for thread(SLAVE) id: ", thread_id, " is ready");
        }
        return false;
    }

    // MASTER thread
    LOG_DEBUG("Process Group for thread(MASTER) id: ", thread_id, " is ready");
    return true;
}

bool gpu_comm_attr::delegate_sync_register_communicator(
    std::shared_ptr<communicator_interface> comm) {
    ccl::device_indices_type device_group_indices;

    std::unique_lock<std::mutex> lock(barrier.thread_group_mutex);

    //sanity check for formed group
    if (thread_communicators.count(thread_id) >= thread_device_group_sizes[thread_id]) {
        //find device_id, we cannot add more device into group than expected
        auto range = thread_communicators.equal_range(thread_id);
        auto it = std::find_if(
            range.first,
            range.second,
            [&comm](const typename thread_comm_storage::value_type& existing_comm) {
                return comm->get_device_path() == existing_comm.second->get_device_path();
            });

        if (it == range.second) {
            LOG_ERROR("Attempt to create communicator by new device id: ",
                      comm->get_device_path(),
                      " in fully formed comm_group is unaccepted!");
            throw ccl::exception("cannot create communicator for requested device");
        }

        //set rank & size for duplicated communicator
        comm->visit(*this);
        return true;
    }

    //group is not formed yet
    thread_communicators.insert({ thread_id, comm });
    size_t registered = thread_communicators.count(thread_id);
    size_t expected = thread_device_group_sizes[thread_id];
    LOG_DEBUG("Thread id: ",
              thread_id,
              " register communicators count: [",
              registered,
              "/",
              expected,
              "]");
    if (registered != expected) {
        return false; //comm group is not reached expected size
    }

    // current thread create all own communicators, start sync context
    auto range = thread_communicators.equal_range(thread_id);
    for (auto it = range.first; it != range.second; ++it) {
        device_group_indices.insert(
            it->second->get_device_path()); //.get_device_properties().deviceId);
    }
    {
        /* TODO: enable back */
        // std::stringstream ss;
        // for(const auto &path : device_group_indices)
        // {
        //     ss << path << ", ";
        // }
        // LOG_DEBUG("Thread id: ", thread_id, " collected device indices: ", ss.str());
    }

    //bind addr
    context_comm_addr bind_thread_addr;
    bind_thread_addr.thread_idx = thread_id;
    bind_thread_addr.thread_count = expected_threads_count;
    if (!ctx->sync_barrier(device_group_indices, bind_thread_addr)) {
        //SLAVE thread waits
        LOG_DEBUG("Thread (SLAVE) id: ", thread_id, " waits on barrier");
        barrier.thread_group_sync_condition.wait(lock, [this]() {
            return barrier.communicator_ready;
        });

        //flush cache
        auto ready_count = std::count_if(thread_communicators.begin(),
                                         thread_communicators.end(),
                                         [](const typename thread_comm_storage::value_type& comm) {
                                             return comm.second->is_ready();
                                         });
        if ((size_t)ready_count != expected_process_device_size) {
            LOG_ERROR("Thread(SLAVE) id: ",
                      thread_id,
                      " not all communicators ready: (",
                      ready_count,
                      "/",
                      expected_process_device_size,
                      "). Abort");
            abort();
        }
        LOG_DEBUG("Thread(SLAVE) id: ",
                  thread_id,
                  " detected communicators ready: (",
                  ready_count,
                  "/",
                  expected_process_device_size,
                  ")");
        return false;
    }

    //MASTER threads

    //finalize communicator creation
    LOG_INFO("Finalize communicators creation, total count:", thread_communicators.size());
    for (auto comm_it = thread_communicators.begin(); comm_it != thread_communicators.end();
         ++comm_it) {
        comm_it->second->visit(*this);
    }

    ccl::stream::impl_value_t empty_stream{};
    ccl_communicator->barrier_impl(empty_stream, ccl::default_barrier_attr, {});

    //notify SLAVES thread ready
    barrier.communicator_ready = true;
    barrier.thread_group_sync_condition.notify_all();
    return true;
}

size_t gpu_comm_attr::get_expected_process_device_size() const noexcept {
    return expected_process_device_size;
}

std::shared_ptr<native::process_group_context> gpu_comm_attr::get_process_context() {
    return ctx;
}

} // namespace ccl
