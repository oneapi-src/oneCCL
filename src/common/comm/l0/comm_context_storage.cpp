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
#include "common/comm/host_communicator/host_communicator.hpp"

#include "common/comm/comm.hpp"
#include "common/comm/l0/comm_context.hpp"
#include "common/comm/l0/comm_context_storage.hpp"

#include "common/global/global.hpp"

namespace ccl {
group_context& group_context::instance() {
    static group_context inst;
    return inst;
}

group_context::comm_group_t group_context::group_by_kvs(
    const std::vector<size_t>& local_thread_device_group_ranks,
    size_t cluster_device_group_size,
    std::shared_ptr<kvs_interface> kvs) {
    //TODO
    static ccl_comm_id_storage::comm_id TODO_TMP_ID = ccl::global_data::get().comm_ids->acquire();

    //barrier operation acquire: wait while all threads from all processes enters here...
    std::shared_ptr<host_communicator> host_comm =
        std::make_shared<host_communicator>(std::make_shared<ccl_comm>(
            local_thread_device_group_ranks, cluster_device_group_size, kvs, TODO_TMP_ID.clone()));
    //barrier operation release: every threads continue its execution here...
    LOG_INFO("Thread released by barrier");

    // register group slot in global context table, based on communicator id
    comm_group_t group = group_context::group_by_comm(host_comm);

    // sync existing group: blocking operation - wait for all groups
    LOG_INFO("group thread barrier acquired: ", static_cast<void*>(group.get()));
    group->sync_group_size(local_thread_device_group_ranks.size());
    LOG_INFO("group thread barrier released: ", static_cast<void*>(group.get()));
    return group;
}

group_context::comm_group_t group_context::group_by_comm(shared_communicator_t host_comm) {
    group_context::group_unique_key unique_id = host_comm->comm_impl->id();
    size_t threads_count = host_comm->comm_impl->thread_count();
    size_t on_process_ranks_count = host_comm->comm_impl->on_process_ranks_count();

    comm_group_t group;
    {
        std::unique_lock<ccl_spinlock> lock(mutex);
        auto ctx_it = communicator_group_map.find(unique_id);
        if (ctx_it == communicator_group_map.end()) {
            group.reset(
                new ccl::comm_group(host_comm, threads_count, on_process_ranks_count, unique_id));
            communicator_group_map.insert({ unique_id, group });
            LOG_INFO("Comm group: ",
                     static_cast<void*>(group.get()),
                     " has been created for unique_id: ",
                     unique_id,
                     ", expected thread count: ",
                     threads_count,
                     ", on process rank count: ",
                     on_process_ranks_count);
        }
        else {
            group = ctx_it->second;
            LOG_INFO("get existing comm group: ",
                     static_cast<void*>(group.get()),
                     " for unique_id: ",
                     unique_id);
        }
    }
    return group;
}

group_context::comm_group_t group_context::get_existing_group_by_id(
    const group_unique_key& unique_id) {
    comm_group_t group;
    LOG_DEBUG("get existing comm group by id: ",
              unique_id,
              ", total groups: ",
              communicator_group_map.size());
    {
        std::unique_lock<ccl_spinlock> lock(mutex);
        auto ctx_it = communicator_group_map.find(unique_id);
        if (ctx_it == communicator_group_map.end()) {
            std::stringstream ss;
            ss << "Cannot find `comm_group_t` by id: " << unique_id << std::endl;
            const std::string mess = ss.str();
            LOG_ERROR(mess);
            throw ccl_error(std::string(__FUNCTION__) + " - " + mess);
        }
        else {
            group = ctx_it->second;
            LOG_DEBUG("get existing comm group: ",
                      static_cast<void*>(group.get()),
                      " for unique_id: ",
                      unique_id);
        }
    }
    return group;
}
} // namespace ccl
