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
    const std::vector<int>& local_thread_device_group_ranks,
    int cluster_device_group_size,
    std::shared_ptr<kvs_interface> kvs) {
    LOG_INFO("Thread acquire by barrier");
    std::shared_ptr<ikvs_wrapper> kvs_wrap = std::shared_ptr<ikvs_wrapper>(new users_kvs(kvs));
    std::shared_ptr<atl_wrapper> atl = std::shared_ptr<atl_wrapper>(
        new atl_wrapper(cluster_device_group_size, local_thread_device_group_ranks, kvs_wrap));

    /* Indicate that multiple devices are not supported, don't throw anything if kernel_path env variable
     * is set to enable our testing with partial functionality.
     * Most of the cases are handled in communicator_impl_details.hpp, but here we check the case
     * when we have multiple threads and each of them has 1 device. And we don't know the total number
     * of ranks in the process until we sync them above */
    if (atl->get_ranks_per_process() > 1 && ccl::global_data::env().kernel_path.empty()) {
        throw ccl::unimplemented("API", "create_communicators", "for multiple devices");
    }

    LOG_INFO("Thread released by barrier");
    LOG_INFO("Cluster_device_group size: ",
             cluster_device_group_size,
             "\nThread device group ranks size: ",
             local_thread_device_group_ranks.size());
    for (size_t i = 0; i < local_thread_device_group_ranks.size(); i++) {
        LOG_INFO("\nLocal thread device group ranks: ", local_thread_device_group_ranks[i]);
    }
    // register group slot in global context table, based on communicator id
    comm_group_t group = group_context::group_by_comm(atl);

    // sync existing group: blocking operation - wait for all groups
    LOG_INFO("group thread barrier acquired: ", static_cast<void*>(group.get()));
    group->sync_group_size(local_thread_device_group_ranks.size());
    LOG_INFO("group thread barrier released: ", static_cast<void*>(group.get()));
    return group;
}

group_context::comm_group_t group_context::group_by_comm(std::shared_ptr<atl_wrapper> atl) {
    LOG_INFO("\n",
             "\nATL info:",
             "\n  threads per process: ",
             atl->get_threads_per_process(),
             "\n  ranks per process:   ",
             atl->get_ranks_per_process(),
             "\n  atl size:            ",
             atl->get_size(),
             "\n  rank:                ",
             atl->get_rank(),
             "\n  unique id of atl:     ",
             atl->get_id(),
             "\n")

    comm_group_t group;
    {
        // mutex
        std::unique_lock<ccl_spinlock> lock(mutex);
        size_t threads_per_process = atl->get_threads_per_process();
        size_t ranks_per_process = atl->get_ranks_per_process();
        group_context::group_unique_key unique_id = atl->get_id();

        auto ctx_it = communicator_group_map.find(unique_id);
        if (ctx_it == communicator_group_map.end()) {
            std::shared_ptr<host_communicator> host_comm = std::make_shared<host_communicator>(atl);
            group.reset(
                new ccl::comm_group(host_comm, threads_per_process, ranks_per_process, unique_id));
            communicator_group_map.insert({ unique_id, group });
            LOG_INFO("Comm group: ",
                     static_cast<void*>(group.get()),
                     " has been created for unique_id: ",
                     unique_id,
                     ", threads per process: ",
                     threads_per_process,
                     ", ranks per process: ",
                     ranks_per_process);
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
            throw ccl::exception(std::string(__FUNCTION__) + " - " + mess);
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
