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

#include "common/utils/spinlock.hpp"
#include "common/comm/atl_tag.hpp"
#include "atl/atl_wrapper.h"

namespace ccl {
namespace v1 {
class kvs_interface;
}

class host_communicator;
class comm_group;

struct group_context {
    /* TODO
     * In multithreading scenario we use different comm_group_t objects in different threads.
     * But we need to match different groups created for the same world in different threads
     * The assumption is done: if different groups created from the same communicator color, than they
     * should be interpreted as the same groups in the same world.
     *
     *
     * In the final solution the 'group_unique_key' should be equal to unique KVS idenditifier
     */
    //    using group_unique_key = typename ccl::ccl_host_attributes_traits<ccl_host_color>::type;
    using group_unique_key = ccl_comm_id_t;
    using comm_group_t = std::shared_ptr<comm_group>;
    std::map<group_unique_key, comm_group_t> communicator_group_map;
    ccl_spinlock mutex;

    comm_group_t group_by_kvs(const std::vector<int>& local_thread_device_group_ranks,
                              int cluster_device_group_size,
                              std::shared_ptr<kvs_interface> kvs);
    comm_group_t group_by_comm(std::shared_ptr<atl_wrapper> atl);
    comm_group_t get_existing_group_by_id(const group_unique_key& id);
    static group_context& instance();

private:
    group_context() = default;
    group_context(group_context&& src) = delete;
    group_context& operator=(group_context&& src) = delete;
};
} // namespace ccl
