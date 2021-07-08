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
#include "common/comm/l0/devices/devices_declaration.hpp"
#include "common/comm/l0/device_community.hpp"

template <ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          template <class, ccl::group_split_type>
          class algorithm>
struct communication_thread_device_expander {
    template <class device_t, class... Args>
    void operator()(native::device_t_ptr<device_t>& comm_device,
                    std::shared_ptr<native::thread_group_context>& ctx,
                    typename native::device_community_container<class_id>::element_type community,
                    size_t thread_id,
                    Args&&... args) {
        if (comm_device) {
            LOG_DEBUG("Invoke: ", comm_device->to_string());

            using gpu_entry = algorithm<device_t, group_id>;

            schedule = ctx->scheduler_impl
                           ->submit_entry<gpu_entry, ccl_sched_add_back, group_id, class_id>(
                               thread_id, *community, comm_device, std::forward<Args>(args)...);
        }
    }

    std::shared_ptr<ccl_gpu_sched> schedule;
};

template <ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          template <class, ccl::group_split_type>
          class algorithm,
          class... Args>
std::unique_ptr<ccl::event_impl> do_collective_op(
    native::device_variant_t<native::ccl_gpu_comm, native::ccl_virtual_gpu_comm>&
        communication_device,
    std::shared_ptr<native::thread_group_context>& ctx,
    typename native::device_community_container<class_id>::element_type community,
    size_t thread_id,
    native::ccl_driver_context_ptr native_context,
    Args&&... args) {
    communication_thread_device_expander<group_id, class_id, algorithm> expander;
    ccl_tuple_for_each_args(communication_device,
                            expander,
                            ctx,
                            community,
                            thread_id,
                            native_context,
                            std::forward<Args>(args)...);
    if (expander.schedule) {
        LOG_DEBUG("Device group finalized");
    }
    return std::unique_ptr<ccl::event_impl>(
        new ccl::gpu_shared_event_impl(std::move(expander.schedule)));
}
