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

template <class kernel_params,
          ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          template <class, class, ccl::group_split_type>
          class algorithm>
struct communication_process_device_expander {
    template <class device_t, class... Args>
    void operator()(native::device_t_ptr<device_t>& comm_device,
                    std::shared_ptr<native::process_group_context>& ctx,
                    typename native::device_community_container<class_id>::element_type community,
                    size_t process_id,
                    size_t thread_id,
                    Args&&... args) {
        if (comm_device) {
            LOG_DEBUG("Invoke: ", comm_device->to_string());

            using gpu_entry = algorithm<kernel_params, device_t, group_id>;

            schedule = ctx->scheduler_impl
                           ->submit_entry<gpu_entry, ccl_sched_add_back, group_id, class_id>(
                               process_id,
                               thread_id,
                               *community,
                               comm_device,
                               std::forward<Args>(args)...);
        }
    }

    std::shared_ptr<ccl_gpu_sched> schedule;
};

template <class kernel_params,
          ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          template <class, class, ccl::group_split_type>
          class algorithm,
          class... Args>
std::unique_ptr<ccl::event_impl> do_collective_op(
    native::device_variant_t<native::ccl_gpu_comm,
                             native::ccl_virtual_gpu_comm,
                             native::ccl_ipc_source_gpu_comm<native::ccl_gpu_comm>,
                             native::ccl_ipc_source_gpu_comm<native::ccl_virtual_gpu_comm>,
                             native::ccl_numa_proxy<native::ccl_gpu_comm>,
                             native::ccl_numa_proxy<native::ccl_virtual_gpu_comm>,
                             native::ccl_scaleout_proxy<native::ccl_gpu_comm>,
                             native::ccl_scaleout_proxy<native::ccl_virtual_gpu_comm>>&
        communication_device,
    std::shared_ptr<native::process_group_context>& ctx,
    typename native::device_community_container<class_id>::element_type community,
    size_t process_id,
    size_t thread_id,
    native::ccl_driver_context_ptr native_context,
    Args&&... args) {
    communication_process_device_expander<kernel_params, group_id, class_id, algorithm> expander;
    ccl_tuple_for_each_args(communication_device,
                            expander,
                            ctx,
                            community,
                            process_id,
                            thread_id,
                            native_context,
                            std::forward<Args>(args)...);
    if (expander.schedule) {
        LOG_DEBUG("Device group finalized");
    }
    return std::unique_ptr<ccl::event_impl>(
        new ccl::gpu_shared_event_impl(std::move(expander.schedule)));
}

template <class buffer_type,
          ccl::group_split_type group_id,
          ccl::device_topology_type class_id,
          template <class, class, ccl::group_split_type>
          class algorithm,
          class... Args>
std::unique_ptr<ccl::event_impl> do_collective_op_reductions(
    ccl::reduction reduction,
    native::device_variant_t<native::ccl_gpu_comm,
                             native::ccl_virtual_gpu_comm,
                             native::ccl_ipc_source_gpu_comm<native::ccl_gpu_comm>,
                             native::ccl_ipc_source_gpu_comm<native::ccl_virtual_gpu_comm>,
                             native::ccl_numa_proxy<native::ccl_gpu_comm>,
                             native::ccl_numa_proxy<native::ccl_virtual_gpu_comm>,
                             native::ccl_scaleout_proxy<native::ccl_gpu_comm>,
                             native::ccl_scaleout_proxy<native::ccl_virtual_gpu_comm>>&
        communication_device,
    std::shared_ptr<native::process_group_context>& ctx,
    typename native::device_community_container<class_id>::element_type community,
    size_t process_id,
    size_t thread_id,
    native::ccl_driver_context_ptr native_context,
    Args&&... args) {
    switch (reduction) {
        case ccl::reduction::sum:
            return do_collective_op<
                kernel_reduction_params_traits<buffer_type, ccl_coll_reduction::sum>,
                group_id,
                class_id,
                algorithm>(communication_device,
                           ctx,
                           community,
                           process_id,
                           thread_id,
                           native_context,
                           std::forward<Args>(args)...);
            break;
        case ccl::reduction::prod:
            return do_collective_op<
                kernel_reduction_params_traits<buffer_type, ccl_coll_reduction::prod>,
                group_id,
                class_id,
                algorithm>(communication_device,
                           ctx,
                           community,
                           process_id,
                           thread_id,
                           native_context,
                           std::forward<Args>(args)...);
            break;
        case ccl::reduction::min:
            return do_collective_op<
                kernel_reduction_params_traits<buffer_type, ccl_coll_reduction::min>,
                group_id,
                class_id,
                algorithm>(communication_device,
                           ctx,
                           community,
                           process_id,
                           thread_id,
                           native_context,
                           std::forward<Args>(args)...);
            break;
        case ccl::reduction::max:
            return do_collective_op<
                kernel_reduction_params_traits<buffer_type, ccl_coll_reduction::max>,
                group_id,
                class_id,
                algorithm>(communication_device,
                           ctx,
                           community,
                           process_id,
                           thread_id,
                           native_context,
                           std::forward<Args>(args)...);
            break;
        // TODO: make support of custom reduction in *.cl
        // case ccl::reduction::custom:
        //     return do_collective_op<kernel_reduction_params_traits<buffer_type, ccl_coll_reduction::custom>,
        //                            group_id, class_id, algorithm>(
        //                                                      communication_device,
        //                                                      ctx,
        //                                                      community,
        //                                                      process_id,
        //                                                      thread_id,
        //                                                      native_context,
        //                                                      std::forward<Args>(args)...);
        //     break;
        default:
            throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                     "Obtained reduction by user is incorrect!");
    }
}
