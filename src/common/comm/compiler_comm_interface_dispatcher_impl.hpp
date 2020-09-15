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
#include "common/comm/comm_attributes.hpp"

#include "common/comm/compiler_comm_interface_dispatcher.hpp"

#include "common/comm/comm_interface.hpp"
#ifdef MULTI_GPU_SUPPORT
#include "common/comm/l0/communicator/device_group/device_ring_communicator.hpp"
#include "common/comm/l0/communicator/device_group/device_a2a_communicator.hpp"
#include "common/comm/l0/communicator/thread_group/thread_ring_communicator.hpp"
#include "common/comm/l0/communicator/thread_group/thread_a2a_communicator.hpp"
#include "common/comm/l0/communicator/process_group/process_ring_communicator.hpp"
#include "common/comm/l0/communicator/process_group/process_a2a_communicator.hpp"
#endif
#include "common/comm/host_communicator/host_communicator_impl.hpp"

namespace ccl {
communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(
    const comm_attr_t& attr) {
    return communicator_interface_ptr(new host_communicator(attr));
}

#ifdef MULTI_GPU_SUPPORT
template <class DeviceType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type>
communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(
    const DeviceType& device,
    size_t thread_idx,
    size_t process_idx,
    const ccl::device_comm_attr_t& attr) {
    static_assert(std::is_same<typename unified_device_type::device_t, DeviceType>::value,
                  "Unsupported 'DeviceType'");

    return communicator_interface_dispatcher::create_communicator_from_unified_device(
        unified_device_type(device), thread_idx, process_idx, attr);
}

template <class DeviceType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type>
communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(
    DeviceType device_id,
    size_t thread_idx,
    size_t process_idx,
    const ccl::device_comm_attr_t& attr) {
#ifdef CCL_ENABLE_SYCL
    return communicator_interface_dispatcher::create_communicator_from_unified_device(
        unified_device_type(device_id, cl::sycl::info::device_type::gpu),
        thread_idx,
        process_idx,
        attr);
#else
    static_assert(std::is_same<typename unified_device_type::device_t, DeviceType>::value,
                  "Unsupported 'DeviceType'");
    return communicator_interface_dispatcher::create_communicator_from_unified_device(
        unified_device_type(device_id), thread_idx, process_idx, attr);
#endif
}

communicator_interface_ptr
communicator_interface_dispatcher::create_communicator_from_unified_device(
    ccl::unified_device_type&& device_id,
    size_t thread_idx,
    size_t process_idx,
    const ccl::device_comm_attr_t& attr) {
    ccl::device_topology_type preferred_topology_class;
    ccl::device_group_split_type preferred_topology_group;
    if (attr) {
        preferred_topology_class = attr->get_value<ccl_device_preferred_topology_class>();
        preferred_topology_group = attr->get_value<ccl_device_preferred_group>();
    }
    else {
        preferred_topology_class = device_attr_impl::class_default();
        preferred_topology_group = device_attr_impl::group_default();
    }

    //TODO check device_id or sycl device validity before communicator creation
    (void)preferred_topology_class;
    (void)preferred_topology_group;
    ccl::device_group_split_type preferred_topology = ccl::device_group_split_type::process;
    switch (preferred_topology_class) {
        case device_topology_type::ring: {
            switch (preferred_topology) {
                case ccl::device_group_split_type::thread:
                    return communicator_interface_ptr(new device_group_ring_communicator(
                        std::move(device_id), thread_idx, process_idx, attr));
                case ccl::device_group_split_type::process:
                    return communicator_interface_ptr(new thread_device_group_ring_communicator(
                        std::move(device_id), thread_idx, process_idx, attr));
                case ccl::device_group_split_type::cluster:
                    return communicator_interface_ptr(new process_ring_communicator(
                        std::move(device_id), thread_idx, process_idx, attr));
                default:
                    throw ccl_error(
                        std::string(
                            "Invalid `device_comm_attr_t` value for `ccl_device_preferred_group`: ") +
                        std::to_string(preferred_topology));
            }
            break;
        }
        case device_topology_type::a2a: {
            switch (preferred_topology) {
                case ccl::device_group_split_type::thread:
                    return communicator_interface_ptr(new device_group_a2a_communicator(
                        std::move(device_id), thread_idx, process_idx, attr));
                case ccl::device_group_split_type::process:
                    return communicator_interface_ptr(new thread_device_group_a2a_communicator(
                        std::move(device_id), thread_idx, process_idx, attr));
                case ccl::device_group_split_type::cluster:
                    return communicator_interface_ptr(new process_a2a_communicator(
                        std::move(device_id), thread_idx, process_idx, attr));
                default:
                    throw ccl_error(
                        std::string(
                            "Invalid `device_comm_attr_t` value for `ccl_device_preferred_group`: ") +
                        std::to_string(preferred_topology));
            }
            break;
        }
        default: {
            throw ccl_error(
                std::string(
                    "Invalid `device_comm_attr_t` value for `ccl_device_preferred_topology_class`: ") +
                std::to_string(preferred_topology_class));
        }
    }

    return std::unique_ptr<communicator_interface>();
}

#define COMMUNICATOR_INTERFACE_DISPATCHER_CLASS_EXPLICIT_INSTANTIATION(DeviceType) \
    template ccl::communicator_interface_ptr \
    ccl::communicator_interface_dispatcher::create_communicator_impl( \
        const DeviceType& device, \
        size_t thread_idx, \
        size_t process_idx, \
        const ccl::device_comm_attr_t& attr);

#define COMMUNICATOR_INTERFACE_DISPATCHER_NON_CLASS_EXPLICIT_INSTANTIATION(DeviceType) \
    template ccl::communicator_interface_ptr \
    ccl::communicator_interface_dispatcher::create_communicator_impl( \
        DeviceType device_id, \
        size_t thread_idx, \
        size_t process_idx, \
        const ccl::device_comm_attr_t& attr);

#endif //MULTI_GPU_SUPPORT
} // namespace ccl
