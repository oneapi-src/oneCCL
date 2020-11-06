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
#include "common/comm/compiler_comm_interface_dispatcher.hpp"

#include "common/comm/comm_interface.hpp"
#include "unified_device_impl.hpp"

#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "common/comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"

#include "common/global/global.hpp"
#ifdef MULTI_GPU_SUPPORT
#include "common/comm/l0/communicator/device_group/device_ring_communicator.hpp"
#include "common/comm/l0/communicator/device_group/device_a2a_communicator.hpp"
#include "common/comm/l0/communicator/thread_group/thread_ring_communicator.hpp"
#include "common/comm/l0/communicator/thread_group/thread_a2a_communicator.hpp"
#include "common/comm/l0/communicator/process_group/process_ring_communicator.hpp"
#include "common/comm/l0/communicator/process_group/process_a2a_communicator.hpp"
#include "supported_topologies.hpp"

#endif

#include "common/comm/single_device_communicator/single_device_communicator.hpp"
#include "common/comm/host_communicator/host_communicator_impl.hpp"

namespace ccl {

communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl() {
    return communicator_interface_ptr(new host_communicator());
}

communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(const size_t size,
                                                               shared_ptr_class<kvs_interface> kvs) {
    return communicator_interface_ptr(new host_communicator(size, kvs));
}

communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(const size_t size,
                                                               const size_t rank,
                                                               shared_ptr_class<kvs_interface> kvs) {
    return communicator_interface_ptr(new host_communicator(size, rank, kvs));
}

template <class DeviceType,
          class ContextType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type>
communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(
    const DeviceType& device,
    ContextType context,
    size_t thread_idx,
    size_t process_idx,
    const ccl::comm_split_attr& attr,
    std::shared_ptr<atl_wrapper> atl) {
    static_assert(std::is_same<typename unified_device_type::ccl_native_t, DeviceType>::value,
                  "Unsupported 'DeviceType'");

    return communicator_interface_dispatcher::create_communicator_from_unified_device(
        unified_device_type(device), unified_device_context_type(context), thread_idx, process_idx, attr, atl);
}

template <class DeviceType,
          class ContextType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type>
communicator_interface_ptr communicator_interface_dispatcher::create_communicator_impl(
    DeviceType device_id,
    ContextType context,
    size_t thread_idx,
    size_t process_idx,
    const ccl::comm_split_attr& attr,
    std::shared_ptr<atl_wrapper> atl) {
#ifdef CCL_ENABLE_SYCL
    return communicator_interface_dispatcher::create_communicator_from_unified_device(
        unified_device_type(device_id, cl::sycl::info::device_type::gpu),
        unified_device_context_type(context),
        thread_idx,
        process_idx,
        attr,
        atl);
#else
//.    static_assert(std::is_same<typename unified_device_type::handle_t, DeviceType>::value,
//                  "Unsupported 'DeviceType'");
    return communicator_interface_dispatcher::create_communicator_from_unified_device(
        unified_device_type(device_id), unified_device_context_type(context), thread_idx, process_idx, attr, atl);
#endif
}

communicator_interface_ptr
communicator_interface_dispatcher::create_communicator_from_unified_device(
    ccl::unified_device_type&& device_id,
    ccl::unified_device_context_type&& context,
    size_t thread_idx,
    size_t process_idx,
    const ccl::comm_split_attr& attr,
    std::shared_ptr<atl_wrapper> atl) {
    // TODO ring by default at now. Choose preferred a2a if availbale
    ccl::device_topology_type preferred_topology_class = ccl::device_topology_type::ring;
    ccl::group_split_type preferred_topology_group = ccl::group_split_type::cluster;

    // read comm split attributes
    if (attr.is_valid<ccl::comm_split_attr_id::group>()) {
        preferred_topology_group = attr.get<ccl::comm_split_attr_id::group>();
        if (attr.is_valid<ccl::comm_split_attr_id::color>()) {
            throw ccl::exception(std::string(
                "Invalid `comm_split_attr`: both `color` and `group` set. Only one is supported"));
        }
    }
    else if (attr.is_valid<ccl::comm_split_attr_id::color>()) {
        throw ccl::exception(std::string(__FUNCTION__) + " - not implemented for 'color'");
    }

    // TODO creation host communicator from device
    // if (device is host ?)
    // return new host_communicator(atl);

    //TODO check device_id or sycl device validity before communicator creation
    switch (preferred_topology_class) {
        case device_topology_type::ring: {
            switch (preferred_topology_group) {
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
                case ccl::group_split_type::undetermined: {
                    auto comm_impl = new single_device_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr);
                    ccl::global_data& data = ccl::global_data::get();
                    auto comm = std::shared_ptr<ccl_comm>(
                        new ccl_comm(thread_idx, process_idx, data.comm_ids->acquire(), atl));
                    comm_impl->set_ccl_comm(std::move(comm));
                    return communicator_interface_ptr(comm_impl);
                }
#endif
#ifdef MULTI_GPU_SUPPORT
                case ccl::group_split_type::thread:
                    return communicator_interface_ptr(new device_group_ring_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
                case ccl::group_split_type::process:
                    return communicator_interface_ptr(new thread_device_group_ring_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
                case ccl::group_split_type::cluster:
                    return communicator_interface_ptr(new process_ring_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
#endif //MULTI_GPU_SUPPORT
                default:
                    throw ccl::exception(
                        std::string(
                            "Invalid `comm_split_attr` value for `ccl_device_preferred_group`: ") +
                        ::to_string(preferred_topology_group));
            }
            break;
        }
        case device_topology_type::a2a: {
            switch (preferred_topology_group) {
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
                case ccl::group_split_type::undetermined:
                    return communicator_interface_ptr(new single_device_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
#endif
#ifdef MULTI_GPU_SUPPORT
                case ccl::group_split_type::thread:
                    return communicator_interface_ptr(new device_group_a2a_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
                case ccl::group_split_type::process:
                    return communicator_interface_ptr(new thread_device_group_a2a_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
                case ccl::group_split_type::cluster:
                    return communicator_interface_ptr(new process_a2a_communicator(
                        std::move(device_id), std::move(context), thread_idx, process_idx, attr));
#endif
                default:
                    throw ccl::exception(
                        std::string(
                            "Invalid `comm_split_attr` value for `ccl_device_preferred_group`: ") +
                        ::to_string(preferred_topology_group));
            }
            break;
        }
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
        case device_topology_type::undetermined: {
            auto comm_impl =
                new single_device_communicator(std::move(device_id), std::move(context), thread_idx, process_idx, attr);
            ccl::global_data& data = ccl::global_data::get();
            auto comm = std::shared_ptr<ccl_comm>(
                new ccl_comm(thread_idx, process_idx, data.comm_ids->acquire(), atl));
            comm_impl->set_ccl_comm(std::move(comm));
            return communicator_interface_ptr(comm_impl);
        }
#endif //MULTI_GPU_SUPPORT
        default: {
            throw ccl::exception(
                std::string(
                    "Invalid `comm_split_attr` value for `ccl_device_preferred_topology_class`: ") +
                ::to_string(preferred_topology_class));
        }
    }

    return std::unique_ptr<communicator_interface>();
}

#define COMMUNICATOR_INTERFACE_DISPATCHER_CLASS_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::communicator_interface_ptr \
    ccl::communicator_interface_dispatcher::create_communicator_impl( \
        const DeviceType& device, \
        ContextType context, \
        size_t thread_idx, \
        size_t process_idx, \
        const ccl::comm_split_attr& attr, \
        std::shared_ptr<atl_wrapper> atl);

#define COMMUNICATOR_INTERFACE_DISPATCHER_NON_CLASS_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::communicator_interface_ptr \
    ccl::communicator_interface_dispatcher::create_communicator_impl( \
        DeviceType device_id, \
        ContextType context, \
        size_t thread_idx, \
        size_t process_idx, \
        const ccl::comm_split_attr& attr, \
        std::shared_ptr<atl_wrapper> atl);

} // namespace ccl
