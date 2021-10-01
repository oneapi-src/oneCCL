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
#include <cstddef>
#include <memory>

#include "oneapi/ccl/types.hpp"
#include "supported_topologies.hpp"
#include "communicator_traits.hpp"
#include "atl/atl_wrapper.h"

namespace native {
struct ccl_device;
}
namespace ccl {
namespace v1 {
class comm_split_attr;
}

#ifdef MULTI_GPU_SUPPORT
struct gpu_comm_attr;
#endif
struct communicator_interface;

using communicator_interface_ptr = std::shared_ptr<communicator_interface>;

struct communicator_interface_dispatcher {
    using device_t = typename ccl::unified_device_type::ccl_native_t;
    using context_t = typename ccl::unified_context_type::ccl_native_t;

    virtual ~communicator_interface_dispatcher() = default;

#ifdef MULTI_GPU_SUPPORT
    virtual void visit(ccl::gpu_comm_attr& comm_attr) = 0;
#endif //MULTI_GPU_SUPPORT

    virtual ccl::device_index_type get_device_path() const = 0;
    virtual device_t get_device() const = 0;
    virtual context_t get_context() const = 0;
    virtual const comm_split_attr& get_comm_split_attr() const = 0;
    virtual group_split_type get_topology_type() const = 0;
    virtual device_topology_type get_topology_class() const = 0;

    // create communicator for device & cpu types (from device class)
    template <class DeviceType,
              class ContextType,
              typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                       device_index_type>::value,
                                      int>::type = 0>
    static communicator_interface_ptr create_communicator_impl(
        const DeviceType& device,
        ContextType context,
        size_t thread_idx,
        size_t process_idx,
        const comm_split_attr& attr,
        std::shared_ptr<atl_wrapper> atl,
        ccl::group_split_type preferred_topology_group = ccl::group_split_type::undetermined);

    // create communicator for device & cpu types (from device index)
    template <class DeviceType,
              class ContextType,
              typename std::enable_if<
                  std::is_same<typename std::remove_cv<DeviceType>::type, device_index_type>::value,
                  int>::type = 0>
    static communicator_interface_ptr create_communicator_impl(
        DeviceType device_id,
        ContextType ctx,
        size_t thread_idx,
        size_t process_idx,
        const comm_split_attr& attr,
        std::shared_ptr<atl_wrapper> atl,
        ccl::group_split_type preferred_topology_group = ccl::group_split_type::undetermined);

    // create communicator for host
    static communicator_interface_ptr create_communicator_impl();

    // create communicator for host
    static communicator_interface_ptr create_communicator_impl(const size_t size,
                                                               shared_ptr_class<ikvs_wrapper> kvs);

    // create communicator for host
    static communicator_interface_ptr create_communicator_impl(const size_t size,
                                                               const int rank,
                                                               shared_ptr_class<ikvs_wrapper> kvs);

private:
    static communicator_interface_ptr create_communicator_from_unified_device(
        unified_device_type&& device_id,
        unified_context_type&& context_id,
        size_t thread_idx,
        size_t process_idx,
        const comm_split_attr& attr,
        std::shared_ptr<atl_wrapper> atl,
        ccl::group_split_type preferred_topology_group = ccl::group_split_type::undetermined);
};
} // namespace ccl
