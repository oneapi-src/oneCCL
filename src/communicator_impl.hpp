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
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "oneapi/ccl/communicator.hpp"

#include "common/comm/l0/comm_context_id.hpp"
#include "communicator_impl_details.hpp"

//TODO
/*
namespace ccl
{
struct comm_split_attr_impl
{
    constexpr static int color_default()
    {
        return 0;
    }
    ccl::library_version version;
};

struct device_attr_impl
{
    constexpr static device_topology_type class_default()
    {
        return device_topology_type::ring;
    }
    constexpr static group_split_type group_default()
    {
        return group_split_type::process;
    }
    device_topology_type current_preferred_topology_class = class_default();
    group_split_type current_preferred_topology_group = group_default();
};
}*/

namespace ccl {

namespace v1 {

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const vector_class<DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs) {
    vector_class<communicator> ret;
    throw std::runtime_error(std::string(__FUNCTION__) + " - not implemented");
    return ret;
}

using rank_t = int;

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const vector_class<pair_class<int, DeviceType>>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs) {
    return comm_impl_dispatch_selector<CL_BACKEND_TYPE>::create_communicators_selector(
        size, devices, context, kvs);
#if 0
    vector_class<int> local_thread_ranks;
    local_thread_ranks.reserve(devices.size());
    std::transform(
        devices.begin(),
        devices.end(),
        std::back_inserter(local_thread_ranks),
        [](const typename vector_class<pair_class<int, DeviceType>>::value_type& val) {
            return val.first;
        });
    group_context::comm_group_t thread_group =
        group_context::instance().group_by_kvs(local_thread_ranks, size, kvs);

    vector_class<DeviceType> local_thread_devices;
    local_thread_devices.reserve(devices.size());
    std::transform(
        devices.begin(),
        devices.end(),
        std::back_inserter(local_thread_devices),
        [](const typename vector_class<pair_class<int, DeviceType>>::value_type& val) {
            return val.second;
        });

    auto ret = thread_group->create_communicators(local_thread_devices);
    return ret;
#endif
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const map_class<int, DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs)

{
    return comm_impl_dispatch_selector<CL_BACKEND_TYPE>::create_communicators_selector(
        size, devices, context, kvs);
#if 0
    vector_class<int> local_thread_ranks;
    local_thread_ranks.reserve(devices.size());
    std::transform(devices.begin(),
                   devices.end(),
                   std::back_inserter(local_thread_ranks),
                   [](const typename map_class<int, DeviceType>::value_type& val) {
                       return val.first;
                   });
    group_context::comm_group_t thread_group =
        group_context::instance().group_by_kvs(local_thread_ranks, size, kvs);

    vector_class<DeviceType> local_thread_devices;
    local_thread_devices.reserve(devices.size());
    std::transform(devices.begin(),
                   devices.end(),
                   std::back_inserter(local_thread_devices),
                   [](const typename map_class<int, DeviceType>::value_type& val) {
                       return val.second;
                   });

    auto ret = thread_group->create_communicators(local_thread_devices);
    return ret;
#endif
}

/*CCL_API bool communicator::is_ready() const
{
    return get_impl()->is_ready();
}*/

/**
 * Creates a new host communicator with externally provided size, rank and kvs.
 * Implementation is platform specific and non portable.
 * @return host communicator
 */
communicator communicator::create_communicator(const comm_attr& attr) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    LOG_DEBUG("Create host communicator");

    communicator_interface_ptr impl = communicator_interface::create_communicator_impl();

    return communicator(std::move(impl));
}

/**
 * Creates a new host communicator with user supplied size and kvs.
 * Rank will be assigned automatically.
 * @param size user-supplied total number of ranks
 * @param kvs key-value store for ranks wire-up
 * @return host communicator
 */
communicator communicator::create_communicator(const int size,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    LOG_DEBUG("Create host communicator");

    communicator_interface_ptr impl = communicator_interface::create_communicator_impl(size, kvs);

    return communicator(std::move(impl));
}

/**
 * Creates a new host communicator with user supplied size, rank and kvs.
 * @param size user-supplied total number of ranks
 * @param rank user-supplied rank
 * @param kvs key-value store for ranks wire-up
 * @return host communicator
 */
communicator communicator::create_communicator(const int size,
                                               const int rank,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr) {
    LOG_DEBUG("Create host communicator: size ", size, ", rank ", rank);

    communicator_interface_ptr impl =
        communicator_interface::create_communicator_impl(size, rank, kvs);

    return communicator(std::move(impl));
}

} // namespace v1

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_COMM_CREATE_WO_RANK_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API ccl::communicator::create_communicators( \
        const int comm_size, \
        const ccl::vector_class<DeviceType>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs);

#define API_COMM_CREATE_WITH_RANK_IN_VECTOR_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API ccl::communicator::create_communicators( \
        const int comm_size, \
        const ccl::vector_class<ccl::pair_class<int, DeviceType>>& local_rank_device_map, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs);

#define API_COMM_CREATE_WITH_RANK_IN_MAP_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API ccl::communicator::create_communicators( \
        const int comm_size, \
        const ccl::map_class<int, DeviceType>& local_rank_device_map, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs);
