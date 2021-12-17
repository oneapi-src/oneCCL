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

#include "kvs_impl.hpp"
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
    shared_ptr_class<ikvs_wrapper> kvs_tmp;
    if (std::dynamic_pointer_cast<ccl::v1::kvs>(kvs) != nullptr) {
        kvs_tmp = std::dynamic_pointer_cast<v1::kvs>(kvs)->get_impl().get();
    }
    else {
        kvs_tmp = std::shared_ptr<ikvs_wrapper>(new users_kvs(kvs));
    }

    CCL_THROW_IF_NOT(devices.size() == 1, "multiple devices per process are not supported");

    LOG_TRACE("create communicator");

    ccl::communicator_interface_ptr impl = ccl::communicator_interface::create_communicator_impl(
        size, devices.begin()->first, kvs_tmp);
    ccl::vector_class<ccl::communicator> ret;
    ret.push_back(ccl::communicator(std::move(impl)));
    return ret;
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const map_class<int, DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs) {
    std::vector<pair_class<int, DeviceType>> vec_devices;
    for (const auto& d : devices) {
        vec_devices.push_back(std::make_pair(d.first, d.second));
    }
    return create_communicators(size, vec_devices, context, kvs);
}

/**
 * Creates a new host communicator with externally provided size, rank and kvs.
 * Implementation is platform specific and non portable.
 * @return host communicator
 */
communicator communicator::create_communicator(const comm_attr& attr) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    LOG_DEBUG("create host communicator");

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

    LOG_DEBUG("create host communicator");

    shared_ptr_class<ikvs_wrapper> kvs_tmp;
    if (std::dynamic_pointer_cast<ccl::v1::kvs>(kvs) != nullptr) {
        kvs_tmp = std::dynamic_pointer_cast<ccl::v1::kvs>(kvs)->get_impl().get();
    }
    else {
        kvs_tmp = std::shared_ptr<ikvs_wrapper>(new users_kvs(kvs));
    }

    communicator_interface_ptr impl =
        communicator_interface::create_communicator_impl(size, kvs_tmp);

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
    LOG_DEBUG("size ", size, ", rank ", rank);

    shared_ptr_class<ikvs_wrapper> kvs_tmp;
    if (std::dynamic_pointer_cast<ccl::v1::kvs>(kvs) != nullptr) {
        kvs_tmp = std::dynamic_pointer_cast<ccl::v1::kvs>(kvs)->get_impl().get();
    }
    else {
        kvs_tmp = std::shared_ptr<ikvs_wrapper>(new users_kvs(kvs));
    }

    communicator_interface_ptr impl =
        communicator_interface::create_communicator_impl(size, rank, kvs_tmp);

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
