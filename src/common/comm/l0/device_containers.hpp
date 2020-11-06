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
#include <tuple>
#include <vector>

#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/native_device_api/l0/declarations.hpp"
#include "common/utils/tuple.hpp"
#include "common/utils/utils.hpp"
#include "common/comm/l0/device_types.hpp"

namespace native {

template <class device_t>
using device_container = std::multimap<ccl_device::handle_t, device_t_ptr<device_t>>;

template <class... device_t>
using device_storage_t = std::tuple<device_container<device_t>...>;

// non-enumeration devices
template <class device_t>
struct plain_device_container : std::vector<device_t_ptr<device_t>> {};
template <class... device_t>
using plain_device_storage = std::tuple<plain_device_container<device_t>...>;

// enumeration devices
template <class device_t>
struct indexed_device_container : std::map<size_t, device_t_ptr<device_t>> {};
/* TODO no need actually at now
template<>  //thread safe for concurrent
struct indexed_device_container<ccl_thread_comm<ccl_gpu_comm>> : std::map<size_t, device_t_ptr<ccl_thread_comm<ccl_gpu_comm>> > {};
template<>  //thread safe for concurrent
struct indexed_device_container<ccl_thread_comm<ccl_virtual_gpu_comm>> : std::map<size_t, device_t_ptr<ccl_thread_comm<ccl_virtual_gpu_comm>> > {};
*/
template <class... device_t>
using indexed_device_storage = std::tuple<indexed_device_container<device_t>...>;

namespace details {
//TODO - use traits
template <class device_t, class... total_devices_t>
inline size_t get_size(const native::device_storage_t<total_devices_t...>& gpu_device_storage) {
    return ccl_tuple_get<native::device_container<device_t>>(gpu_device_storage).size();
}

template <class device_t, class... total_devices_t>
inline size_t get_size(const native::plain_device_storage<total_devices_t...>& gpu_device_storage) {
    return ccl_tuple_get<native::plain_device_container<device_t>>(gpu_device_storage).size();
}

template <class device_t, class... total_devices_t>
inline size_t get_size(
    const native::indexed_device_storage<total_devices_t...>& gpu_device_storage) {
    return ccl_tuple_get<native::indexed_device_container<device_t>>(gpu_device_storage).size();
}

template <class Container>
inline size_t get_aggregated_size(const Container& gpu_device_storage) {
    return 0;
}

template <class Container, class DeviceType, class... Types>
inline size_t get_aggregated_size(const Container& gpu_device_storage) {
    return get_size<DeviceType>(gpu_device_storage) +
           get_aggregated_size<Container, Types...>(gpu_device_storage);
}

} // namespace details
} // namespace native
