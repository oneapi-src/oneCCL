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

#include "common/comm/l0/device_community_holder.hpp"
#include "common/comm/l0/device_community_holder_utils.hpp"

namespace native {
#define TEMPLATE_DECL_ARG \
    ccl::group_split_type group_id, ccl::device_topology_type... class_id
#define TEMPLATE_DEF_ARG group_id, class_id...

// community impl
template <ccl::device_topology_type class_id>
template <ccl::group_split_type group_id>
void device_community_container<class_id>::register_device_by_id(
    const ccl::device_index_type& device_id,
    ccl::context_comm_addr& registered_addr) {
    storage->template register_device_by_id<group_id>(device_id, registered_addr);
}

template <ccl::group_split_type group_id>
void device_community_container<ccl::device_topology_type::ring>::register_device_by_id(
    const ccl::device_index_type& device_id,
    ccl::context_comm_addr& registered_addr) {
    for (auto it = closed_rings.begin(); it != closed_rings.end(); ++it) {
        (*it)->template register_device_by_id<group_id>(device_id, registered_addr);
    }

    for (auto it = torn_apart_rings.begin(); it != torn_apart_rings.end(); ++it) {
        (*it)->template register_device_by_id<group_id>(device_id, registered_addr);
    }
}
// implementation
template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type requested_id>
const device_community_container<requested_id>&
device_group_community_holder<TEMPLATE_DEF_ARG>::get_community() const {
    return std::get<requested_id>(typed_communities);
}

template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type requested_id>
device_community_container<requested_id>&
device_group_community_holder<TEMPLATE_DEF_ARG>::get_community() {
    return const_cast<device_community_container<requested_id>&>(
        static_cast<const self_t*>(this)->get_community<requested_id>());
}

template <TEMPLATE_DECL_ARG>
std::string device_group_community_holder<TEMPLATE_DEF_ARG>::to_string() const {
    std::stringstream ss;
    details::device_community_container_print_helper<group_id> p(ss);
    ccl_tuple_for_each(typed_communities, p);
    return ss.str();
}

template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type requested_class_id>
void device_group_community_holder<TEMPLATE_DEF_ARG>::register_device_by_id(
    const ccl::device_index_type& device_id,
    ccl::context_comm_addr& registered_addr) {
    device_community_container<requested_class_id>& container =
        this->template get_community<requested_class_id>();
    container->register_device_by_id(device_id, registered_addr);
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG
} // namespace native
