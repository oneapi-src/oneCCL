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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "common/comm/single_device_communicator/single_device_base.hpp"

#define TEMPLATE_DECL_ARG class comm_impl, class communicator_traits
#define TEMPLATE_DEF_ARG  comm_impl, communicator_traits

template <TEMPLATE_DECL_ARG>
typed_single_device_base_communicator<TEMPLATE_DEF_ARG>::typed_single_device_base_communicator(
    ccl::unified_device_type&& owned_device,
    ccl::unified_context_type&& context,
    size_t thread_idx,
    size_t process_idx,
    const ccl::comm_split_attr& attr)
        : base_communicator(std::move(owned_device),
                            std::move(context),
                            thread_idx,
                            process_idx /*, comm_attr*/,
                            attr) {
    try {
        LOG_INFO("sheduled for create, device id: ",
                 device.get_id(),
                 ", thread_id: ",
                 thread_idx,
                 ", process id:",
                 process_idx);
    }
    catch (...) {
        LOG_INFO("sheduled for create single device communicator , thread_id: ",
                 thread_idx,
                 ", process id:",
                 process_idx);
    }
}

template <TEMPLATE_DECL_ARG>
bool typed_single_device_base_communicator<TEMPLATE_DEF_ARG>::is_ready() const {
    return true;
}

template <TEMPLATE_DECL_ARG>
ccl::group_split_type typed_single_device_base_communicator<TEMPLATE_DEF_ARG>::get_topology_type()
    const {
    return self_t::topology_type();
}

template <TEMPLATE_DECL_ARG>
ccl::device_topology_type
typed_single_device_base_communicator<TEMPLATE_DEF_ARG>::get_topology_class() const {
    return self_t::topology_class();
}

template <TEMPLATE_DECL_ARG>
std::string typed_single_device_base_communicator<TEMPLATE_DEF_ARG>::to_string() const {
    return std::string("single device communicator, rank (") + std::to_string(rank()) + "/" +
           std::to_string(size());
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG
