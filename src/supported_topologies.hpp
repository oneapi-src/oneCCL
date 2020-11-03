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
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "common/utils/enums.hpp"

namespace ccl {

//TODO I do not want to rename 90% f code at now
using group_split_type = group_split_type;

#define SUPPORTED_HW_TOPOLOGIES_DECL_LIST \
    ccl::group_split_type::thread, ccl::group_split_type::process, \
        ccl::group_split_type::cluster

#define SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST \
    ccl::device_topology_type::ring, ccl::device_topology_type::a2a
} // namespace ccl

using device_group_split_type_names = utils::enum_to_str<
    static_cast<typename std::underlying_type<ccl::group_split_type>::type>(
        ccl::group_split_type::last_value)>;
inline std::string to_string(ccl::group_split_type type) {
    return device_group_split_type_names({
                                             "TG",
                                             "PG",
                                             "CG",
                                         })
        .choose(type, "INVALID_VALUE");
}

using device_topology_type_names = utils::enum_to_str<ccl::device_topology_type::last_class_value>;
inline std::string to_string(ccl::device_topology_type class_value) {
    return device_topology_type_names({ "RING_CLASS", "A2A_CLASS" })
        .choose(class_value, "INVALID_VALUE");
}
