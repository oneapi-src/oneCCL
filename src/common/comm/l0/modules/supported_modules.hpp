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
#include "common/comm/l0/modules/base_entry_module.hpp"

namespace native {
// alias for topologies
template <template <ccl_coll_type, ccl::group_split_type, ccl::device_topology_type>
          class module_impl,
          ccl_coll_type type,
          ccl::group_split_type group_id,
          ccl::device_topology_type... class_ids>
using topology_device_classes_modules_for_group_id =
    std::tuple<std::shared_ptr<module_impl<type, group_id, class_ids>>...>;

template <template <ccl_coll_type, ccl::group_split_type, ccl::device_topology_type>
          class module_impl,
          ccl_coll_type type,
          ccl::group_split_type... top_types>
using topology_device_group_modules = std::tuple<
    topology_device_classes_modules_for_group_id<module_impl,
                                                 type,
                                                 top_types,
                                                 SUPPORTED_TOPOLOGY_CLASSES_DECL_LIST>...>;

// alias for coll types
template <template <ccl_coll_type, ccl::group_split_type, ccl::device_topology_type>
          class module_impl,
          ccl_coll_type... types>
using supported_topology_device_modules = std::tuple<
    topology_device_group_modules<module_impl, types, SUPPORTED_HW_TOPOLOGIES_DECL_LIST>...>;

// alias for implemented modules
template <template <ccl_coll_type, ccl::group_split_type, ccl::device_topology_type>
          class module_impl>
using supported_device_modules = supported_topology_device_modules<module_impl, CCL_COLL_TYPE_LIST>;
} // namespace native
