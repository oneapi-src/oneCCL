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
#include "common/comm/l0/context/scaling_ctx/numa_ctx.hpp"
#include "common/utils/tuple.hpp"

namespace native {

#define TEMPLATE_DECL_ARG class Impl, ccl::device_topology_type... types
#define TEMPLATE_DEF_ARG  Impl, types...

template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type class_id, class device_t>
void numa_ctx<TEMPLATE_DEF_ARG>::register_observer_impl(observer_t<device_t>* observer_ptr) {
    observer::container_t<observer_t<device_t>>& container =
        scaling_ctx_base_t::template get_types_container<observer_t<device_t>, class_id>(
            observables);
    container.insert(observer_ptr);
}

template <TEMPLATE_DECL_ARG>
void numa_ctx<TEMPLATE_DEF_ARG>::invoke_ctx_observer(
    observer_t<ccl_gpu_comm>* observer_ptr,
    std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::ring> val) {
    observer::container_t<observer_t<ccl_gpu_comm>>& container =
        scaling_ctx_base_t::template get_types_container<observer_t<ccl_gpu_comm>,
                                                         ccl::device_topology_type::ring>(
            observables);

    auto it = container.find(observer_ptr);
    if (it == container.end()) {
        throw std::runtime_error(std::string("invalid proxy: ") + observer_ptr->to_string());
    }

    throw std::runtime_error(std::string("Valid proxy: ") + observer_ptr->to_string());
}

template <TEMPLATE_DECL_ARG>
void numa_ctx<TEMPLATE_DEF_ARG>::invoke_ctx_observer(
    observer_t<ccl_virtual_gpu_comm>* observer_ptr,
    std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::ring> val) {
    throw std::runtime_error(std::string("Valid proxy: ") + observer_ptr->to_string());
}

template <TEMPLATE_DECL_ARG>
void numa_ctx<TEMPLATE_DEF_ARG>::invoke_ctx_observer(
    observer_t<ccl_gpu_comm>* observer_ptr,
    std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::a2a> val) {
    throw std::runtime_error(std::string("Valid proxy: ") + observer_ptr->to_string());
}

template <TEMPLATE_DECL_ARG>
void numa_ctx<TEMPLATE_DEF_ARG>::invoke_ctx_observer(
    observer_t<ccl_virtual_gpu_comm>* observer_ptr,
    std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::a2a> val) {
    throw std::runtime_error(std::string("Valid proxy: ") + observer_ptr->to_string());
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG

} // namespace native
