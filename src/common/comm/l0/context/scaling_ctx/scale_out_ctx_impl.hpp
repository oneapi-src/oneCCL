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
#include "common/comm/l0/context/scaling_ctx/scale_out_ctx.hpp"
#include "common/log/log.hpp"

namespace native {

#define TEMPLATE_DECL_ARG class Impl, ccl::device_topology_type... types
#define TEMPLATE_DEF_ARG  Impl, types...

// observer_ptr interface implementations
template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type topology_type, class device_t>
void scale_out_ctx<TEMPLATE_DEF_ARG>::register_observer_impl(size_t rank_addr,
                                                             observer_t<device_t>* observer_ptr) {
    observer::container_t<observer_t<device_t>>& container =
        scaling_ctx_base_t::template get_types_container<observer_t<device_t>, topology_type>(
            observables);
    container.insert(observer_ptr);
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG
} // namespace native
