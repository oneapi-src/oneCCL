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
#include "common/comm/l0/context/base_scaling_ctx.hpp"

namespace native {

class ccl_gpu_comm;
class ccl_virtual_gpu_comm;

template <class device>
class ccl_gpu_scaleup_proxy;

template <class device>
class ccl_numa_proxy;

template <class Impl, ccl::device_topology_type... types>
class scale_up_ctx : public observer::base_scaling_ctx<
                         scale_up_ctx<Impl, types...>,
                         ccl_gpu_scaleup_proxy<ccl_gpu_comm>,
                         ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>,
                         ccl_gpu_scaleup_proxy<ccl_numa_proxy<ccl_gpu_comm>>,
                         ccl_gpu_scaleup_proxy<ccl_numa_proxy<ccl_virtual_gpu_comm>>> {
public:
    using context_impl = Impl;

    template <class device_t>
    using observer_t = ccl_gpu_scaleup_proxy<device_t>;

    using scaling_ctx_base_t =
        observer::base_scaling_ctx<scale_up_ctx<Impl, types...>,
                                   observer_t<ccl_gpu_comm>,
                                   observer_t<ccl_virtual_gpu_comm>,
                                   observer_t<ccl_numa_proxy<ccl_gpu_comm>>,
                                   observer_t<ccl_numa_proxy<ccl_virtual_gpu_comm>>>;
    using observable_scale_up_topologies =
        typename scaling_ctx_base_t::template observable_topologies<types...>;

    observable_scale_up_topologies observables;

    //observer subject interface implementations
    template <class device_t, ccl::device_topology_type topology_type>
    void attach_ctx_observer(size_t rank_addr,
                             observer_t<device_t>* observer_ptr,
                             std::integral_constant<ccl::device_topology_type, topology_type> val) {
        register_observer_impl<topology_type>(rank_addr, observer_ptr);
    }

    void invoke_ctx_observer(
        observer_t<ccl_gpu_comm>* observer_ptr,
        std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::ring> val);
    void invoke_ctx_observer(
        observer_t<ccl_virtual_gpu_comm>* observer_ptr,
        std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::ring> val);
    void invoke_ctx_observer(
        observer_t<ccl_gpu_comm>* observer_ptr,
        std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::a2a> val);
    void invoke_ctx_observer(
        observer_t<ccl_virtual_gpu_comm>* observer_ptr,
        std::integral_constant<ccl::device_topology_type, ccl::device_topology_type::a2a> val);

private:
    template <ccl::device_topology_type topology_type, class device_t>
    void register_observer_impl(size_t rank_addr, observer_t<device_t>* observer_ptr);
};
} // namespace native
