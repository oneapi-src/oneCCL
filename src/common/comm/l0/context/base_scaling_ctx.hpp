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
#include <tuple>
#include <memory>
#include <stdexcept>
#include <vector>
#include "common/comm/l0/devices/proxy_observer.hpp"

namespace native {

template <class device>
class ccl_gpu_scaleup_proxy;

namespace observer {

template <class device_t>
using proxy_observer_ptr = typename std::add_pointer<device_t>::type;

template <class device_t>
using container_t = std::set<proxy_observer_ptr<device_t>>;

// Static interface used to register proxy_observers
template <class ctx_impl_t, class... proxy_observer_device_t>
class base_scaling_ctx {
public:
    using own_t = base_scaling_ctx<ctx_impl_t, proxy_observer_device_t...>;

    template <ccl::device_topology_type class_id>
    struct observables_types : std::tuple<container_t<proxy_observer_device_t>...> {};

    template <ccl::device_topology_type... class_id>
    using observable_topologies = std::tuple<observables_types<class_id>...>;

    ctx_impl_t* get_this() {
        return static_cast<ctx_impl_t*>(this);
    }

    const ctx_impl_t* get_this() const {
        return static_cast<const ctx_impl_t*>(this);
    }

    template <ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class device_t>
    void attach(device_t* obj) {
        static_assert(std::is_base_of<proxy_observer<device_t>, device_t>::value,
                      "Only `proxy_observer` derived class can be attached to context");

        get_this()->attach_ctx_observer(
            obj, std::integral_constant<ccl::device_topology_type, class_id>{});
    }

    template <ccl::group_split_type group_id, class device_t, class... Args>
    void invoke_proxy(device_t* obj, Args&&... args) {
        static_assert(std::is_base_of<proxy_observer<device_t>, device_t>::value,
                      "Only `proxy_observer` derived class can invoke context");

        get_this()->invoke_ctx_observer(
            obj,
            std::integral_constant<ccl::group_split_type, group_id>{},
            std::forward<Args>(args)...);
    }

    // helpers
    template <ccl::device_topology_type specific_type, ccl::device_topology_type... class_id>
    static observables_types<specific_type>& get_types(
        observable_topologies<class_id...>& tops) noexcept {
        return ccl_tuple_get<observables_types<specific_type>>(tops);
    }

    template <class observer_device_t, ccl::device_topology_type specific_type>
    container_t<observer_device_t>& get_container(
        observables_types<specific_type>& types) noexcept {
        return ccl_tuple_get<container_t<observer_device_t>>(types);
    }

    template <class observer_device_t,
              ccl::device_topology_type specific_type,
              ccl::device_topology_type... class_id>
    container_t<observer_device_t>& get_types_container(
        observable_topologies<class_id...>& tops) noexcept {
        return get_container<observer_device_t>(get_types<specific_type>(tops));
    }
};
} // namespace observer
} // namespace native
