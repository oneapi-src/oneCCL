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
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include "common/comm/l0/devices/proxy_observer.hpp"
#include "common/comm/l0/context/base_ctx_actor.hpp"

namespace native {

template <class device>
class ccl_gpu_scaleup_proxy;

namespace observer {

template <class device_t, class actor_t>
using device_thread_map = std::map<device_t*, std::unique_ptr<actor_t>>;

template <class actor_t, class... devices_types>
using multiple_device_thread_map_t = std::tuple<device_thread_map<devices_types, actor_t>...>;

template <class device_t>
using proxy_observer_ptr = typename std::add_pointer<device_t>::type;

template <class device_t>
using container_t = std::set<proxy_observer_ptr<device_t>>;

template <class... device_t>
using container_tuple_t = std::tuple<container_t<device_t>...>;

template <class device_t>
using indexed_container_t = std::map<size_t /* rank */, proxy_observer_ptr<device_t>>;

template <class... device_t>
using indexed_container_tuple_t = std::tuple<indexed_container_t<device_t>...>;

// Static interface used to register proxy_observers
template <class ctx_impl_t, class... proxy_observer_device_t>
class base_scaling_ctx {
public:
    using own_t = base_scaling_ctx<ctx_impl_t, proxy_observer_device_t...>;

    using device_types_t = std::tuple<proxy_observer_device_t...>;

    template <ccl::device_topology_type class_id>
    struct observables_types : container_tuple_t<proxy_observer_device_t...> {};

    template <ccl::device_topology_type class_id>
    struct indexed_observables_types : indexed_container_tuple_t<proxy_observer_device_t...> {};

    template <ccl::device_topology_type... class_id>
    using observable_topologies = std::tuple<observables_types<class_id>...>;

    /* TODO use templated tepmlated container */
    template <ccl::device_topology_type... class_id>
    using indexed_observable_topologies = std::tuple<indexed_observables_types<class_id>...>;

    template <class device_t>
    static constexpr bool is_registered_device_t() {
        return is_one_of<device_t, proxy_observer_device_t...>::value;
    }

    ctx_impl_t* get_this() {
        return static_cast<ctx_impl_t*>(this);
    }

    const ctx_impl_t* get_this() const {
        return static_cast<const ctx_impl_t*>(this);
    }

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id, class device_t>
    void attach(device_t* obj) {
        static_assert(std::is_base_of<proxy_observer<device_t>, device_t>::value,
                      "Only `proxy_observer` derived class can be attached to context");

        get_this()->attach_ctx_observer(
            std::numeric_limits<size_t>::max(), /* unassigned addr at moment */
            obj,
            std::integral_constant<ccl::device_topology_type, class_id>{});
    }

    /* Workaround:
     * topology constructor invoke `attach` straight toward after observer device creation
     * But there are unassigneed rank addr in this case
     * Rank will be assigned after indexer execution in topology constructor
     * Need to remove `attach_ctx_observer` with  unassigned addr version and use assigning after indexer only
     */

    template <ccl::group_split_type group_id, ccl::device_topology_type class_id, class device_t>
    void reattach_with_addr(size_t rank, device_t* obj) {
        static_assert(std::is_base_of<proxy_observer<device_t>, device_t>::value,
                      "Only `proxy_observer` derived class can be attached to context");

        get_this()->attach_ctx_observer(
            rank, obj, std::integral_constant<ccl::device_topology_type, class_id>{});
    }

    template <class device_t,
              class = typename std::enable_if<is_registered_device_t<device_t>()>::type>
    own_t* get_ctx_selector() {
        return this;
    }

    template <ccl::group_split_type group_id,
              ccl::device_topology_type class_id,
              class device_t,
              class... Args>
    //class = typename std::enable_if<is_registered_device_t<device_t>()>::type>
    void invoke_proxy(device_t* obj, Args&&... args) {
        static_assert(is_one_of<device_t, proxy_observer_device_t...>::value, "Unsupported");
        static_assert(std::is_base_of<proxy_observer<device_t>, device_t>::value,
                      "Only `proxy_observer` derived class can invoke context");

        get_this()->invoke_ctx_observer(
            obj,
            // TODO std::integral_constant<ccl::group_split_type, group_id>{},
            std::integral_constant<ccl::device_topology_type, class_id>{},
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

    template <ccl::device_topology_type specific_type, ccl::device_topology_type... class_id>
    static indexed_observables_types<specific_type>& get_types(
        indexed_observable_topologies<class_id...>& tops) noexcept {
        return ccl_tuple_get<indexed_observables_types<specific_type>>(tops);
    }

    template <class observer_device_t, ccl::device_topology_type specific_type>
    indexed_container_t<observer_device_t>& get_container(
        indexed_observables_types<specific_type>& types) noexcept {
        return ccl_tuple_get<indexed_container_t<observer_device_t>>(types);
    }

    template <class observer_device_t,
              ccl::device_topology_type specific_type,
              ccl::device_topology_type... class_id>
    indexed_container_t<observer_device_t>& get_types_container(
        indexed_observable_topologies<class_id...>& tops) noexcept {
        return get_container<observer_device_t>(get_types<specific_type>(tops));
    }
};

namespace detail {

struct actor_visitor {
    template <class device_t, class actor_t>
    void operator()(device_thread_map<device_t, actor_t>& actors, actor_t* subscriber) {
        for (auto& a : actors) {
            a.second->subscribe_on(subscriber);
        }
    }
};

template <class message_type, class mailbox_message_type>
struct actor_publisher {
    template <class device_t, class... message_args>
    void operator()(
        device_thread_map<device_t, subscribed_actor<message_type, mailbox_message_type>>& actors,
        size_t topic_tag,
        size_t publisher_id,
        message_args&&... args) {
        for (auto& a : actors) {
            a.second->put_message(publisher_id, topic_tag, std::forward<message_args>(args)...);
        }
    }
};
} // namespace detail
} // namespace observer
} // namespace native
