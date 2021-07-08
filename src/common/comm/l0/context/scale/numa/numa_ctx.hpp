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
#include "common/comm/l0/context/scale/base/base_session.hpp"
#include "common/comm/l0/context/scale/base/base_session_table.hpp"
#include "common/comm/l0/context/scale/numa/numa_session.hpp"

namespace native {

class ccl_gpu_comm;
class ccl_virtual_gpu_comm;

template <class device>
class ccl_numa_proxy;

#define NUMA_CTX_DEVICE_PROXY_TYPES(observer_type) \
    observer_type<ccl_gpu_comm>, observer_type<ccl_virtual_gpu_comm>

template <class Impl, ccl::device_topology_type... types>
class numa_ctx : public observer::base_scaling_ctx<numa_ctx<Impl, types...>,
                                                   NUMA_CTX_DEVICE_PROXY_TYPES(ccl_numa_proxy)> {
public:
    static_assert(sizeof...(types), "types must be not 0");
    using context_impl = Impl;

    template <class device_t>
    using observer_t = ccl_numa_proxy<device_t>;

    using scaling_ctx_base_t = observer::base_scaling_ctx<numa_ctx<Impl, types...>,
                                                          NUMA_CTX_DEVICE_PROXY_TYPES(observer_t)>;

    using session_t = observer::numa_session_iface; //TODO: numa_session
    using session_ptr_t = std::shared_ptr<session_t>;
    using base_session_table_t = observer::session_table<session_t>;
    using base_session_table_ptr_t = std::shared_ptr<base_session_table_t>;

    using numa_actor = observer::subscribed_actor<session_ptr_t, observer::session_notification>;

    using observable_scale_up_topologies =
        typename scaling_ctx_base_t::template observable_topologies<types...>;

    using indexed_observable_topologies =
        typename scaling_ctx_base_t::template indexed_observable_topologies<types...>;

    observable_scale_up_topologies observables;
    indexed_observable_topologies indexed_observables;

    // session data
    template <class NUMA_source_device_t, ccl_coll_type coll_type>
    struct device_session_data {
        std::map<NUMA_source_device_t*, base_session_table_ptr_t> source_sessions;
    };

    //TODO make table PER thread!!!
    template <ccl_coll_type coll_type, class... devices_types>
    using session_table_t = std::tuple<device_session_data<devices_types, coll_type>...>;

    template <ccl_coll_type... coll_type>
    using session_table_typed_storage_t =
        std::tuple<session_table_t<coll_type, NUMA_CTX_DEVICE_PROXY_TYPES(observer_t)>...>;

    struct session_table_initializer {
        template <ccl_coll_type coll_type, class device_t>
        void operator()(session_table_t<coll_type, NUMA_CTX_DEVICE_PROXY_TYPES(observer_t)>& table,
                        observer_t<device_t>* observer_ptr) {
            auto& sessions_table =
                ccl_tuple_get<device_session_data<observer_t<device_t>, coll_type>>(table);
            sessions_table.source_sessions.emplace(
                observer_ptr, std::make_shared<base_session_table_t>(base_session_table_t{}));
        }
    };

    session_table_typed_storage_t<CCL_COLL_LIST> collective_sessions;

    //observer subject interface implementations
    template <class device_t, ccl::device_topology_type topology_type>
    void attach_ctx_observer(size_t rank_addr,
                             observer_t<device_t>* observer_ptr,
                             std::integral_constant<ccl::device_topology_type, topology_type> val) {
        register_observer_impl<topology_type>(rank_addr, observer_ptr);
    }

    template <class device_t, ccl::device_topology_type class_id, class numa_invoke_params_t>
    void invoke_ctx_observer(observer_t<device_t>* observer_ptr,
                             std::integral_constant<ccl::device_topology_type, class_id> val,
                             const observer::session_key& sess_key,
                             numa_invoke_params_t& param) {
        throw std::runtime_error("TODO - NUMA use-case is not implemented");

        // sanity - check registered proxy
        observer::container_t<observer_t<device_t>>& container =
            scaling_ctx_base_t::template get_types_container<observer_t<device_t>, class_id>(
                observables);

        auto it = container.find(observer_ptr);
        if (it == container.end()) {
            throw std::runtime_error(std::string("Observer is not registered: ") +
                                     observer_ptr->to_string() +
                                     " total count: " + std::to_string(container.size()));
        }
        size_t registered_index = std::distance(container.begin(), it);

        static constexpr ccl_coll_type coll_type = numa_invoke_params_t::get_coll_type();
        //Try to find existing session owner for coll type
        auto& sessions_table = ccl_tuple_get<device_session_data<observer_t<device_t>, coll_type>>(
            std::get<coll_type>(collective_sessions));

        // In general way sessions_table.source_sessions.find(observer_ptr) has multithreading access,
        // But it has write access only in wire up-phase, when observers are inserted from topology construction
        // Multithreading access here is served by model "multiple-readers - no writers"
        // and can be used without mutex protection
        auto session_table_it = sessions_table.source_sessions.find(observer_ptr);
        if (session_table_it == sessions_table.source_sessions.end()) {
            std::stringstream ss;
            ss << "sessions count: " << sessions_table.source_sessions.size() << std::endl;
            for (const auto& val : sessions_table.source_sessions) {
                ss << val.first->to_string() << ", " << val.second->to_string() << std::endl;
            }
            LOG_ERROR("session_key: ",
                      sess_key.to_string(),
                      ", cannot find source session for device: ",
                      observer_ptr->to_string(),
                      ". Available keys: ",
                      ss.str());
            abort();
        }

        base_session_table_ptr_t table = session_table_it->second;
        if (!table) {
            LOG_ERROR("session_key: ", sess_key.to_string(), ", session table is empty. Abort");
            abort();
        }

        session_ptr_t sess;
        LOG_DEBUG("session_key: ",
                  sess_key.to_string(),
                  ", current sessions count: ",
                  table->sessions.size());
        auto session_it = table->sessions.find(sess_key);
        if (session_it == table->sessions.end()) {
            //create new session
            sess = table->template create_session<observer::numa_session, class_id>(
                sess_key, param, registered_index, registered_devices_count);
        }
        else {
            //renew existing
            sess = session_it->second;
            sess->prepare(
                registered_index, registered_devices_count, reinterpret_cast<void*>(&param));

            //param.reset_staged_counters(registered_index, container.size());
        }

        // notify actor-owner
        const auto& thread_map =
            ccl_tuple_get<observer::device_thread_map<observer_t<device_t>, numa_actor>>(
                numa_workers);
        auto actor_it = thread_map.find(observer_ptr);
        if (actor_it == thread_map.end()) {
            LOG_ERROR("Unregistered observer: ",
                      observer_ptr->to_string(),
                      ", thread_map size: ",
                      thread_map.size(),
                      " . Abort");
            abort();
        }

        actor_it->second->start_job(sess);
    }

private:
    template <ccl::device_topology_type topology_type, class device_t>
    void register_observer_impl(size_t rank_addr, observer_t<device_t>* observer_ptr);

    using specific_device_tuple_thread_map_t =
        observer::multiple_device_thread_map_t<numa_actor, NUMA_CTX_DEVICE_PROXY_TYPES(observer_t)>;
    specific_device_tuple_thread_map_t numa_workers;

    template <class device_t>
    void worker(observer_t<device_t>* device,
                numa_actor* actor_ptr,
                typename numa_actor::storage_t& todo_list);
    size_t registered_devices_count{};
};
} // namespace native
