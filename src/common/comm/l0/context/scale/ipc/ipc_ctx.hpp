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

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>
#include <vector>
#include "common/comm/l0/context/base_scaling_ctx.hpp"
#include "common/comm/l0/context/scale/ipc/ipc_session_key.hpp"
#include "common/comm/l0/context/scale/ipc/ipc_ctx_session.hpp"

namespace ccl {
class host_communicator;
}

namespace native {

class ccl_gpu_comm;
class ccl_virtual_gpu_comm;

class ccl_ipc_gpu_comm;

template <class device>
class ccl_ipc_source_gpu_comm;

struct session_table;
class session;

template <class Impl, ccl::device_topology_type... types>
class ipc_ctx : public observer::base_scaling_ctx<ipc_ctx<Impl, types...>,
                                                  ccl_ipc_source_gpu_comm<ccl_gpu_comm>,
                                                  ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>,
                                                  ccl_ipc_gpu_comm> {
public:
    static_assert(sizeof...(types), "types must be not 0");
    using context_impl = Impl;

    template <class device_t>
    using observer_t = ccl_ipc_source_gpu_comm<device_t>;

    using scaling_ctx_base_t = observer::base_scaling_ctx<ipc_ctx<Impl, types...>,
                                                          observer_t<ccl_gpu_comm>,
                                                          observer_t<ccl_virtual_gpu_comm>,
                                                          ccl_ipc_gpu_comm>;

    using observable_ipc_topologies =
        typename scaling_ctx_base_t::template observable_topologies<types...>;

    using indexed_observable_ipc_topologies =
        typename scaling_ctx_base_t::template indexed_observable_topologies<types...>;

    observable_ipc_topologies observables;
    indexed_observable_ipc_topologies indexed_observables;

    ipc_ctx() {}

    ~ipc_ctx() {
        stop.store(true);
        delivery_condition.notify_all();

        for (auto& thread : listener_thread_map) {
            thread.second->join();
        }
    }

    void initialize_ctx(std::shared_ptr<ccl::host_communicator> communicator);

    // session data
    template <class IPC_source_device_t>
    struct ipc_src_session_data {
        std::map<IPC_source_device_t*, std::shared_ptr<session_table>> source_sessions;
    };

    using session_table_t = std::tuple<ipc_src_session_data<observer_t<ccl_gpu_comm>>,
                                       ipc_src_session_data<observer_t<ccl_virtual_gpu_comm>>>;
    std::map<ccl_coll_type, session_table_t> collective_sessions;

    //observer subject interface implementations
    template <class device_t, ccl::device_topology_type topology_type>
    void attach_ctx_observer(size_t rank_addr,
                             observer_t<device_t>* observer_ptr,
                             std::integral_constant<ccl::device_topology_type, topology_type> val) {
        register_observer_impl<topology_type>(rank_addr, observer_ptr);
    }

    template <ccl::device_topology_type topology_type>
    void attach_ctx_observer(size_t rank_addr,
                             ccl_ipc_gpu_comm* observer_ptr,
                             std::integral_constant<ccl::device_topology_type, topology_type> val) {
        register_observer_impl<topology_type>(rank_addr, observer_ptr);
    }

    template <class device_t, ccl::device_topology_type class_id, class ipc_invoke_params_t>
    void invoke_ctx_observer(observer_t<device_t>* observer_ptr,
                             std::integral_constant<ccl::device_topology_type, class_id> val,
                             const ipc_session_key& session_key,
                             ipc_invoke_params_t&& param) {
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

        //Try to find existing session owner for coll type
        auto coll_session_table_it = collective_sessions.find(ipc_invoke_params_t::get_coll_type());
        if (coll_session_table_it == collective_sessions.end()) {
            std::stringstream ss;
            for (const auto& val : collective_sessions) {
                ss << ccl_coll_type_to_str(val.first) << ", ";
            }
            LOG_ERROR("session_key: ",
                      session_key.to_string(),
                      ", cannot find collective session table for key: ",
                      ccl_coll_type_to_str(ipc_invoke_params_t::get_coll_type()),
                      ". Available keys: ",
                      ss.str());
            abort();
        }

        auto& sessions_table = ccl_tuple_get<ipc_src_session_data<observer_t<device_t>>>(
            coll_session_table_it->second);
        auto session_table_it = sessions_table.source_sessions.find(observer_ptr);
        if (session_table_it == sessions_table.source_sessions.end()) {
            std::stringstream ss;
            ss << "sessions count: " << sessions_table.source_sessions.size() << std::endl;
            for (const auto& val : sessions_table.source_sessions) {
                ss << val.first->to_string() << ", " << val.second->to_string() << std::endl;
            }
            LOG_ERROR("session_key: ",
                      session_key.to_string(),
                      ", cannot find source session for device: ",
                      observer_ptr->to_string(),
                      ". Available keys: ",
                      ss.str());
            abort();
        }

        std::shared_ptr<session_table> table = session_table_it->second;
        if (!table) {
            LOG_ERROR("session_key: ", session_key.to_string(), ", session table is empty. Abort");
            abort();
        }

        // TODO: WA: destroy all sessions that were before
        // (only one session is always active)
        // without this WA, we hang in kernels when reusing sessions
        // because other sessions have the same key accidentally.
        // It will works for GPU cache enabled but invalid without cache
        table->sessions.clear();

        std::shared_ptr<session> sess;
        auto session_it = table->sessions.find(session_key);
        if (session_it == table->sessions.end()) {
            LOG_DEBUG("create new session session_key: ",
                      session_key.to_string(),
                      ", current sessions count: ",
                      table->sessions.size());
            const auto& comm_addr =
                observer_ptr->template get_comm_data<ccl::group_split_type::cluster,
                                                     ccl::device_topology_type::ring>();

            size_t rank_peer_addr = comm_addr.rank;

            std::string peer_addr = create_ipc_addr_for_rank(rank_peer_addr);
            sess = table->create_session<class_id>(
                session_key, observer_ptr, peer_addr, std::move(param), comm_addr.rank);
        }
        else {
            //renew existing
            sess = session_it->second;
            LOG_DEBUG("session reuse: session_key: ",
                      session_key.to_string(),
                      ", current sessions count: ",
                      table->sessions.size());
        }

        append_session_for_processing(session_key, sess);
    }

    void send_stop();

private:
    std::string create_ipc_addr_for_rank(size_t rank) const;
    template <ccl::device_topology_type topology_type, class device_t>
    void register_observer_impl(size_t rank_addr, observer_t<device_t>* observer_ptr);

    template <ccl::device_topology_type topology_type>
    void register_observer_impl(size_t rank_addr, ccl_ipc_gpu_comm* observer_ptr);

    std::atomic<bool> stop;
    std::mutex delivery_mutex;
    std::condition_variable delivery_condition;
    std::list<std::shared_ptr<session>> processing_queue;

    std::map<ccl_ipc_gpu_comm*, std::unique_ptr<std::thread>> listener_thread_map;

    void listener(ccl_ipc_gpu_comm* listener_device);

    void append_session_for_processing(const ipc_session_key& session_key,
                                       std::shared_ptr<session> sess);
};
} // namespace native
