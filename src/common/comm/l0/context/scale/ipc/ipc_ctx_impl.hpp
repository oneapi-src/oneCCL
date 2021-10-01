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
#include "common/comm/l0/context/scale/ipc/ipc_ctx.hpp"
#include "common/utils/tuple.hpp"

#include "common/comm/l0/context/scale/ipc/ipc_ctx_session.hpp"
#include "common/log/log.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_client.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_server.hpp"
#include "common/comm/l0/devices/communication_structs/ipc_connection.hpp"

namespace native {

#define TEMPLATE_DECL_ARG class Impl, ccl::device_topology_type... types
#define TEMPLATE_DEF_ARG  Impl, types...

/*
template <TEMPLATE_DECL_ARG>
ipc_ctx<TEMPLATE_DEF_ARG>::~ipc_ctx() {

    send_stop();
    delivery_thread.join();
}
*/
template <TEMPLATE_DECL_ARG>
void ipc_ctx<TEMPLATE_DEF_ARG>::initialize_ctx(
    std::shared_ptr<ccl::host_communicator> communicator) {
    (void)communicator;
    //send_stop();
    stop.store(false);
    LOG_DEBUG("IPC context Initialized for mpi rank: (",
              std::to_string(communicator->rank()),
              "/",
              std::to_string(communicator->size()),
              ")");
}

template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type class_id, class device_t>
void ipc_ctx<TEMPLATE_DEF_ARG>::register_observer_impl(size_t rank_addr,
                                                       observer_t<device_t>* observer_ptr) {
    LOG_DEBUG(
        "device rank addr: ", std::to_string(rank_addr), ", device: ", observer_ptr->to_string());
    observer::container_t<observer_t<device_t>>& container =
        scaling_ctx_base_t::template get_types_container<observer_t<device_t>, class_id>(
            observables);
    auto cont_it = container.find(observer_ptr);
    if (cont_it == container.end()) {
        container.insert(observer_ptr);

        // prepare IPC session tables
        for (size_t i = static_cast<size_t>(ccl_coll_allgatherv);
             i < static_cast<size_t>(ccl_coll_internal);
             i++) {
            ccl_coll_type type = static_cast<ccl_coll_type>(i);

            auto& tuple_sessions = collective_sessions[type];
            auto& sessions_table =
                ccl_tuple_get<ipc_src_session_data<observer_t<device_t>>>(tuple_sessions);
            sessions_table.source_sessions.emplace(
                observer_ptr, std::make_shared<session_table>(session_table{}));
        }

        if (rank_addr == std::numeric_limits<size_t>::max()) {
            return; //nothing to do more
        }
    }

    //reassign with index
    assert(rank_addr != std::numeric_limits<size_t>::max() &&
           "Reassign with assigned address failed");

    observer::indexed_container_t<observer_t<device_t>>& indexed_container =
        scaling_ctx_base_t::template get_types_container<observer_t<device_t>, class_id>(
            indexed_observables);

    auto indexed_it = indexed_container.find(rank_addr);
    if (indexed_it != indexed_container.end()) {
        // collect troubleshooting info
        std::stringstream ss;
        for (const auto& indexed_dev : indexed_container) {
            ss << "rank: " << indexed_dev.first << ", dev: " << indexed_dev.second->to_string()
               << "\n";
        }
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 "- Cannot reassing rank: " + std::to_string(rank_addr) +
                                 " for device:\n" + observer_ptr->to_string() +
                                 "\nBecause it registered already:\n" + ss.str());
    }

    indexed_container.emplace(rank_addr, observer_ptr);
}

template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type class_id>
void ipc_ctx<TEMPLATE_DEF_ARG>::register_observer_impl(size_t rank_addr,
                                                       ccl_ipc_gpu_comm* observer_ptr) {
    LOG_DEBUG("DST device rank addr: ",
              std::to_string(rank_addr),
              ", DST device: ",
              observer_ptr->to_string());
    observer::container_t<ccl_ipc_gpu_comm>& container =
        scaling_ctx_base_t::template get_types_container<ccl_ipc_gpu_comm, class_id>(observables);
    auto cont_it = container.find(observer_ptr);
    if (cont_it == container.end()) {
        container.insert(observer_ptr);

        if (rank_addr == std::numeric_limits<size_t>::max()) {
            return; //nothing to do more
        }
    }

    //reassign with index
    assert(rank_addr != std::numeric_limits<size_t>::max() &&
           "Reassign with assigned address failed");

    observer::indexed_container_t<ccl_ipc_gpu_comm>& indexed_container =
        scaling_ctx_base_t::template get_types_container<ccl_ipc_gpu_comm, class_id>(
            indexed_observables);

    auto indexed_it = indexed_container.find(rank_addr);
    if (indexed_it != indexed_container.end()) {
        // collect troubleshooting info
        std::stringstream ss;
        for (const auto& indexed_dev : indexed_container) {
            ss << "rank: " << indexed_dev.first << ", dev: " << indexed_dev.second->to_string()
               << "\n";
        }
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 "- Cannot reassing rank: " + std::to_string(rank_addr) +
                                 " for device:\n" + observer_ptr->to_string() +
                                 "\nBecause it registered already:\n" + ss.str());
    }

    indexed_container.emplace(rank_addr, observer_ptr);

    //Start IPC server for each DST device for listening incoming conections from SRC devices
    std::string addr = create_ipc_addr_for_rank(rank_addr);
    LOG_DEBUG("Start IPC listener for device:\n", observer_ptr->to_string(), "\nAddr: ", addr);
    try {
        observer_ptr->start(addr);
        auto it = listener_thread_map.find(observer_ptr);
        if (it != listener_thread_map.end()) {
            throw std::runtime_error(std::string("IPC listener already exist for device: ") +
                                     observer_ptr->to_string());
        }

        listener_thread_map.emplace(
            observer_ptr,
            new std::thread(&ipc_ctx<TEMPLATE_DEF_ARG>::listener, this, observer_ptr));
        LOG_DEBUG("Listener thread started on addr: ", addr);
    }
    catch (const std::exception& ex) {
        LOG_ERROR("Cannot start IPC listener on: ",
                  addr,
                  " for device: ",
                  observer_ptr->to_string(),
                  ", error: ",
                  ex.what());
        throw;
    }
}

template <TEMPLATE_DECL_ARG>
std::string ipc_ctx<TEMPLATE_DEF_ARG>::create_ipc_addr_for_rank(size_t rank) const {
    std::string ret;

    //TODO make unique for KVS group
    ret += "IPC_[";
    ret += std::to_string(rank);
    ret += "]";
    return ret;
}

template <TEMPLATE_DECL_ARG>
void ipc_ctx<TEMPLATE_DEF_ARG>::append_session_for_processing(const ipc_session_key& session_key,
                                                              std::shared_ptr<session> sess) {
    LOG_DEBUG("session_key: ", session_key.to_string(), ", selected session: ", sess->to_string());
    {
        std::lock_guard<std::mutex> lk(delivery_mutex);
        processing_queue.push_back(sess);
        delivery_condition.notify_one();
    }
}

template <TEMPLATE_DECL_ARG>
void ipc_ctx<TEMPLATE_DEF_ARG>::send_stop() {
    stop.store(true);
    delivery_condition.notify_all();
}

template <TEMPLATE_DECL_ARG>
void ipc_ctx<TEMPLATE_DEF_ARG>::listener(ccl_ipc_gpu_comm* listener_device) {
    LOG_DEBUG("Start IPC context listener worker, Listener device: ", listener_device->to_string());

    // TODO ring only, peer-to-peer case: one SRC connects to one DST
    std::unique_ptr<net::ipc_rx_connection> incoming_connection;
    while (!incoming_connection) {
        try {
            auto incoming = listener_device->process_connection();
            if (incoming) {
                LOG_DEBUG("Got connection on device: ", listener_device->to_string());
                incoming_connection = std::move(incoming);
            }
        }
        catch (const std::exception& ex) {
            LOG_DEBUG("Stop requested at serving connection stage");
            return;
        }

        if (stop.load()) {
            LOG_DEBUG("Stop requested at serving connection stage");
            return;
        }
    }

    // processing incoming data from connected clients
    LOG_DEBUG("Start IPC context processing worker, Listener device: ",
              listener_device->to_string());
    while (!stop.load()) {
        //TODO choose std::list
        decltype(processing_queue) sessions_to_execute;
        {
            std::unique_lock<std::mutex> lk(delivery_mutex);
            delivery_condition.wait(lk, [this]() {
                return !processing_queue.empty() || stop.load();
            });

            sessions_to_execute.splice(sessions_to_execute.end(), processing_queue);
        }

        LOG_DEBUG("Sessions for processing: ",
                  sessions_to_execute.size(),
                  " stop flag status: ",
                  stop.load());
        for (auto sess_it = sessions_to_execute.begin();
             sess_it != sessions_to_execute.end() and !stop.load();) {
            shared_session_ptr_t sess = *sess_it;

            // try restore IPC handles
            LOG_DEBUG("process session: ", sess->to_string());
            if (!sess->process(listener_device, incoming_connection.get())) {
                ++sess_it;
                continue;
            }

            // bind restored IPC handles to kernel
            sess->visit(listener_device, listener_device->get_registered_modules());
            LOG_DEBUG("session processed: ", sess->to_string());

            // find next session
            sess_it = sessions_to_execute.erase(sess_it);
        }
    }
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG

} // namespace native
