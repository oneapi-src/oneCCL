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
#include "common/comm/l0/context/scale/numa/numa_ctx.hpp"
#include "common/utils/tuple.hpp"
#include "common/log/log.hpp"

namespace native {

#define TEMPLATE_DECL_ARG class Impl, ccl::device_topology_type... types
#define TEMPLATE_DEF_ARG  Impl, types...

template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type class_id, class device_t>
void numa_ctx<TEMPLATE_DEF_ARG>::register_observer_impl(size_t rank_addr,
                                                        observer_t<device_t>* observer_ptr) {
    LOG_INFO(
        "device rank addr: ", std::to_string(rank_addr), ", device: ", observer_ptr->to_string());
    observer::container_t<observer_t<device_t>>& container =
        scaling_ctx_base_t::template get_types_container<observer_t<device_t>, class_id>(
            observables);
    auto cont_it = container.find(observer_ptr);
    if (cont_it == container.end()) {
        container.insert(observer_ptr);

        // remember total count
        registered_devices_count++;

        // prepare session tables
        session_table_initializer init;
        ccl_tuple_for_each_args(collective_sessions, init, observer_ptr);

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
                                 " for NUMA device:\n" + observer_ptr->to_string() +
                                 "\nBecause it registered already:\n" + ss.str());
    }

    indexed_container.emplace(rank_addr, observer_ptr);

    // start NUMA worker
    auto& thread_map =
        ccl_tuple_get<observer::device_thread_map<observer_t<device_t>, numa_actor>>(numa_workers);
    {
        std::unique_ptr<numa_actor> new_actor{ new numa_actor(
            rank_addr, &numa_ctx<TEMPLATE_DEF_ARG>::worker<device_t>, this, observer_ptr) };
        observer::detail::actor_visitor visitor;
        ccl_tuple_for_each_args(numa_workers, visitor, new_actor.get());
        thread_map[observer_ptr] = std::move(new_actor);
    }
}

template <TEMPLATE_DECL_ARG>
template <class device_t>
void numa_ctx<TEMPLATE_DEF_ARG>::worker(observer_t<device_t>* listener_device,
                                        numa_actor* actor_ptr,
                                        typename numa_actor::storage_t& todo_list) {
    size_t total_actors_count = actor_ptr->get_subscriptions_count();

    LOG_DEBUG("Start NUMA context worker, Listener device: ",
              listener_device->to_string(),
              ",\nactor_id: ",
              actor_ptr->get_id(),
              ", total_actors_count: ",
              total_actors_count,
              ",\ntotal devices: ",
              registered_devices_count,
              ",\ntodo list size: ",
              todo_list.size());

    for (auto sess_it = todo_list.begin(); sess_it != todo_list.end();) {
        void* partial_chunk = nullptr;
        size_t partial_chunk_size = 0;

        // get own device partial chunk data
        (*sess_it)->produce_data(&partial_chunk, partial_chunk_size);
        if (partial_chunk_size > 0) {
            // notify other actor for data_ready
            observer::detail::actor_publisher<session_ptr_t, observer::session_notification>
                visitor;
            ccl_tuple_for_each_args(numa_workers,
                                    visitor,
                                    (*sess_it)->get_send_tag(),
                                    actor_ptr->get_id(),
                                    partial_chunk,
                                    partial_chunk_size);
        }

        // consume data_ready from other actor: starting from myself(!)
        bool session_finished = false;
        for (size_t actor_index = actor_ptr->get_id();
             actor_index < actor_ptr->get_id() + total_actors_count;
             actor_index++) {
            std::list<typename numa_actor::mailbox_message_t> messages;
            actor_ptr->get_mailbox_messages(
                actor_index % total_actors_count, (*sess_it)->get_send_tag(), messages);

            for (auto mess_it = messages.begin(); mess_it != messages.end(); ++mess_it) {
                (*sess_it)->consume_data(
                    0 /*TODO !!!! */, mess_it->host_src_ptr, mess_it->src_size_bytes);
                session_finished = (*sess_it)->is_consumed();
                assert(not(session_finished && std::next(mess_it, 1) != messages.end()) &&
                       "Session are filled too early");
            }
        }

        if (session_finished) {
            sess_it = todo_list.erase(sess_it);
            //TODO invoke BCast !!!
        }
        else {
            ++sess_it;
        }
    }
}
#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG

} // namespace native
