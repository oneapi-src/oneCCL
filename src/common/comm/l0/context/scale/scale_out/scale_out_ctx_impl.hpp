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
#include "common/comm/l0/context/scale/scale_out/scale_out_ctx.hpp"
#include "common/log/log.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"

namespace native {

#define TEMPLATE_DECL_ARG class Impl, ccl::device_topology_type... types
#define TEMPLATE_DEF_ARG  Impl, types...

template <TEMPLATE_DECL_ARG>
void scale_out_ctx<TEMPLATE_DEF_ARG>::initialize_ctx(
    std::shared_ptr<ccl::host_communicator> communicator) {
    process_communicator = communicator;

    LOG_DEBUG("SCALE-OUT context Initialized for mpi rank: (",
              std::to_string(communicator->rank()),
              "/",
              std::to_string(communicator->size()),
              ")");
}

// observer_ptr interface implementations
template <TEMPLATE_DECL_ARG>
template <ccl::device_topology_type class_id, class device_t>
void scale_out_ctx<TEMPLATE_DEF_ARG>::register_observer_impl(size_t rank_addr,
                                                             observer_t<device_t>* observer_ptr) {
    LOG_DEBUG("scaleout device rank addr: ",
              std::to_string(rank_addr),
              ", device: ",
              observer_ptr->to_string());
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
                                 " for SCALEOUT device:\n" + observer_ptr->to_string() +
                                 "\nBecause it registered already:\n" + ss.str());
    }

    indexed_container.emplace(rank_addr, observer_ptr);

    // start SCALEOUT worker
    auto& thread_map =
        ccl_tuple_get<observer::device_thread_map<observer_t<device_t>, scaleout_actor>>(
            scaleout_workers);
    {
        std::unique_ptr<scaleout_actor> new_actor{ new scaleout_actor(
            rank_addr, &scale_out_ctx<TEMPLATE_DEF_ARG>::worker<device_t>, this, observer_ptr) };
        thread_map[observer_ptr] = std::move(new_actor);
    }
}

template <TEMPLATE_DECL_ARG>
template <class device_t>
void scale_out_ctx<TEMPLATE_DEF_ARG>::worker(observer_t<device_t>* listener_device,
                                             scaleout_actor* actor_ptr,
                                             typename scaleout_actor::storage_t& todo_list) {
    LOG_DEBUG("Start SCALEOUT context worker, Listener device: ",
              listener_device->to_string(),
              ",\nactor_id: ",
              actor_ptr->get_id(),
              ",\ntodo list size: ",
              todo_list.size());

    // invoke CPU collective on data chunk
    for (auto sess_it = todo_list.begin(); sess_it != todo_list.end();) {
        session_ptr_t sess = *sess_it;

        sess->produce_data(process_communicator);
        ++sess_it;
    }

    // check CPU collective accomplishment
    for (auto sess_it = todo_list.begin(); sess_it != todo_list.end();) {
        (*sess_it)->consume_data(0 /*TODO !!!! */, process_communicator);
        if ((*sess_it)->is_consumed()) {
            sess_it = todo_list.erase(sess_it);
        }
        else {
            ++sess_it;
        }
    }
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG
} // namespace native
