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
#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "common/comm/l0/device_community.hpp"
#include "common/comm/l0/topology/ring/thread_group_ring_creator.hpp"
#include "common/comm/l0/context/device_storage.hpp"

#include "common/comm/l0/scheduler/thread_group_scheduler.hpp"

namespace native
{

thread_group_context::~thread_group_context()
{
}

bool thread_group_context::sync_barrier(const ccl::device_indices_t& device_indices_t,
                                             ccl::context_comm_addr& comm_addr, device_storage& devices)
{
    std::shared_ptr<specific_plain_device_storage> thread_device_list;

    // Collect per thread data

    //comm_addr.thread_idx = thread_device_group_ctx.size();
    aggregate_device_indices(comm_addr.thread_idx, device_indices_t);


    //check on group creation final condition
    device_group_ctx_ptr group_ctx = device_group_context::create(comm_addr, device_indices_t, devices);
    if(false == thread_device_group_ctx.insert({comm_addr.thread_idx, group_ctx}).second)
    {
        LOG_ERROR("cannot register devices group ctx for thread idx: ", comm_addr.thread_idx);
        abort();
    }

    LOG_DEBUG("Thread ", comm_addr.to_string(), " reached thread group communicator barrier");

    //Make device communities
    thread_device_topology[comm_addr.thread_idx] =
            std::make_tuple(std::make_shared<device_community<ccl::device_topology_type::thread_group_ring>>(comm_addr),
                            std::make_shared<device_community<ccl::device_topology_type::thread_group_torn_apart_ring>>(comm_addr),
                            std::make_shared<device_community<ccl::device_topology_type::a2a_thread_group>>(comm_addr));

    if(thread_device_group_ctx.size() != comm_addr.thread_count)
    {
        // not all threads are registered yet - wait for all
        LOG_DEBUG("Thread ", comm_addr.to_string(), " waits on barrier");
        return false;   //slave thread
    }

    //Current thread finalize communicator creation
    LOG_INFO("Thread ", comm_addr.to_string(), " starts hardware topologies creation");
    {
        std::stringstream ss;
        thread_group_ring_topology top (*this, devices/*.get_size<ccl_gpu_comm>() +
                                            devices.get_size<ccl_virtual_gpu_comm>()*/);
        auto matrix = top.build_p2p_capability_matrix(ss, per_thread_indices);
        if (!top.build(ss,  per_thread_indices, matrix))
        {
            LOG_ERROR("Cannot build THREAD_GROUP_RING. Build log:\n", ss.str());
        }

        LOG_DEBUG("Topologies RING created successfully. Log:\n", ss.str());
    }

    {
        //TODO Create A2A topology
        LOG_INFO("Thread Context Topologies A2A TODO");
    }

    {
        std::stringstream out;
        dump_thread_topologies(out);
        LOG_INFO("Thread (MASTER): ", comm_addr.to_string(), " finalized thread topology creation\n", out.str());
    }
    // create scheduler in final step
    scheduler_impl.reset(new thread_group_scheduler(comm_addr.thread_count));
    LOG_DEBUG("Final thread ", comm_addr.to_string(), " ready to communicate");

    return true; //master thread
}


void thread_group_context::aggregate_device_indices(size_t thread_id, const ccl::device_indices_t& new_indices)
{
    per_thread_indices.insert({thread_id, new_indices});
}

const ccl::process_device_indices_t& thread_group_context::get_thread_group_device_indices() const
{
    return per_thread_indices;
}

const ccl::device_indices_t& thread_group_context::get_device_group_indices(size_t thread_id) const
{
    auto it = per_thread_indices.find(thread_id);
    if(it == per_thread_indices.end())
    {
        LOG_ERROR("Cannot find device group for thread: ", thread_id, ". Empty indices");
        static const ccl::device_indices_t empty;
        return empty;
    }
    return it->second;
}

thread_group_context::device_group_ctx_ptr thread_group_context::get_device_group_ctx(size_t thread_id)
{
    auto it = thread_device_group_ctx.find(thread_id);
    if(it == thread_device_group_ctx.end())
    {
        LOG_ERROR("Cannot find device group for thread: ", thread_id, ". Empty context");
        return {};
    }
    return it->second;
}

void thread_group_context::dump_thread_topologies(std::ostream& out) const
{
    out << "Threads count: " << thread_device_topology.size() << std::endl;
    for(auto it = thread_device_topology.begin();
        it != thread_device_topology.end();
        ++it)
    {
        const auto& top = it->second;
        size_t thread = it->first;

        out << "\nThread Group: " << thread << " topology:\n";
        details::device_community_printer printer(out);
        ccl_tuple_for_each(top, printer);
    }
}


// observer interface implementations
void thread_group_context::attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val)
{
    register_observer_impl<ccl::device_topology_type::thread_group_ring>(observer);
}
void thread_group_context::attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val)
{
    register_observer_impl<ccl::device_topology_type::thread_group_ring>(observer);
}


void thread_group_context::attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val)
{
    register_observer_impl<ccl::device_topology_type::thread_group_torn_apart_ring>(observer);
}
void thread_group_context::attach_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val)
{
    register_observer_impl<ccl::device_topology_type::thread_group_torn_apart_ring>(observer);
}


void thread_group_context::invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val)
{
    auto &topologu_specific_observers =
            std::get<top_to_index(ccl::device_topology_type::thread_group_ring)>(observables);
    observers_container_t<ccl_gpu_comm>& container = std::get<ccl_gpu_comm::type_idx()>(topologu_specific_observers);
    auto it = container.find(observer);
    if(it == container.end())
    {
        throw std::runtime_error(std::string("invalid proxy: ") + observer->get_this()->get_device().to_string());
    }

    throw std::runtime_error(std::string("Valid proxy: ") + observer->get_this()->get_device().to_string());
}

void thread_group_context::invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_ring> val)
{
    throw std::runtime_error(std::string("Valid proxy: ") + observer->get_this()->get_device().to_string());
}

void thread_group_context::invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val)
{
    throw std::runtime_error(std::string("Valid proxy: ") + observer->get_this()->get_device().to_string());
}

void thread_group_context::invoke_scaleup_proxy_observer(proxy_observer<ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>>* observer,
                                       std::integral_constant<ccl::device_topology_type,
                                                              ccl::device_topology_type::thread_group_torn_apart_ring> val)
{
    throw std::runtime_error(std::string("Valid proxy: ") + observer->get_this()->get_device().to_string());
}
}
