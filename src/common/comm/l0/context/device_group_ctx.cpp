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
#include <sstream>

#include "common/comm/l0/devices/devices_declaration.hpp"
#include "common/comm/l0/context/device_group_ctx.hpp"
#include "common/comm/l0/context/device_storage.hpp"
#include "common/comm/l0/topology/ring/device_group_ring_creator.hpp"
#include "common/comm/l0/device_community_holder_impl.hpp"

#include "common/comm/l0/scheduler/device_group_scheduler.hpp"

#include "common/comm/l0/context/scaling_ctx/numa_ctx_impl.hpp"

namespace native {

std::shared_ptr<device_group_context> device_group_context::create(
    const ccl::context_comm_addr& comm_addr,
    const ccl::device_indices_type& group_device_ids,
    device_storage& devices) {
    std::shared_ptr<device_group_context> ret(
        new device_group_context(comm_addr, group_device_ids));

    //TODO More intellectual topology creation required
    //Ring
    {
        device_group_ring_topology top(*ret, devices);

        std::stringstream ss;
        auto matrix = top.build_p2p_capability_matrix(ss, group_device_ids);
        ss << "\nMatrix\n" << matrix << std::endl;

        if (!top.build(ss, comm_addr, group_device_ids, matrix)) {
            LOG_ERROR(
                "Cannot build DEVICE_GROUP_RING. Devices cannot communicate for current setup!\nBuild log:\n",
                ss.str());
            abort();
        }
        LOG_DEBUG("Device Group Context for ",
                  comm_addr.to_string(),
                  " build RING topology. Log:\n ",
                  ss.str());

        /*        native::detail::printer<device_group_ring_topology::type()> p;
        ccl_tuple_for_each(ring_device_topology->get_device_storage(), p);
        LOG_INFO("Device Group ", context_addr.to_string(), " RING topology:\n", p.to_string());
*/
    }

    //A2A
    {
        /* TODO
        auto a2a_device_topology = std::make_shared<device_community<ccl::group_split_type::a2a_device_group>>(context_addr);
        device_group_a2a_topology top(*this, plain_gpu_comms, ring_device_topology->get_device_storage_ptr());
        std::stringstream ss;
        auto matrix = top.build_p2p_capability_matrix(ss, group_device_ids);
        ss << "\nMatrix\n" << matrix << std::endl;
        if(!top.build(ss, 0, group_device_ids, matrix))
        {
            LOG_ERROR("Cannot build DEVICE_GROUP_RING. Devices cannot communicate for current setup!\nBuild log:\n", ss.str());
            abort();
        }
        LOG_DEBUG("Device Group Context for ", context_addr.to_string(), " build RING topology. Log:\n ", ss.str());
        native::detail::printer<device_group_ring_topology::type()> p;
        ccl_tuple_for_each(ring_device_topology->get_device_storage(), p);
        LOG_INFO("Device Group ", context_addr.to_string(), " RING topology:\n", p.to_string());
        LOG_INFO("Device Group ", context_addr.to_string(), " A2A topology:\nTODO!");
        //remember
        std::get<ccl::device_topology_type::a2a>(device_topology) = a2a_device_topology;
        */
    }

    return ret;
}

device_group_context::device_group_context(const ccl::context_comm_addr& comm_addr,
                                           const ccl::device_indices_type& group_device_ids)
        : scaling_context_base(),
          device_indices(group_device_ids),
          context_addr(comm_addr) {
    //scheduler
    scheduler_impl.reset(new device_group_scheduler);
}

device_group_context::~device_group_context() {}

const ccl::device_indices_type& device_group_context::get_group_device_indices() const {
    return device_indices;
}

device_group_context::scaling_context_base& device_group_context::get_numa_ctx() {
    return *this;
}
const device_group_context::scaling_context_base& device_group_context::get_numa_ctx() const {
    return *this;
}
} // namespace native
