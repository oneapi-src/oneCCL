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
#include "ccl.hpp"
#include "ccl_type_traits.hpp"
#include "common/comm/l0/communicator/typed_base_communicator.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"

#define TEMPLATE_DECL_ARG              class comm_impl, ccl::device_topology_type topology, class communicator_traits
#define TEMPLATE_DEF_ARG               comm_impl, topology, communicator_traits


template<TEMPLATE_DECL_ARG>
typed_base_communicator<TEMPLATE_DEF_ARG>::typed_base_communicator(ccl::unified_device_type&& owned_device,
                                                           size_t thread_idx, size_t process_idx,
                                                           const ccl::device_comm_attr_t& attr) :
 base_communicator(std::move(owned_device),
                   thread_idx, process_idx/*, comm_attr*/, attr)
{
    LOG_INFO("sheduled for create, device id: ", device.get_id(), ", thread_id: ",thread_idx, ", process id:", process_idx);
}

template<TEMPLATE_DECL_ARG>
void typed_base_communicator<TEMPLATE_DEF_ARG>::initialize_comm_addr(const ccl::device_index_type& device_id,
                                                             std::shared_ptr<native::device_community<topology>> new_community)
{
    native::details::rank_getter<topology> initializer(device_id,
                                                   new_community->registered_device_id);

    {
        std::unique_lock<ccl_spinlock> lock(ready_mutex);
        device_community_impl = new_community;

        if (!device_community_impl->get_device_storage_ptr())
        {
            std::string err_str;
            {
                std::stringstream str;
                ccl_logger::format(str, "Cannot initialize comm_addr for device id: ", device_id,
                                       " on topology: ", ::to_string(self_t::get_topology_type()),
                                       ", empty device storage has got from context");
                err_str = str.str();
            }
            LOG_ERROR(err_str);
            throw std::runtime_error(err_str);
        }
        ccl_tuple_for_each(device_community_impl->get_device_storage(), initializer);
    }

    comm_rank = initializer.rank;
    comm_size = initializer.size;

    //TODO
    //ADD communicator device binding

    LOG_INFO("Communicator finalized. Rank (", comm_rank, "/", comm_size,
             ") on {dev: ", device_id, ", thr: ",thread_id, ", proc: ", process_id, "}");
}

template<TEMPLATE_DECL_ARG>
bool typed_base_communicator<TEMPLATE_DEF_ARG>::is_ready() const
{
    if(!device_community_impl.get())
    {
        std::unique_lock<ccl_spinlock> lock(ready_mutex);
        return device_community_impl.get();
    }
    return true;
}

template<TEMPLATE_DECL_ARG>
ccl::device_topology_type typed_base_communicator<TEMPLATE_DEF_ARG>::get_topology_type() const
{
    return self_t::topology_type();
}

template<TEMPLATE_DECL_ARG>
template<class device_t>
size_t typed_base_communicator<TEMPLATE_DEF_ARG>::get_device_count() const
{
    return ccl_tuple_get<native::indexed_device_container<device_t>>(device_community_impl->get_device_storage()).size();
}

template<TEMPLATE_DECL_ARG>
template<class device_t>
native::indexed_device_container<device_t>& typed_base_communicator<TEMPLATE_DEF_ARG>::get_devices()
{
    return std::get<device_t::type_idx()>(device_community_impl->get_device_storage());
}

template<TEMPLATE_DECL_ARG>
std::string typed_base_communicator<TEMPLATE_DEF_ARG>::to_string() const
{
    native::details::printer<self_t::topology_type()> p;
    ccl_tuple_for_each(device_community_impl->get_device_storage(), p);
    return std::string("Rank (") + std::to_string(rank()) + "/" + std::to_string(size()) +
            "\nTopology: " + ::to_string(self_t::get_topology_type()) + ":\n" + p.to_string();
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG
