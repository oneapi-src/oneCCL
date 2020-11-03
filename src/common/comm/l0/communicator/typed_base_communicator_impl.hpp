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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "common/comm/l0/communicator/typed_base_communicator.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/context/thread_group_ctx.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"
#include "common/comm/l0/comm_context_storage.hpp"
#include "common/comm/l0/comm_context.hpp"

#define TEMPLATE_DECL_ARG \
    class comm_impl, ccl::group_split_type topology, ccl::device_topology_type class_id, \
        class communicator_traits
#define TEMPLATE_DEF_ARG comm_impl, topology, class_id, communicator_traits

template <TEMPLATE_DECL_ARG>
typed_base_communicator<TEMPLATE_DEF_ARG>::typed_base_communicator(
    ccl::unified_device_type&& owned_device,
    ccl::unified_device_context_type&& owned_ctx,
    size_t thread_idx,
    size_t process_idx,
    const ccl::comm_split_attr& attr)
        : base_communicator(std::move(owned_device), std::move(owned_ctx),
                            thread_idx,
                            process_idx /*, comm_attr*/,
                            attr) {
    try {
        LOG_INFO("sheduled for create, device id: ",
                 device.get_id(),
                 ", thread_id: ",
                 thread_idx,
                 ", process id:",
                 process_idx);
    }
    catch (...) {
        LOG_INFO("sheduled for create single device communicator , thread_id: ",
                 thread_idx,
                 ", process id:",
                 process_idx);
    }
}

template <TEMPLATE_DECL_ARG>
void typed_base_communicator<TEMPLATE_DEF_ARG>::initialize_comm_addr(
    const ccl::device_index_type& device_id,
    native::device_community_container<class_id>& new_community) {
    // Iterate over community container, find device and assing rank, size from topology.
    // Lets register woned deive in each toplogy, but return as PUBLIC the only onw
    // It is not matter, what speficic topology use select here for PUBLIC rank & size
    // for rank assigning in auto-ranking mode. SO, use clsoed ring t first, then goesn into torn_apart

    ccl::context_comm_addr registered_addr;
    {
        std::unique_lock<ccl_spinlock> lock(ready_mutex);
        device_community_impl = new_community;
        device_community_impl.template register_device_by_id<topology_type()>(device_id,
                                                                              registered_addr);
    }

    //TODO multiple topologies in curr class_id
    comm_rank = registered_addr.comm_rank;
    comm_size = registered_addr.comm_size;

    //TODO
    //ADD communicator device binding

    LOG_INFO("Communicator finalized. Rank (",
             comm_rank,
             "/",
             comm_size,
             ") on {dev: ",
             device_id,
             ", thr: ",
             thread_id,
             ", proc: ",
             process_id,
             "}");
}

template <TEMPLATE_DECL_ARG>
bool typed_base_communicator<TEMPLATE_DEF_ARG>::is_ready() const {
    /* TODO!!!!
    if(!device_community_impl.get())
    {
        std::unique_lock<ccl_spinlock> lock(ready_mutex);
        return device_community_impl.get();
    }
    */
    return true;
}

template <TEMPLATE_DECL_ARG>
ccl::group_split_type typed_base_communicator<TEMPLATE_DEF_ARG>::get_topology_type() const {
    return self_t::topology_type();
}

template <TEMPLATE_DECL_ARG>
ccl::device_topology_type typed_base_communicator<TEMPLATE_DEF_ARG>::get_topology_class() const {
    return self_t::topology_class();
}
/*
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
*/
template <TEMPLATE_DECL_ARG>
std::string typed_base_communicator<TEMPLATE_DEF_ARG>::to_string() const {
    native::details::printer<self_t::topology_type(), self_t::topology_class()> p;
    ccl_tuple_for_each(device_community_impl->get_device_storage(), p);
    return std::string("Rank (") + std::to_string(rank()) + "/" + std::to_string(size()) +
           "\nGroup id: " + ::to_string(self_t::topology_type()) +
           "\nClassId: " + ::to_string(self_t::topology_class()) + ":\n" + p.to_string();
}

template <TEMPLATE_DECL_ARG>
ccl::communicator_interface_ptr
typed_base_communicator<TEMPLATE_DEF_ARG>::split(const ccl::comm_split_attr& attr) {
    if (!attr.is_valid<ccl::comm_split_attr_id::group>()) {
        throw ccl::exception(std::string(__FUNCTION__) +
                        " - TODO `comm_split_attr`: supports `group` only");
    }
    //TODO
    #ifdef MULTI_GPU_SUPPORT
        auto id = get_impl()->get_comm_group_id();
        ccl::group_context::comm_group_t my_group =
            ccl::group_context::instance().get_existing_group_by_id(id);
        #ifdef CCL_ENABLE_SYCL
            auto ctx = get_impl()->get_context();
            return my_group->create_communicator_from_group<cl::sycl::device>(get_device(), ctx, attr);
        #else
            #ifdef MULTI_GPU_SUPPORT
                auto ctx = get_impl()->get_context();
                return my_group->create_communicator_from_group(get_impl()->get_device_path(), ctx, attr);
            #endif
        #endif
    #else
        throw ccl::exception(std::string(__FUNCTION__) + " - TODO `comm_split_attr`: unsupported");
        return this;
    #endif
}

#undef TEMPLATE_DECL_ARG
#undef TEMPLATE_DEF_ARG
