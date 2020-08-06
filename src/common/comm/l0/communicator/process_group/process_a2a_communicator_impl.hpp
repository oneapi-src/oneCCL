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
#include "common/comm/l0/communicator/process_group/process_a2a_communicator.hpp"
#include "common/comm/l0/communicator/typed_base_communicator_impl.hpp"

#include "common/comm/l0/devices/devices_declaration.hpp"
#include "common/comm/l0/device_community.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"
#include "common/comm/l0/scheduler/allied_process_group_scheduler.hpp"
#include "common/request/gpu_request.hpp"

/* allgatherv */
template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::allgatherv_impl(
    const buffer_type* send_buf,
    size_t send_count,
    buffer_type* recv_buf,
    const size_t* recv_counts,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    buffer_type& recv_buf,
    const size_t* recv_counts,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* allreduce */
template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::allreduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    using namespace native;

    static constexpr ccl::device_group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::ccl_error(std::string(
            "Device communicator for group_id: " + ::to_string(group_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    size_t comm_rank = rank();
    LOG_DEBUG("communicator for device idx: ", get_device_path(), ", rank idx: ", comm_rank);

    //TODO make const!
    ccl_buffer send_entry_buffer(const_cast<buffer_type**>(&send_buf),
                                 count * sizeof(buffer_type),
                                 0,
                                 ccl_buffer_type::INDIRECT);
    ccl_buffer recv_entry_buffer(
        &recv_buf, count * sizeof(buffer_type), 0, ccl_buffer_type::INDIRECT);

    using community_t = typename device_community_container<class_id>::element_type;
    community_t community = device_community_impl.get_topology();

    const auto& in_process_gpu_storage = community->get_devices<ccl_gpu_comm>();
    const auto& virtual_process_gpu_storage = community->get_devices<ccl_virtual_gpu_comm>();

    auto& ipc_gpu_storage = community->get_devices<ccl_ipc_gpu_comm>();
    (void)ipc_gpu_storage;
    auto& in_process_ipc_source_real_gpu_storage =
        community->get_devices<ccl_ipc_source_gpu_comm<ccl_gpu_comm>>();
    auto& in_process_ipc_source_virtual_gpu_storage =
        community->get_devices<ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>>();

    allied_process_group_scheduler::thread_schedule_ptr schedule;
    //source for collective operation is ipc sources, real gpu or virtual gpu
    auto ipc_src_real_it = in_process_ipc_source_real_gpu_storage.find(comm_rank);
    if (ipc_src_real_it != in_process_ipc_source_real_gpu_storage.end()) {
        LOG_DEBUG("Invoke: ", ipc_src_real_it->second->to_string());
        /*
        using gpu_allreduce_entry = l0_allreduce_typed_entry<buffer_type,
                                                             ccl_ipc_source_gpu_comm<ccl_gpu_comm>,
                                                             group_id>;

        schedule =
                ctx->scheduler_impl->submit_entry_ipc<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                               thread_id,
                                                                                               *device_community_impl,
                                                                                               ipc_src_real_it->second,
                                                                                               send_entry_buffer,
                                                                                               recv_entry_buffer,
                                                                                               count,
                                                                                               static_cast<ccl_reduction_t>(reduction));
*/
    }
    else {
        auto ipc_src_virt_it = in_process_ipc_source_virtual_gpu_storage.find(comm_rank);
        if (ipc_src_virt_it != in_process_ipc_source_virtual_gpu_storage.end()) {
            LOG_DEBUG("Invoke: ", ipc_src_virt_it->second->to_string());
            /*
        using gpu_allreduce_entry = l0_allreduce_typed_entry<buffer_type,
                                                             ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>,
                                                             group_id>;

        schedule =
                ctx->scheduler_impl->submit_entry_ipc<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                            thread_id,
                                                                                           *device_community_impl,
                                                                                           ipc_src_virt_it->second,
                                                                                           send_entry_buffer,
                                                                                           recv_entry_buffer,
                                                                                           count,
                                                                                           static_cast<ccl_reduction_t>(reduction));
*/
        }
        else {
            auto real_device_it = in_process_gpu_storage.find(comm_rank);
            if (real_device_it != in_process_gpu_storage.end()) {
                LOG_DEBUG("Invoke: ", real_device_it->second->to_string());
                /*
        using gpu_allreduce_entry = l0_allreduce_typed_entry<buffer_type, ccl_gpu_comm, group_id>;

        schedule =
                ctx->scheduler_impl->submit_entry<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                           thread_id,
                                                                                           *device_community_impl,
                                                                                           real_device_it->second,send_entry_buffer,
                                                                                           recv_entry_buffer,
                                                                                           count,
                                                                                           static_cast<ccl_reduction_t>(reduction));
*/
            }
            else {
                auto virtual_device_it = virtual_process_gpu_storage.find(comm_rank);
                if (virtual_device_it != virtual_process_gpu_storage.end()) {
                    LOG_DEBUG("Invoke: ", virtual_device_it->second->to_string());
                    /*
        using gpu_allreduce_entry = l0_allreduce_typed_entry<buffer_type, ccl_virtual_gpu_comm, group_id>;


        schedule =
            ctx->scheduler_impl->submit_entry<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                       thread_id,
                                                                                       *device_community_impl,
                                                                                       virtual_device_it->second,send_entry_buffer,
                                                                                       recv_entry_buffer,
                                                                                       count,
                                                                                       static_cast<ccl_reduction_t>(reduction));
    */
                }
            }
        }
    }

    //if sched is not ready - send NULL
    if (schedule) {
        LOG_DEBUG("Device group finalized");
    }
    return std::unique_ptr<ccl::gpu_shared_request_impl>(
        new ccl::gpu_shared_request_impl(std::move(schedule)));
}

template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::allreduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoall */
template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::alltoall_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::alltoall_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::alltoallv_impl(
    const buffer_type* send_buf,
    const size_t* send_counts,
    buffer_type* recv_buf,
    const size_t* recv_counts,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::alltoallv_impl(
    const buffer_type& send_buf,
    const size_t* send_counts,
    buffer_type& recv_buf,
    const size_t* recv_counts,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::bcast_impl(
    buffer_type* buf,
    size_t count,
    size_t root,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::bcast_impl(
    buffer_type& buf,
    size_t count,
    size_t root,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* reduce */
template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::reduce_impl(
    const buffer_type* send_buf,
    buffer_type* recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::reduce_impl(
    const buffer_type& send_buf,
    buffer_type& recv_buf,
    size_t count,
    ccl::reduction reduction,
    size_t root,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* sparse_allreduce */
template <class index_buffer_type, class value_buffer_type>
ccl::communicator::coll_request_t process_a2a_communicator::sparse_allreduce_impl(
    const index_buffer_type* send_ind_buf,
    size_t send_ind_count,
    const value_buffer_type* send_val_buf,
    size_t send_val_count,
    index_buffer_type* recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_type* recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class index_buffer_container_type, class value_buffer_container_type>
ccl::communicator::coll_request_t process_a2a_communicator::sparse_allreduce_impl(
    const index_buffer_container_type& send_ind_buf,
    size_t send_ind_count,
    const value_buffer_container_type& send_val_buf,
    size_t send_val_count,
    index_buffer_container_type& recv_ind_buf,
    size_t recv_ind_count,
    value_buffer_container_type& recv_val_buf,
    size_t recv_val_count,
    ccl::reduction reduction,
    const ccl::coll_attr* attr,
    ccl::stream::impl_t& stream) {
    throw ccl::ccl_error(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
