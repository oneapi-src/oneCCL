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
#include "oneapi/ccl.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "common/comm/l0/communicator/process_group/process_a2a_communicator_impl.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/context/process_group_ctx.hpp"

using namespace ccl;

process_a2a_communicator::process_a2a_communicator(ccl::unified_device_type&& device,
                                                   ccl::unified_context_type&& ctx,
                                                   size_t thread_idx,
                                                   size_t process_idx,
                                                   const ccl::comm_split_attr& attr)
        : base_t(std::move(device), std::move(ctx), thread_idx, process_idx, /*comm_attr, */ attr) {
}

void process_a2a_communicator::visit(ccl::gpu_comm_attr& comm_attr) {
    ctx = comm_attr.get_process_context();

    //get rank & size
    auto topology = ctx->get_process_topology<base_t::topology_class()>(process_id, thread_id);
    this->initialize_comm_addr(get_device_path(), topology);

    this->set_comm_group_id(comm_attr.get_unique_id());
}

ccl::event process_a2a_communicator::barrier(const ccl::stream::impl_value_t& stream,
                                             const ccl::barrier_attr& attr,
                                             const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented yet");
}

/* allgatherv */
ccl::event process_a2a_communicator::allgatherv_impl(const void* send_buf,
                                                     size_t send_count,
                                                     void* recv_buf,
                                                     const ccl::vector_class<size_t>& recv_counts,
                                                     ccl::datatype dtype,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::allgatherv_attr& attr,
                                                     const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
ccl::event process_a2a_communicator::allgatherv_impl(const void* send_buf,
                                                     size_t send_count,
                                                     const ccl::vector_class<void*>& recv_bufs,
                                                     const ccl::vector_class<size_t>& recv_counts,
                                                     ccl::datatype dtype,
                                                     const ccl::stream::impl_value_t& stream,
                                                     const ccl::allgatherv_attr& attr,

                                                     const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* allreduce */
ccl::event process_a2a_communicator::allreduce_impl(const void* send_buf,
                                                    void* recv_buf,
                                                    size_t count,
                                                    ccl::datatype dtype,
                                                    ccl::reduction reduction,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::allreduce_attr& attr,
                                                    const ccl::vector_class<ccl::event>& deps) {
    using namespace native;

    static constexpr ccl::group_split_type group_id = base_t::topology_type();
    static constexpr ccl::device_topology_type class_id = base_t::topology_class();

    if (!is_ready()) {
        throw ccl::exception(std::string(
            "Device communicator for group_id: " + ::to_string(group_id) +
            " is not ready yet. Not all сommunicators are created in group. Please create them before usage"));
    }

    int comm_rank = rank();
    LOG_DEBUG("communicator for device idx: ", get_device_path(), ", rank idx: ", comm_rank);

    //TODO make const!
    ccl_buffer send_entry_buffer(const_cast<void**>(&send_buf),
                                 count * ccl::get_datatype_size(dtype),
                                 0,
                                 ccl_buffer_type::INDIRECT);
    ccl_buffer recv_entry_buffer(
        &recv_buf, count * ccl::get_datatype_size(dtype), 0, ccl_buffer_type::INDIRECT);

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
        using gpu_allreduce_entry = l0_allreduce_typed_entry<ccl_ipc_source_gpu_comm<ccl_gpu_comm>,
                                                             group_id>;

        schedule =
                ctx->scheduler_impl->submit_entry_ipc<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                               thread_id,
                                                                                               *device_community_impl,
                                                                                               ipc_src_real_it->second,
                                                                                               send_entry_buffer,
                                                                                               recv_entry_buffer,
                                                                                               count,
                                                                                               dtype,
                                                                                               reduction);
        */
    }
    else {
        auto ipc_src_virt_it = in_process_ipc_source_virtual_gpu_storage.find(comm_rank);
        if (ipc_src_virt_it != in_process_ipc_source_virtual_gpu_storage.end()) {
            LOG_DEBUG("Invoke: ", ipc_src_virt_it->second->to_string());
            /*
            using gpu_allreduce_entry = l0_allreduce_typed_entry<ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>,
                                                                group_id>;

            schedule =
                    ctx->scheduler_impl->submit_entry_ipc<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                                thread_id,
                                                                                            *device_community_impl,
                                                                                            ipc_src_virt_it->second,
                                                                                            send_entry_buffer,
                                                                                            recv_entry_buffer,
                                                                                            count,
                                                                                            dtype,
                                                                                            reduction);
            */
        }
        else {
            auto real_device_it = in_process_gpu_storage.find(comm_rank);
            if (real_device_it != in_process_gpu_storage.end()) {
                LOG_DEBUG("Invoke: ", real_device_it->second->to_string());
                /*
                using gpu_allreduce_entry = l0_allreduce_typed_entry<ccl_gpu_comm, group_id>;

                schedule =
                        ctx->scheduler_impl->submit_entry<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                                thread_id,
                                                                                                *device_community_impl,
                                                                                                real_device_it->second,send_entry_buffer,
                                                                                                recv_entry_buffer,
                                                                                                count,
                                                                                                dtype,
                                                                                                reduction);
                */
            }
            else {
                auto virtual_device_it = virtual_process_gpu_storage.find(comm_rank);
                if (virtual_device_it != virtual_process_gpu_storage.end()) {
                    LOG_DEBUG("Invoke: ", virtual_device_it->second->to_string());
                    /*
                    using gpu_allreduce_entry = l0_allreduce_typed_entry<ccl_virtual_gpu_comm, group_id>;

                    schedule =
                        ctx->scheduler_impl->submit_entry<gpu_allreduce_entry, ccl_sched_add_back>(process_id,
                                                                                                thread_id,
                                                                                                *device_community_impl,
                                                                                                virtual_device_it->second,send_entry_buffer,
                                                                                                recv_entry_buffer,
                                                                                                count,
                                                                                                dtype,
                                                                                                reduction);
                    */
                }
            }
        }
    }

    //if sched is not ready - send NULL
    if (schedule) {
        LOG_DEBUG("Device group finalized");
    }
    return std::unique_ptr<ccl::event_impl>(new ccl::gpu_shared_event_impl(std::move(schedule)));
}

/* alltoall */
ccl::event process_a2a_communicator::alltoall_impl(const void* send_buf,
                                                   void* recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::alltoall_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
ccl::event process_a2a_communicator::alltoall_impl(const ccl::vector_class<void*>& send_buf,
                                                   const ccl::vector_class<void*>& recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   const ccl::stream::impl_value_t& stream,
                                                   const ccl::alltoall_attr& attr,
                                                   const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
ccl::event process_a2a_communicator::alltoallv_impl(const void* send_buf,
                                                    const ccl::vector_class<size_t>& send_counts,
                                                    void* recv_buf,
                                                    const ccl::vector_class<size_t>& recv_counts,
                                                    ccl::datatype dtype,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::alltoallv_attr& attr,
                                                    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}
ccl::event process_a2a_communicator::alltoallv_impl(const ccl::vector_class<void*>& send_buf,
                                                    const ccl::vector_class<size_t>& send_counts,
                                                    ccl::vector_class<void*> recv_buf,
                                                    const ccl::vector_class<size_t>& recv_counts,
                                                    ccl::datatype dtype,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::alltoallv_attr& attr,

                                                    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
ccl::event process_a2a_communicator::broadcast_impl(void* buf,
                                                    size_t count,
                                                    ccl::datatype dtype,
                                                    int root,
                                                    const ccl::stream::impl_value_t& stream,
                                                    const ccl::broadcast_attr& attr,
                                                    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* reduce */
ccl::event process_a2a_communicator::reduce_impl(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl::reduction reduction,
                                                 int root,
                                                 const ccl::stream::impl_value_t& stream,
                                                 const ccl::reduce_attr& attr,
                                                 const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* reduce_scatter */
ccl::event process_a2a_communicator::reduce_scatter_impl(
    const void* send_buf,
    void* recv_buf,
    size_t recv_count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::reduce_scatter_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* sparse_allreduce */
ccl::event process_a2a_communicator::sparse_allreduce_impl(
    const void* send_ind_buf,
    size_t send_ind_count,
    const void* send_val_buf,
    size_t send_val_count,
    void* recv_ind_buf,
    size_t recv_ind_count,
    void* recv_val_buf,
    size_t recv_val_count,
    ccl::datatype index_dtype,
    ccl::datatype value_dtype,
    ccl::reduction reduction,
    const ccl::stream::impl_value_t& stream,
    const ccl::sparse_allreduce_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

COMM_INTERFACE_COLL_INSTANTIATION(process_a2a_communicator);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(process_a2a_communicator);
#endif /* CCL_ENABLE_SYCL */
