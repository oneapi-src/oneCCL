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
#include "atl/atl_base_comm.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"
#include "exec/exec.hpp"
#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"
#include "coll/ccl_allgather_op_attr.hpp"
#include "comm/comm.hpp"
#include "comm/comm_impl.hpp"
#include "common/global/global.hpp"
#include "common/event/impls/host_event.hpp"
#include "common/request/request.hpp"
#include "sched/sched.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/kvs.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "kvs_impl.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

// ccl_comm_env

ccl_comm_env::ccl_comm_env(std::shared_ptr<ccl::device> device) : device(device) {
#ifdef CCL_ENABLE_SYCL
    enable_topo_algo = ccl::global_data::env().enable_topo_algo;
    ze_copy_engine = ccl::global_data::env().ze_copy_engine;
    ze_h2d_copy_engine = ccl::global_data::env().ze_h2d_copy_engine;

    if (device &&
        (device.get()->get_native().get_backend() == ccl::utils::get_level_zero_backend())) {
        auto ze_device =
            sycl::get_native<ccl::utils::get_level_zero_backend()>(device.get()->get_native());
        CCL_THROW_IF_NOT(ze_device, "null ze device");

        if ((ccl::ze::get_device_family(ze_device) == ccl::device_family::unknown) ||
            (ccl::ze::get_device_family(ze_device) == ccl::device_family::family1)) {
            ze_copy_engine = ccl::ze::copy_engine_mode::none;
            ze_h2d_copy_engine = ccl::ze::h2d_copy_engine_mode::none;
        }
    }
    else {
        enable_topo_algo = 0;
        ze_copy_engine = ccl::ze::copy_engine_mode::none;
        ze_h2d_copy_engine = ccl::ze::h2d_copy_engine_mode::none;
    }
#endif // CCL_ENABLE_SYCL
}

std::string ccl_comm_env::to_string() const {
    std::stringstream ss;
    ss << "{";

#ifdef CCL_ENABLE_SYCL
    if (device) {
        ss << " enable_topo_algo: " << enable_topo_algo;
        ss << ", ze_copy_engine: " << ccl::ze::copy_engine_names[ze_copy_engine];
        ss << ", ze_h2d_copy_engine: " << ccl::ze::h2d_copy_engine_names[ze_h2d_copy_engine];
        ss << " ";
    }
#endif // CCL_ENABLE_SYCL

    ss << "}";

    return ss.str();
}

// ccl_internal_comm

ccl_internal_comm::ccl_internal_comm(int comm_id,
                                     int rank,
                                     int size,
                                     std::shared_ptr<atl_base_comm> comm)
        : m_dtree(size, rank) {
    atl_comm = atl_comm_manager::create_with_id(comm, comm_id);
    reset(rank, size);

    if (comm_id == comm->get_comm_id()) {
        LOG_DEBUG("comm.id == explicit_id, reset comm.id ", comm_id);
        comm->reset_comm_id();
    }
}

void ccl_internal_comm::reset(int rank, int size) {
    m_rank = rank;
    m_size = size;
    m_pof2 = ccl::utils::pof2(m_size);
}

// ccl_comm

void ccl_comm::init(int comm_id,
                    std::shared_ptr<atl_base_comm> atl_comm,
                    bool share_resources,
                    bool is_sub_communicator) {
    comm_rank = atl_comm->get_rank();
    comm_size = atl_comm->get_size();

    next_sched_id_internal = ccl_comm::max_sched_count / 2;
    next_sched_id_external = 0;

    if (comm_rank >= comm_size || comm_size <= 0) {
        throw ccl::exception("incorrect rank or size when creating \
                             communicator: rank: " +
                             std::to_string(comm_rank) + ", size: " + std::to_string(comm_size));
    }

    comm_impl = std::unique_ptr<ccl_internal_comm>(
        new ccl_internal_comm(comm_id, comm_rank, comm_size, atl_comm));

    if (!share_resources) {
        allocate_resources();
    }

    if (!is_sub_communicator) {
        topo_manager.init(atl_comm, device_ptr, context_ptr);
        if (!comm_rank && device_ptr) {
            LOG_INFO("topo_manager:", topo_manager.to_string());
        }
        create_topo_subcomms();
    }
    else {
        local2global_map = atl_comm->get_rank2rank_map();
    }

    env = std::make_shared<ccl_comm_env>(device_ptr);

    if (comm_rank == 0) {
        LOG_DEBUG(to_string_ext());
    }
}

ccl_comm::ccl_comm(int comm_id,
                   std::shared_ptr<atl_base_comm> atl_comm,
                   bool share_resources,
                   bool is_sub_communicator) {
    init(comm_id, atl_comm, share_resources, is_sub_communicator);
}

ccl_comm::ccl_comm(std::shared_ptr<atl_base_comm> atl_comm,
                   bool share_resources,
                   bool is_sub_communicator)
        : ccl_comm(atl_comm->create_comm_id(), atl_comm, share_resources, is_sub_communicator) {}

ccl_comm::ccl_comm(device_t device, context_t context, std::shared_ptr<atl_base_comm> atl_comm)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)) {
    init(atl_comm->create_comm_id(), atl_comm);
}

ccl_comm::ccl_comm(int size, int rank, ccl::shared_ptr_class<ikvs_wrapper> kvs)
        : ccl_comm(atl_comm_manager::create(size, { rank }, kvs)) {}

ccl_comm::ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs)
        : ccl_comm(atl_comm_manager::create(size, { 0 }, kvs)) {}

ccl_comm::ccl_comm() : ccl_comm(atl_comm_manager::create()) {}

ccl_comm::ccl_comm(const ccl_comm& src, int comm_id)
        : ccl_comm(comm_id, src.get_atl_comm(), true, true) {
    r2r_comm = src.r2r_comm;
    node_comm = src.node_comm;
    even_comm = src.even_comm;
    pair_comm = src.pair_comm;
}

std::shared_ptr<ikvs_wrapper> ccl_comm::get_kvs_wrapper(std::shared_ptr<ccl::kvs_interface> kvs) {
    ccl::shared_ptr_class<ikvs_wrapper> kvs_tmp;
    if (std::dynamic_pointer_cast<ccl::v1::kvs>(kvs) != nullptr) {
        kvs_tmp = ccl::get_kvs_impl_typed<ccl::native_kvs_impl>(
                      std::dynamic_pointer_cast<ccl::v1::kvs>(kvs))
                      ->get();
    }
    else {
        kvs_tmp = std::shared_ptr<ikvs_wrapper>(new users_kvs(kvs));
    }

    return kvs_tmp;
}

ccl_comm* ccl_comm::create(device_t device,
                           context_t context,
                           int size,
                           int rank,
                           ccl::shared_ptr_class<ccl::kvs_interface> kvs) {
    return new ccl_comm(
        device, context, atl_comm_manager::create(size, { rank }, get_kvs_wrapper(kvs)));
}

ccl_comm* ccl_comm::create(int size, int rank, ccl::shared_ptr_class<ccl::kvs_interface> kvs) {
    return new ccl_comm(size, rank, get_kvs_wrapper(kvs));
}

ccl_comm* ccl_comm::create(int size, ccl::shared_ptr_class<ccl::kvs_interface> kvs) {
    return new ccl_comm(size, get_kvs_wrapper(kvs));
}

void ccl_comm::create_topo_subcomms() {
    std::shared_ptr<atl_base_comm> atl_comm = get_atl_comm();
    r2r_comm = std::shared_ptr<ccl_comm>(create_subcomm(atl_comm->get_r2r_color()));
    node_comm = std::shared_ptr<ccl_comm>(create_subcomm(topo_manager.get_host_idx()));
    even_comm = std::shared_ptr<ccl_comm>(
        create_subcomm(topo_manager.get_inter_card_color(atl_comm->get_rank())));
    pair_comm = std::shared_ptr<ccl_comm>(create_subcomm(
        topo_manager.get_intra_card_color(atl_comm->get_rank()),
        topo_manager.get_inter_card_color(atl_comm->get_rank()) % topo_manager.max_ranks_per_card));
}

ccl_comm* ccl_comm::create_subcomm(int color, int key) const {
    std::shared_ptr<atl_base_comm> new_atl_comm = get_atl_comm()->comm_split(color, key);
    ccl_comm* comm = new ccl_comm(
        new_atl_comm->get_comm_id(), new_atl_comm, true /*share_resources*/, true /*subcomm*/);
    comm->set_parent_comm(const_cast<ccl_comm*>(this));
    LOG_DEBUG("new subcomm: color ", color, ", ", comm->to_string());
    return comm;
}

std::shared_ptr<ccl_comm> ccl_comm::clone_with_new_id(int comm_id) {
    return std::shared_ptr<ccl_comm>(new ccl_comm(*this, comm_id));
}

// NOTE: allocate_resources must be done on ccl_comm level
// if it's called on ccl_internal_comm level
// the ccl_comm object that we need won't be fully constructed
void ccl_comm::allocate_resources() {
    if (ccl::global_data::env().enable_unordered_coll) {
        comm_impl->unordered_coll_manager.reset(new ccl_unordered_coll_manager(*this));
    }
    ccl::global_data::env().print(rank());
}

ccl::comm_interface_ptr ccl_comm::split(const ccl::comm_split_attr& attr) {
    if (!attr.is_valid<ccl::comm_split_attr_id::color>()) {
        CCL_THROW(std::string(__FUNCTION__) +
                  " - 'color' split attribute for communicator is not set");
    }

    auto new_comm = create_subcomm(attr.get<ccl::comm_split_attr_id::color>());

    return std::shared_ptr<ccl_comm>(new_comm);
}

std::string ccl_comm::to_string() const {
    std::stringstream ss;
    ss << "{ rank: " << rank() << ", size: " << size() << ", id: " << id() << " }";
    return ss.str();
}

std::string ccl_comm::to_string_ext() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "   " << to_string() << "\n";
    ss << "   r2r_comm: " << (r2r_comm ? r2r_comm->to_string() : "{}") << "\n";
    ss << "   node_comm: " << (node_comm ? node_comm->to_string() : "{}") << "\n";
    ss << "   even_comm: " << (even_comm ? even_comm->to_string() : "{}") << "\n";
    ss << "   pair_comm: " << (pair_comm ? pair_comm->to_string() : "{}") << "\n";
    ss << "   env: " << (env ? env->to_string() : "{}") << "\n";
    ss << "}";

    return ss.str();
}

int ccl_comm::get_global_rank(int rank) const {
    if (local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return rank;
    }

    CCL_THROW_IF_NOT((int)local2global_map.size() > rank,
                     "no rank ",
                     rank,
                     " was found in comm ",
                     this,
                     ", id ",
                     id());
    int global_rank = local2global_map[rank];
    LOG_DEBUG("comm ", this, ", id ", id(), ", map rank ", rank, " to global ", global_rank);
    return global_rank;
}

int ccl_comm::get_rank_from_global(int global_rank) const {
    if (local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return global_rank;
    }

    int rank = ccl_comm::invalid_rank;

    for (size_t i = 0; i < local2global_map.size(); ++i) {
        if (local2global_map[i] == global_rank) {
            rank = static_cast<int>(i);
            break;
        }
    }

    CCL_THROW_IF_NOT(rank != ccl_comm::invalid_rank, "can not find rank");

    return rank;
}

ccl_sched_id_t ccl_comm::get_sched_id(bool use_internal_space) {
    ccl_sched_id_t& next_sched_id =
        (use_internal_space) ? next_sched_id_internal : next_sched_id_external;

    ccl_sched_id_t first_sched_id =
        (use_internal_space) ? static_cast<ccl_sched_id_t>(0) : ccl_comm::max_sched_count / 2;

    ccl_sched_id_t max_sched_id =
        (use_internal_space) ? ccl_comm::max_sched_count / 2 : ccl_comm::max_sched_count;

    ccl_sched_id_t id = next_sched_id;

    ++next_sched_id;

    if (next_sched_id == max_sched_id) {
        /* wrap the sched numbers around to the start */
        next_sched_id = first_sched_id;
    }

    LOG_DEBUG("sched_id ", id, ", comm_id ", this->id(), ", next sched_id ", next_sched_id);

    return id;
}

/* barrier */
ccl::event ccl_comm::barrier(const ccl::stream::impl_value_t& stream,
                             const ccl::barrier_attr& attr,
                             const ccl::vector_class<ccl::event>& deps) {
    return barrier_impl(stream, attr, deps);
}

ccl::event ccl_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                  const ccl::barrier_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    ccl_barrier_impl(this, stream.get(), deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(nullptr));
}

/* allgatherv */
ccl::event ccl_comm::allgatherv_impl(const void* send_buf,
                                     size_t send_count,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allgatherv_impl(send_buf,
                                           send_count,
                                           recv_buf,
                                           recv_counts.data(),
                                           dtype,
                                           attr,
                                           this,
                                           get_stream_ptr(stream),
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event ccl_comm::allgatherv_impl(const void* send_buf,
                                     size_t send_count,
                                     const ccl::vector_class<void*>& recv_bufs,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           (void*)(recv_bufs.data()),
                                           recv_counts.data(),
                                           dtype,
                                           internal_attr,
                                           this,
                                           get_stream_ptr(stream),
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* allreduce */
ccl::event ccl_comm::allreduce_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    ccl::reduction reduction,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allreduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allreduce_impl(
        send_buf, recv_buf, count, dtype, reduction, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* alltoall */
ccl::event ccl_comm::alltoall_impl(const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoall_impl(
        send_buf, recv_buf, count, dtype, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event ccl_comm::alltoall_impl(const ccl::vector_class<void*>& send_buf,
                                   const ccl::vector_class<void*>& recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    ccl_request* req = ccl_alltoall_impl((void*)(send_buf.data()),
                                         (void*)(recv_buf.data()),
                                         count,
                                         dtype,
                                         internal_attr,
                                         this,
                                         get_stream_ptr(stream),
                                         deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* alltoallv */
ccl::event ccl_comm::alltoallv_impl(const void* send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    void* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_alltoallv_impl(send_buf,
                                          send_counts.data(),
                                          recv_buf,
                                          recv_counts.data(),
                                          dtype,
                                          attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event ccl_comm::alltoallv_impl(const ccl::vector_class<void*>& send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    ccl::vector_class<void*> recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    ccl_request* req = ccl_alltoallv_impl((void*)send_buf.data(),
                                          send_counts.data(),
                                          (void*)recv_buf.data(),
                                          recv_counts.data(),
                                          dtype,
                                          internal_attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* bcast */
ccl::event ccl_comm::broadcast_impl(void* buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req =
        ccl_broadcast_impl(buf, count, dtype, root, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce */
ccl::event ccl_comm::reduce_impl(const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 ccl::reduction reduction,
                                 int root,
                                 const ccl::stream::impl_value_t& stream,
                                 const ccl::reduce_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_reduce_impl(send_buf,
                                       recv_buf,
                                       count,
                                       dtype,
                                       reduction,
                                       root,
                                       attr,
                                       this,
                                       get_stream_ptr(stream),
                                       deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce_scatter */
ccl::event ccl_comm::reduce_scatter_impl(const void* send_buf,
                                         void* recv_buf,
                                         size_t recv_count,
                                         ccl::datatype dtype,
                                         ccl::reduction reduction,
                                         const ccl::stream::impl_value_t& stream,
                                         const ccl::reduce_scatter_attr& attr,
                                         const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_reduce_scatter_impl(
        send_buf, recv_buf, recv_count, dtype, reduction, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#endif // CCL_ENABLE_SYCL
