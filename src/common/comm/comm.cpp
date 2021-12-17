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
#include "common/comm/comm.hpp"
#include "common/comm/comm_impl.hpp"
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

ccl_comm_internal::ccl_comm_internal(int rank, int size, std::shared_ptr<atl_base_comm> atl)
        : ccl_comm_internal(rank, size, atl->get_rank2rank_map(), atl) {}

ccl_comm_internal::ccl_comm_internal(int rank,
                                     int size,
                                     ccl_rank2rank_map&& rank_map,
                                     std::shared_ptr<atl_base_comm> atl)
        : atl(atl),
          m_local2global_map(std::move(rank_map)),
          m_dtree(size, rank) {
    reset(rank, size);
}

ccl_comm_internal::ccl_comm_internal(const std::vector<int>& local_ranks,
                                     int comm_size,
                                     std::shared_ptr<ccl::kvs_interface> kvs_instance)
        : m_local2global_map(),
          m_dtree(local_ranks.size(), comm_size) {
    std::shared_ptr<ikvs_wrapper> kvs_wrapper(new users_kvs(kvs_instance));

    atl = atl_comm_manager::create_comm(comm_size, local_ranks, kvs_wrapper);

    reset(atl->get_rank(), atl->get_size());
}

//TODO: will fix it after OFI refactoring
int ccl_comm::get_global_rank(int rank, bool only_global) const {
    // TODO: move map to ccl_comm?
    const auto& local2global_map = comm_impl->get_local2global_map();

    if (local2global_map.empty() || !only_global) {
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
    const auto& local2global_map = comm_impl->get_local2global_map();

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

    CCL_THROW_IF_NOT(rank != ccl_comm::invalid_rank, "can't find rank");

    return rank;
}

using ccl::preview::create_comm_split_attr;

ccl_comm::ccl_comm()
        : device(ccl::device_index_type(ccl::unused_index_value,
                                        ccl::unused_index_value,
                                        ccl::unused_index_value)),
          comm_attr(create_comm_split_attr()) {}

ccl_comm::ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs)
        : device(ccl::device_index_type(ccl::unused_index_value,
                                        ccl::unused_index_value,
                                        ccl::unused_index_value)),
          comm_attr(create_comm_split_attr()),
          comm_rank(0),
          comm_size(size),
          next_sched_id_internal(ccl_comm_internal::max_sched_count / 2),
          next_sched_id_external(0) {
    if (size <= 0) {
        throw ccl::exception("Incorrect size value when creating a host communicator");
    }
}

ccl_comm::ccl_comm(int size, int rank, ccl::shared_ptr_class<ikvs_wrapper> kvs)
        : ccl_comm(atl_comm_manager::create_comm(size, { rank }, kvs)) {}

ccl_comm::ccl_comm(ccl::unified_device_type&& d,
                   ccl::unified_context_type&& c,
                   std::shared_ptr<atl_base_comm> atl)
        : device(std::move(d)),
          context(std::move(c)),
          comm_attr(create_comm_split_attr()),
          comm_rank(atl->get_rank()),
          comm_size(atl->get_size()),
          comm_id(std::unique_ptr<ccl_comm_id_storage::comm_id>(
              new ccl_comm_id_storage::comm_id(ccl::global_data::get().comm_ids->acquire()))),
          next_sched_id_internal(ccl_comm_internal::max_sched_count / 2),
          next_sched_id_external(0) {
    int rank = atl->get_rank();
    int size = atl->get_size();

    if (rank > size || size <= 0) {
        throw ccl::exception("incorrect rank or size when creating \
                             a host communicator: rank: " +
                             std::to_string(rank) + ", size: " + std::to_string(size));
    }

    LOG_DEBUG("ctor");

    comm_impl = std::unique_ptr<ccl_comm_internal>(new ccl_comm_internal(rank, size, atl));

    allocate_resources();
    create_sub_comms(atl);
}

ccl_comm::ccl_comm(std::shared_ptr<atl_base_comm> atl)
        : ccl_comm(ccl::device_index_type(ccl::unused_index_value,
                                          ccl::unused_index_value,
                                          ccl::unused_index_value),
                   {},
                   atl) {}

ccl_comm::ccl_comm(int rank,
                   int size,
                   ccl_comm_id_storage::comm_id&& id,
                   std::shared_ptr<atl_base_comm> atl,
                   bool share_resources,
                   bool is_sub_communicator)
        : comm_impl(std::make_shared<ccl_comm_internal>(rank, size, atl->get_rank2rank_map(), atl)),
          device(ccl::device_index_type(ccl::unused_index_value,
                                        ccl::unused_index_value,
                                        ccl::unused_index_value)),
          comm_attr(create_comm_split_attr()),
          comm_rank(rank),
          comm_size(size),
          comm_id(std::unique_ptr<ccl_comm_id_storage::comm_id>(
              new ccl_comm_id_storage::comm_id(std::move(id)))),
          next_sched_id_internal(ccl_comm_internal::max_sched_count / 2),
          next_sched_id_external(0) {
    if (!share_resources) {
        allocate_resources();
    }

    if (!is_sub_communicator) {
        create_sub_comms(comm_impl.get()->atl);
    }
}

ccl_comm::ccl_comm(int rank,
                   int size,
                   ccl_comm_id_storage::comm_id&& id,
                   ccl_rank2rank_map&& rank_map,
                   std::shared_ptr<atl_base_comm> atl,
                   bool share_resources,
                   bool is_sub_communicator)
        : comm_impl(std::make_shared<ccl_comm_internal>(rank, size, std::move(rank_map), atl)),
          device(ccl::device_index_type(ccl::unused_index_value,
                                        ccl::unused_index_value,
                                        ccl::unused_index_value)),
          comm_attr(create_comm_split_attr()),
          comm_rank(rank),
          comm_size(size),
          comm_id(std::unique_ptr<ccl_comm_id_storage::comm_id>(
              new ccl_comm_id_storage::comm_id(std::move(id)))),
          next_sched_id_internal(ccl_comm_internal::max_sched_count / 2),
          next_sched_id_external(0) {
    if (!share_resources) {
        allocate_resources();
    }

    if (!is_sub_communicator) {
        create_sub_comms(get_atl_comm());
    }
}

ccl_comm::ccl_comm(const ccl_comm& src, ccl_comm_id_storage::comm_id&& id)
        : comm_impl(src.comm_impl),
          device(ccl::device_index_type(ccl::unused_index_value,
                                        ccl::unused_index_value,
                                        ccl::unused_index_value)),
          r2r_comm(src.r2r_comm),
          node_comm(src.node_comm),
          even_comm(src.even_comm),
          pair_comm(src.pair_comm),
          comm_attr(create_comm_split_attr()),
          comm_rank(src.rank()),
          comm_size(src.size()),
          comm_id(std::unique_ptr<ccl_comm_id_storage::comm_id>(
              new ccl_comm_id_storage::comm_id(std::move(id)))),
          next_sched_id_internal(ccl_comm_internal::max_sched_count / 2),
          next_sched_id_external(0) {}

ccl::device_index_type ccl_comm::get_device_path() const {
    return ccl::device_index_type{ ccl::unused_index_value,
                                   ccl::unused_index_value,
                                   ccl::unused_index_value };
}

ccl::communicator_interface::device_t ccl_comm::get_device() const {
    CCL_THROW(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
    static ccl::communicator_interface::device_t empty;
    return empty;
}

ccl::communicator_interface::context_t ccl_comm::get_context() const {
    CCL_THROW(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
    static ccl::communicator_interface::context_t empty;
    return empty;
}

void ccl_comm::create_sub_comms(std::shared_ptr<atl_base_comm> atl) {
    ccl::global_data& data = ccl::global_data::get();

    r2r_comm = std::shared_ptr<ccl_comm>(
        this->create_with_color(atl->get_r2r_color(), data.comm_ids.get(), true));
    node_comm = std::shared_ptr<ccl_comm>(
        this->create_with_color(atl->get_host_color(), data.comm_ids.get(), true));
    even_comm = std::shared_ptr<ccl_comm>(this->create_with_color(
        atl->get_host_color() + atl->get_rank() % 2, data.comm_ids.get(), true));
    pair_comm = std::shared_ptr<ccl_comm>(this->create_with_color(
        atl->get_host_color() + atl->get_rank() / 2, data.comm_ids.get(), true));
}

ccl_comm* ccl_comm::create_with_color(int color,
                                      ccl_comm_id_storage* comm_ids,
                                      bool share_resources) const {
    std::shared_ptr<atl_base_comm> atl_comm = get_atl_comm()->comm_split(color);
    ccl_comm* comm = new ccl_comm(atl_comm->get_rank(),
                                  atl_comm->get_size(),
                                  comm_ids->acquire(),
                                  atl_comm->get_rank2rank_map(),
                                  atl_comm,
                                  share_resources,
                                  true);

    LOG_DEBUG("new comm: color ",
              color,
              ", rank ",
              comm->rank(),
              ", size ",
              comm->size(),
              ", comm_id ",
              comm->id());

    return comm;
}

ccl::communicator_interface_ptr ccl_comm::split(const ccl::comm_split_attr& attr) {
    if (!attr.is_valid<ccl::comm_split_attr_id::color>()) {
        CCL_THROW(std::string(__FUNCTION__) +
                  " - 'Color' split attribute for host communicator is not set");
    }

    ccl::global_data& data = ccl::global_data::get();
    auto new_comm = this->create_with_color(
        attr.get<ccl::comm_split_attr_id::color>(), data.comm_ids.get(), true);

    comm_attr = attr;

    return std::shared_ptr<ccl_comm>(new_comm);
}

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
    // TODO not implemented
    CCL_THROW(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
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
                                    const ccl::vector_class<ccl::event>& dep) {
    // TODO not implemented
    CCL_THROW(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
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

/* sparse_allreduce */
ccl::event ccl_comm::sparse_allreduce_impl(const void* send_ind_buf,
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
    ccl_request* req = ccl_sparse_allreduce_impl(send_ind_buf,
                                                 send_ind_count,
                                                 send_val_buf,
                                                 send_val_count,
                                                 recv_ind_buf,
                                                 recv_ind_count,
                                                 recv_val_buf,
                                                 recv_val_count,
                                                 index_dtype,
                                                 value_dtype,
                                                 reduction,
                                                 attr,
                                                 this,
                                                 get_stream_ptr(stream),
                                                 deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

std::shared_ptr<atl_base_comm> ccl_comm::get_atl_comm() const {
    return comm_impl->atl;
}

std::shared_ptr<ccl_comm> ccl_comm::get_r2r_comm() {
    return r2r_comm;
}

std::shared_ptr<ccl_comm> ccl_comm::get_node_comm() {
    return node_comm;
}

std::shared_ptr<ccl_comm> ccl_comm::get_pair_comm() {
    return pair_comm;
}

std::shared_ptr<ccl_comm> ccl_comm::get_even_comm() {
    return even_comm;
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
    ss << "}";

    return ss.str();
}

// NOTE: allocate_resources must be done on ccl_comm level, if it's called on ccl_comm_internal level
// the ccl_comm object that we need won't be fully constructed
void ccl_comm::allocate_resources() {
    if (ccl::global_data::env().enable_unordered_coll) {
        comm_impl->unordered_coll_manager.reset(new ccl_unordered_coll_manager(*this));
    }

    auto& env_object = ccl::global_data::env();

    comm_impl->allreduce_2d_builder.reset(new ccl_allreduce_2d_builder(
        (env_object.allreduce_2d_base_size != CCL_ENV_SIZET_NOT_SPECIFIED)
            ? env_object.allreduce_2d_base_size
            : ccl::global_data::get().executor->get_local_proc_count(),
        env_object.allreduce_2d_switch_dims,
        this));

    env_object.print(rank());
}

std::shared_ptr<ccl_comm> ccl_comm::clone_with_new_id(ccl_comm_id_storage::comm_id&& id) {
    return std::shared_ptr<ccl_comm>(new ccl_comm(*this, std::move(id)));
}

COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#endif // CCL_ENABLE_SYCL
