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
#include "common/global/global.hpp"
#include "common/comm/host_communicator/host_communicator_impl.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"
#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"
#include "coll/ccl_allgather_op_attr.hpp"

#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "atl/atl_wrapper.h"

#include "common/comm/comm.hpp"

#ifdef MULTI_GPU_SUPPORT
#include "common/comm/l0/gpu_comm_attr.hpp"
#endif

namespace ccl {

using ccl::preview::create_comm_split_attr;

host_communicator::host_communicator() : comm_attr(create_comm_split_attr()) {}

host_communicator::host_communicator(int size, shared_ptr_class<ikvs_wrapper> kvs)
        : comm_attr(create_comm_split_attr()),
          comm_rank(0),
          comm_size(size) {
    if (size <= 0) {
        throw ccl::exception("Incorrect size value when creating a host communicator");
    }
}

host_communicator::host_communicator(int size, int rank, shared_ptr_class<ikvs_wrapper> kvs)
        : comm_attr(create_comm_split_attr()),
          comm_rank(rank),
          comm_size(size) {
    if (rank > size || size <= 0) {
        throw ccl::exception("Incorrect rank or size value when creating a host communicator");
    }

    LOG_DEBUG("ctor");

    ccl::global_data& data = ccl::global_data::get();
    std::shared_ptr<atl_wrapper> atl_tmp =
        std::shared_ptr<atl_wrapper>(new atl_wrapper(size, { rank }, kvs));
    comm_impl =
        std::shared_ptr<ccl_comm>(new ccl_comm(rank, size, data.comm_ids->acquire(), atl_tmp));
}

host_communicator::host_communicator(std::shared_ptr<atl_wrapper> atl)
        : comm_attr(create_comm_split_attr()),
          comm_rank(atl->get_rank()),
          comm_size(atl->get_size()) {
    int rank = atl->get_rank();
    int size = atl->get_size();

    if (rank > size || size <= 0) {
        throw ccl::exception("Incorrect rank or size value when creating \
                             a host communicator: rank" +
                             std::to_string(rank) + " size: " + std::to_string(size));
    }

    LOG_DEBUG("ctor");

    ccl::global_data& data = ccl::global_data::get();
    comm_impl = std::shared_ptr<ccl_comm>(new ccl_comm(rank, size, data.comm_ids->acquire(), atl));
}

host_communicator::host_communicator(std::shared_ptr<ccl_comm> impl)
        : comm_impl(impl),
          comm_attr(create_comm_split_attr()),
          comm_rank(impl->rank()),
          comm_size(impl->size()) {}

int host_communicator::rank() const {
    return comm_rank;
}

int host_communicator::size() const {
    return comm_size;
}

#ifdef MULTI_GPU_SUPPORT
void host_communicator::visit(ccl::gpu_comm_attr& comm_attr) {
    (void)(comm_attr);
}
#endif

ccl::device_index_type host_communicator::get_device_path() const {
    return ccl::device_index_type{ ccl::unused_index_value,
                                   ccl::unused_index_value,
                                   ccl::unused_index_value };
}

ccl::communicator_interface::device_t host_communicator::get_device() const {
    throw ccl::exception(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
    static ccl::communicator_interface::device_t empty;
    return empty;
}

ccl::communicator_interface::context_t host_communicator::get_context() const {
    throw ccl::exception(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
    static ccl::communicator_interface::context_t empty;
    return empty;
}

void host_communicator::exchange_colors(std::vector<int>& colors) {
    size_t send_count = 1;
    vector_class<size_t> recv_counts(colors.size(), send_count);
    auto attr =
        create_operation_attr<allgatherv_attr>(attr_val<operation_attr_id::to_cache>(false));

    this->allgatherv_impl(colors.data(), send_count, colors.data(), recv_counts, {}, attr, {})
        .wait();
}

ccl_comm* host_communicator::create_with_color(int color,
                                               ccl_comm_id_storage* comm_ids,
                                               const ccl_comm* parent_comm) {
    if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        throw ccl::exception(
            "MPI transport doesn't support creation of communicator with color yet");
    }

    std::vector<int> colors(this->size());
    colors[this->rank()] = color;
    this->exchange_colors(colors);

    // TODO we can replace this func with own
    return ccl_comm::create_with_colors(colors, comm_ids, parent_comm, true);
}

ccl::communicator_interface_ptr host_communicator::split(const comm_split_attr& attr) {
    if (!attr.is_valid<comm_split_attr_id::color>()) {
        throw ccl::exception(std::string(__FUNCTION__) +
                             " - 'Color' split attribute for host communicator is not set");
    }

    ccl::global_data& data = ccl::global_data::get();
    auto new_comm = this->create_with_color(
        attr.get<ccl::comm_split_attr_id::color>(), data.comm_ids.get(), comm_impl.get());

    comm_attr = attr;

    return std::shared_ptr<host_communicator>(
        new host_communicator(std::shared_ptr<ccl_comm>(new_comm)));
}

ccl::event host_communicator::barrier(const ccl::stream::impl_value_t& op_stream,
                                      const ccl::barrier_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    return get_impl()->barrier_impl(op_stream, attr, deps);
}

ccl::event host_communicator::barrier_impl(const ccl::stream::impl_value_t& op_stream,
                                           const ccl::barrier_attr& attr,
                                           const ccl::vector_class<ccl::event>& deps) {
    // TODO what exactly we need to do with 'attr' here?

    ccl_barrier_impl(comm_impl.get(), op_stream.get(), deps);

    // TODO what exactly we need to return here? ccl_barrier_impl() is void func
    ccl_request* req = nullptr;
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* allgatherv */
ccl::event host_communicator::allgatherv_impl(const void* send_buf,
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
                                           comm_impl.get(),
                                           nullptr,
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event host_communicator::allgatherv_impl(const void* send_buf,
                                              size_t send_count,
                                              const ccl::vector_class<void*>& recv_bufs,
                                              const ccl::vector_class<size_t>& recv_counts,
                                              ccl::datatype dtype,
                                              const ccl::stream::impl_value_t& stream,
                                              const ccl::allgatherv_attr& attr,
                                              const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.vector_buf = 1;

    ccl_request* req = ccl_allgatherv_impl(reinterpret_cast<const void*>(send_buf),
                                           send_count,
                                           (void*)(recv_bufs.data()),
                                           recv_counts.data(),
                                           dtype,
                                           internal_attr,
                                           comm_impl.get(),
                                           nullptr,
                                           deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* allreduce */
ccl::event host_communicator::allreduce_impl(const void* send_buf,
                                             void* recv_buf,
                                             size_t count,
                                             ccl::datatype dtype,
                                             ccl::reduction reduction,
                                             const ccl::stream::impl_value_t& stream,
                                             const ccl::allreduce_attr& attr,
                                             const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allreduce_impl(
        send_buf, recv_buf, count, dtype, reduction, attr, comm_impl.get(), nullptr, deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* alltoall */
ccl::event host_communicator::alltoall_impl(const void* send_buf,
                                            void* recv_buf,
                                            size_t count,
                                            ccl::datatype dtype,
                                            const ccl::stream::impl_value_t& stream,
                                            const ccl::alltoall_attr& attr,
                                            const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req =
        ccl_alltoall_impl(send_buf, recv_buf, count, dtype, attr, comm_impl.get(), nullptr, deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event host_communicator::alltoall_impl(const ccl::vector_class<void*>& send_buf,
                                            const ccl::vector_class<void*>& recv_buf,
                                            size_t count,
                                            ccl::datatype dtype,
                                            const ccl::stream::impl_value_t& stream,
                                            const ccl::alltoall_attr& attr,
                                            const ccl::vector_class<ccl::event>& deps) {
    // TODO not implemented
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
ccl::event host_communicator::alltoallv_impl(const void* send_buf,
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
                                          comm_impl.get(),
                                          nullptr,
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event host_communicator::alltoallv_impl(const ccl::vector_class<void*>& send_buf,
                                             const ccl::vector_class<size_t>& send_counts,
                                             ccl::vector_class<void*> recv_buf,
                                             const ccl::vector_class<size_t>& recv_counts,
                                             ccl::datatype dtype,
                                             const ccl::stream::impl_value_t& stream,
                                             const ccl::alltoallv_attr& attr,
                                             const ccl::vector_class<ccl::event>& dep) {
    // TODO not implemented
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* bcast */
ccl::event host_communicator::broadcast_impl(void* buf,
                                             size_t count,
                                             ccl::datatype dtype,
                                             int root,
                                             const ccl::stream::impl_value_t& stream,
                                             const ccl::broadcast_attr& attr,
                                             const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req =
        ccl_broadcast_impl(buf, count, dtype, root, attr, comm_impl.get(), nullptr, deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce */
ccl::event host_communicator::reduce_impl(const void* send_buf,
                                          void* recv_buf,
                                          size_t count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          int root,
                                          const ccl::stream::impl_value_t& stream,
                                          const ccl::reduce_attr& attr,
                                          const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_reduce_impl(
        send_buf, recv_buf, count, dtype, reduction, root, attr, comm_impl.get(), nullptr, deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* reduce_scatter */
ccl::event host_communicator::reduce_scatter_impl(const void* send_buf,
                                                  void* recv_buf,
                                                  size_t recv_count,
                                                  ccl::datatype dtype,
                                                  ccl::reduction reduction,
                                                  const ccl::stream::impl_value_t& stream,
                                                  const ccl::reduce_scatter_attr& attr,
                                                  const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_reduce_scatter_impl(
        send_buf, recv_buf, recv_count, dtype, reduction, attr, comm_impl.get(), nullptr, deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* sparse_allreduce */
ccl::event host_communicator::sparse_allreduce_impl(const void* send_ind_buf,
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
                                                 comm_impl.get(),
                                                 nullptr,
                                                 deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

std::shared_ptr<atl_wrapper> host_communicator::get_atl() {
    return comm_impl->atl;
}

std::string host_communicator::to_string() const {
    return std::string("host communicator, rank (") + std::to_string(rank()) + "/" +
           std::to_string(size());
}

COMM_INTERFACE_COLL_INSTANTIATION(host_communicator);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(host_communicator);
#endif /* CCL_ENABLE_SYCL */

} // namespace ccl
