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
#include "coll/attr/ccl_common_op_attrs.hpp"
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
        : m_dtree(size, rank)
#ifdef CCL_ENABLE_SYCL
          ,
          m_barrier_data(rank, size)
#endif // CCL_ENABLE_SYCL
{
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
                    const std::shared_ptr<atl_base_comm>& atl_comm,
                    bool share_resources,
                    bool is_sub_communicator) {
    comm_rank = atl_comm->get_rank();
    comm_size = atl_comm->get_size();

    next_sched_id_internal = atl_comm->tag_creator->get_max_sched_count() / 2;
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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        // init of fd manager is based on node comm,
        // it initializes for every creation of comm in multi comms case
        init_ipc_exchange_mode(node_comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
    init(comm_id, std::move(atl_comm), share_resources, is_sub_communicator);
}

ccl_comm::ccl_comm(std::shared_ptr<atl_base_comm> atl_comm,
                   bool share_resources,
                   bool is_sub_communicator)
        : ccl_comm(atl_comm->create_comm_id(), atl_comm, share_resources, is_sub_communicator) {}

ccl_comm::ccl_comm(device_t device, context_t context, std::shared_ptr<atl_base_comm> atl_comm)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)) {
    int id = atl_comm->create_comm_id();
    init(id, std::move(atl_comm));
}

ccl_comm::ccl_comm(int size, int rank, ccl::shared_ptr_class<ikvs_wrapper> kvs)
        : ccl_comm(atl_comm_manager::create(size, { rank }, std::move(kvs))) {}

ccl_comm::ccl_comm(int size, ccl::shared_ptr_class<ikvs_wrapper> kvs)
        : ccl_comm(atl_comm_manager::create(size, { 0 }, std::move(kvs))) {}

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
                      std::dynamic_pointer_cast<ccl::v1::kvs>(std::move(kvs)))
                      ->get();
    }
    else {
        kvs_tmp = std::shared_ptr<ikvs_wrapper>(new users_kvs(std::move(kvs)));
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

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
void ccl_comm::init_ipc_exchange_mode(std::shared_ptr<ccl_comm> comm) {
    if (device_ptr && context_ptr) {
        LOG_DEBUG("initialize ipc_exchange_mode");
        if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::pidfd &&
            ccl::ze::fd_manager::is_pidfd_supported()) {
            LOG_DEBUG("pidfd exchange mode is verified successfully");
        }
#ifdef CCL_ENABLE_DRM
        else if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::drmfd) {
            fd_manager = std::make_shared<ccl::ze::fd_manager>(comm->get_atl_comm());
            // update physical_idx for each logical device, by default it is invalid
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
            auto& devices = ccl::global_data::get().ze_data->devices;
            for (size_t idx = 0; idx < devices.size(); idx++) {
                devices[idx].physical_idx = ccl::ze::fd_manager::get_physical_device_idx(
                    fd_manager->get_physical_devices(), devices[idx].pci);
            }
#endif // ZE_PCI_PROPERTIES_EXT_NAME
            LOG_DEBUG("drmfd exchange mode is verified successfully");
        }
#endif // CCL_ENABLE_DRM
    }
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

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

    // TODO: Add reverse map to speed this up
    for (size_t i = 0; i < local2global_map.size(); ++i) {
        if (local2global_map[i] == global_rank) {
            rank = static_cast<int>(i);
            break;
        }
    }

    CCL_THROW_IF_NOT(rank != ccl_comm::invalid_rank, "can not find rank");

    return rank;
}

bool ccl_comm::try_get_rank_from_global(int global_rank) const {
    bool ret = false;
    if (local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return ret;
    }

    for (size_t i = 0; i < local2global_map.size(); ++i) {
        if (local2global_map[i] == global_rank) {
            return true;
        }
    }

    return ret;
}

int ccl_comm::get_node_rank(int rank) const {
    if (this == get_node_comm().get()) {
        CCL_THROW("untested get_node_rank() on node_comm");
        // This is the node_comm, mapping is direct
        return rank;
    }

    // First, get global_rank from rank
    int global_rank = get_global_rank(rank);

    // Then, map global_rank to node_comm's rank
    return get_node_comm()->get_rank_from_global(global_rank);
}

ccl_sched_id_t ccl_comm::get_sched_id(bool use_internal_space, bool is_pt2pt) {
    std::shared_ptr<atl_base_comm> atl_comm = get_atl_comm();
    ccl_sched_id_t& next_sched_id =
        (use_internal_space) ? next_sched_id_internal : next_sched_id_external;

    ccl_sched_id_t max_sched_count = atl_comm->tag_creator->get_max_sched_count();

    ccl_sched_id_t first_sched_id =
        (use_internal_space) ? static_cast<ccl_sched_id_t>(0) : max_sched_count / 2;

    ccl_sched_id_t max_sched_id = (use_internal_space) ? max_sched_count / 2 : max_sched_count;

    ccl_sched_id_t id = next_sched_id;

    // is_pt2pt flag is required in the case
    // to avoid when send-recv communication between ranks
    // less comm_size, the ++next_sched_id op is skipped if
    // is_pt2pt = true
    if (!is_pt2pt) {
        ++next_sched_id;
    }

    if (next_sched_id == max_sched_id) {
        /* wrap the sched numbers around to the start */
        next_sched_id = first_sched_id;
    }

    LOG_DEBUG("sched_id ", id, ", comm_id ", this->id(), ", next sched_id ", next_sched_id);

    return id;
}

#ifdef CCL_ENABLE_SYCL
void* ccl_scaleout_host_bufs::get_scaleout_host_buf() {
    if (!host_bufs[index]) {
        CCL_THROW_IF_NOT(get_scaleout_host_buf_size() > 0,
                         "CCL_SCALEOUT_HOST_BUF_SIZE must be greater than zero");

        switch (ccl::global_data::env().sycl_scaleout_buf_alloc_mode) {
            case ccl::utils::alloc_mode::hwloc: {
                int numa_node_os_idx = 0; // TODO: determine which hbm_node_os_idx to set;
                host_bufs[index] = ccl::global_data::get().hwloc_wrapper->alloc_memory(
                    CCL_REG_MSG_ALIGNMENT, buf_size, numa_node_os_idx);
            } break;
            case ccl::utils::alloc_mode::malloc: host_bufs[index] = malloc(buf_size); break;
            case ccl::utils::alloc_mode::memalign:
                // internally, CCL_MALLOC calls posix_memalign
                host_bufs[index] = CCL_MALLOC(buf_size, "scaleout_host_buf");
                break;
            default: CCL_THROW("unexpected alloc_mode");
        }
        CCL_THROW_IF_NOT(host_bufs[index] != nullptr, "Cannot allocate host buffer");

        if (ccl::global_data::get().ze_data->external_pointer_registration_enabled) {
            ccl::global_data::get().ze_data->import_external_pointer(host_bufs[index], buf_size);
        }
    }

    auto old_index = index;
    index = (index + 1) % 3;
    return host_bufs[old_index];
}

size_t ccl_scaleout_host_bufs::get_scaleout_host_buf_size() {
    if (buf_size == 0) {
        buf_size = ccl::global_data::env().sycl_scaleout_host_buf_size;
    }
    return buf_size;
}

ccl_scaleout_host_bufs::~ccl_scaleout_host_bufs() {
    for (int i = 0; i < buf_count; ++i) {
        if (host_bufs[i] != nullptr) {
            if (ccl::global_data::get().ze_data->external_pointer_registration_enabled) {
                ccl::global_data::get().ze_data->release_imported_pointer(host_bufs[i]);
            }

            switch (ccl::global_data::env().sycl_scaleout_buf_alloc_mode) {
                case ccl::utils::alloc_mode::hwloc:
                    ccl::global_data::get().hwloc_wrapper->dealloc_memory(host_bufs[i]);
                    break;
                case ccl::utils::alloc_mode::malloc: free(host_bufs[i]); break;
                case ccl::utils::alloc_mode::memalign: CCL_FREE(host_bufs[i]); break;
                default:
                    // destructors cannot throw exceptions
                    LOG_ERROR("unexpected alloc_mode");
            }
        }
    }
}
#endif // CCL_ENABLE_SYCL

/* barrier */
ccl::event ccl_comm::barrier(const ccl::stream::impl_value_t& stream,
                             const ccl::barrier_attr& attr,
                             const ccl::vector_class<ccl::event>& deps) {
    return barrier_impl(stream, attr, deps);
}

ccl::event ccl_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                  const ccl::barrier_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_barrier_impl(this, stream.get(), deps);
    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* allgather */
ccl::event ccl_comm::allgather_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_allgather_impl(
        send_buf, recv_buf, count, dtype, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

ccl::event ccl_comm::allgather_impl(const void* send_buf,
                                    const ccl::vector_class<void*>& recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    ccl_request* req = ccl_allgather_impl(reinterpret_cast<const void*>(send_buf),
                                          (void*)(recv_buf.data()),
                                          count,
                                          dtype,
                                          internal_attr,
                                          this,
                                          get_stream_ptr(stream),
                                          deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
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

/* bcastExt */
ccl::event ccl_comm::broadcastExt_impl(void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       int root,
                                       const ccl::stream::impl_value_t& stream,
                                       const ccl::broadcastExt_attr& attr,
                                       const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req = ccl_broadcastExt_impl(
        send_buf, recv_buf, count, dtype, root, attr, this, get_stream_ptr(stream), deps);

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

/* recv */
ccl::event ccl_comm::recv_impl(void* recv_buf,
                               size_t recv_count,
                               ccl::datatype dtype,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req =
        ccl_recv_impl(recv_buf, recv_count, dtype, peer, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

/* send */
ccl::event ccl_comm::send_impl(void* send_buf,
                               size_t send_count,
                               ccl::datatype dtype,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    ccl_request* req =
        ccl_send_impl(send_buf, send_count, dtype, peer, attr, this, get_stream_ptr(stream), deps);

    return std::unique_ptr<ccl::event_impl>(new ccl::host_event_impl(req));
}

COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#ifdef CCL_ENABLE_SYCL
SYCL_COMM_INTERFACE_COLL_INSTANTIATION(ccl_comm);
#endif // CCL_ENABLE_SYCL
