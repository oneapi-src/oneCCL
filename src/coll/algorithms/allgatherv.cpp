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
#include "coll/algorithms/algorithms.hpp"
#include "common/comm/comm.hpp"
#include "sched/entry/coll/coll_entry_helper.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "coll/coll_util.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

#include <numeric>

ccl::status ccl_coll_build_direct_allgatherv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             size_t send_count,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm) {
    LOG_DEBUG("build direct allgatherv");

    entry_factory::create<allgatherv_entry>(
        sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_build_naive_allgatherv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            size_t send_count,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm) {
    LOG_DEBUG("build naive allgatherv");

    ccl::status status = ccl::status::success;

    int comm_size = comm->size();
    int comm_rank = comm->rank();
    size_t dtype_size = dtype.size();
    std::vector<size_t> offsets(comm_size);

    offsets[0] = 0;
    for (int rank = 1; rank < comm_size; rank++) {
        offsets[rank] = offsets[rank - 1] + recv_counts[rank - 1] * dtype_size;
    }

    if (send_buf != recv_buf) {
        // out-of-place case
        entry_factory::create<copy_entry>(
            sched, send_buf, recv_buf + offsets[comm_rank], send_count, dtype);
    }

    for (int idx = 1; idx < comm_size; idx++) {
        int dst = (comm_rank + idx) % comm_size;
        int src = (comm_rank - idx + comm_size) % comm_size;

        // send own buffer to other ranks
        entry_factory::create<send_entry>(
            sched, recv_buf + offsets[comm_rank], send_count, dtype, dst, comm);

        // recv other's rank buffer
        entry_factory::create<recv_entry>(
            sched, recv_buf + offsets[src], recv_counts[src], dtype, src, comm);
    }

    return status;
}

ccl::status ccl_coll_build_ring_allgatherv(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           size_t send_count,
                                           ccl_buffer recv_buf,
                                           const size_t* recv_counts,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm) {
    LOG_DEBUG("build ring allgatherv, send_count ", send_count);

    ccl::status status = ccl::status::success;
    int comm_size, rank;
    size_t dtype_size = dtype.size();
    int src, dst;

    comm_size = comm->size();
    rank = comm->rank();

    size_t* offsets = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    offsets[0] = 0;
    for (int rank_idx = 1; rank_idx < comm_size; ++rank_idx) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * dtype_size;
    }

    if (send_buf != recv_buf) {
        entry_factory::create<copy_entry>(
            sched, send_buf, recv_buf + offsets[rank], send_count, dtype);
    }

    ccl_buffer sbuf, rbuf;

    src = (comm_size + rank - 1) % comm_size;
    dst = (comm_size + rank + 1) % comm_size;

    size_t block_idx =
        rank; // start send with 'rank' block and recv with 'rank-1' block and move blocks left
    size_t send_block_idx, recv_block_idx;
    size_t send_block_count, recv_block_count;
    size_t send_block_offset, recv_block_offset;

    for (int idx = 0; idx < (comm_size - 1); idx++) {
        send_block_idx = block_idx;
        recv_block_idx = (comm_size + block_idx - 1) % comm_size;
        send_block_count = recv_counts[send_block_idx];
        recv_block_count = recv_counts[recv_block_idx];
        send_block_offset = offsets[send_block_idx];
        recv_block_offset = offsets[recv_block_idx];
        sbuf = recv_buf + send_block_offset;
        rbuf = recv_buf + recv_block_offset;

        entry_factory::create<send_entry>(sched, sbuf, send_block_count, dtype, dst, comm);
        entry_factory::create<recv_entry>(sched, rbuf, recv_block_count, dtype, src, comm);
        sched->add_barrier();

        block_idx = (comm_size + block_idx - 1) % comm_size; // move left
    }

    CCL_FREE(offsets);
    return status;
}

ccl::status ccl_coll_get_allgatherv_bufs_and_offsets(const ccl_coll_param& coll_param,
                                                     std::vector<ccl_buffer>& recv_bufs,
                                                     std::vector<size_t>& recv_offsets) {
    int comm_size = coll_param.comm->size();
    size_t dtype_size = coll_param.dtype.size();

    recv_bufs.resize(comm_size);
    recv_offsets.resize(comm_size);

    if (coll_param.recv_bufs.size() > 1) {
        CCL_THROW_IF_NOT((int)coll_param.recv_bufs.size() == comm_size,
                         "unexpected recv_bufs.size ",
                         coll_param.recv_bufs.size(),
                         ", expected ",
                         comm_size);

        for (int idx = 0; idx < comm_size; idx++) {
            recv_bufs[idx].set(coll_param.get_recv_buf_ptr(idx),
                               coll_param.get_recv_count(idx) * dtype_size,
                               ccl_buffer_type::INDIRECT);
            recv_offsets[idx] = 0;
        }
    }
    else {
        size_t offset = 0;
        size_t dtype_size = coll_param.dtype.size();
        for (int idx = 0; idx < comm_size; idx++) {
            size_t bytes = coll_param.get_recv_count(idx) * dtype_size;
            recv_bufs[idx].set(
                coll_param.get_recv_buf_ptr(), offset + bytes, offset, ccl_buffer_type::INDIRECT);
            recv_offsets[idx] = offset;
            offset += bytes;
        }
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_flat_allgatherv(ccl_master_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param) {
    LOG_DEBUG("build flat allgatherv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    bool inplace = coll_param.is_inplace();

    std::vector<ccl_buffer> recv_bufs;
    std::vector<size_t> recv_offsets;
    ccl_coll_get_allgatherv_bufs_and_offsets(coll_param, recv_bufs, recv_offsets);

    auto send_seg = ccl_buffer(coll_param.get_send_buf_ptr(),
                               coll_param.get_send_count() * dtype_size,
                               ccl_buffer_type::INDIRECT);

    if (!inplace) {
        entry_factory::create<copy_entry>(scheds[2 * comm_rank % sched_count],
                                          ccl_buffer(coll_param.get_send_buf_ptr(),
                                                     coll_param.get_send_count() * dtype_size,
                                                     ccl_buffer_type::INDIRECT),
                                          recv_bufs[comm_rank],
                                          coll_param.get_recv_count(comm_rank),
                                          dtype);
    }
    else {
        size_t total_recv_bytes =
            std::accumulate(coll_param.recv_counts.begin(), coll_param.recv_counts.end(), 0) *
            dtype_size;
        send_seg = ccl_buffer(coll_param.get_send_buf_ptr(),
                              total_recv_bytes,
                              recv_offsets[comm_rank],
                              ccl_buffer_type::INDIRECT);
    }

    CCL_THROW_IF_NOT(static_cast<int>(sched_count) == comm_size,
                     "unexpected sched_count ",
                     sched_count,
                     ", expected ",
                     comm_size);

    for (size_t idx = 0; idx < sched_count; idx++) {
        if (static_cast<int>(idx) == comm_rank)
            continue;

        entry_factory::create<recv_entry>(scheds[(comm_rank + idx) % sched_count],
                                          recv_bufs[idx],
                                          coll_param.get_recv_count(idx),
                                          dtype,
                                          idx,
                                          comm);

        entry_factory::create<send_entry>(scheds[(comm_rank + idx) % sched_count],
                                          send_seg,
                                          coll_param.get_recv_count(comm_rank),
                                          dtype,
                                          idx,
                                          comm);
    }
    main_sched->sync_partial_scheds();

    return ccl::status::success;
}

ccl::status ccl_coll_build_multi_bcast_allgatherv(ccl_master_sched* main_sched,
                                                  std::vector<ccl_sched*>& scheds,
                                                  const ccl_coll_param& coll_param,
                                                  size_t data_partition_count) {
    LOG_DEBUG("build multi_bcast allgatherv");

    CCL_THROW_IF_NOT(data_partition_count > 0, "data_partition_count should be > 0 ");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    bool inplace = coll_param.is_inplace();

    std::vector<ccl_buffer> recv_bufs;
    std::vector<size_t> recv_offsets;
    ccl_coll_get_allgatherv_bufs_and_offsets(coll_param, recv_bufs, recv_offsets);

    if (!inplace) {
        std::vector<size_t> copy_counts(data_partition_count);
        std::vector<size_t> copy_offsets(data_partition_count);
        for (size_t idx = 0; idx < data_partition_count; idx++) {
            copy_counts[idx] = coll_param.get_recv_count(comm_rank) / data_partition_count;
            copy_offsets[idx] = idx * copy_counts[idx] * dtype_size;
        }
        copy_counts[data_partition_count - 1] +=
            coll_param.get_recv_count(comm_rank) % data_partition_count;

        CCL_ASSERT(scheds.size() >= data_partition_count);

        for (size_t idx = 0; idx < data_partition_count; idx++) {
            entry_factory::create<copy_entry>(scheds[idx],
                                              ccl_buffer(coll_param.get_send_buf_ptr(),
                                                         coll_param.get_send_count() * dtype_size,
                                                         copy_offsets[idx],
                                                         ccl_buffer_type::INDIRECT),
                                              recv_bufs[comm_rank] + copy_offsets[idx],
                                              copy_counts[idx],
                                              dtype);
        }
        main_sched->sync_partial_scheds();
    }

    for (int idx = 0; idx < comm_size; idx++) {
        ccl_coll_entry_param param{};
        param.ctype = ccl_coll_bcast;
        param.recv_buf = recv_bufs[idx];
        param.count = coll_param.get_recv_count(idx);
        param.dtype = dtype;
        param.root = idx;
        param.comm = comm;
        param.stream = coll_param.stream;
        coll_entry_helper::add_coll_entry<ccl_coll_bcast>(scheds[idx % sched_count], param);
    }

    return ccl::status::success;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_allgatherv(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           size_t send_count,
                                           ccl_buffer recv_buf,
                                           const size_t* recv_counts,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm) {
    LOG_DEBUG("build topo allgatherv");

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    int comm_size = comm->size();
    int pair_comm_size = pair_comm->size();
    int node_comm_size = node_comm->size();
    int r2r_comm_size = r2r_comm->size();

    bool is_inplace = send_buf == recv_buf;
    bool is_single_node = comm_size == node_comm_size;

    const std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
        { recv_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 1
    };

    size_t send_buf_idx = 0;
    size_t recv_buf_idx = 1;

    ccl::add_handle_exchange(sched, node_comm, in_buffers);

    if (is_single_node) {
        std::vector<ze_event_handle_t> wait_events;
        entry_factory::create<ze_a2a_allgatherv_entry>(sched,
                                                       send_buf,
                                                       send_count,
                                                       recv_buf,
                                                       recv_counts,
                                                       dtype,
                                                       comm,
                                                       wait_events,
                                                       recv_buf_idx);
        sched->add_barrier();
        ccl::add_comm_barrier(sched, comm);
        return ccl::status::success;
    }

    // helper function
    auto get_distance = [&](int from, int to) {
        CCL_THROW_IF_NOT(from >= 0, "from: ", from, " to: ", to);
        CCL_THROW_IF_NOT(from <= to, "from: ", from, " to: ", to);
        CCL_THROW_IF_NOT(to <= comm_size, "from: ", from, " to: ", to);
        return std::accumulate(recv_counts + from, recv_counts + to, 0);
    };

    if (pair_comm->rank() == ccl::global_data::env().kernel_1s_lead) {
        /* 1. allocate send && recv tmp host buffers for host bcast stage */
        int pair_start = pair_comm->get_global_rank(0, true);
        size_t host_send_buf_count = get_distance(pair_start, pair_start + pair_comm_size);
        size_t host_send_buf_bytes = host_send_buf_count * dtype.size();

        size_t host_recv_buf_count{}; // calculate max pair size in recv_count
        for (int rank = 0; rank < comm_size; rank += pair_comm_size) {
            size_t count = get_distance(rank, rank + pair_comm_size);
            host_recv_buf_count = std::max(host_recv_buf_count, count);
        }
        size_t host_recv_buf_bytes = host_recv_buf_count * dtype.size();

        LOG_DEBUG("alloc host tmp buffers for bcast: send_buf: ",
                  host_send_buf_bytes,
                  ", recv_buf: ",
                  host_recv_buf_bytes);
        ccl::alloc_param host_send_buf_alloc(
            host_send_buf_bytes, ccl::buffer_type::regular, ccl::buffer_place::host);
        ccl_buffer send_host_buf = sched->alloc_buffer(host_send_buf_alloc);

        ccl::alloc_param host_recv_buf_alloc(
            host_recv_buf_bytes, ccl::buffer_type::regular, ccl::buffer_place::host);
        ccl_buffer recv_host_buf = sched->alloc_buffer(host_recv_buf_alloc);

        /* 2. copy to host */
        for (int peer_rank = 0, dst_offset{}; peer_rank < pair_comm_size; ++peer_rank) {
            int global_rank = pair_comm->get_global_rank(peer_rank, true) -
                              ccl::global_data::env().kernel_1s_lead;
            size_t copy_count = recv_counts[global_rank];
            ccl_buffer src{};
            size_t src_offset = (is_inplace) ? get_distance(0, global_rank) : 0;
            copy_attr attr(
                peer_rank, send_buf_idx, copy_direction::d2h, pair_comm, src_offset, dst_offset);
            if (peer_rank == pair_comm->rank()) {
                src = send_buf;
                attr = copy_attr(copy_direction::d2h, src_offset);
            }
            LOG_DEBUG("copy to host: from global rank: ", global_rank, ", count: ", copy_count);
            entry_factory::create<copy_entry>(sched, src, send_host_buf, copy_count, dtype, attr);
            dst_offset += copy_count;
        }
        sched->add_barrier();

        /* 3. bcast between nodes */
        for (int peer_rank = 0; peer_rank < r2r_comm_size; ++peer_rank) {
            ccl_buffer buf = recv_host_buf;
            if (peer_rank == r2r_comm->rank()) {
                buf = send_host_buf;
            }

            int global_rank = r2r_comm->get_global_rank(peer_rank, true);
            int r2r_start = global_rank - ccl::global_data::env().kernel_1s_lead;
            size_t copy_count = get_distance(r2r_start, r2r_start + pair_comm_size);
            LOG_DEBUG("bcast: peer_rank: ", global_rank, ", count ", copy_count);
            ccl_coll_build_bcast(sched, buf, copy_count, dtype, peer_rank, r2r_comm);
            sched->add_barrier();

            size_t dst_offset = get_distance(0, r2r_start);
            LOG_DEBUG("copy to device: offset: ", dst_offset, ", count: ", copy_count);
            entry_factory::create<copy_entry>(sched,
                                              buf,
                                              recv_buf,
                                              copy_count,
                                              dtype,
                                              copy_attr(copy_direction::h2d, 0, dst_offset));
            sched->add_barrier();
        }
        ccl::add_comm_barrier(sched, even_comm);

        /* 4. allgatherv in even_comm */
        for (int node_idx = 0; node_idx < r2r_comm_size; ++node_idx) {
            int from = (comm->rank() - ccl::global_data::env().kernel_1s_lead +
                        node_idx * node_comm_size) %
                       comm_size; // TODO: fix lead
            int to = from + pair_comm_size;
            size_t count = get_distance(from, to);
            size_t offset = get_distance(0, from);
            for (int i = 0; i < even_comm->size() - 1; ++i) {
                int peer_rank = (even_comm->rank() + i + 1) % even_comm->size();
                copy_attr attr(
                    peer_rank, recv_buf_idx, copy_direction::d2d, even_comm, offset, offset);
                entry_factory::create<copy_entry>(
                    sched, recv_buf, ccl_buffer(), count, dtype, attr);
            }
        }
        sched->add_barrier();
        ccl::add_comm_barrier(sched, even_comm);

        /* 5. copy to peer pair rank */
        size_t copy_count = get_distance(0, comm_size);
        int peer_rank = (pair_comm->rank() + 1) % pair_comm_size;
        copy_attr attr(peer_rank, recv_buf_idx, copy_direction::d2d, pair_comm);
        entry_factory::create<copy_entry>(sched, recv_buf, ccl_buffer(), copy_count, dtype, attr);
        sched->add_barrier();
    }
    ccl::add_comm_barrier(sched, pair_comm);

    return ccl::status::success;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
