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

/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <numeric>

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

ccl::status ccl_coll_build_direct_alltoallv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            const size_t* send_counts,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm) {
    LOG_DEBUG("build direct alltoallv");

    entry_factory::create<alltoallv_entry>(
        sched, send_buf, send_counts, recv_buf, recv_counts, dtype, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_calculate_alltoallv_counts(const ccl_coll_param& coll_param,
                                                std::vector<size_t>& send_counts,
                                                std::vector<size_t>& recv_counts,
                                                std::vector<size_t>& send_offsets,
                                                std::vector<size_t>& recv_offsets,
                                                size_t& total_send_count,
                                                size_t& total_recv_count,
                                                size_t& total_send_bytes,
                                                size_t& total_recv_bytes) {
    ccl_coll_type coll_type = coll_param.ctype;
    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_size = comm->size();
    size_t dtype_size = dtype.size();

    if (coll_type == ccl_coll_alltoall) {
        send_counts.resize(comm_size, coll_param.get_send_count());
        recv_counts.resize(comm_size, coll_param.get_recv_count());
    }
    else if (coll_type == ccl_coll_alltoallv) {
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.send_counts.size()) == comm_size,
                         "unexpected send_counts size ",
                         coll_param.send_counts.size(),
                         ", expected ",
                         comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_counts.size()) == comm_size,
                         "unexpected recv_counts size ",
                         coll_param.recv_counts.size(),
                         ", expected ",
                         comm_size);
        send_counts = coll_param.send_counts;
        recv_counts = coll_param.recv_counts;
    }

    send_offsets.resize(comm_size, 0);
    recv_offsets.resize(comm_size, 0);

    for (int idx = 1; idx < comm_size; idx++) {
        send_offsets[idx] = send_offsets[idx - 1] + send_counts[idx - 1] * dtype_size;
        recv_offsets[idx] = recv_offsets[idx - 1] + recv_counts[idx - 1] * dtype_size;
    }

    total_send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0);
    total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    total_send_bytes = total_send_count * dtype_size;
    total_recv_bytes = total_recv_count * dtype_size;

    LOG_DEBUG("total_send_count ",
              total_send_count,
              ", total_recv_count ",
              total_recv_count,
              ", total_send_bytes ",
              total_send_bytes,
              ", total_recv_bytes ",
              total_recv_bytes);

    return ccl::status::success;
}

ccl::status ccl_coll_build_naive_alltoallv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param) {
    LOG_DEBUG("build naive alltoallv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    bool inplace = coll_param.is_inplace();

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::create<copy_entry>(scheds[sched_idx],
                                          ccl_buffer(coll_param.get_send_buf_ptr(),
                                                     total_send_bytes,
                                                     send_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                     total_recv_bytes,
                                                     recv_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          send_counts[comm_rank],
                                          dtype);
    }

    for (int idx = 0; idx < comm_size; idx++) {
        if (idx == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + idx) % sched_count;

        ccl_buffer recv_buf;

        if (inplace)
            recv_buf = scheds[sched_idx]->alloc_buffer(
                { recv_counts[idx] * dtype_size, coll_param.get_recv_buf() });
        else
            recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                  total_recv_bytes,
                                  recv_offsets[idx],
                                  ccl_buffer_type::INDIRECT);

        entry_factory::make_chunked_recv_entry(
            scheds, sched_idx, recv_buf, recv_counts[idx], dtype, idx, comm);

        entry_factory::make_chunked_send_entry(scheds,
                                               sched_idx,
                                               ccl_buffer(coll_param.get_send_buf_ptr(),
                                                          total_send_bytes,
                                                          send_offsets[idx],
                                                          ccl_buffer_type::INDIRECT),
                                               send_counts[idx],
                                               dtype,
                                               idx,
                                               comm);

        if (inplace) {
            scheds[sched_idx]->add_barrier();
            entry_factory::create<copy_entry>(scheds[sched_idx],
                                              recv_buf,
                                              ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                         total_recv_bytes,
                                                         recv_offsets[idx],
                                                         ccl_buffer_type::INDIRECT),
                                              recv_counts[idx],
                                              dtype);
            scheds[sched_idx]->add_barrier();
        }
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_scatter_alltoallv(ccl_sched* main_sched,
                                             std::vector<ccl_sched*>& scheds,
                                             const ccl_coll_param& coll_param) {
    LOG_DEBUG("build scatter alltoallv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    ssize_t max_ops = ccl::global_data::env().alltoall_scatter_max_ops;
    if (max_ops != CCL_ENV_SIZET_NOT_SPECIFIED) {
        for (size_t idx = 0; idx < sched_count; idx++) {
            scheds[idx]->flow_control.set_max_credits(max_ops);
        }
    }

    bool inplace = coll_param.is_inplace();

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    std::vector<ccl_buffer> recv_bufs;
    if (inplace)
        recv_bufs.resize(comm_size);

    std::vector<ccl_sched*> recv_scheds(sched_count);
    std::vector<ccl_sched*> send_scheds(sched_count);

    for (size_t idx = 0; idx < sched_count; idx++) {
        auto recv_sched = entry_factory::create<subsched_entry>(
                              scheds[idx], 0, [](ccl_sched* s) {}, "A2AV_RECV")
                              ->get_subsched();

        recv_scheds[idx] = recv_sched;

        auto send_sched = entry_factory::create<subsched_entry>(
                              scheds[idx], 0, [](ccl_sched* s) {}, "A2AV_SEND")
                              ->get_subsched();

        send_scheds[idx] = send_sched;
    }

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::create<copy_entry>(recv_scheds[sched_idx],
                                          ccl_buffer(coll_param.get_send_buf_ptr(),
                                                     total_send_bytes,
                                                     send_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                     total_recv_bytes,
                                                     recv_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          send_counts[comm_rank],
                                          dtype);
    }

    for (int idx = 0; idx < comm_size; idx++) {
        int src = (comm_rank + idx) % comm_size;
        if (src == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + src) % sched_count;

        ccl_buffer recv_buf;

        if (inplace) {
            recv_buf = scheds[sched_idx]->alloc_buffer(
                { recv_counts[src] * dtype_size, coll_param.get_recv_buf() });
            recv_bufs[src] = recv_buf;
        }
        else
            recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                  total_recv_bytes,
                                  recv_offsets[src],
                                  ccl_buffer_type::INDIRECT);

        entry_factory::make_chunked_recv_entry(
            recv_scheds, sched_idx, recv_buf, recv_counts[src], dtype, src, comm);
    }

    for (int idx = 0; idx < comm_size; idx++) {
        int dst = (comm_rank - idx + comm_size) % comm_size;
        if (dst == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + dst) % sched_count;

        entry_factory::make_chunked_send_entry(send_scheds,
                                               sched_idx,
                                               ccl_buffer(coll_param.get_send_buf_ptr(),
                                                          total_send_bytes,
                                                          send_offsets[dst],
                                                          ccl_buffer_type::INDIRECT),
                                               send_counts[dst],
                                               dtype,
                                               dst,
                                               comm);
    }

    if (!inplace)
        return ccl::status::success;

    main_sched->sync_subscheds();

    for (int idx = 0; idx < comm_size; idx++) {
        int src = (comm_rank + idx) % comm_size;
        if (src == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + src) % sched_count;

        entry_factory::create<copy_entry>(scheds[sched_idx],
                                          recv_bufs[src],
                                          ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                     total_recv_bytes,
                                                     recv_offsets[src],
                                                     ccl_buffer_type::INDIRECT),
                                          recv_counts[src],
                                          dtype);
    }

    return ccl::status::success;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_alltoallv(ccl_sched* main_sched,
                                          std::vector<ccl_sched*>& scheds,
                                          const ccl_coll_param& coll_param) {
    ccl_comm* comm = coll_param.comm;
    ccl_sched* sched = scheds.front();
    CCL_THROW_IF_NOT(scheds.size() == 1, "unexpected scheds size: ", scheds.size());
    const ccl_datatype& dtype = coll_param.dtype;
    const bool is_inplace = coll_param.is_inplace();

    int comm_rank = comm->rank();
    int comm_size = comm->size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    std::vector<ccl_buffer> send_bufs(comm_size);
    std::vector<ccl_buffer> recv_bufs(comm_size);
    std::vector<ccl_buffer> tmp_bufs;

    if (is_inplace) {
        CCL_THROW_IF_NOT(send_counts == recv_counts, "unexpected send_counts");
        for (int idx = 0; idx < comm_size; idx++) {
            recv_bufs[idx].set(coll_param.get_send_buf(), total_recv_bytes, recv_offsets[idx]);
            send_bufs[idx] = recv_bufs[idx];
        }

        tmp_bufs.resize(comm_size);
        for (int idx = 0; idx < comm_size; idx++) {
            ccl::alloc_param alloc_param(
                send_counts[idx] * dtype.size(), ccl::buffer_type::ze, ccl::buffer_place::device);
            tmp_bufs[idx] = sched->alloc_buffer(alloc_param);
        }
    }
    else {
        CCL_THROW_IF_NOT(send_counts[comm_rank] == recv_counts[comm_rank],
                         "unexpected send_counts");
        for (int idx = 0; idx < comm_size; idx++) {
            send_bufs[idx].set(coll_param.get_send_buf(), total_send_bytes, send_offsets[idx]);
            recv_bufs[idx].set(coll_param.get_recv_buf(), total_recv_bytes, recv_offsets[idx]);
        }
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();

    const int even_comm_size = even_comm->size();
    bool is_multi_card = (even_comm_size > 1);
    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    CCL_THROW_IF_NOT(topo_manager.is_single_card != is_multi_card);

    // IPC exchange
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

    const size_t send_buf_idx_start = in_buffers.size();
    in_buffers.reserve(send_buf_idx_start + send_bufs.size());
    for (const auto& buf : send_bufs) {
        in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
    }

    const size_t recv_buf_idx_start = in_buffers.size();
    in_buffers.reserve(recv_buf_idx_start + recv_bufs.size());
    for (const auto& buf : recv_bufs) {
        in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
    }

    size_t tmp_buf_idx_start = -1;
    if (is_inplace) {
        tmp_buf_idx_start = in_buffers.size();
        in_buffers.reserve(tmp_buf_idx_start + tmp_bufs.size());
        for (const auto& buf : tmp_bufs) {
            in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
        }
    }

    ccl::add_handle_exchange(sched, node_comm, in_buffers);
    sched->try_enable_ze_single_list();
    std::vector<ze_event_handle_t> wait_events;
    std::list<ze_event_handle_t> parallel_copy_events;

    auto copy_to_peers = [&](std::vector<ccl_buffer>& bufs, std::vector<size_t>& counts) {
        auto card_count = even_comm->size();
        auto tile_count = pair_comm->size();
        for (int card_idx = 0; card_idx < card_count; card_idx++) {
            for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) {
                auto peer_rank = (card_idx * tile_count + tile_idx);
                if (peer_rank == comm_rank)
                    continue;
                copy_attr attr{};
                attr.peer_rank = peer_rank;
                attr.peer_buf_idx = recv_buf_idx_start + comm_rank;
                attr.direction = copy_direction::d2d;
                attr.map_comm = comm;
                attr.hint_queue_index = parallel_copy_events.size();
                attr.is_peer_card_copy = true;
                auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                                  bufs[peer_rank],
                                                                  ccl_buffer(),
                                                                  counts[peer_rank],
                                                                  dtype,
                                                                  attr,
                                                                  wait_events);
                parallel_copy_events.push_back(entry->entry_event);
            }
        }
    };

    auto add_sched_barrier_for_parallel_copies = [&]() {
        wait_events.insert(
            wait_events.end(), parallel_copy_events.begin(), parallel_copy_events.end());
        parallel_copy_events.clear();
        sched->add_barrier();
    };

    auto copy_to_self = [&](ccl_buffer& send, ccl_buffer& recv, const size_t count) {
        copy_attr attr{};
        attr.hint_queue_index = parallel_copy_events.size();
        attr.is_peer_card_copy = true;
        auto entry = entry_factory::create<ze_copy_entry>(
            sched, send, recv, count, dtype, attr, wait_events);
        parallel_copy_events.push_back(entry->entry_event);
    };

    auto barrier_comm = [&](ccl_comm* comm) {
        auto barrier_event = ccl::add_comm_barrier(sched, comm, wait_events);
        wait_events.push_back(barrier_event);
    };

    if (is_inplace) {
        for (int idx = 0; idx < comm_size; idx++) {
            CCL_THROW_IF_NOT(send_bufs[idx].get_ptr() == recv_bufs[idx].get_ptr(),
                             "unexpected send_buf ptr for inplace case");
        }

        // copy from all recvs to tmps
        for (int idx = 0; idx < comm_size; idx++) {
            copy_to_self(recv_bufs[idx], tmp_bufs[idx], send_counts[idx]);
        }
        add_sched_barrier_for_parallel_copies();
        barrier_comm(node_comm);

        // copy from my tmp buffer to peer recv
        copy_to_peers(tmp_bufs, send_counts);
    }
    else {
        // copy from own send to own recv
        copy_to_self(send_bufs[comm->rank()], recv_bufs[comm->rank()], send_counts[comm->rank()]);
        // copy from peer rank send to peer rank recv
        copy_to_peers(send_bufs, send_counts);
    }

    add_sched_barrier_for_parallel_copies();
    barrier_comm(node_comm);

    return ccl::status::success;
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
