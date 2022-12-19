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
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

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
            recv_bufs[idx].set(coll_param.get_recv_buf(idx),
                               coll_param.get_recv_count(idx) * dtype_size);
            recv_offsets[idx] = 0;
        }
    }
    else {
        size_t offset = 0;
        for (int idx = 0; idx < comm_size; idx++) {
            size_t bytes = coll_param.get_recv_count(idx) * dtype_size;
            recv_bufs[idx].set(coll_param.get_recv_buf(), offset + bytes, offset);
            recv_offsets[idx] = offset;
            offset += bytes;
        }
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_flat_allgatherv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param) {
    LOG_DEBUG("build flat allgatherv");
    CCL_THROW_IF_NOT(main_sched || (!main_sched && scheds.size() == 1));

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

    auto send_seg = ccl_buffer(coll_param.get_send_buf(), coll_param.get_send_count() * dtype_size);

    if (!inplace) {
        entry_factory::create<copy_entry>(
            scheds[2 * comm_rank % sched_count],
            ccl_buffer(coll_param.get_send_buf(), coll_param.get_send_count() * dtype_size),
            recv_bufs[comm_rank],
            coll_param.get_recv_count(comm_rank),
            dtype);
    }
    else {
        size_t total_recv_bytes =
            std::accumulate(coll_param.recv_counts.begin(), coll_param.recv_counts.end(), 0) *
            dtype_size;
        send_seg = ccl_buffer(coll_param.get_send_buf(), total_recv_bytes, recv_offsets[comm_rank]);
    }

    CCL_THROW_IF_NOT(static_cast<int>(sched_count) == comm_size || !main_sched,
                     "unexpected sched_count ",
                     sched_count,
                     ", expected ",
                     comm_size);

    size_t total_ranks = (main_sched) ? sched_count : comm_size;
    for (size_t idx = 0; idx < total_ranks; idx++) {
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
    if (main_sched) {
        main_sched->sync_subscheds();
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_multi_bcast_allgatherv(ccl_sched* main_sched,
                                                  std::vector<ccl_sched*>& scheds,
                                                  const ccl_coll_param& coll_param,
                                                  size_t data_partition_count) {
    LOG_DEBUG("build multi_bcast allgatherv");
    CCL_THROW_IF_NOT(main_sched || (!main_sched && scheds.size() == 1));

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
        if (main_sched) {
            main_sched->sync_subscheds();
        }
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
        ccl::add_coll_entry(scheds[idx % sched_count], param);
    }

    return ccl::status::success;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_allgatherv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param) {
    LOG_DEBUG("build topo allgatherv");
    CCL_THROW_IF_NOT(scheds.size() == 1);

    ccl_comm* comm = coll_param.comm;
    ccl_sched* sched = scheds.front();
    const ccl_datatype& dtype = coll_param.dtype;
    const bool is_inplace = coll_param.is_inplace();

    std::vector<ccl_buffer> recv_bufs;
    std::vector<size_t> recv_offsets;
    const std::vector<size_t>& recv_counts = coll_param.recv_counts;
    ccl_coll_get_allgatherv_bufs_and_offsets(coll_param, recv_bufs, recv_offsets);

    const size_t send_count = recv_counts[comm->rank()];
    ccl_buffer send_buf;
    if (is_inplace) {
        send_buf = recv_bufs[comm->rank()];
    }
    else {
        send_buf = ccl_buffer(coll_param.get_send_buf(), send_count * dtype.size());
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    const int lead_rank = ccl::global_data::env().kernel_1s_lead;
    const bool is_lead_rank = pair_comm->rank() == lead_rank;

    const int even_comm_size = even_comm->size();
    const bool is_multi_card = (even_comm_size > 1);
    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    bool is_single_node = topo_manager.is_single_node;

    /* IPC exchange */
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
    };
    constexpr size_t send_buf_idx = 0;
    in_buffers.reserve(in_buffers.size() + recv_bufs.size());
    const size_t recv_buf_idx_start = in_buffers.size();
    for (const auto& buf : recv_bufs) {
        in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
    }
    ccl::add_handle_exchange(sched, node_comm, in_buffers);

    sched->try_enable_ze_single_list();
    std::vector<ze_event_handle_t> wait_events;
    // events from commands that must be executed in parallel are placed
    // to parallel_copy_events container. after commands are appended,
    // these events must be moved to the general container wait_events
    // using add_sched_barrier_for_parallel_copies function
    std::list<ze_event_handle_t> parallel_copy_events;

    auto add_sched_barrier_for_parallel_copies = [&]() {
        wait_events.insert(
            wait_events.end(), parallel_copy_events.begin(), parallel_copy_events.end());
        parallel_copy_events.clear();
        sched->add_barrier();
    };

    if (!is_single_node) {
        if (!is_inplace) {
            // copy data from my send_buf to my recv_buf
            copy_attr attr{};
            attr.direction = copy_direction::d2d;
            auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                              send_buf,
                                                              recv_bufs[comm->rank()],
                                                              recv_counts[comm->rank()],
                                                              dtype,
                                                              attr,
                                                              wait_events);
            parallel_copy_events.push_back(entry->entry_event);
        }

        // pack data to be used for scaleout
        std::vector<ccl_buffer> recv_bufs_r2r;
        std::vector<size_t> recv_counts_r2r;
        for (int i = 0; i < r2r_comm->size(); i++) {
            const int global_rank = r2r_comm->get_global_rank(i);
            recv_bufs_r2r.push_back(recv_bufs[global_rank]);
            recv_counts_r2r.push_back(recv_counts[global_rank]);
        }

        if (!is_single_node) {
            ccl_coll_entry_param coll_param_scaleout{};
            coll_param_scaleout.ctype = ccl_coll_allgatherv;
            coll_param_scaleout.send_buf = send_buf;
            coll_param_scaleout.recv_bufs = recv_bufs_r2r;
            coll_param_scaleout.send_count = send_count;
            coll_param_scaleout.recv_counts = recv_counts_r2r.data();
            coll_param_scaleout.dtype = dtype;
            coll_param_scaleout.comm = r2r_comm;

            ccl::add_scaleout(sched, coll_param_scaleout, is_single_node, wait_events);
        }

        ccl::add_comm_barrier(sched, even_comm, wait_events);
        auto recv_send_peers = [&](ccl_comm* recv_comm,
                                   ccl_comm* send_comm,
                                   size_t scaleout_offset = 0,
                                   bool is_inplace = false) {
            for (int peer_idx = 1; peer_idx < recv_comm->size(); peer_idx++) {
                // copy data from all peers in even_comm
                const int peer_rank = (recv_comm->rank() + peer_idx) % recv_comm->size();
                CCL_THROW_IF_NOT(recv_comm->rank() != peer_rank, "Do not copy from own rank");
                const int global_rank =
                    (recv_comm->get_global_rank(peer_rank) + scaleout_offset) % comm->size();
                copy_attr attr{};
                attr.peer_rank = peer_rank;
                if (is_inplace)
                    attr.peer_buf_idx = recv_buf_idx_start + global_rank;
                else
                    attr.peer_buf_idx = send_buf_idx;
                attr.direction = copy_direction::c2c;
                attr.map_comm = recv_comm;
                attr.hint_queue_index = parallel_copy_events.size();
                auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                                  ccl_buffer(),
                                                                  recv_bufs[global_rank],
                                                                  recv_counts[global_rank],
                                                                  dtype,
                                                                  attr,
                                                                  wait_events);
                parallel_copy_events.push_back(entry->entry_event);

                // do not do mdfi copy if only one tile is used
                if (send_comm->size() == 1) {
                    continue;
                }

                // copy the data recieved from even_comm peer (xelink) to pair_comm peer (mdfi)
                int send_rank = (send_comm->rank() + 1) % send_comm->size();
                copy_attr attr_send{};
                attr_send.peer_rank = send_rank;
                attr_send.peer_buf_idx = recv_buf_idx_start + global_rank;
                attr_send.direction = copy_direction::t2t;
                attr_send.map_comm = send_comm;
                auto entry_send =
                    entry_factory::create<ze_copy_entry>(sched,
                                                         recv_bufs[global_rank],
                                                         ccl_buffer(),
                                                         recv_counts[global_rank],
                                                         dtype,
                                                         attr_send,
                                                         wait_events,
                                                         std::vector{ entry->entry_event });
                parallel_copy_events.push_back(entry_send->entry_event);
            }
        };

        size_t node_offset = 0;
        // in case of scaleout, data is already copied to recv_buf and we can use in_place
        bool is_use_inplace = is_inplace || !is_single_node;
        for (int r2r_rank = 0; r2r_rank < r2r_comm->size(); r2r_rank++) {
            // copy data from even_comm peers (xelink) that they recieved during scaleout
            // and write the copied data to pair_comm peer (mdfi)
            recv_send_peers(even_comm, pair_comm, node_offset, is_use_inplace);
            node_offset += node_comm->size();

            // do not do mdfi copy if only one tile is used
            if (pair_comm->size() == 1) {
                continue;
            }

            // write the data recieved during scaleout to pair_comm peer (mdfi)
            int send_rank = (pair_comm->rank() + 1) % pair_comm->size();
            copy_attr attr_send{};
            attr_send.peer_rank = send_rank;
            const int global_rank = r2r_comm->get_global_rank(r2r_rank);
            attr_send.peer_buf_idx = recv_buf_idx_start + global_rank;
            attr_send.direction = copy_direction::t2t;
            attr_send.map_comm = pair_comm;
            auto entry_send = entry_factory::create<ze_copy_entry>(sched,
                                                                   recv_bufs[global_rank],
                                                                   ccl_buffer(),
                                                                   recv_counts[global_rank],
                                                                   dtype,
                                                                   attr_send,
                                                                   wait_events);

            parallel_copy_events.push_back(entry_send->entry_event);
        }
        add_sched_barrier_for_parallel_copies();
        ccl::add_comm_barrier(sched, pair_comm, wait_events);

        return ccl::status::success;
    }

    // for small msg sizes we get more performance without main CE using (main CE has overhead)
    const bool can_use_small_msg_optimization = (send_count * dtype.size()) <= (1 * 1024 * 1024);

    // we use small scale algorithm by default and enable large scale algorithm using knob,
    // because small scale algorithm show more perfomance for today
    // TODO: and also we need to replace this knob with more intelligent switching in the future
    const bool can_use_large_scale_algorithm = ccl::global_data::env().allgatherv_topo_large_scale;
    //comm->get_env()->get_ze_copy_engine() != ccl::ze::copy_engine_mode::none && is_multi_card &&
    //ccl::global_data::env().ze_max_copy_queues /* here must be real queue count */ >= even_comm_size-1) or unspecified;

    auto send_to_peers = [&](ccl_comm* comm, ccl_buffer in_buf, size_t count, size_t peer_buf_idx) {
        for (int peer_idx = 0; peer_idx < comm->size() - 1; peer_idx++) {
            const int peer_rank = (comm->rank() + peer_idx + 1) % comm->size();
            CCL_THROW_IF_NOT(comm->rank() != peer_rank);
            copy_attr attr{};
            attr.peer_rank = peer_rank;
            attr.peer_buf_idx = peer_buf_idx;
            // using of link CE for small msgs give us more performance
            const bool use_c2c_direction = (comm == even_comm) || can_use_small_msg_optimization;
            attr.direction = (use_c2c_direction) ? copy_direction::c2c : copy_direction::d2d;
            attr.map_comm = comm;
            attr.hint_queue_index = parallel_copy_events.size();
            auto entry = entry_factory::create<ze_copy_entry>(
                sched, in_buf, ccl_buffer(), count, dtype, attr, wait_events);
            parallel_copy_events.push_back(entry->entry_event);
        }
    };

    auto recv_from_peers = [&](ccl_comm* comm) {
        for (int peer_idx = 0; peer_idx < comm->size() - 1; peer_idx++) {
            const int peer_rank = (comm->rank() + peer_idx + 1) % comm->size();
            CCL_THROW_IF_NOT(comm->rank() != peer_rank);
            const int global_rank = comm->get_global_rank(peer_rank);
            copy_attr attr{};
            attr.peer_rank = peer_rank;
            attr.peer_buf_idx = send_buf_idx;
            const bool use_c2c_direction = (comm == even_comm) || can_use_small_msg_optimization;
            attr.direction = (use_c2c_direction) ? copy_direction::c2c : copy_direction::d2d;
            attr.map_comm = comm;
            attr.hint_queue_index = parallel_copy_events.size();
            auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                              ccl_buffer(),
                                                              recv_bufs[global_rank],
                                                              recv_counts[global_rank],
                                                              dtype,
                                                              attr,
                                                              wait_events);
            parallel_copy_events.push_back(entry->entry_event);
        }
    };

    const bool do_self_copy = !is_inplace;
    if (do_self_copy) {
        /* copy data from my send_buf to my recv_buf */
        copy_attr attr{};
        attr.hint_queue_index = parallel_copy_events.size();
        attr.direction = copy_direction::t2t;
        auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                          send_buf,
                                                          recv_bufs[comm->rank()],
                                                          recv_counts[comm->rank()],
                                                          dtype,
                                                          attr,
                                                          wait_events);
        parallel_copy_events.push_back(entry->entry_event);
    }

    const bool is_small_scale_algorithm = is_multi_card && !can_use_large_scale_algorithm;
    if (is_small_scale_algorithm) {
        LOG_DEBUG("use small scale algorithm");
    }
    const bool is_large_scale_algorithm = is_multi_card && can_use_large_scale_algorithm;
    if (is_large_scale_algorithm) {
        LOG_DEBUG("use large scale algorithm");
    }

    if (is_small_scale_algorithm) {
        /* Small scale algorithm: step 1. inter-card copy */
        LOG_DEBUG("topo/scale_up/inter: copy to self from peers");
        recv_from_peers(even_comm);
        add_sched_barrier_for_parallel_copies();

        /* Small scale algorithm: step 2 & 3. intra-card copy */
        LOG_DEBUG("topo/scale_up/intra: copy from self to peers");
        if (!is_lead_rank && !ccl::global_data::env().enable_ze_bidir_algo) {
            ccl::add_comm_barrier(sched, pair_comm, wait_events);
        }

        for (int rank = pair_comm->rank(); rank < comm->size(); rank += pair_comm->size()) {
            send_to_peers(pair_comm, recv_bufs[rank], recv_counts[rank], recv_buf_idx_start + rank);
        }
        add_sched_barrier_for_parallel_copies();

        if (is_lead_rank && !ccl::global_data::env().enable_ze_bidir_algo) {
            ccl::add_comm_barrier(sched, pair_comm, wait_events);
        }
    }
    else {
        /* Single GPU algorithm */
        /* Large scale algorithm: step 1 & 2. intra-card copy */
        LOG_DEBUG("topo/scale_up/intra: copy to self from peers");
        if (!is_lead_rank && !ccl::global_data::env().enable_ze_bidir_algo) {
            ccl::add_comm_barrier(sched, pair_comm, wait_events);
        }
        recv_from_peers(pair_comm);
        add_sched_barrier_for_parallel_copies();
        if (is_lead_rank && !ccl::global_data::env().enable_ze_bidir_algo) {
            ccl::add_comm_barrier(sched, pair_comm, wait_events);
        }
    }

    if (is_large_scale_algorithm) {
        /* Large scale algorithm: step 3. inter-card copy */
        LOG_DEBUG("topo/scale_up/inter: copy from self to peers");
        size_t start_rank = comm->rank() - pair_comm->rank();
        CCL_THROW_IF_NOT(start_rank < static_cast<size_t>(comm->size()));
        for (int idx = 0; idx < pair_comm->size(); idx++) {
            send_to_peers(even_comm,
                          recv_bufs[start_rank + idx],
                          recv_counts[start_rank + idx],
                          recv_buf_idx_start + start_rank + idx);
        }
        add_sched_barrier_for_parallel_copies();
    }

    ccl_comm* barrier_comm = (is_large_scale_algorithm) ? even_comm : pair_comm;
    ccl::add_comm_barrier(sched, barrier_comm, wait_events);

    return ccl::status::success;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
