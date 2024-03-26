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

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/entry_factory.hpp"

using namespace ccl::utils;

ccl::status ccl_coll_build_direct_reduce_scatter(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 ccl_buffer recv_buf,
                                                 size_t recv_count,
                                                 const ccl_datatype& dtype,
                                                 ccl::reduction reduction,
                                                 ccl_comm* comm) {
    LOG_DEBUG("build direct reduce_scatter");

    if (recv_count == 0)
        return ccl::status::success;

    entry_factory::create<reduce_scatter_entry>(
        sched, send_buf, recv_buf, recv_count, dtype, reduction, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_build_ring_reduce_scatter_block(ccl_sched* sched,
                                                     ccl_buffer send_buf,
                                                     ccl_buffer recv_buf,
                                                     size_t recv_count,
                                                     const ccl_datatype& dtype,
                                                     ccl::reduction op,
                                                     ccl_comm* comm) {
    if (recv_count == 0) {
        return ccl::status::success;
    }

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    int inplace = (send_buf == recv_buf) ? 1 : 0;
    LOG_DEBUG("build ring reduce_scatter_block: ", inplace ? "in-place" : "out-of-place");

    ccl::status status = ccl::status::success;
    int comm_size, rank, idx;
    size_t dtype_size = dtype.size();

    int src, dst;

    comm_size = comm->size();
    rank = comm->rank();

    if (!inplace) {
        /* copy local data into recv_buf */
        entry_factory::create<copy_entry>(
            sched, send_buf + rank * recv_count * dtype_size, recv_buf, recv_count, dtype);
    }

    /* allocate temporary buffer to store incoming data */
    ccl_buffer tmp_buf = sched->alloc_buffer({ recv_count * dtype_size, recv_buf });

    for (idx = 1; idx < comm_size; idx++) {
        src = (comm_size + rank - idx) % comm_size;
        dst = (rank + idx) % comm_size;

        /* send the data that dst needs. recv data that this process
         * needs from src into tmp_recvbuf */
        if (!inplace) {
            entry_factory::create<send_entry>(
                sched, send_buf + dst * recv_count * dtype_size, recv_count, dtype, dst, comm);

            entry_factory::create<recv_entry>(sched, tmp_buf, recv_count, dtype, src, comm);
        }
        else {
            entry_factory::create<send_entry>(
                sched, recv_buf + dst * recv_count * dtype_size, recv_count, dtype, dst, comm);

            entry_factory::create<recv_entry>(sched, tmp_buf, recv_count, dtype, src, comm);
        }

        sched->add_barrier();

        if (!inplace) {
            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, recv_count, recv_buf, nullptr, dtype, op);
        }
        else {
            entry_factory::create<reduce_local_entry>(sched,
                                                      tmp_buf,
                                                      recv_count,
                                                      recv_buf + rank * recv_count * dtype_size,
                                                      nullptr,
                                                      dtype,
                                                      op);
        }
    }

    /* if inplace, move output data to the beginning of
     * recv_buf. already done for rank 0 */
    if (inplace && (rank != 0)) {
        entry_factory::create<copy_entry>(
            sched, recv_buf + rank * recv_count * dtype_size, recv_buf, recv_count, dtype);
    }

    return status;
}

/* behaves like reduce_scatter_block but last block may contain more elements */
ccl::status ccl_coll_build_ring_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t recv_count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction op,
                                               ccl_comm* comm) {
    LOG_DEBUG("build ring reduce_scatter");

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    ccl::status status = ccl::status::success;
    int comm_size, rank;
    size_t dtype_size = dtype.size();

    comm_size = comm->size();
    rank = comm->rank();

    int src = (comm_size + rank - 1) % comm_size;
    int dst = (comm_size + rank + 1) % comm_size;

    size_t count = recv_count;
    size_t bytes = count * dtype_size;

    if (comm_size == 0) {
        return status;
    }

    size_t chunk_count =
        (bytes >= ccl::global_data::env().rs_min_chunk_size &&
         count >= ccl::global_data::env().rs_chunk_count && (int)count >= comm_size)
            ? ccl::global_data::env().rs_chunk_count
            : 1;

    while ((chunk_count > 1) &&
           (bytes / (comm_size * chunk_count) < ccl::global_data::env().rs_min_chunk_size)) {
        chunk_count--;
    }

    if (chunk_count == 0) {
        LOG_ERROR("unexpected chunk_count");
        chunk_count = 1;
    }

    int inplace = (send_buf == recv_buf) ? 1 : 0;
    LOG_DEBUG("build ring reduce_scatter: ",
              inplace ? "in-place" : "out-of-place",
              ", chunk_count ",
              chunk_count);

    if (comm_size == 1) {
        if (!inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
            sched->add_barrier();
        }
        return ccl::status::success;
    }

    ccl_buffer tmp_buf;

    if (inplace) {
        tmp_buf = sched->alloc_buffer({ count * dtype_size, recv_buf });
    }

    ccl_buffer sbuf, rbuf;
    ccl_buffer reduce_in_buf, reduce_inout_buf;
    ccl_buffer recv_reduce_local_buf, recv_reduce_comm_buf;

    /* start send and recv from such positions to have */
    /* the final reduction result on last iteration in corresponsing block */

    /* block = group of ~ equal-sized chunks */
    int block_idx = (rank + comm_size - 1) % comm_size;
    size_t main_block_size = count / comm_size;
    size_t last_block_size = main_block_size + count % comm_size;
    int send_block_idx, recv_block_idx;
    size_t send_block_size, recv_block_size;
    size_t send_block_offset, recv_block_offset;

    size_t send_main_chunk_size, send_last_chunk_size;
    size_t recv_main_chunk_size, recv_last_chunk_size;

    size_t send_chunk_size, recv_chunk_size = 0, reduce_chunk_size;
    size_t send_chunk_offset, recv_chunk_offset = 0, reduce_chunk_offset;

    /* if chunk_count > 1 then make reduction with 1 chunk delay to get comp/comp overlapping */
    bool use_prev = (chunk_count > 1) ? true : false;
    size_t prev_recv_chunk_size, prev_recv_chunk_offset;

    ccl_recv_reduce_result_buf_type recv_reduce_result_type;

    for (int idx = 0; idx < (comm_size - 1); idx++) {
        send_block_idx = block_idx;
        recv_block_idx = (comm_size + block_idx - 1) % comm_size;

        send_block_size = (send_block_idx == (comm_size - 1)) ? last_block_size : main_block_size;
        recv_block_size = (recv_block_idx == (comm_size - 1)) ? last_block_size : main_block_size;

        send_block_offset = main_block_size * send_block_idx * dtype_size;
        recv_block_offset = main_block_size * recv_block_idx * dtype_size;

        send_main_chunk_size = send_block_size / chunk_count;
        send_last_chunk_size = send_main_chunk_size + send_block_size % chunk_count;

        recv_main_chunk_size = recv_block_size / chunk_count;
        recv_last_chunk_size = recv_main_chunk_size + recv_block_size % chunk_count;

        for (size_t chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            prev_recv_chunk_size = recv_chunk_size;
            send_chunk_size =
                (chunk_idx == (chunk_count - 1)) ? send_last_chunk_size : send_main_chunk_size;
            recv_chunk_size =
                (chunk_idx == (chunk_count - 1)) ? recv_last_chunk_size : recv_main_chunk_size;
            reduce_chunk_size = (use_prev) ? prev_recv_chunk_size : recv_chunk_size;

            prev_recv_chunk_offset = recv_chunk_offset;
            send_chunk_offset = send_block_offset + send_main_chunk_size * chunk_idx * dtype_size;
            recv_chunk_offset = recv_block_offset + recv_main_chunk_size * chunk_idx * dtype_size;
            reduce_chunk_offset = (use_prev) ? prev_recv_chunk_offset : recv_chunk_offset;

            if (inplace) {
                sbuf = recv_buf;
                rbuf = tmp_buf;

                reduce_in_buf = tmp_buf;
                reduce_inout_buf = recv_buf;

                recv_reduce_local_buf = recv_buf;
                recv_reduce_comm_buf = rbuf;
                recv_reduce_result_type = ccl_recv_reduce_local_buf;
            }
            else {
                sbuf = (idx == 0) ? send_buf : recv_buf;
                rbuf = recv_buf;

                reduce_in_buf = send_buf;
                reduce_inout_buf = recv_buf;

                recv_reduce_local_buf = send_buf;
                recv_reduce_comm_buf = rbuf;
                recv_reduce_result_type = ccl_recv_reduce_comm_buf;
            }

            sbuf += send_chunk_offset;
            rbuf += recv_chunk_offset;

            reduce_in_buf += reduce_chunk_offset;
            reduce_inout_buf += reduce_chunk_offset;

            recv_reduce_local_buf += reduce_chunk_offset;
            recv_reduce_comm_buf += reduce_chunk_offset;

            entry_factory::create<send_entry>(sched, sbuf, send_chunk_size, dtype, dst, comm);

            if (!use_prev) {
                CCL_ASSERT(recv_chunk_size == reduce_chunk_size);
                entry_factory::create<recv_reduce_entry>(sched,
                                                         recv_reduce_local_buf,
                                                         recv_chunk_size,
                                                         dtype,
                                                         op,
                                                         src,
                                                         comm,
                                                         recv_reduce_comm_buf,
                                                         recv_reduce_result_type);
            }
            else {
                entry_factory::create<recv_entry>(sched, rbuf, recv_chunk_size, dtype, src, comm);

                if (idx + chunk_idx > 0) {
                    entry_factory::create<reduce_local_entry>(sched,
                                                              reduce_in_buf,
                                                              reduce_chunk_size,
                                                              reduce_inout_buf,
                                                              nullptr,
                                                              dtype,
                                                              op);
                    sched->add_barrier();
                }

                if ((idx == comm_size - 2) && (chunk_idx == chunk_count - 1)) {
                    /* tail reduction for last recv operation */
                    sched->add_barrier();

                    if (inplace) {
                        reduce_in_buf = tmp_buf;
                        reduce_inout_buf = recv_buf;
                    }
                    else {
                        reduce_in_buf = send_buf;
                        reduce_inout_buf = recv_buf;
                    }

                    reduce_in_buf += recv_chunk_offset;
                    reduce_inout_buf += recv_chunk_offset;

                    entry_factory::create<reduce_local_entry>(sched,
                                                              reduce_in_buf,
                                                              recv_chunk_size,
                                                              reduce_inout_buf,
                                                              nullptr,
                                                              dtype,
                                                              op);
                }
            }

            sched->add_barrier();
        }

        /* move blocks left */
        block_idx = (comm_size + block_idx - 1) % comm_size;
    }

    return status;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_reduce_scatter_fill(ccl_sched* sched,
                                                    ccl_buffer send_buf,
                                                    ccl_buffer recv_buf,
                                                    size_t recv_count,
                                                    const ccl_datatype& dtype,
                                                    ccl::reduction op,
                                                    ccl_comm* comm) {
    LOG_DEBUG("build topo reduce_scatter, recv_count ", recv_count);
    if (recv_count == 0) {
        return ccl::status::success;
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    const int comm_size = comm->size();
    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();
    const int node_comm_size = node_comm->size();
    const int r2r_comm_size = r2r_comm->size();

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    const bool is_single_node = topo_manager.is_single_node;
    const bool is_single_card = topo_manager.is_single_card;
    const bool is_multi_card = (even_comm_size > 1);

    const size_t count = comm_size * recv_count;
    std::vector<ze_event_handle_t> wait_events;
    ze_event_handle_t out_event;

    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
        { recv_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 1
    };

    // TODO: fix - reduce_scatter pipeline uses xelink write which seems to fail with int8
    const bool use_reduce_scatter_pipeline =
        ccl::global_data::env().reduce_scatter_monolithic_pipeline_kernel && pair_comm_size > 1 &&
        dtype != ccl::datatype::int8 && ccl::global_data::env().enable_ze_bidir_algo;
    LOG_DEBUG("topo/reduce_scatter pipeline ", use_reduce_scatter_pipeline);

    // optimized non-fallback algorithm is currently supported for bidirectional case
    const bool use_non_fallback_algo = !ccl::global_data::env().reduce_scatter_fallback_algo &&
                                       ccl::global_data::env().enable_ze_bidir_algo;
    LOG_DEBUG("topo/reduce_scatter fallback algo ", !use_non_fallback_algo);

    const bool is_inplace = send_buf == recv_buf;
    LOG_DEBUG("topo/reduce_scatter is_inplace ", is_inplace);

    // temporary buffer is used for scaleout, fallback algorithm and inplace.
    // for non fallback version, temp buffer is not used for single card and single plane
    const bool use_tmp_buf = !is_single_node || !use_non_fallback_algo || !is_single_card ||
                             pair_comm_size > 1 || is_inplace;

    bool is_rs_write = !ccl::global_data::env().reduce_scatter_topo_read &&
                       !use_reduce_scatter_pipeline && use_non_fallback_algo &&
                       dtype != ccl::datatype::int8;
    LOG_DEBUG("topo/reduce_scatter is_rs_write:",
              is_rs_write,
              ccl::global_data::env().reduce_scatter_topo_read,
              use_reduce_scatter_pipeline,
              use_non_fallback_algo,
              dtype != ccl::datatype::int8);
    size_t base_count = count;
    size_t pair_comm_offset = 0;
    size_t pair_comm_offset_bytes = 0;

    if (ccl::global_data::env().enable_ze_bidir_algo) {
        base_count = count / pair_comm->size();
        pair_comm_offset = base_count * pair_comm->rank();
        pair_comm_offset_bytes = pair_comm_offset * dtype.size();

        if (pair_comm->rank() == pair_comm->size() - 1)
            base_count += count % pair_comm->size();
    }

    size_t main_block_count = base_count / even_comm_size;
    size_t block_count = main_block_count;
    if (even_comm->rank() == even_comm_size - 1) {
        block_count += base_count % even_comm_size;
    }

    // setup tmp buffer for write copy mode
    ccl_buffer tmp_write_buf;

    if (is_rs_write) {
        size_t tmp_buf_bytes = 0;
        tmp_buf_bytes = dtype.size() * ((even_comm_size - 1) * block_count);
        // workaround with dummy 1 byte to avoid allocation with 0 byte
        if (tmp_buf_bytes == 0) {
            tmp_buf_bytes = 1;
        }
        ccl::alloc_param alloc_param(
            tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
        tmp_write_buf = sched->alloc_buffer(alloc_param);
        LOG_DEBUG("topo/reduce_scatter: allocate temp write buffer");
    }

    size_t tmp_write_buf_idx = -1;
    if (is_rs_write) {
        in_buffers.push_back({ tmp_write_buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
        tmp_write_buf_idx = in_buffers.size() - 1;
    }
    std::vector<ccl_buffer> tmp_bufs;
    size_t tmp_buf_idx_start = -1;
    ccl_buffer tmp_buf;
    // reduce_scatter pipeline entry require distinct temp buffers for each peer
    if (use_non_fallback_algo && use_reduce_scatter_pipeline) {
        ze_utils::alloc_tmp_bufs(
            sched, comm, tmp_bufs, in_buffers, tmp_buf_idx_start, count, dtype);
        if (!is_single_node) {
            // TODO: add device memory manager support or any mechanism that
            // allows to control the memory consumption once the pipleine chunking is implemented
            size_t tmp_buf_bytes = count * dtype.size();
            ccl::alloc_param alloc_param(
                tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
            tmp_buf = sched->alloc_buffer(alloc_param);
            // scaleout rearranges send_buf into tmp_buf and uses this rearranged buf as input
            in_buffers[0] = { tmp_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }; // 0
        }
    }
    else if (use_tmp_buf) {
        size_t tmp_buf_bytes = count * dtype.size();
        // single-node non-fallback algo only needs temp buffer for a plane
        if (use_non_fallback_algo && is_single_node) {
            tmp_buf_bytes /= pair_comm_size;
        }
        ccl::alloc_param alloc_param(
            tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
        tmp_buf = sched->alloc_buffer(alloc_param);

        const size_t tmp_buf_size_per_rank = recv_count * r2r_comm_size * dtype.size();
        if (use_non_fallback_algo && !is_single_node) {
            // plane 0 works on even partitions and plane 1 works on odd partitions of tmp_buf
            tmp_bufs.push_back(tmp_buf + tmp_buf_size_per_rank * pair_comm->rank());

            // scaleout rearranges send_buf into tmp_buf and uses this rearranged buf as input
            in_buffers[0] = { tmp_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }; // 0
        }
        else {
            tmp_bufs.push_back(tmp_buf);
        }
        tmp_buf_idx_start = in_buffers.size();
        in_buffers.push_back({ tmp_bufs.front().get_ptr(), ccl::ze::ipc_mem_type::memory }); // 2

        // for scaleout, plane 0 works on even partitions and plane 1 works on odd partitions
        // of tmp_buf and therefore we need to skip two partitions to reach the next one.
        const size_t skip_multiplier = is_single_node ? 1 : pair_comm_size;
        for (int i = 0; i < even_comm_size - 1; i++) {
            tmp_bufs.push_back(tmp_bufs.back() + tmp_buf_size_per_rank * skip_multiplier);
        }
    }

    size_t ipc_event_count{};
    size_t max_ipc_event_count{ 6 };
    ze_event_pool_handle_t ipc_event_pool{};
    if (ccl::global_data::env().enable_ze_barrier) {
        ipc_event_pool = sched->get_memory().ipc_event_pool_manager.create(max_ipc_event_count);
        in_buffers.push_back({ static_cast<void*>(ipc_event_pool), ccl::ze::ipc_mem_type::pool });
    }

    int skip_rank = ccl_comm::invalid_rank;
    if (ccl::global_data::env().enable_kernel_1s_ipc_wa && is_single_card) {
        skip_rank = ccl::global_data::env().kernel_1s_lead;
    }

    sched->try_enable_ze_single_list();

    ccl::add_handle_exchange(sched,
                             node_comm,
                             wait_events,
                             out_event,
                             in_buffers,
                             skip_rank,
                             ipc_event_pool,
                             ipc_event_count++);

    CCL_THROW_IF_NOT(comm_size % 2 == 0, "unexpected comm_size ", comm_size);
    CCL_THROW_IF_NOT(node_comm_size % 2 == 0, "unexpected node_comm_size ", node_comm_size);

    if (use_non_fallback_algo) {
        if (!is_single_node) {
            // rearrange data from send_buf to tmp_buf
            // this is essentially a transpose of input data visualized as
            // node_comm_size X r2r_comm_size to r2r_comm_size X node_comm_size
            std::vector<ze_event_handle_t> parallel_copy_events;
            for (int node_comm_idx = 0; node_comm_idx < node_comm_size; node_comm_idx++) {
                for (int r2r_comm_idx = 0; r2r_comm_idx < r2r_comm_size; r2r_comm_idx++) {
                    const size_t recv_bytes = recv_count * dtype.size();
                    const size_t dst_offset =
                        (node_comm_idx * r2r_comm_size + r2r_comm_idx) * recv_bytes;
                    const size_t src_offset =
                        (r2r_comm_idx * node_comm_size + node_comm_idx) * recv_bytes;
                    copy_attr attr{};
                    attr.direction = copy_direction::d2d;
                    //TODO: make offset calculation more general and robust
                    auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                                      send_buf + src_offset,
                                                                      tmp_buf + dst_offset,
                                                                      recv_count,
                                                                      dtype,
                                                                      attr,
                                                                      wait_events);
                    parallel_copy_events.push_back(entry->entry_event);
                }
            }

            // make sure all ranks have finished rearranging send_buf
            ccl::add_comm_barrier(sched, node_comm, parallel_copy_events, out_event);
            clear_and_push_back(wait_events, out_event);
        }

        // ze_a2a_pipeline_read_write_entry only works for even_comm_size > 1 and
        // therefore we need to deal with even_comm_size == 1 separately
        if (even_comm_size == 1) {
            LOG_DEBUG("topo/scale_up/intra: use ze_onesided_reduce");
            const size_t pair_comm_count = recv_count * r2r_comm_size;
            const size_t pair_comm_local_offset = pair_comm_count * pair_comm->rank();
            // for single-node use send_buf and for multi-node use rearranged tmp_buf
            const ccl_buffer pair_comm_send_buf =
                is_single_node ? send_buf + pair_comm_local_offset * dtype.size()
                               : tmp_buf + pair_comm_local_offset * dtype.size();

            // for inplace, write to tmp buffer instead of recv_buf since
            // the other rank is reading from send_buf which is same as recv_buf
            // for multi-node, write to tmp_buf that will be later used for scaleout
            auto entry = entry_factory::create<ze_onesided_reduce_entry>(
                sched,
                pair_comm_send_buf,
                is_inplace || !is_single_node ? *tmp_bufs.begin() : recv_buf,
                pair_comm_count,
                dtype,
                op,
                pair_comm->rank(),
                pair_comm,
                wait_events,
                pair_comm_local_offset);
            clear_and_push_back(wait_events, entry->entry_event);

            // for single-node inplace, copy output from tmp_buf to recv_buf
            if (is_inplace && is_single_node) {
                ccl::add_comm_barrier(sched, pair_comm, wait_events, out_event);
                clear_and_push_back(wait_events, out_event);
                copy_attr attr{};
                attr.direction = copy_direction::d2d;
                auto copy_entry = entry_factory::create<ze_copy_entry>(
                    sched, *tmp_bufs.begin(), recv_buf, recv_count, dtype, attr, wait_events);
                clear_and_push_back(wait_events, copy_entry->entry_event);
            }
        }
        else if (pair_comm_size > 1) {
            LOG_DEBUG("topo/scale_up/intra: use ze_a2a_pipeline_read_write_entry");

            // allreduce and reduce divides whole data into two continous chunks,
            // whereas reduce_scatter divides them as even and odd chunks.
            // when using pipeline, write data to peer using xelink after mdfi reduce.
            // when not using pipeline, write data to local buffer after mdfi reduce.
            ze_a2a_pipeline_read_write_entry::attr attrs{ .use_continous_data = false,
                                                          .use_remote_target =
                                                              use_reduce_scatter_pipeline };
            auto entry = entry_factory::create<ze_a2a_pipeline_read_write_entry>(
                sched,
                comm,
                is_single_node ? send_buf : tmp_buf,
                tmp_bufs,
                tmp_buf_idx_start,
                count,
                dtype,
                op,
                wait_events,
                attrs);
            clear_and_push_back(wait_events, entry->entry_event);

            ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
            clear_and_push_back(wait_events, out_event);
        }

        ccl_buffer out_tmp_buf = *tmp_bufs.begin();
        if (even_comm_size > 1 && use_reduce_scatter_pipeline) {
            LOG_DEBUG("topo/scale_up/intra: use ze_a2a_pipeline_reduce_entry");
            // reduce from remote even_comm peer buffers or from local buffer
            ccl_buffer dst_recv_buf = is_single_node ? recv_buf : out_tmp_buf;
            auto entry = entry_factory::create<ze_a2a_pipeline_reduce_entry>(
                sched, comm, dst_recv_buf, tmp_bufs, count, dtype, op, wait_events);
            clear_and_push_back(wait_events, entry->entry_event);
        }
        else if (even_comm_size > 1) {
            const size_t tmp_buf_count_per_rank = recv_count * r2r_comm_size;
            const size_t tmp_buf_size_per_rank = tmp_buf_count_per_rank * dtype.size();
            if (!is_single_node) {
                // plane 0 worked on even partitions and plane 1
                // worked on odd partitions, but we need continous
                // data for ze_a2a_reduce_scatter_entry and therefore
                // we pack alternate paritions into continous data
                std::vector<ze_event_handle_t> parallel_copy_events;
                for (int even_comm_idx = 1; even_comm_idx < even_comm_size; even_comm_idx++) {
                    copy_attr attr{};
                    attr.direction = copy_direction::d2d;
                    auto entry = entry_factory::create<ze_copy_entry>(
                        sched,
                        tmp_bufs[even_comm_idx],
                        tmp_bufs.front() + even_comm_idx * tmp_buf_size_per_rank,
                        tmp_buf_count_per_rank,
                        dtype,
                        attr,
                        wait_events);
                    parallel_copy_events.push_back(entry->entry_event);
                }
                ccl::add_comm_barrier(sched, even_comm, parallel_copy_events, out_event);
                clear_and_push_back(wait_events, out_event);
            }

            // perform xelink read and followed by reduce
            ccl_buffer src_send_buf = is_single_node ? send_buf : tmp_buf;
            size_t send_buf_idx = 0;
            // when both tiles are used, we need to read from temporary
            // which contains the result of mdfi reduce
            if (pair_comm_size > 1) {
                src_send_buf = tmp_bufs.front();
                send_buf_idx = tmp_buf_idx_start;
            }
            std::vector<size_t> block_counts(even_comm_size, recv_count * r2r_comm_size);

            if (!is_rs_write) {
                LOG_DEBUG("topo/scale_up/intra: use ze_a2a_reduce_scatter_entry");
                out_tmp_buf = src_send_buf + even_comm->rank() * tmp_buf_size_per_rank;
                auto entry = entry_factory::create<ze_a2a_reduce_scatter_entry>(
                    sched,
                    src_send_buf,
                    is_single_node ? recv_buf : out_tmp_buf,
                    block_counts.data(),
                    dtype,
                    op,
                    even_comm,
                    wait_events,
                    send_buf_idx,
                    0); // pair_comm_offset
                clear_and_push_back(wait_events, entry->entry_event);
            }
            else {
                // copy using write
                LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_write_copy_entry");
                reduce_scatter_args rs_args = { even_comm, block_counts, dtype, op };
                reduce_scatter_bufs rs_bufs = { src_send_buf,
                                                is_single_node ? recv_buf : out_tmp_buf,
                                                tmp_write_buf,
                                                tmp_write_buf_idx,
                                                0 }; //pair_comm_offset

                auto copy_entry = entry_factory::create<ze_a2a_reduce_scatter_write_copy_entry>(
                    sched, rs_args, rs_bufs, wait_events);
                clear_and_push_back(wait_events, copy_entry->entry_event);
                ccl::add_comm_barrier(
                    sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
                clear_and_push_back(wait_events, out_event);

                LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_write_kernel_entry");
                // local reduction
                auto kernel_entry = entry_factory::create<ze_a2a_reduce_scatter_write_kernel_entry>(
                    sched, rs_args, rs_bufs, wait_events);
                clear_and_push_back(wait_events, kernel_entry->entry_event);
                ccl::add_comm_barrier(
                    sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
                clear_and_push_back(wait_events, out_event);
            }
        }

        if (!is_single_node) {
            ccl_coll_param coll_param{ false };
            coll_param.ctype = ccl_coll_reduce_scatter;
            coll_param.send_buf = out_tmp_buf;
            coll_param.recv_buf = recv_buf;
            coll_param.count = recv_count;
            coll_param.dtype = dtype;
            coll_param.reduction = op;
            coll_param.comm = r2r_comm;
            coll_param.hint_algo.reduce_scatter = ccl_coll_reduce_scatter_direct;

            ccl::add_scaleout(sched, coll_param, is_single_node, wait_events, out_event);
            if (out_event) {
                clear_and_push_back(wait_events, out_event);
            }
        }

        return ccl::status::success;
    }

    // note: start section common with allreduce topo
    // the following section is based on the allreduce topo implementation
    // it uses steps from allreduce topo to collect allreduce results on tmp buffer

    if (!ccl::global_data::env().enable_ze_bidir_algo &&
        pair_comm->rank() != ccl::global_data::env().kernel_1s_lead) {
        ccl::add_comm_barrier(
            sched, pair_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);

        CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                         "unexpected ipc_event_count ",
                         ipc_event_count,
                         ", expected max ",
                         max_ipc_event_count);
        return ccl::status::success;
    }
    const ccl_buffer tmp_recv_buf = tmp_bufs.front();
    const size_t tmp_recv_buf_idx = tmp_buf_idx_start;
    size_t even_comm_offset_bytes = main_block_count * even_comm->rank() * dtype.size();
    ccl_buffer pair_comm_send_buf = send_buf + pair_comm_offset_bytes;
    ccl_buffer pair_comm_recv_buf = tmp_recv_buf + pair_comm_offset_bytes;
    ccl_buffer even_comm_recv_buf = tmp_recv_buf + pair_comm_offset_bytes + even_comm_offset_bytes;

    LOG_DEBUG("rank: ",
              pair_comm->rank(),
              ", count: ",
              base_count,
              ", pair_comm_offset: ",
              pair_comm_offset);
    if (is_single_card) {
        auto entry = entry_factory::create<ze_onesided_allreduce_entry>(sched,
                                                                        pair_comm_send_buf,
                                                                        pair_comm_recv_buf,
                                                                        base_count,
                                                                        dtype,
                                                                        op,
                                                                        pair_comm,
                                                                        wait_events,
                                                                        pair_comm_offset);
        clear_and_push_back(wait_events, entry->entry_event);
    }
    else {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_reduce");
        auto entry = entry_factory::create<ze_onesided_reduce_entry>(sched,
                                                                     pair_comm_send_buf,
                                                                     pair_comm_recv_buf,
                                                                     base_count,
                                                                     dtype,
                                                                     op,
                                                                     pair_comm->rank(),
                                                                     pair_comm,
                                                                     wait_events,
                                                                     pair_comm_offset);
        clear_and_push_back(wait_events, entry->entry_event);
    }
    sched->add_barrier();

    bool is_read_allgatherv = ccl::global_data::env().allgatherv_topo_read;
    if (is_multi_card) {
        ccl::add_comm_barrier(
            sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
        // cannot use ze_a2a_allreduce_entry with allgatherv_read
        // since comm_barrier is not available inside the entry
        if (is_single_node && !is_read_allgatherv) {
            LOG_DEBUG("topo/scale_up/intra: use ze_a2a_allreduce_entry");
            auto entry = entry_factory::create<ze_a2a_allreduce_entry>(sched,
                                                                       pair_comm_recv_buf,
                                                                       pair_comm_recv_buf,
                                                                       base_count,
                                                                       dtype,
                                                                       op,
                                                                       even_comm,
                                                                       wait_events,
                                                                       tmp_recv_buf_idx,
                                                                       tmp_recv_buf_idx,
                                                                       pair_comm_offset);
            clear_and_push_back(wait_events, entry->entry_event);
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }
        else {
            LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_entry");
            std::vector<size_t> block_counts(even_comm->size(), main_block_count);
            block_counts.back() += base_count % even_comm_size;
            auto entry = entry_factory::create<ze_a2a_reduce_scatter_entry>(sched,
                                                                            pair_comm_recv_buf,
                                                                            even_comm_recv_buf,
                                                                            block_counts.data(),
                                                                            dtype,
                                                                            op,
                                                                            even_comm,
                                                                            wait_events,
                                                                            tmp_recv_buf_idx,
                                                                            pair_comm_offset);
            clear_and_push_back(wait_events, entry->entry_event);
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }
    }

    ccl_coll_param coll_param{ false };
    coll_param.ctype = ccl_coll_allreduce;
    coll_param.send_buf = even_comm_recv_buf;
    coll_param.recv_buf = even_comm_recv_buf;
    coll_param.count = block_count;
    coll_param.dtype = dtype;
    coll_param.reduction = op;
    coll_param.comm = r2r_comm;

    ccl::add_scaleout(sched, coll_param, is_single_node, wait_events, out_event);
    if (out_event) {
        clear_and_push_back(wait_events, out_event);
    }

    if (is_multi_card && (!is_single_node || is_read_allgatherv)) {
        LOG_DEBUG("topo/scale_up/inter: use ze_a2a_allgatherv");
        // for multinode with allgatherv_read, use a comm_barrier to make sure all
        // r2r scaleout within even_comm has finished so that remote reads are valid
        if (!is_single_node && is_read_allgatherv) {
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }

        std::vector<size_t> recv_counts(even_comm_size, main_block_count);
        recv_counts.back() += base_count % even_comm_size;
        // allgatherv entry takes an array of recv bufs with one buffer for each rank
        // rather than a single large recv buf that needs to be divided
        std::vector<ccl_buffer> recv_bufs;
        for (int i = 0; i < even_comm_size; i++) {
            recv_bufs.push_back(pair_comm_recv_buf + i * main_block_count * dtype.size());
        }
        auto entry = entry_factory::create<ze_a2a_allgatherv_entry>(sched,
                                                                    recv_bufs[even_comm->rank()],
                                                                    block_count,
                                                                    recv_bufs,
                                                                    recv_counts,
                                                                    dtype,
                                                                    even_comm,
                                                                    wait_events,
                                                                    tmp_recv_buf_idx,
                                                                    pair_comm_offset);
        clear_and_push_back(wait_events, entry->entry_event);
        ccl::add_comm_barrier(
            sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
    }

    if (!is_single_card && pair_comm->size() > 1) {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_bcast");
        int peer_rank = (pair_comm->rank() + 1) % pair_comm->size();
        auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                          tmp_recv_buf,
                                                          ccl_buffer(),
                                                          base_count,
                                                          dtype,
                                                          copy_attr(peer_rank,
                                                                    tmp_recv_buf_idx,
                                                                    copy_direction::t2t,
                                                                    false, /*pt2pt_op*/
                                                                    pair_comm,
                                                                    pair_comm_offset,
                                                                    pair_comm_offset),
                                                          wait_events);
        clear_and_push_back(wait_events, entry->entry_event);
        sched->add_barrier();
    }

    ccl::add_comm_barrier(
        sched, pair_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
    clear_and_push_back(wait_events, out_event);

    CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                     "unexpected ipc_event_count ",
                     ipc_event_count,
                     ", expected max ",
                     max_ipc_event_count);

    // note: end section common with allreduce topo

    sched->add_barrier();
    ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
    clear_and_push_back(wait_events, out_event);

    // copy part of temp buffer to recv buffer
    copy_attr attr{};
    attr.direction = copy_direction::d2d;
    size_t allred_offset_bytes = recv_count * comm->rank() * dtype.size();
    ccl_buffer allred_recv_buf = tmp_recv_buf + allred_offset_bytes;

    auto entry_copy = entry_factory::create<ze_copy_entry>(
        sched, allred_recv_buf, recv_buf, recv_count, dtype, attr, wait_events);

    clear_and_push_back(wait_events, entry_copy->entry_event);
    sched->add_barrier();
    ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);

    return ccl::status::success;
}

ccl::status ccl_coll_build_topo_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction op,
                                               ccl_comm* comm) {
    return ccl_build_topo_uniform_buff_size_op(
        sched,
        send_buf,
        recv_buf,
        count,
        dtype.size(),
        ccl::global_data::env().reduce_scatter_pipe_chunk_count,
        "REDUCE_SCATTER",
        ccl::global_data::get().metrics_profiler->reduce_scatter_pipe,
        [dtype, op, comm](ccl_sched* sched, ccl_buffer send_buf, ccl_buffer recv_buf, size_t count)
            -> ccl::status {
            return ccl_coll_build_topo_reduce_scatter_fill(
                sched, send_buf, recv_buf, count, dtype, op, comm);
        });
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
