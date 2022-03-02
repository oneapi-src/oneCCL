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

ccl::status ccl_coll_build_direct_reduce_scatter(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 ccl_buffer recv_buf,
                                                 size_t recv_count,
                                                 const ccl_datatype& dtype,
                                                 ccl::reduction reduction,
                                                 ccl_comm* comm) {
    LOG_DEBUG("build direct reduce_scatter");

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

    if (recv_count == 0) {
        return ccl::status::success;
    }

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

ccl::status ccl_coll_build_topo_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t recv_count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction reduction,
                                               ccl_comm* comm) {
    LOG_DEBUG("build topo reduce_scatter, recv_count ", recv_count);

    const std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
    };

    size_t send_buf_idx = 0;

    ccl::add_handle_exchange(sched, comm, in_buffers);

    std::vector<ze_event_handle_t> wait_events;
    std::vector<size_t> blocks_count(comm->size(), recv_count);
    entry_factory::create<ze_a2a_reduce_scatter_entry>(sched,
                                                       send_buf,
                                                       recv_buf,
                                                       blocks_count.data(),
                                                       dtype,
                                                       reduction,
                                                       comm,
                                                       wait_events,
                                                       send_buf_idx);

    ccl::add_comm_barrier(sched, comm);

    return ccl::status::success;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
