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
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

ccl_status_t ccl_coll_build_direct_allgatherv(ccl_sched* sched,
                                              ccl_buffer send_buf,
                                              size_t send_count,
                                              ccl_buffer recv_buf,
                                              const size_t* recv_counts,
                                              const ccl_datatype& dtype,
                                              ccl_comm* comm) {
    LOG_DEBUG("build direct allgatherv");

    entry_factory::make_entry<allgatherv_entry>(
        sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm);
    return ccl_status_success;
}

ccl_status_t ccl_coll_build_naive_allgatherv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             size_t send_count,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm) {
    LOG_DEBUG("build naive allgatherv");

    size_t comm_size = comm->size();
    size_t this_rank = comm->rank();
    size_t dtype_size = dtype.size();
    size_t* offsets = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    ccl_status_t status = ccl_status_success;

    offsets[0] = 0;
    for (size_t rank_idx = 1; rank_idx < comm_size; ++rank_idx) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * dtype_size;
    }

    if (send_buf != recv_buf) {
        // out-of-place case
        entry_factory::make_entry<copy_entry>(
            sched, send_buf, recv_buf + offsets[this_rank], send_count, dtype);
    }

    for (size_t rank_idx = 0; rank_idx < comm_size; ++rank_idx) {
        if (rank_idx != this_rank) {
            // send own buffer to other ranks
            entry_factory::make_chunked_send_entry(
                sched, recv_buf + offsets[this_rank], send_count, dtype, rank_idx, comm);
            // recv other's rank buffer
            entry_factory::make_chunked_recv_entry(
                sched, recv_buf + offsets[rank_idx], recv_counts[rank_idx], dtype, rank_idx, comm);
        }
    }

    CCL_FREE(offsets);
    return status;
}

ccl_status_t ccl_coll_build_ring_allgatherv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            size_t send_count,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm) {
    LOG_DEBUG("build ring allgatherv, send_count ", send_count);

    ccl_status_t status = ccl_status_success;
    size_t comm_size, rank;
    size_t dtype_size = dtype.size();
    size_t idx = 0;
    size_t src, dst;

    comm_size = comm->size();
    rank = comm->rank();

    size_t* offsets = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    offsets[0] = 0;
    for (size_t rank_idx = 1; rank_idx < comm_size; ++rank_idx) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * dtype_size;
    }

    if (send_buf != recv_buf) {
        entry_factory::make_entry<copy_entry>(
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

    for (idx = 0; idx < (comm_size - 1); idx++) {
        send_block_idx = block_idx;
        recv_block_idx = (comm_size + block_idx - 1) % comm_size;
        send_block_count = recv_counts[send_block_idx];
        recv_block_count = recv_counts[recv_block_idx];
        send_block_offset = offsets[send_block_idx];
        recv_block_offset = offsets[recv_block_idx];
        sbuf = recv_buf + send_block_offset;
        rbuf = recv_buf + recv_block_offset;

        entry_factory::make_entry<send_entry>(sched, sbuf, send_block_count, dtype, dst, comm);
        entry_factory::make_entry<recv_entry>(sched, rbuf, recv_block_count, dtype, src, comm);
        sched->add_barrier();

        block_idx = (comm_size + block_idx - 1) % comm_size; // move left
    }

    CCL_FREE(offsets);
    return status;
}
