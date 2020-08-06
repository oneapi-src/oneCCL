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
#include "sched/entry/factory/entry_factory.hpp"

/* not exposed in CCL API */

/* behaves like reduce_scatter_block but last block may contain more elements */
ccl_status_t ccl_coll_build_ring_reduce_scatter(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t send_count,
                                                const ccl_datatype& dtype,
                                                ccl_reduction_t op,
                                                ccl_comm* comm) {
    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    ccl_status_t status = ccl_status_success;
    size_t comm_size, rank;
    size_t dtype_size = dtype.size();

    comm_size = comm->size();
    rank = comm->rank();

    size_t src = (comm_size + rank - 1) % comm_size;
    size_t dst = (comm_size + rank + 1) % comm_size;

    size_t count = send_count;
    size_t bytes = count * dtype_size;

    size_t chunk_count = (bytes >= ccl::global_data::env().rs_min_chunk_size &&
                          count >= ccl::global_data::env().rs_chunk_count && count >= comm_size)
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
            entry_factory::make_entry<copy_entry>(sched, send_buf, recv_buf, count, dtype);
            sched->add_barrier();
        }
        return ccl_status_success;
    }

    ccl_buffer tmp_buf;
    if (inplace) {
        tmp_buf = sched->alloc_buffer(count * dtype_size);
    }

    ccl_buffer sbuf, rbuf;
    ccl_buffer reduce_in_buf, reduce_inout_buf;
    ccl_buffer recv_reduce_local_buf, recv_reduce_comm_buf;

    /* start send and recv from such positions to have */
    /* the final reduction result on last iteration in corresponsing block */

    /* block = group of ~ equal-sized chunks */
    size_t block_idx = (rank + comm_size - 1) % comm_size;
    size_t main_block_size = count / comm_size;
    size_t last_block_size = main_block_size + count % comm_size;
    size_t send_block_idx, recv_block_idx;
    size_t send_block_size, recv_block_size;
    size_t send_block_offset, recv_block_offset;

    size_t send_main_chunk_size, send_last_chunk_size;
    size_t recv_main_chunk_size, recv_last_chunk_size;

    size_t send_chunk_size, recv_chunk_size, reduce_chunk_size;
    size_t send_chunk_offset, recv_chunk_offset = 0, reduce_chunk_offset;

    /* if chunk_count > 1 then make reduction with 1 chunk delay to get comp/comp overlapping */
    bool use_prev = (chunk_count > 1) ? true : false;
    size_t prev_recv_chunk_size, prev_recv_chunk_offset;

    ccl_recv_reduce_result_buf_type recv_reduce_result_type;

    for (size_t idx = 0; idx < (comm_size - 1); idx++) {
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

            entry_factory::make_entry<send_entry>(sched, sbuf, send_chunk_size, dtype, dst, comm);

            if (!use_prev) {
                CCL_ASSERT(recv_chunk_size == reduce_chunk_size);
                entry_factory::make_entry<recv_reduce_entry>(sched,
                                                             recv_reduce_local_buf,
                                                             recv_chunk_size,
                                                             nullptr, /* out_cnt */
                                                             dtype,
                                                             op,
                                                             src,
                                                             recv_reduce_comm_buf,
                                                             comm,
                                                             recv_reduce_result_type);
            }
            else {
                entry_factory::make_entry<recv_entry>(
                    sched, rbuf, recv_chunk_size, dtype, src, comm);

                if (idx + chunk_idx > 0) {
                    entry_factory::make_entry<reduce_local_entry>(sched,
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

                    entry_factory::make_entry<reduce_local_entry>(sched,
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
