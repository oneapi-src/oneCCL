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
#include "coll/algorithms/algorithm_utils.hpp"
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

ccl::status ccl_coll_build_direct_allreduce(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl::reduction op,
                                            ccl_comm* comm) {
    LOG_DEBUG("build direct allreduce");

    entry_factory::create<allreduce_entry>(sched, send_buf, recv_buf, count, dtype, op, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_build_rabenseifner_allreduce(ccl_sched* sched,
                                                  ccl_buffer send_buf,
                                                  ccl_buffer recv_buf,
                                                  size_t count,
                                                  const ccl_datatype& dtype,
                                                  ccl::reduction op,
                                                  ccl_comm* comm) {
    LOG_DEBUG("build Rabenseifner's allreduce");
    CCL_ASSERT(sched != nullptr, "empty sched");

    ccl::status status = ccl::status::success;
    int comm_size, rank, newrank, pof2, rem;
    int i, send_idx, recv_idx, last_idx, mask, newdst, dst, send_cnt, recv_cnt;
    int *cnts = NULL, *disps = NULL;
    size_t dtype_size = dtype.size();

    comm_size = comm->size();
    rank = comm->rank();
    ccl_buffer tmp_buf = sched->alloc_buffer({ count * dtype_size, send_buf });

    /* copy local data into recv_buf */

    if (send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    if (comm_size == 1)
        return status;

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm->pof2();

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* even */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
            sched->add_barrier();

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = CCL_INVALID_PROC_IDX;
        }
        else { /* odd */
            entry_factory::create<recv_entry>(sched, tmp_buf, count, dtype, rank - 1, comm);
            sched->add_barrier();

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, count, recv_buf, nullptr, dtype, op);
            sched->add_barrier();

            /* change the rank */
            newrank = rank / 2;
        }
    }
    else /* rank >= 2*rem */
        newrank = rank - rem;

    if (newrank != CCL_INVALID_PROC_IDX) {
        /* for the reduce-scatter, calculate the count that
         * each process receives and the displacement within
         * the buffer */
        cnts = static_cast<int*>(CCL_MALLOC(pof2 * sizeof(int), "counts"));
        disps = static_cast<int*>(CCL_MALLOC(pof2 * sizeof(int), "displacements"));

        /* the cnts calculations assume this */
        CCL_ASSERT(count >= static_cast<size_t>(pof2), "count ", count, ", pof2 ", pof2);

        for (i = 0; i < (pof2 - 1); i++)
            cnts[i] = count / pof2;
        cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

        disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }
            else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            int can_use_recv_reduce = 0;
            ccl_buffer buf1 = (recv_buf + disps[send_idx] * dtype_size);
            ccl_buffer buf2 = (recv_buf + disps[recv_idx] * dtype_size);
            if (buf1 != buf2 &&
                ((buf1 + send_cnt * dtype_size <= buf2) || (buf2 + recv_cnt * dtype_size <= buf1)))
                can_use_recv_reduce = 1;

            CCL_ASSERT(can_use_recv_reduce);

            if (can_use_recv_reduce) {
                entry_factory::create<recv_reduce_entry>(sched,
                                                         (recv_buf + disps[recv_idx] * dtype_size),
                                                         recv_cnt,
                                                         dtype,
                                                         op,
                                                         dst,
                                                         comm);
                entry_factory::create<send_entry>(
                    sched, (recv_buf + disps[send_idx] * dtype_size), send_cnt, dtype, dst, comm);
                sched->add_barrier();
            }

            else {
                /* Send data from recv_buf. Recv into tmp_buf */
                entry_factory::create<recv_entry>(
                    sched, (tmp_buf + disps[recv_idx] * dtype_size), recv_cnt, dtype, dst, comm);
                /* sendrecv, no barrier here */
                entry_factory::create<send_entry>(
                    sched, (recv_buf + disps[send_idx] * dtype_size), send_cnt, dtype, dst, comm);
                sched->add_barrier();

                /* tmp_buf contains data received in this step.
                 * recv_buf contains data accumulated so far */

                /* This algorithm is used only for predefined ops
                 * and predefined ops are always commutative. */
                entry_factory::create<reduce_local_entry>(sched,
                                                          (tmp_buf + disps[recv_idx] * dtype_size),
                                                          recv_cnt,
                                                          (recv_buf + disps[recv_idx] * dtype_size),
                                                          nullptr,
                                                          dtype,
                                                          op);
                sched->add_barrier();
            }

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }
            else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            entry_factory::create<recv_entry>(
                sched, (recv_buf + disps[recv_idx] * dtype_size), recv_cnt, dtype, dst, comm);
            /* sendrecv, no barrier here */
            entry_factory::create<send_entry>(
                sched, (recv_buf + disps[send_idx] * dtype_size), send_cnt, dtype, dst, comm);
            sched->add_barrier();

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }

    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) { /* odd */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank - 1, comm);
        }
        else { /* even */
            entry_factory::create<recv_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
        }
    }

    CCL_FREE(cnts);
    CCL_FREE(disps);

    return status;
}

ccl::status ccl_coll_build_nreduce_allreduce(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             ccl_buffer recv_buf,
                                             size_t count,
                                             const ccl_datatype& dtype,
                                             ccl::reduction op,
                                             ccl_comm* comm) {
    LOG_DEBUG("build nreduce allreduce");

    ccl::status status = ccl::status::success;
    int comm_size = comm->size();
    int comm_rank = comm->rank();
    std::vector<size_t> elem_counts(comm_size);
    std::vector<size_t> elem_offsets(comm_size);
    size_t dtype_size = dtype.size();
    bool is_inplace = (send_buf == recv_buf);

    if (comm_size == 1) {
        if (!is_inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        }
        return status;
    }

    int use_buffering = ccl::global_data::env().allreduce_nreduce_buffering;

    size_t segment_size = 2 * 1024 * 1024;
    if (ccl::global_data::env().allreduce_nreduce_segment_size != CCL_ENV_SIZET_NOT_SPECIFIED) {
        segment_size = ccl::global_data::env().allreduce_nreduce_segment_size;
    }

    std::vector<size_t> segment_sizes;
    ccl_get_segment_sizes(dtype_size, count, segment_size, segment_sizes);

    size_t tmp_buf_size = *segment_sizes.rbegin() * comm_size * dtype_size;
    ccl_buffer tmp_buf = sched->alloc_buffer({ tmp_buf_size, send_buf });

    size_t seg_offset = 0;

    for (size_t seg_idx = 0; seg_idx < segment_sizes.size(); seg_idx++) {
        size_t seg_size = segment_sizes[seg_idx];

        ccl_buffer seg_send_buf = send_buf + seg_offset;
        ccl_buffer seg_recv_buf = recv_buf + seg_offset;
        ccl_buffer seg_tmp_buf = tmp_buf;

        seg_offset += seg_size * dtype_size;

        // calculate counts and offsets for each rank
        size_t common_buffer_count = seg_size / comm_size;
        for (int idx = 0; idx < comm_size; idx++) {
            elem_counts[idx] = common_buffer_count;
            elem_offsets[idx] = idx * elem_counts[idx] * dtype_size;
        }
        elem_counts[comm_size - 1] += seg_size % comm_size;

        size_t elem_count = elem_counts[comm_rank];

        ccl_buffer reduce_buf;
        if (use_buffering) {
            reduce_buf = seg_tmp_buf + elem_count * comm_rank * dtype_size;
        }
        else {
            reduce_buf = seg_recv_buf + elem_offsets[comm_rank];
        }

        if (!is_inplace || use_buffering) {
            entry_factory::create<copy_entry>(sched,
                                              seg_send_buf + elem_offsets[comm_rank],
                                              reduce_buf,
                                              elem_counts[comm_rank],
                                              dtype);
            sched->add_barrier();
        }

        // reduce-scatter
        for (int idx = 1; idx < comm_size; idx++) {
            // send part of buffer to other rank
            int dst = (comm_rank - idx + comm_size) % comm_size;
            entry_factory::create<send_entry>(
                sched, seg_send_buf + elem_offsets[dst], elem_counts[dst], dtype, dst, comm);

            // recv part of buffer from other rank and perform reduce
            int src = (comm_rank + idx) % comm_size;
            entry_factory::create<recv_reduce_entry>(sched,
                                                     reduce_buf,
                                                     elem_count,
                                                     dtype,
                                                     op,
                                                     src,
                                                     comm,
                                                     seg_tmp_buf + elem_count * src * dtype_size);
        }

        sched->add_barrier();

        // allgatherv
        if (use_buffering) {
            copy_attr attr;
            attr.direction = copy_direction::h2h;
            attr.use_nontemporal = true;

            // copy own result from tmp to recv buffer
            entry_factory::create<copy_entry>(
                sched, reduce_buf, seg_recv_buf + elem_offsets[comm_rank], elem_count, dtype, attr);
            sched->add_barrier();

            for (int idx = 1; idx < comm_size; idx++) {
                int dst = (comm_rank + idx) % comm_size;
                int src = (comm_rank - idx + comm_size) % comm_size;

                // send own result to other ranks
                entry_factory::create<send_entry>(
                    sched, reduce_buf, elem_counts[comm_rank], dtype, dst, comm);

                // recv other's rank result into tmp buffer and copy to recv buffer
                entry_factory::create<recv_copy_entry>(sched,
                                                       seg_tmp_buf + elem_offsets[src],
                                                       seg_recv_buf + elem_offsets[src],
                                                       elem_counts[src] * dtype_size,
                                                       src,
                                                       comm,
                                                       attr);
            }
        }
        else {
            CCL_CALL(ccl_coll_build_naive_allgatherv(sched,
                                                     seg_recv_buf,
                                                     elem_counts[comm_rank],
                                                     seg_recv_buf,
                                                     elem_counts.data(),
                                                     dtype,
                                                     comm));
        }

        sched->add_barrier();
    }

    return status;
}

ccl::status ccl_coll_build_ring_allreduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction op,
                                          ccl_comm* comm) {
    int inplace = (send_buf == recv_buf) ? 1 : 0;
    LOG_DEBUG("build ring allreduce ", inplace ? "in-place" : "out-of-place");

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    ccl::status status = ccl::status::success;

    ccl_coll_build_ring_reduce_scatter(sched, send_buf, recv_buf, count, dtype, op, comm);

    sched->add_barrier();

    int comm_size = comm->size();
    size_t main_block_count = count / comm_size;
    size_t last_block_count = main_block_count + count % comm_size;
    std::vector<size_t> recv_counts(comm_size, main_block_count);
    if (count % comm_size) {
        recv_counts[comm_size - 1] = last_block_count;
    }

    ccl_coll_build_ring_allgatherv(
        sched, recv_buf, recv_counts[comm->rank()], recv_buf, recv_counts.data(), dtype, comm);

    sched->add_barrier();

    return status;
}

ccl::status ccl_coll_build_recursive_doubling_allreduce(ccl_sched* sched,
                                                        ccl_buffer send_buf,
                                                        ccl_buffer recv_buf,
                                                        size_t count,
                                                        const ccl_datatype& dtype,
                                                        ccl::reduction op,
                                                        ccl_comm* comm) {
    LOG_DEBUG("build recursive_doubling allreduce");

    ccl::status status = ccl::status::success;

    int pof2, rem, comm_size, rank;
    int newrank, mask, newdst, dst;

    comm_size = comm->size();
    rank = comm->rank();

    size_t dtype_size = dtype.size();

    ccl_buffer tmp_buf = sched->alloc_buffer({ count * dtype_size, send_buf });

    /* copy local data into recv_buf */
    if (send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    if (comm_size == 1)
        return status;

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm->pof2();
    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { /* even */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
            sched->add_barrier();

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        }
        else { /* odd */
            entry_factory::create<recv_entry>(sched, tmp_buf, count, dtype, rank - 1, comm);
            sched->add_barrier();

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */

            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, count, recv_buf, nullptr, dtype, op);
            sched->add_barrier();

            /* change the rank */
            newrank = rank / 2;
        }
    }
    else /* rank >= 2*rem */
        newrank = rank - rem;

    if (newrank != -1) {
        mask = 0x1;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            /* Send the most current data, which is in recv_buf. Recv
             * into tmp_buf */
            entry_factory::create<recv_entry>(sched, tmp_buf, count, dtype, dst, comm);
            /* sendrecv, no barrier here */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, dst, comm);
            sched->add_barrier();

            /* tmp_buf contains data received in this step.
             * recv_buf contains data accumulated so far */
            entry_factory::create<reduce_local_entry>(
                sched, tmp_buf, count, recv_buf, nullptr, dtype, op);
            sched->add_barrier();

            mask <<= 1;
        }
    }

    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2) { /* odd */
            entry_factory::create<send_entry>(sched, recv_buf, count, dtype, rank - 1, comm);
        }
        else { /* even */
            entry_factory::create<recv_entry>(sched, recv_buf, count, dtype, rank + 1, comm);
        }
        sched->add_barrier();
    }

    return status;
}

static void ccl_allreduce_2d_add_allreduce_allgather(ccl_sched* sched,
                                                     ccl_buffer send_buf,
                                                     ccl_buffer recv_buf,
                                                     size_t count,
                                                     const ccl_datatype& dtype,
                                                     ccl::reduction op,
                                                     ccl_comm* comm,
                                                     ccl_comm* first_dim_comm,
                                                     ccl_comm* second_dim_comm,
                                                     size_t chunk_idx,
                                                     size_t chunk_count) {
    size_t dtype_size = dtype.size();
    size_t main_chunk_size = count / chunk_count;
    size_t last_chunk_size = main_chunk_size + count % chunk_count;
    size_t cnt = (chunk_idx == (chunk_count - 1)) ? last_chunk_size : main_chunk_size;
    ccl_buffer rbuf = recv_buf + chunk_idx * main_chunk_size * dtype_size;

    size_t main_block_count = cnt / first_dim_comm->size();
    size_t last_block_count = main_block_count + cnt % first_dim_comm->size();
    size_t ar_count = (first_dim_comm->rank() == (first_dim_comm->size() - 1)) ? last_block_count
                                                                               : main_block_count;

    if (ar_count) {
        // TODO: add second level selection to distinguish high and low level algorithms
        ccl_buffer ar_buf = rbuf + first_dim_comm->rank() * main_block_count * dtype_size;
        ccl_coll_build_nreduce_allreduce(
            sched, ar_buf, ar_buf, ar_count, dtype, op, second_dim_comm);
        sched->add_barrier();
    }

    std::vector<size_t> ag_recv_counts(first_dim_comm->size(), main_block_count);
    ag_recv_counts[first_dim_comm->size() - 1] = last_block_count;

    // TODO: skip direct algo since it may be started
    // with different order on different ranks
    sched->hint_algo.allgatherv = ccl_coll_allgatherv_ring;
    ccl_coll_build_allgatherv(
        sched, rbuf, ar_count, rbuf, ag_recv_counts.data(), dtype, first_dim_comm);
    sched->hint_algo.allgatherv = ccl_coll_allgatherv_undefined;
}

static void ccl_allreduce_2d_add_reduce_scatter_allreduce_allgather(ccl_sched* sched,
                                                                    ccl_buffer send_buf,
                                                                    ccl_buffer recv_buf,
                                                                    size_t count,
                                                                    const ccl_datatype& dtype,
                                                                    ccl::reduction op,
                                                                    ccl_comm* comm,
                                                                    ccl_comm* first_dim_comm,
                                                                    ccl_comm* second_dim_comm,
                                                                    size_t chunk_idx,
                                                                    size_t chunk_count) {
    size_t dtype_size = dtype.size();
    size_t main_chunk_size = count / chunk_count;
    size_t last_chunk_size = main_chunk_size + count % chunk_count;
    size_t cnt = (chunk_idx == (chunk_count - 1)) ? last_chunk_size : main_chunk_size;
    ccl_buffer sbuf = send_buf + chunk_idx * main_chunk_size * dtype_size;
    ccl_buffer rbuf = recv_buf + chunk_idx * main_chunk_size * dtype_size;

    ccl_coll_build_reduce_scatter(sched, sbuf, rbuf, cnt, dtype, op, first_dim_comm, true);
    sched->add_barrier();

    if (chunk_idx == (chunk_count - 1) || (chunk_count == 1)) {
        ccl_allreduce_2d_add_allreduce_allgather(sched,
                                                 send_buf,
                                                 recv_buf,
                                                 count,
                                                 dtype,
                                                 op,
                                                 comm,
                                                 first_dim_comm,
                                                 second_dim_comm,
                                                 chunk_idx,
                                                 chunk_count);
    }
    else {
        entry_factory::create<subsched_entry>(
            sched,
            chunk_idx,
            [send_buf,
             recv_buf,
             count,
             &dtype,
             op,
             comm,
             first_dim_comm,
             second_dim_comm,
             chunk_idx,
             chunk_count](ccl_sched* s) {
                ccl_allreduce_2d_add_allreduce_allgather(s,
                                                         send_buf,
                                                         recv_buf,
                                                         count,
                                                         dtype,
                                                         op,
                                                         comm,
                                                         first_dim_comm,
                                                         second_dim_comm,
                                                         chunk_idx,
                                                         chunk_count);
            },
            "AR_AG");

        entry_factory::create<subsched_entry>(
            sched,
            chunk_idx + 1,
            [send_buf,
             recv_buf,
             count,
             &dtype,
             op,
             comm,
             first_dim_comm,
             second_dim_comm,
             chunk_idx,
             chunk_count](ccl_sched* s) {
                ccl_allreduce_2d_add_reduce_scatter_allreduce_allgather(s,
                                                                        send_buf,
                                                                        recv_buf,
                                                                        count,
                                                                        dtype,
                                                                        op,
                                                                        comm,
                                                                        first_dim_comm,
                                                                        second_dim_comm,
                                                                        chunk_idx + 1,
                                                                        chunk_count);
            },
            "RS_AR_AG");
    }
}

ccl::status ccl_coll_build_2d_allreduce(ccl_sched* sched,
                                        ccl_buffer send_buf,
                                        ccl_buffer recv_buf,
                                        size_t count,
                                        const ccl_datatype& dtype,
                                        ccl::reduction op,
                                        ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    size_t chunk_count = ccl::global_data::env().allreduce_2d_chunk_count;

    bool switch_dims = ccl::global_data::env().allreduce_2d_switch_dims;
    ccl_comm* first_dim_comm =
        (switch_dims) ? comm->get_r2r_comm().get() : comm->get_node_comm().get();
    ccl_comm* second_dim_comm =
        (switch_dims) ? comm->get_node_comm().get() : comm->get_r2r_comm().get();

    LOG_DEBUG("build 2d allreduce: chunk_count: ",
              chunk_count,
              ", switch_dims: ",
              switch_dims,
              ", comm: ",
              comm->to_string(),
              ", 1st dim comm: ",
              first_dim_comm->to_string(),
              ", 2nd dim comm: ",
              second_dim_comm->to_string());

    ccl_allreduce_2d_add_reduce_scatter_allreduce_allgather(sched,
                                                            send_buf,
                                                            recv_buf,
                                                            count,
                                                            dtype,
                                                            op,
                                                            comm,
                                                            first_dim_comm,
                                                            second_dim_comm,
                                                            0 /* chunk_idx */,
                                                            chunk_count);

    return status;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_allreduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction op,
                                          ccl_comm* comm) {
    LOG_DEBUG("build topo allreduce");

    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
        { recv_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 1
    };

    size_t ipc_event_count{};
    size_t max_ipc_event_count{ 6 };
    ze_event_pool_handle_t ipc_event_pool{};
    if (ccl::global_data::env().enable_ze_barrier) {
        ipc_event_pool = sched->get_memory().ipc_event_pool_manager.create(max_ipc_event_count);
        in_buffers.push_back({ static_cast<void*>(ipc_event_pool), ccl::ze::ipc_mem_type::pool });
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    int comm_size = comm->size();
    int even_comm_size = even_comm->size();
    int node_comm_size = node_comm->size();

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    bool is_single_node = topo_manager.is_single_node;
    bool is_single_card = topo_manager.is_single_card;
    bool is_multi_card = (even_comm_size > 1);

    size_t recv_buf_idx = 1;

    int skip_rank = ccl_comm::invalid_rank;
    if (ccl::global_data::env().enable_kernel_1s_ipc_wa && is_single_card) {
        skip_rank = ccl::global_data::env().kernel_1s_lead;
    }

    ccl::add_handle_exchange(
        sched, node_comm, in_buffers, skip_rank, ipc_event_pool, ipc_event_count++);

    CCL_THROW_IF_NOT(comm_size % 2 == 0, "unexpected comm_size ", comm_size);
    CCL_THROW_IF_NOT(node_comm_size % 2 == 0, "unexpected node_comm_size ", node_comm_size);

    sched->try_enable_ze_single_list();
    std::vector<ze_event_handle_t> wait_events;

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

    else if (pair_comm->rank() != ccl::global_data::env().kernel_1s_lead) {
        ccl::add_comm_barrier(sched, pair_comm, ipc_event_pool, ipc_event_count++);
        CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                         "unexpected ipc_event_count ",
                         ipc_event_count,
                         ", expected max ",
                         max_ipc_event_count);
        return ccl::status::success;
    }

    size_t main_block_count = base_count / even_comm_size;
    size_t block_count = main_block_count;
    if (even_comm->rank() == even_comm_size - 1) {
        block_count += base_count % even_comm_size;
    }
    size_t even_comm_offset_bytes = main_block_count * even_comm->rank() * dtype.size();
    ccl_buffer pair_comm_send_buf = send_buf + pair_comm_offset_bytes;
    ccl_buffer pair_comm_recv_buf = recv_buf + pair_comm_offset_bytes;
    ccl_buffer even_comm_recv_buf = recv_buf + pair_comm_offset_bytes + even_comm_offset_bytes;

    LOG_DEBUG("rank: ",
              pair_comm->rank(),
              ", count: ",
              base_count,
              ", pair_comm_offset: ",
              pair_comm_offset);
    if (is_single_card) {
        entry_factory::create<ze_onesided_allreduce_entry>(sched,
                                                           pair_comm_send_buf,
                                                           pair_comm_recv_buf,
                                                           base_count,
                                                           dtype,
                                                           op,
                                                           pair_comm,
                                                           wait_events,
                                                           pair_comm_offset);
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
        wait_events.push_back(entry->entry_event);
    }
    sched->add_barrier();

    bool is_read_allgatherv = ccl::global_data::env().allgatherv_topo_read;
    if (is_multi_card) {
        ccl::add_comm_barrier(sched, even_comm, wait_events, ipc_event_pool, ipc_event_count++);
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
                                                                       recv_buf_idx,
                                                                       recv_buf_idx,
                                                                       pair_comm_offset);
            wait_events.push_back(entry->entry_event);
            ccl::add_comm_barrier(sched, even_comm, wait_events, ipc_event_pool, ipc_event_count++);
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
                                                                            recv_buf_idx,
                                                                            pair_comm_offset);
            wait_events.push_back(entry->entry_event);
            ccl::add_comm_barrier(sched, even_comm, wait_events, ipc_event_pool, ipc_event_count++);
        }
    }

    ccl_coll_entry_param coll_param{ .ctype = ccl_coll_allreduce,
                                     .send_buf = even_comm_recv_buf,
                                     .recv_buf = even_comm_recv_buf,
                                     .count = block_count,
                                     .dtype = dtype,
                                     .reduction = op,
                                     .comm = r2r_comm };

    ccl::add_scaleout(sched, coll_param, is_single_node, wait_events);

    if (is_multi_card && (!is_single_node || is_read_allgatherv)) {
        LOG_DEBUG("topo/scale_up/inter: use ze_a2a_allgatherv");
        // for multinode with allgatherv_read, use a comm_barrier to make sure all
        // r2r scaleout within even_comm has finished so that remote reads are valid
        if (!is_single_node && is_read_allgatherv) {
            ccl::add_comm_barrier(sched, even_comm, wait_events, ipc_event_pool, ipc_event_count++);
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
                                                                    recv_buf_idx,
                                                                    pair_comm_offset);
        wait_events.push_back(entry->entry_event);
        ccl::add_comm_barrier(sched, even_comm, wait_events, ipc_event_pool, ipc_event_count++);
    }

    if (!is_single_card && pair_comm->size() > 1) {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_bcast");
        int peer_rank = (pair_comm->rank() + 1) % pair_comm->size();
        auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                          recv_buf,
                                                          ccl_buffer(),
                                                          base_count,
                                                          dtype,
                                                          copy_attr(peer_rank,
                                                                    recv_buf_idx,
                                                                    copy_direction::t2t,
                                                                    pair_comm,
                                                                    pair_comm_offset,
                                                                    pair_comm_offset),
                                                          wait_events);
        wait_events.push_back(entry->entry_event);
        sched->add_barrier();
    }

    ccl::add_comm_barrier(sched, pair_comm, ipc_event_pool, ipc_event_count++);

    CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                     "unexpected ipc_event_count ",
                     ipc_event_count,
                     ", expected max ",
                     max_ipc_event_count);

    return ccl::status::success;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
