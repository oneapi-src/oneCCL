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
#include "sched/entry/factory/entry_factory.hpp"

/* An implementation of Rabenseifner's reduce algorithm (see
   http://www.hlrs.de/mpi/myreduce.html).

   This algorithm implements the reduce in two steps: first a
   reduce-scatter, followed by a gather to the root. A
   recursive-halving algorithm (beginning with processes that are
   distance 1 apart) is used for the reduce-scatter, and a binomial tree
   algorithm is used for the gather. The non-power-of-two case is
   handled by dropping to the nearest lower power-of-two: the first
   few odd-numbered processes send their data to their left neighbors
   (rank-1), and the reduce-scatter happens among the remaining
   power-of-two processes. If the root is one of the excluded
   processes, then after the reduce-scatter, rank 0 sends its result to
   the root and exits; the root now acts as rank 0 in the binomial tree
   algorithm for gather.

   For the power-of-two case, the cost for the reduce-scatter is
   lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma. The cost for the
   gather to root is lgp.alpha + n.((p-1)/p).beta. Therefore, the
   total cost is:
   Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma

   For the non-power-of-two case, assuming the root is not one of the
   odd-numbered processes that get excluded in the reduce-scatter,
   Cost = (2.floor(lgp)+1).alpha + (2.((p-1)/p) + 1).n.beta +
           n.(1+(p-1)/p).gamma
*/

ccl_status_t ccl_coll_build_direct_reduce(ccl_sched *sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl_reduction_t reduction,
                                          size_t root,
                                          ccl_comm* comm)
{
    LOG_DEBUG("build direct reduce");

    entry_factory::make_entry<reduce_entry>(sched, send_buf, recv_buf, count,
                                            dtype, reduction, root, comm);
    return ccl_status_success;
}

ccl_status_t ccl_coll_build_rabenseifner_reduce(ccl_sched *sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t count,
                                                const ccl_datatype& dtype,
                                                ccl_reduction_t reduction,
                                                size_t root,
                                                ccl_comm* comm)
{
    LOG_DEBUG("build Rabenseifner's reduce");

    ccl_status_t status = ccl_status_success;

    int i, j, comm_size, rank, local_root, pof2;
    int rem, dst, new_rank, new_dst, mask, send_idx, recv_idx, last_idx;
    int send_cnt, recv_cnt, newroot, newdst_tree_root, newroot_tree_root;
    int *cnts = NULL, *disps = NULL;
    size_t dtype_size = dtype.size();
    local_root = static_cast<int>(root);

    comm_size = comm->size();
    rank = comm->rank();

    ccl_buffer tmp_buf = sched->alloc_buffer(count * dtype_size);

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = comm->pof2();
    CCL_THROW_IF_NOT(count >= static_cast<size_t>(pof2), "count ", count, ", pof2 ", pof2);
    rem = comm_size - pof2;

    /* If I'm not the root, then my recv_buf may not be valid, therefore
     * I have to allocate a temporary one */
    if (rank != local_root) {
        recv_buf = sched->alloc_buffer(count * dtype_size);
    }

    if ((rank != local_root) || (send_buf != recv_buf))
        entry_factory::make_entry<copy_entry>(sched, send_buf, recv_buf, count, dtype);

    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send their data to
     * (rank-1). These odd-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two.
     *
     * Note that in MPI_Allreduce we have the even-numbered processes
     * send data to odd-numbered processes. That is better for
     * non-commutative operations because it doesn't require a
     * buffer copy. However, for MPI_Reduce, the most common case
     * is commutative operations with root=0. Therefore we want
     * even-numbered processes to participate the computation for
     * the root=0 case, in order to avoid an extra send-to-root
     * communication after the reduce-scatter. In MPI_Allreduce it
     * doesn't matter because all processes must get the result. */

    if (rank < 2 * rem) {
        if (rank % 2 != 0) {    /* odd */
            entry_factory::make_entry<send_entry>(sched, recv_buf, count, dtype, rank - 1, comm);
            sched->add_barrier();

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            new_rank = CCL_INVALID_PROC_IDX;
        } else {        /* even */
            entry_factory::make_entry<recv_entry>(sched, tmp_buf, count, dtype, rank + 1, comm);
            sched->add_barrier();

            /* do the reduction on received data. */
            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            entry_factory::make_entry<reduce_local_entry>(sched, tmp_buf, count,
                                                          recv_buf, nullptr, dtype, reduction);
            sched->add_barrier();

            /* change the rank */
            new_rank = rank / 2;
        }
    } else {    /* rank >= 2*rem */
        new_rank = rank - rem;
    }

    /* for the reduce-scatter, calculate the count that
     * each process receives and the displacement within
     * the buffer */

    /* We allocate these arrays on all processes, even if new_rank=-1,
     * because if root is one of the excluded processes, we will
     * need them on the root later on below. */
    cnts = static_cast<int*>(CCL_MALLOC(pof2 * sizeof(int), "counts"));
    disps = static_cast<int*>(CCL_MALLOC(pof2 * sizeof(int), "displacements"));

    last_idx = send_idx = 0;    /* suppress spurious compiler warnings */

    if (new_rank != CCL_INVALID_PROC_IDX) {
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
            new_dst = new_rank ^ mask;
            /* find real rank of dest */
            dst = (new_dst < rem) ? new_dst * 2 : new_dst + rem;

            send_cnt = recv_cnt = 0;
            if (new_rank < new_dst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            /* Send data from recv_buf. Recv into tmp_buf */
            entry_factory::make_entry<send_entry>(sched, (recv_buf + disps[send_idx] * dtype_size),
                                                  send_cnt, dtype, dst, comm);
            /* sendrecv, no barrier here */
            entry_factory::make_entry<recv_entry>(sched, (tmp_buf + disps[recv_idx] * dtype_size),
                                                  recv_cnt, dtype, dst, comm);
            sched->add_barrier();


            /* tmp_buf contains data received in this step.
             * recv_buf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            entry_factory::make_entry<reduce_local_entry>(sched, (tmp_buf + disps[recv_idx] * dtype_size), recv_cnt,
                                                          (recv_buf + disps[recv_idx] * dtype_size),
                                                          nullptr, dtype, reduction);
            sched->add_barrier();

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the gather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }
    }

    /* now do the gather to root */

    /* Is root one of the processes that was excluded from the
     * computation above? If so, send data from new_rank=0 to
     * the root and have root take on the role of new_rank = 0 */

    if (local_root < 2 * rem) {
        if (local_root % 2 != 0) {
            if (rank == local_root) { /* recv */
                /* initialize the arrays that weren't initialized */
                for (i = 0; i < (pof2 - 1); i++)
                    cnts[i] = count / pof2;
                cnts[pof2 - 1] = count - (count / pof2) * (pof2 - 1);

                disps[0] = 0;
                for (i = 1; i < pof2; i++)
                    disps[i] = disps[i - 1] + cnts[i - 1];

                entry_factory::make_entry<recv_entry>(sched, recv_buf, cnts[0], dtype, 0, comm);
                sched->add_barrier();

                new_rank = 0;
                send_idx = 0;
                last_idx = 2;
            } else if (new_rank == 0) {  /* send */
                entry_factory::make_entry<send_entry>(sched, recv_buf, cnts[0], dtype, local_root, comm);
                sched->add_barrier();

                new_rank = CCL_INVALID_PROC_IDX;
            }
            newroot = 0;
        } else
            newroot = local_root / 2;
    } else {
        newroot = local_root - rem;
    }

    if (new_rank != CCL_INVALID_PROC_IDX) {
        j = 0;
        mask = 0x1;
        while (mask < pof2) {
            mask <<= 1;
            j++;
        }
        mask >>= 1;
        j--;
        while (mask > 0) {
            new_dst = new_rank ^ mask;

            /* find real rank of dest */
            dst = (new_dst < rem) ? new_dst * 2 : new_dst + rem;
            /* if root is playing the role of new_dst=0, adjust for
             * it */
            if ((new_dst == 0) && (local_root < 2 * rem) && (local_root % 2 != 0))
                dst = local_root;

            /* if the root of new_dst's half of the tree is the
             * same as the root of newroot's half of the tree, send to
             * new_dst and exit, else receive from new_dst. */

            newdst_tree_root = new_dst >> j;
            newdst_tree_root <<= j;

            newroot_tree_root = newroot >> j;
            newroot_tree_root <<= j;

            send_cnt = recv_cnt = 0;
            if (new_rank < new_dst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            if (newdst_tree_root == newroot_tree_root) {
                /* send and exit */
                /* Send data from recv_buf. Recv into tmp_buf */
                entry_factory::make_entry<send_entry>(sched, (recv_buf + disps[send_idx] * dtype_size),
                                                      send_cnt, dtype, dst, comm);
                sched->add_barrier();
                break;
            } else {
                /* recv and continue */
                entry_factory::make_entry<recv_entry>(sched, (recv_buf + disps[recv_idx] * dtype_size),
                                                      recv_cnt, dtype, dst, comm);
                sched->add_barrier();
            }

            if (new_rank > new_dst)
                send_idx = recv_idx;

            mask >>= 1;
            j--;
        }
    }

    CCL_FREE(cnts);
    CCL_FREE(disps);

    return status;
}

ccl_status_t ccl_coll_build_binomial_reduce(ccl_sched *sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl_reduction_t reduction,
                                            size_t root,
                                            ccl_comm* comm)
{
    LOG_DEBUG("build binomial reduce");

    ccl_status_t status = ccl_status_success;

    int comm_size, rank, local_root;
    int mask, relrank, source, lroot;

    if (count == 0)
        return status;

    comm_size = comm->size();
    rank = comm->rank();
    local_root = static_cast<int>(root);

    /* Create a temporary buffer */
    size_t dtype_size = dtype.size();
    ccl_buffer tmp_buf = sched->alloc_buffer(count * dtype_size);

    /* If I'm not the root, then my recv_buf may not be valid, therefore
     * I have to allocate a temporary one */
    if (rank != local_root) {
        recv_buf = sched->alloc_buffer(count * dtype_size);
    }

    if ((rank != local_root) || (send_buf != recv_buf)) {
        entry_factory::make_entry<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    /* This code is from MPICH-1. */

    /* Here's the algorithm.  Relative to the root, look at the bit pattern in
     * my rank.  Starting from the right (lsb), if the bit is 1, send to
     * the node with that bit zero and exit; if the bit is 0, receive from the
     * node with that bit set and combine (as long as that node is within the
     * group)
     *
     * Note that by receiving with source selection, we guarentee that we get
     * the same bits with the same input.  If we allowed the parent to receive
     * the children in any order, then timing differences could cause different
     * results (roundoff error, over/underflows in some cases, etc).
     *
     * Because of the way these are ordered, if root is 0, then this is correct
     * for both commutative and non-commutitive operations.  If root is not
     * 0, then for non-commutitive, we use a root of zero and then send
     * the result to the root.  To see this, note that the ordering is
     * mask = 1: (ab)(cd)(ef)(gh)            (odds send to evens)
     * mask = 2: ((ab)(cd))((ef)(gh))        (3,6 send to 0,4)
     * mask = 4: (((ab)(cd))((ef)(gh)))      (4 sends to 0)
     *
     * Comments on buffering.
     * If the dtype is not contiguous, we still need to pass contiguous
     * data to the user routine.
     * In this case, we should make a copy of the data in some format,
     * and send/operate on that.
     *
     * In general, we can't use MPI_PACK, because the alignment of that
     * is rather vague, and the data may not be re-usable.  What we actually
     * need is a "squeeze" operation that removes the skips.
     */
    mask = 0x1;
    lroot = local_root;
    relrank = (rank - lroot + comm_size) % comm_size;

    while (mask < comm_size) {
        /* Receive */
        if ((mask & relrank) == 0) {
            source = (relrank | mask);
            if (source < comm_size) {
                source = (source + lroot) % comm_size;

                entry_factory::make_entry<recv_entry>(sched, tmp_buf, count, dtype, source, comm);
                sched->add_barrier();

                entry_factory::make_entry<reduce_local_entry>(sched, tmp_buf, count,
                                                              recv_buf, nullptr, dtype, reduction);
                sched->add_barrier();

            }
        } else {
            /* I've received all that I'm going to.  Send my result to
             * my parent */
            source = ((relrank & (~mask)) + lroot) % comm_size;
            entry_factory::make_entry<send_entry>(sched, recv_buf, count, dtype, source, comm);
            sched->add_barrier();
            break;
        }
        mask <<= 1;
    }

    return status;
}
