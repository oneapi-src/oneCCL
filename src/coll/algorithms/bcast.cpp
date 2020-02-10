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

#define MIN(a,b) std::min(a,b)

ccl_status_t ccl_coll_build_direct_bcast(ccl_sched *sched, ccl_buffer buf, size_t count,
                                         ccl_datatype_internal_t dtype, size_t root, ccl_comm* comm)
{
    LOG_DEBUG("build direct bcast");

    entry_factory::make_entry<bcast_entry>(sched, buf, count,
                                           dtype, root, comm);
    return ccl_status_success;
}

ccl_status_t ccl_coll_build_naive_bcast(ccl_sched *sched, ccl_buffer buf, size_t count,
                                        ccl_datatype_internal_t dtype, size_t root, ccl_comm* comm)
{
    LOG_DEBUG("build naive bcast");

    ccl_status_t status = ccl_status_success;

    size_t rank = comm->rank();
    size_t comm_size = comm->size();
    size_t idx;

    if (comm_size == 1)
        goto fn_exit;

    if (rank == root)
    {
        for (idx = 0; idx < comm_size; idx++)
        {
            if (idx != rank)
            {
                entry_factory::make_entry<send_entry>(sched, buf, count, dtype, idx, comm);
            }
        }
    }
    else
    {
        entry_factory::make_entry<recv_entry>(sched, buf, count, dtype, root, comm);
    }

  fn_exit:
    return status;
}

ccl_status_t ccl_coll_build_scatter_for_bcast(ccl_sched *sched,
                                              ccl_buffer tmp_buf,
                                              size_t root,
                                              size_t nbytes,
                                              ccl_comm* comm)
{
    LOG_DEBUG("build scatter_for_bcast");

    ccl_status_t status = ccl_status_success;
    int rank, local_root, comm_size, src, dst;
    int relative_rank, mask;
    int scatter_size, curr_size, recv_size, send_size;

    comm_size = comm->size();
    rank = comm->rank();
    local_root = static_cast<int>(root);
    relative_rank = (rank >= local_root) ? rank - local_root : rank - local_root + comm_size;

    /* The scatter algorithm divides the buffer into nprocs pieces and
     * scatters them among the processes. Root gets the first piece,
     * root+1 gets the second piece, and so forth. Uses the same
     * binomial tree algorithm as above. Ceiling division is used to
     * compute the size of each piece. This means some processes may
     * not get any data. For example if bufsize = 97 and nprocs = 16,
     * ranks 15 and 16 will get 0 data. On each process, the scattered
     * data is stored at the same offset in the buffer as it is on the
     * root process. */

    scatter_size = (nbytes + comm_size - 1) / comm_size;        /* ceiling division */
    curr_size = (rank == local_root) ? nbytes : 0;    /* root starts with all the data */

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {

            src = rank - mask;
            if (src < 0)
                src += comm_size;

            /* compute the exact recv_size to avoid writing this NBC
             * in callback style */
            recv_size = nbytes - (relative_rank * scatter_size);
            if (recv_size < 0)
                recv_size = 0;

            curr_size = recv_size;

            if (recv_size > 0) {
                entry_factory::make_entry<recv_entry>(sched, tmp_buf + relative_rank * scatter_size,
                                                      recv_size, ccl_dtype_internal_char, src, comm);
                sched->add_barrier();
            }
            break;
        }
        mask <<= 1;
    }

    /* This process is responsible for all processes that have bits
     * set from the LSB upto (but not including) mask.  Because of the
     * "not including", we start by shifting mask back down one. */

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {

            send_size = curr_size - scatter_size * mask;

            /* mask is also the size of this process's subtree */

            if (send_size > 0) {
                dst = rank + mask;
                if (dst >= comm_size)
                    dst -= comm_size;

                entry_factory::make_entry<send_entry>(sched, tmp_buf + scatter_size * (relative_rank + mask),
                                                      send_size, ccl_dtype_internal_char, dst, comm);
                sched->add_barrier();
                curr_size -= send_size;
            }
        }
        mask >>= 1;
    }

    return status;
}

ccl_status_t ccl_coll_build_scatter_ring_allgather_bcast(ccl_sched *sched,
                                                         ccl_buffer buf,
                                                         size_t count,
                                                         ccl_datatype_internal_t dtype,
                                                         size_t root,
                                                         ccl_comm* comm)
{
    LOG_DEBUG("build scatter_ring_allgather bcast");

    ccl_status_t status = ccl_status_success;

    int comm_size, rank, nbytes;
    int scatter_size, curr_size;
    int i, j, jnext, left, right;
    size_t dtype_size = ccl_datatype_get_size(dtype);

    comm_size = comm->size();
    rank = comm->rank();

    ccl_buffer tmp_buf(buf);

    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;

    nbytes = dtype_size * count;

    CCL_CALL(ccl_coll_build_scatter_for_bcast(sched, tmp_buf, root, nbytes, comm));

    /* this is the block size used for the scatter operation */
    scatter_size = (nbytes + comm_size - 1) / comm_size;        /* ceiling division */

    /* curr_size is the amount of data that this process now has stored in
     * buffer at byte offset (rank*scatter_size) */
    curr_size = MIN(scatter_size, (nbytes - (rank * scatter_size)));
    if (curr_size < 0)
        curr_size = 0;

    /* long-message allgather or medium-size but non-power-of-two. use ring algorithm. */

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    j = rank;
    jnext = left;
    for (i = 1; i < comm_size; i++)
    {
        int left_count, right_count, left_disp, right_disp, rel_j, rel_jnext;

        rel_j = (j - root + comm_size) % comm_size;
        rel_jnext = (jnext - root + comm_size) % comm_size;
        left_count = MIN(scatter_size, (nbytes - rel_jnext * scatter_size));
        if (left_count < 0)
            left_count = 0;
        left_disp = rel_jnext * scatter_size;
        right_count = MIN(scatter_size, (nbytes - rel_j * scatter_size));
        if (right_count < 0)
            right_count = 0;
        right_disp = rel_j * scatter_size;
        entry_factory::make_entry<send_entry>(sched, tmp_buf + right_disp,
                                              right_count, ccl_dtype_internal_char, right, comm);
        /* sendrecv, no barrier here */
        entry_factory::make_entry<recv_entry>(sched, tmp_buf + left_disp,
                                              left_count, ccl_dtype_internal_char, left, comm);
        sched->add_barrier();

        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
    }

  fn_exit:
    return status;
}
