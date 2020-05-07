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

ccl_status_t ccl_coll_build_direct_alltoall(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm)
{
    LOG_DEBUG("build direct alltoall");

    entry_factory::make_entry<alltoall_entry>(sched, send_buf, recv_buf,
                                              count, dtype, comm);
    return ccl_status_success;
}

//TODO: Will be used instead send\recv from parallelizer
#if 0
ccl_status_t ccl_coll_build_scatter_alltoall(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             ccl_buffer recv_buf,
                                             size_t count,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm)
{
    LOG_DEBUG("build scatter alltoall");

    size_t comm_size     = comm->size();
    size_t this_rank     = comm->rank();
    size_t dtype_size    = dtype.size();
    size_t* offsets      = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    ccl_status_t status = ccl_status_success;
    size_t bblock = comm_size;
    size_t ss, dst;
    offsets[0] = 0;
    for (size_t rank_idx = 1; rank_idx < comm_size; ++rank_idx)
    {
        offsets[rank_idx] = offsets[rank_idx - 1] + count * dtype_size;
    }

    if (send_buf != recv_buf)
    {
        // out-of-place case
        entry_factory::make_entry<copy_entry>(sched, send_buf, recv_buf + offsets[this_rank],
                                              count, dtype);
    }
    for (size_t idx = 0; idx < comm_size; idx += bblock) {
        ss = comm_size - idx < bblock ? comm_size - idx : bblock;
        /* do the communication -- post ss sends and receives: */
        for (size_t i = 0; i < ss; i++) {
            dst = (this_rank + i + idx) % comm_size;
            entry_factory::make_entry<recv_entry>(sched, recv_buf + offsets[dst],
                                                  count, dtype, dst);
        }

        for (size_t i = 0; i < ss; i++) {
            dst = (this_rank - i - idx + comm_size) % comm_size;
            entry_factory::make_entry<send_entry>(sched, send_buf + offsets[this_rank],
                                                  count, dtype, dst);
        }
    }


    CCL_FREE(offsets);
    return status;
}
#endif
