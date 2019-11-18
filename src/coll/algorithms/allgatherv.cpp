/*
 Copyright 2016-2019 Intel Corporation
 
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

ccl_status_t ccl_coll_build_direct_allgatherv(ccl_sched* sched,
                                              ccl_buffer send_buf, size_t send_count,
                                              ccl_buffer recv_buf, const size_t* recv_counts,
                                              ccl_datatype_internal_t dtype)
{
    LOG_DEBUG("build direct allgatherv");

    entry_factory::make_entry<allgatherv_entry>(sched, send_buf, send_count, recv_buf, recv_counts, dtype);
    return ccl_status_success;
}

ccl_status_t ccl_coll_build_naive_allgatherv(ccl_sched* sched,
                                             ccl_buffer send_buf, size_t send_count,
                                             ccl_buffer recv_buf, const size_t* recv_counts,
                                             ccl_datatype_internal_t dtype)
{
    LOG_DEBUG("build naive allgatherv");

    size_t comm_size     = sched->coll_param.comm->size();
    size_t this_rank     = sched->coll_param.comm->rank();
    size_t dtype_size    = ccl_datatype_get_size(dtype);
    size_t* offsets      = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    ccl_status_t status = ccl_status_success;

    offsets[0] = 0;
    for (size_t rank_idx = 1; rank_idx < comm_size; ++rank_idx)
    {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * dtype_size;
    }

    if (send_buf != recv_buf)
    {
        // out-of-place case
        entry_factory::make_entry<copy_entry>(sched, send_buf, recv_buf + offsets[this_rank],
                                              send_count, dtype);
    }

    for (size_t rank_idx = 0; rank_idx < sched->coll_param.comm->size(); ++rank_idx)
    {
        if (rank_idx != this_rank)
        {
            // send own buffer to other ranks
            entry_factory::make_entry<send_entry>(sched, recv_buf + offsets[this_rank],
                                                  send_count, dtype, rank_idx);
            // recv other's rank buffer
            entry_factory::make_entry<recv_entry>(sched, recv_buf + offsets[rank_idx],
                                                  recv_counts[rank_idx], dtype, rank_idx);
        }
    }

    CCL_FREE(offsets);
    return status;
}
