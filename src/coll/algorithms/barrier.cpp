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

ccl_status_t ccl_coll_build_direct_barrier(ccl_sched *sched, ccl_comm* comm)
{
    LOG_DEBUG("build direct barrier");

    entry_factory::make_entry<barrier_entry>(sched, comm);
    return ccl_status_success;
}

ccl_status_t ccl_coll_build_dissemination_barrier(ccl_sched *sched, ccl_comm* comm)
{
    LOG_DEBUG("build dissemination barrier");

    ccl_status_t status = ccl_status_success;
    int size, rank, src, dst, mask;
    size = comm->size();
    rank = comm->rank();

    if (size == 1)
        return status;

    mask = 0x1;
    while (mask < size)
    {
        dst = (rank + mask) % size;
        src = (rank - mask + size) % size;
        entry_factory::make_entry<send_entry>(sched, ccl_buffer(), 0, ccl_dtype_internal_char, dst, comm);
        entry_factory::make_entry<recv_entry>(sched, ccl_buffer(), 0, ccl_dtype_internal_char, src, comm);
        sched->add_barrier();
        mask <<= 1;
    }

    return status;
}
