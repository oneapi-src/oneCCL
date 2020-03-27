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

ccl_status_t ccl_coll_build_direct_alltoallv(ccl_sched* sched,
                                             ccl_buffer send_buf, const size_t* send_counts,
                                             ccl_buffer recv_buf, const size_t* recv_counts,
                                             ccl_datatype_internal_t dtype,
                                             ccl_comm* comm)
{
    LOG_DEBUG("build direct alltoallv");

    entry_factory::make_entry<alltoallv_entry>(sched, send_buf, send_counts,
                                               recv_buf, recv_counts, dtype, comm);
    return ccl_status_success;
}
