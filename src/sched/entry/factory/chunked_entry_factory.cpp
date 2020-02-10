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
#include "sched/entry/factory/chunked_entry_factory.hpp"

namespace entry_factory
{
    void make_chunked_send_entry(ccl_sched* sched,
                                 const ccl_buffer buf,
                                 size_t cnt,
                                 ccl_datatype_internal_t dtype,
                                 size_t dst,
                                 ccl_comm* comm)
    {
        CCL_CHUNKED_ENTRY_FUNCTION("send", dtype, cnt,
            make_entry<send_entry>(sched,
                                   buf + chunk_offset,
                                   chunk_size,
                                   dtype,
                                   dst,
                                   comm));
    }

    void make_chunked_recv_entry(ccl_sched* sched,
                                 const ccl_buffer buf,
                                 size_t cnt,
                                 ccl_datatype_internal_t dtype,
                                 size_t src,
                                 ccl_comm* comm)
    {
        CCL_CHUNKED_ENTRY_FUNCTION("recv", dtype, cnt,
            make_entry<recv_entry>(sched,
                                   buf + chunk_offset,
                                   chunk_size,
                                   dtype,
                                   src,
                                   comm));
    }

    void make_chunked_recv_reduce_entry(ccl_sched* sched,
                                        ccl_buffer inout_buf,
                                        size_t cnt,
                                        size_t* out_cnt,
                                        ccl_datatype_internal_t dtype,
                                        ccl_reduction_t reduction_op,
                                        size_t src,
                                        ccl_buffer comm_buf,
                                        ccl_comm* comm,
                                        ccl_recv_reduce_result_buf_type result_buf_type)
    {
        CCL_CHUNKED_ENTRY_FUNCTION("recv_reduce", dtype, cnt,
            make_entry<recv_reduce_entry>(sched,
                                          inout_buf + chunk_offset,
                                          chunk_size,
                                          out_cnt,
                                          dtype,
                                          reduction_op,
                                          src,
                                          comm_buf + chunk_offset,
                                          comm,
                                          result_buf_type));
    }
}
