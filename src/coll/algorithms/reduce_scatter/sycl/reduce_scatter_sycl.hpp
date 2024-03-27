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
#pragma once

#define SYCL_REDUCE_SCATTER_FUNCTIONS(MSGSIZE) \
    void init_reduce_scatter_##MSGSIZE(ccl::datatype dtype, \
                                       sycl::queue &queue, \
                                       ccl_comm *comm, \
                                       ccl_stream *stream, \
                                       uint32_t rank_in, \
                                       uint32_t world_in); \
    ccl::event run_reduce_scatter_##MSGSIZE( \
        ccl::datatype dtype, sycl::queue q, const void *send_buf, void *rev_buf, size_t recv_count, bool &done);

SYCL_REDUCE_SCATTER_FUNCTIONS(small)
SYCL_REDUCE_SCATTER_FUNCTIONS(medium)
SYCL_REDUCE_SCATTER_FUNCTIONS(large)

namespace ccl {
namespace v1 {

ccl::event reduce_scatter_sycl(sycl::queue q,
                               const void *send_buf,
                               void *recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               reduction reduction,
                               const ccl::communicator &comm,
                               const stream &op_stream,
                               bool &done);

}
} // namespace ccl
