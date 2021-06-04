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

#include "coll/algorithms/algorithms_enum.hpp"
#include "common/comm/comm.hpp"
#include "coll/coll_param.hpp"
#include "common/stream/stream.hpp"
#include "common/datatype/datatype.hpp"
#include "common/utils/buffer.hpp"

#include "coll/coll_common_attributes.hpp"

#include "internal_types.hpp"

class ccl_sched;
class ccl_request;

ccl::status ccl_coll_build_allgatherv(ccl_sched* sched,
                                      ccl_buffer send_buf,
                                      size_t send_count,
                                      ccl_buffer recv_buf,
                                      const size_t* recv_counts,
                                      const ccl_datatype& dtype,
                                      ccl_comm* comm);

ccl::status ccl_coll_build_allreduce(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     ccl_buffer recv_buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     ccl::reduction reduction,
                                     ccl_comm* comm);

ccl::status ccl_coll_build_alltoall(ccl_sched* sched,
                                    ccl_buffer send_buf,
                                    ccl_buffer recv_buf,
                                    size_t count,
                                    const ccl_datatype& dtype,
                                    ccl_comm* comm);

ccl::status ccl_coll_build_alltoallv(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     const size_t* send_counts,
                                     ccl_buffer recv_buf,
                                     const size_t* recv_counts,
                                     const ccl_datatype& dtype,
                                     ccl_comm* comm);

ccl::status ccl_coll_build_barrier(ccl_sched* sched, ccl_comm* comm);

ccl::status ccl_coll_build_bcast(ccl_sched* sched,
                                 ccl_buffer buf,
                                 size_t count,
                                 const ccl_datatype& dtype,
                                 int root,
                                 ccl_comm* comm);

ccl::status ccl_coll_build_reduce(ccl_sched* sched,
                                  ccl_buffer send_buf,
                                  ccl_buffer recv_buf,
                                  size_t count,
                                  const ccl_datatype& dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  ccl_comm* comm);

ccl::status ccl_coll_build_reduce_scatter(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction reduction,
                                          ccl_comm* comm,
                                          bool from_allreduce = false);

ccl::status ccl_coll_build_sparse_allreduce(ccl_sched* sched,
                                            ccl_buffer send_ind_buf,
                                            size_t send_ind_count,
                                            ccl_buffer send_val_buf,
                                            size_t send_val_count,
                                            void** recv_ind_buf,
                                            size_t* recv_ind_count,
                                            void** recv_val_buf,
                                            size_t* recv_val_count,
                                            const ccl_datatype& index_dtype,
                                            const ccl_datatype& value_dtype,
                                            ccl::reduction reduction,
                                            ccl_comm* comm);

ccl_request* ccl_allgatherv_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const size_t* recv_counts,
                                 ccl::datatype dtype,
                                 const ccl_coll_attr& attr,
                                 ccl_comm* comm,
                                 const ccl_stream* stream,
                                 const std::vector<ccl::event>& deps,
                                 bool skip_validation = false);

ccl_request* ccl_allreduce_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps,
                                bool skip_validation = false);
template <class gpu_device_type>
ccl_request* ccl_allreduce_gpu_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    ccl::reduction reduction,
                                    const ccl_coll_attr& attr,
                                    ccl_comm* comm,
                                    const ccl_stream* stream,
                                    const std::vector<ccl::event>& deps);

ccl_request* ccl_alltoall_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl::datatype dtype,
                               const ccl_coll_attr& attr,
                               ccl_comm* comm,
                               const ccl_stream* stream,
                               const std::vector<ccl::event>& deps,
                               bool skip_validation = false);

ccl_request* ccl_alltoallv_impl(const void* send_buf,
                                const size_t* send_counts,
                                void* recv_buf,
                                const size_t* recv_counts,
                                ccl::datatype dtype,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps,
                                bool skip_validation = false);

void ccl_barrier_impl(ccl_comm* comm,
                      const ccl_stream* stream,
                      const std::vector<ccl::event>& deps,
                      bool skip_validation = false);

ccl_request* ccl_broadcast_impl(void* buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps,
                                bool skip_validation = false);

ccl_request* ccl_reduce_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::datatype dtype,
                             ccl::reduction reduction,
                             int root,
                             const ccl_coll_attr& attr,
                             ccl_comm* comm,
                             const ccl_stream* stream,
                             const std::vector<ccl::event>& deps,
                             bool skip_validation = false);

ccl_request* ccl_reduce_scatter_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl_coll_attr& attr,
                                     ccl_comm* comm,
                                     const ccl_stream* stream,
                                     const std::vector<ccl::event>& deps,
                                     bool skip_validation = false);

ccl_request* ccl_sparse_allreduce_impl(const void* send_ind_buf,
                                       size_t send_ind_count,
                                       const void* send_val_buf,
                                       size_t send_val_count,
                                       void* recv_ind_buf,
                                       size_t recv_ind_count,
                                       void* recv_val_buf,
                                       size_t recv_val_count,
                                       ccl::datatype index_dtype,
                                       ccl::datatype dtype,
                                       ccl::reduction reduction,
                                       const ccl_coll_attr& attr,
                                       ccl_comm* comm,
                                       const ccl_stream* stream,
                                       const std::vector<ccl::event>& deps,
                                       bool skip_validation = false);
