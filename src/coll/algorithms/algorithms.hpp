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

#include "sched/master_sched.hpp"
#include "sched/sched.hpp"

#include <map>
#include <type_traits>

#define CCL_UNDEFINED_ALGO_ID (-1)

ccl_status_t ccl_coll_build_naive_bcast(ccl_sched* sched,
                                        ccl_buffer buf,
                                        size_t count,
                                        const ccl_datatype& dtype,
                                        size_t root,
                                        ccl_comm* comm);

ccl_status_t ccl_coll_build_scatter_ring_allgather_bcast(ccl_sched* sched,
                                                         ccl_buffer buf,
                                                         size_t count,
                                                         const ccl_datatype& dtype,
                                                         size_t root,
                                                         ccl_comm* comm);

ccl_status_t ccl_coll_build_dissemination_barrier(ccl_sched* sched, ccl_comm* comm);

ccl_status_t ccl_coll_build_rabenseifner_reduce(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t count,
                                                const ccl_datatype& dtype,
                                                ccl_reduction_t reduction,
                                                size_t root,
                                                ccl_comm* comm);

ccl_status_t ccl_coll_build_rabenseifner_allreduce(ccl_sched* sched,
                                                   ccl_buffer send_buf,
                                                   ccl_buffer recv_buf,
                                                   size_t count,
                                                   const ccl_datatype& dtype,
                                                   ccl_reduction_t reduction,
                                                   ccl_comm* comm);

ccl_status_t ccl_coll_build_binomial_reduce(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl_reduction_t reduction,
                                            size_t root,
                                            ccl_comm* comm);

ccl_status_t ccl_coll_build_ring_allreduce(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           ccl_reduction_t reduction,
                                           ccl_comm* comm);

ccl_status_t ccl_coll_build_ring_rma_allreduce(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t count,
                                               const ccl_datatype& dtype,
                                               ccl_reduction_t reduction,
                                               ccl_comm* comm);

ccl_status_t ccl_coll_build_recursive_doubling_allreduce(ccl_sched* sched,
                                                         ccl_buffer send_buf,
                                                         ccl_buffer recv_buf,
                                                         size_t count,
                                                         const ccl_datatype& dtype,
                                                         ccl_reduction_t reduction,
                                                         ccl_comm* comm);

ccl_status_t ccl_coll_build_starlike_allreduce(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t count,
                                               const ccl_datatype& dtype,
                                               ccl_reduction_t reduction,
                                               ccl_comm* comm);

ccl_status_t ccl_coll_build_naive_allgatherv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             size_t send_count,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm);

template <typename i_type, typename v_type>
ccl_status_t ccl_coll_build_sparse_allreduce_ring(ccl_sched* sched,
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
                                                  ccl_reduction_t reduction,
                                                  ccl_comm* comm);

template <typename i_type, typename v_type>
ccl_status_t ccl_coll_build_sparse_allreduce_mask(ccl_sched* sched,
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
                                                  ccl_reduction_t reduction,
                                                  ccl_comm* comm);

template <typename i_type, typename v_type>
ccl_status_t ccl_coll_build_sparse_allreduce_3_allgatherv(ccl_sched* sched,
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
                                                          ccl_reduction_t reduction,
                                                          ccl_comm* comm);

class ccl_double_tree;
ccl_status_t ccl_coll_build_double_tree_op(ccl_sched* sched,
                                           ccl_coll_type coll_type,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           ccl_reduction_t reduction,
                                           const ccl_double_tree& dtree,
                                           ccl_comm* comm);

ccl_status_t ccl_coll_build_ring_reduce_scatter(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t send_count,
                                                const ccl_datatype& dtype,
                                                ccl_reduction_t reduction,
                                                ccl_comm* comm);

ccl_status_t ccl_coll_build_ring_allgatherv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            size_t send_count,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm);

ccl_status_t ccl_coll_build_naive_alltoallv(ccl_master_sched* main_sched,
                                            std::vector<ccl_sched*>& scheds,
                                            const ccl_coll_param& coll_param);

ccl_status_t ccl_coll_build_scatter_alltoallv(ccl_master_sched* main_sched,
                                              std::vector<ccl_sched*>& scheds,
                                              const ccl_coll_param& coll_param);

ccl_status_t ccl_coll_build_scatter_barrier_alltoallv(ccl_master_sched* main_sched,
                                                      std::vector<ccl_sched*>& scheds,
                                                      const ccl_coll_param& coll_param);

/* direct algorithms - i.e. direct mapping on collective API from transport level */

ccl_status_t ccl_coll_build_direct_barrier(ccl_sched* sched, ccl_comm* comm);

ccl_status_t ccl_coll_build_direct_reduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl_reduction_t reduction,
                                          size_t root,
                                          ccl_comm* comm);

ccl_status_t ccl_coll_build_direct_allgatherv(ccl_sched* sched,
                                              ccl_buffer send_buf,
                                              size_t send_count,
                                              ccl_buffer recv_buf,
                                              const size_t* recv_counts,
                                              const ccl_datatype& dtype,
                                              ccl_comm* comm);

ccl_status_t ccl_coll_build_direct_allreduce(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             ccl_buffer recv_buf,
                                             size_t count,
                                             const ccl_datatype& dtype,
                                             ccl_reduction_t reduction,
                                             ccl_comm* comm);

ccl_status_t ccl_coll_build_direct_alltoall(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm);

ccl_status_t ccl_coll_build_direct_alltoallv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             const size_t* send_counts,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm);

ccl_status_t ccl_coll_build_direct_bcast(ccl_sched* sched,
                                         ccl_buffer buf,
                                         size_t count,
                                         const ccl_datatype& dtype,
                                         size_t root,
                                         ccl_comm* comm);
