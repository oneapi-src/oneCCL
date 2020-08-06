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

#include "ccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

ccl_status_t CCL_API ccl_init(void);
ccl_status_t CCL_API ccl_get_version(ccl_version_t* version);
ccl_status_t CCL_API ccl_finalize(void);
#ifdef MULTI_GPU_SUPPORT
ccl_status_t CCL_API ccl_set_device_comm_attr(ccl_device_comm_attr_t* comm_attr,
                                              unsigned long attribute,
                                              ...);
#endif
ccl_status_t CCL_API ccl_set_resize_fn(ccl_resize_fn_t callback);

/* Collective API */

ccl_status_t CCL_API ccl_allgatherv(const void* send_buf,
                                    size_t send_count,
                                    void* recv_buf,
                                    const size_t* recv_counts,
                                    ccl_datatype_t dtype,
                                    const ccl_coll_attr_t* attr,
                                    ccl_comm_t comm,
                                    ccl_stream_t stream,
                                    ccl_request_t* req);

ccl_status_t CCL_API ccl_allreduce(const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl_datatype_t dtype,
                                   ccl_reduction_t reduction,
                                   const ccl_coll_attr_t* attr,
                                   ccl_comm_t comm,
                                   ccl_stream_t stream,
                                   ccl_request_t* req);

ccl_status_t CCL_API ccl_alltoall(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl_datatype_t dtype,
                                  const ccl_coll_attr_t* attr,
                                  ccl_comm_t comm,
                                  ccl_stream_t stream,
                                  ccl_request_t* req);

ccl_status_t CCL_API ccl_alltoallv(const void* send_buf,
                                   const size_t* send_counts,
                                   void* recv_buf,
                                   const size_t* recv_counts,
                                   ccl_datatype_t dtype,
                                   const ccl_coll_attr_t* attr,
                                   ccl_comm_t comm,
                                   ccl_stream_t stream,
                                   ccl_request_t* req);

ccl_status_t CCL_API ccl_barrier(ccl_comm_t comm, ccl_stream_t stream);

ccl_status_t CCL_API ccl_bcast(void* buf,
                               size_t count,
                               ccl_datatype_t dtype,
                               size_t root,
                               const ccl_coll_attr_t* attr,
                               ccl_comm_t comm,
                               ccl_stream_t stream,
                               ccl_request_t* req);

ccl_status_t CCL_API ccl_reduce(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl_datatype_t dtype,
                                ccl_reduction_t reduction,
                                size_t root,
                                const ccl_coll_attr_t* attr,
                                ccl_comm_t comm,
                                ccl_stream_t stream,
                                ccl_request_t* req);

/* WARNING: ccl_sparse_allreduce is currently considered experimental, so the API may change! */
ccl_status_t CCL_API ccl_sparse_allreduce(const void* send_ind_buf,
                                          size_t send_ind_count,
                                          const void* send_val_buf,
                                          size_t send_val_count,
                                          void* recv_ind_buf,
                                          size_t recv_ind_count,
                                          void* recv_val_buf,
                                          size_t recv_val_count,
                                          ccl_datatype_t index_dtype,
                                          ccl_datatype_t dtype,
                                          ccl_reduction_t reduction,
                                          const ccl_coll_attr_t* attr,
                                          ccl_comm_t comm,
                                          ccl_stream_t stream,
                                          ccl_request_t* req);

/* Completion API */

ccl_status_t CCL_API ccl_wait(ccl_request_t req);
ccl_status_t CCL_API ccl_test(ccl_request_t req, int* is_completed);

/* Communicator API */

ccl_status_t CCL_API ccl_comm_create(ccl_comm_t* comm, const ccl_comm_attr_t* attr);
ccl_status_t CCL_API ccl_comm_free(ccl_comm_t comm);

ccl_status_t CCL_API ccl_get_comm_rank(ccl_comm_t comm, size_t* rank);
ccl_status_t CCL_API ccl_get_comm_size(ccl_comm_t comm, size_t* size);

/* Datatype API */

ccl_status_t CCL_API ccl_datatype_create(ccl_datatype_t* type, const ccl_datatype_attr_t* attr);
ccl_status_t CCL_API ccl_datatype_free(ccl_datatype_t type);

ccl_status_t CCL_API ccl_get_datatype_size(ccl_datatype_t type, size_t* size);

/* Stream API */

ccl_status_t CCL_API ccl_stream_create(ccl_stream_type_t type,
                                       void* native_stream,
                                       ccl_stream_t* ccl_stream);
ccl_status_t CCL_API ccl_stream_free(ccl_stream_t stream);

#ifdef __cplusplus
} /*extern C */
#endif
