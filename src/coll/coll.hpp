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

#include "common/comm/comm.hpp"
#include "common/stream/stream.hpp"
#include "common/datatype/datatype.hpp"
#include "common/utils/buffer.hpp"
#include "common/global/global.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
typedef cl::sycl::buffer<char, 1> ccl_sycl_buffer_t;

template<class native_type>
using ccl_sycl_typed_buffer_t = cl::sycl::buffer<native_type, 1>;

/* ordering should be aligned with ccl_datatype_t */
using ccl_sycle_buffer_one_dim_types =
      std::tuple<ccl_sycl_typed_buffer_t<char>,
                 ccl_sycl_typed_buffer_t<int>,
                 ccl_sycl_typed_buffer_t<float>,
                 ccl_sycl_typed_buffer_t<float>,
                 ccl_sycl_typed_buffer_t<double>,
                 ccl_sycl_typed_buffer_t<int64_t>,
                 ccl_sycl_typed_buffer_t<uint64_t>>;
#endif /* CCL_ENABLE_SYCL */

#define CCL_INVALID_PROC_IDX (-1)

class ccl_sched;
class ccl_request;

struct ccl_coll_attr
{
    ccl_coll_attr() = default;
    ccl_coll_attr(const ccl_coll_attr&) = default;
    ccl_coll_attr& operator= (const ccl_coll_attr&) = default;
    ccl_coll_attr(const ccl_coll_attr_t* attr);
    ccl_coll_attr& operator= (const ccl_coll_attr_t* attr);

    ccl_coll_attr(ccl_coll_attr&&) = delete;
    ccl_coll_attr& operator= (ccl_coll_attr&&) = delete;

    ccl_prologue_fn_t prologue_fn = nullptr;
    ccl_epilogue_fn_t epilogue_fn = nullptr;
    ccl_reduction_fn_t reduction_fn = nullptr;
    ccl_sparse_allreduce_completion_fn_t sparse_allreduce_completion_fn = nullptr;
    const void* sparse_allreduce_completion_ctx = nullptr;
    size_t priority = 0;
    int synchronous = 0;
    int to_cache = 0;
    int vector_buf = 0;
    std::string match_id{};
};

struct ccl_coll_sparse_param
{
    const void* send_ind_buf;
    size_t send_ind_count;
    const void* send_val_buf;
    size_t send_val_count;
    void* recv_ind_buf;
    size_t recv_ind_count;
    void* recv_val_buf;
    size_t recv_val_count;
    ccl_datatype itype;
};

struct ccl_coll_param
{
    ccl_coll_type ctype;
    void* buf;
    const void* send_buf;
    void* recv_buf;
    size_t count;
    size_t send_count;
    const size_t* send_counts;
    const size_t* recv_counts;
    ccl_datatype dtype;
    ccl_reduction_t reduction;
    size_t root;
    const ccl_stream* stream;
    ccl_comm* comm;
    ccl_coll_sparse_param sparse_param;

#ifdef CCL_ENABLE_SYCL
    ccl_sycl_buffer_t* sycl_send_buf;
    ccl_sycl_buffer_t* sycl_recv_buf;
    ccl_sycl_buffer_t* sycl_buf;
#endif /* CCL_ENABLE_SYCL */

};

/*
    explicitly split coll_param and coll_param_copy
    to separate coll_param structure which is used for interaction between different modules
    and coll_param_copy which is used as storage for user options
*/
struct ccl_coll_param_copy
{
    /* keep copy of user options which can be invalidated after collective call */

    std::vector<void*> ag_recv_bufs;
    std::vector<size_t> ag_recv_counts;

    std::vector<size_t> a2av_send_counts;
    std::vector<size_t> a2av_recv_counts;
};

ccl_status_t ccl_coll_build_allgatherv(ccl_sched* sched,
                                       ccl_buffer send_buf,
                                       size_t send_count,
                                       ccl_buffer recv_buf,
                                       const size_t* recv_counts,
                                       const ccl_datatype& dtype,
                                       ccl_comm* comm);

ccl_status_t ccl_coll_build_allreduce(ccl_sched* sched,
                                      ccl_buffer send_buf,
                                      ccl_buffer recv_buf,
                                      size_t count,
                                      const ccl_datatype& dtype,
                                      ccl_reduction_t reduction,
                                      ccl_comm* comm);

ccl_status_t ccl_coll_build_alltoall(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     ccl_buffer recv_buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     ccl_comm* comm);

ccl_status_t ccl_coll_build_alltoallv(ccl_sched* sched,
                                      ccl_buffer send_buf,
                                      const size_t* send_counts,
                                      ccl_buffer recv_buf,
                                      const size_t* recv_counts,
                                      const ccl_datatype& dtype,
                                      ccl_comm* comm);

ccl_status_t ccl_coll_build_barrier(ccl_sched* sched, ccl_comm* comm);

ccl_status_t ccl_coll_build_bcast(ccl_sched* sched,
                                  ccl_buffer buf,
                                  size_t count,
                                  const ccl_datatype& dtype,
                                  size_t root,
                                  ccl_comm* comm);

ccl_status_t ccl_coll_build_reduce(ccl_sched* sched,
                                   ccl_buffer send_buf,
                                   ccl_buffer recv_buf,
                                   size_t count,
                                   const ccl_datatype& dtype,
                                   ccl_reduction_t reduction,
                                   size_t root,
                                   ccl_comm* comm);


ccl_status_t ccl_coll_build_reduce_scatter(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t send_count,
                                           const ccl_datatype& dtype,
                                           ccl_reduction_t reduction,
                                           ccl_comm* comm);

ccl_status_t ccl_coll_build_sparse_allreduce(ccl_sched* sched,
                                             ccl_buffer send_ind_buf, size_t send_ind_count,
                                             ccl_buffer send_val_buf, size_t send_val_count,
                                             void** recv_ind_buf, size_t* recv_ind_count,
                                             void** recv_val_buf, size_t* recv_val_count,
                                             const ccl_datatype& index_dtype,
                                             const ccl_datatype& value_dtype,
                                             ccl_reduction_t reduction,
                                             ccl_comm* comm);

ccl_request* ccl_allgatherv_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const size_t* recv_counts,
                                 ccl_datatype_t dtype,
                                 const ccl_coll_attr_t* attr,
                                 ccl_comm* comm,
                                 const ccl_stream* stream);

ccl_request* ccl_allreduce_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl_datatype_t dtype,
                                ccl_reduction_t reduction,
                                const ccl_coll_attr_t* attr,
                                ccl_comm* comm,
                                const ccl_stream* stream);
template<class gpu_device_type>
ccl_request* ccl_allreduce_gpu_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl_datatype_t dtype,
                                ccl_reduction_t reduction,
                                const ccl_coll_attr_t* attr,
                                ccl_comm* comm,
                                const ccl_stream* stream);

ccl_request* ccl_alltoall_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl_datatype_t dtype,
                               const ccl_coll_attr_t* attr,
                               ccl_comm* comm,
                               const ccl_stream* stream);

ccl_request* ccl_alltoallv_impl(const void* send_buf,
                                const size_t* send_counts,
                                void* recv_buf,
                                const size_t* recv_counts,
                                ccl_datatype_t dtype,
                                const ccl_coll_attr_t* attr,
                                ccl_comm* comm,
                                const ccl_stream* stream);

void ccl_barrier_impl(ccl_comm* comm,
                      const ccl_stream* stream);

ccl_request* ccl_bcast_impl(void* buf,
                            size_t count,
                            ccl_datatype_t dtype,
                            size_t root,
                            const ccl_coll_attr_t* attr,
                            ccl_comm* comm,
                            const ccl_stream* stream);

ccl_request* ccl_reduce_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl_datatype_t dtype,
                             ccl_reduction_t reduction,
                             size_t root,
                             const ccl_coll_attr_t* attr,
                             ccl_comm* comm,
                             const ccl_stream* stream);

ccl_request* ccl_sparse_allreduce_impl(const void* send_ind_buf, size_t send_ind_count,
                                       const void* send_val_buf, size_t send_val_count,
                                       void* recv_ind_buf, size_t recv_ind_count,
                                       void* recv_val_buf, size_t recv_val_count,
                                       ccl_datatype_t index_dtype, ccl_datatype_t dtype,
                                       ccl_reduction_t reduction, const ccl_coll_attr_t* attr,
                                       ccl_comm* comm, const ccl_stream* stream);
