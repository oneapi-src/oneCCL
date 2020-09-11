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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_aliases.hpp"

#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "oneapi/ccl/ccl_event_attr_ids.hpp"
#include "oneapi/ccl/ccl_event_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "common/request/request.hpp"

#include "common/comm/comm.hpp"
#include "coll/coll.hpp"
#include "coll/coll_common_attributes.hpp"
#include "coll/ccl_allgather_op_attr.hpp"
#include "coll/ccl_allreduce_op_attr.hpp"
#include "coll/ccl_alltoall_op_attr.hpp"
#include "coll/ccl_alltoallv_op_attr.hpp"
#include "coll/ccl_barrier_attr.hpp"
#include "coll/ccl_bcast_op_attr.hpp"
#include "coll/ccl_reduce_op_attr.hpp"
#include "coll/ccl_reduce_scatter_op_attr.hpp"
#include "coll/ccl_sparse_allreduce_op_attr.hpp"

#include "common/global/global.hpp"

#define COPY_COMMON_OP_ATTRS(from, to) \
    to->prologue_fn = from.get<ccl::operation_attr_id::prologue_fn>().get(); \
    to->epilogue_fn = from.get<ccl::operation_attr_id::epilogue_fn>().get(); \
    to->priority = from.get<ccl::operation_attr_id::priority>(); \
    to->synchronous = from.get<ccl::operation_attr_id::synchronous>(); \
    to->to_cache = from.get<ccl::operation_attr_id::to_cache>(); \
    to->match_id = from.get<ccl::operation_attr_id::match_id>();

ccl_coll_attr::ccl_coll_attr(const ccl::allgatherv_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);

    vector_buf = attr.get<ccl::allgatherv_attr_id::vector_buf>();
}

ccl_coll_attr::ccl_coll_attr(const ccl::allreduce_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);

    reduction_fn = attr.get<ccl::allreduce_attr_id::reduction_fn>().get();
}

ccl_coll_attr::ccl_coll_attr(const ccl::alltoall_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::alltoallv_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::barrier_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::broadcast_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);
}

ccl_coll_attr::ccl_coll_attr(const ccl::reduce_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);

    reduction_fn = attr.get<ccl::reduce_attr_id::reduction_fn>().get();
}

ccl_coll_attr::ccl_coll_attr(const ccl::reduce_scatter_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);

    reduction_fn = attr.get<ccl::reduce_scatter_attr_id::reduction_fn>().get();
}

ccl_coll_attr::ccl_coll_attr(const ccl::sparse_allreduce_attr& attr) {
    COPY_COMMON_OP_ATTRS(attr, this);

    sparse_allreduce_completion_fn = attr.get<ccl::sparse_allreduce_attr_id::completion_fn>().get();
    sparse_allreduce_alloc_fn = attr.get<ccl::sparse_allreduce_attr_id::alloc_fn>().get();
    sparse_allreduce_fn_ctx = attr.get<ccl::sparse_allreduce_attr_id::fn_ctx>();
    sparse_coalesce_mode = attr.get<ccl::sparse_allreduce_attr_id::coalesce_mode>();
}

ccl_request* ccl_allgatherv_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const size_t* recv_counts,
                                 ccl::datatype dtype,
                                 const ccl_coll_attr& attr,
                                 ccl_comm* comm,
                                 const ccl_stream* stream) {
    return nullptr;
}

ccl_request* ccl_allreduce_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream) {
    return nullptr;
}

ccl_request* ccl_alltoall_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl::datatype dtype,
                               const ccl_coll_attr& attr,
                               ccl_comm* comm,
                               const ccl_stream* stream) {
    return nullptr;
}

ccl_request* ccl_alltoallv_impl(const void* send_buf,
                                const size_t* send_counts,
                                void* recv_buf,
                                const size_t* recv_counts,
                                ccl::datatype dtype,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream) {
    return nullptr;
}

void ccl_barrier_impl(ccl_comm* comm, const ccl_stream* stream) {}

ccl_request* ccl_broadcast_impl(void* buf,
                                size_t count,
                                ccl::datatype dtype,
                                size_t root,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream) {
    return nullptr;
}

ccl_request* ccl_reduce_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::datatype dtype,
                             ccl::reduction reduction,
                             size_t root,
                             const ccl_coll_attr& attr,
                             ccl_comm* comm,
                             const ccl_stream* stream) {
    return nullptr;
}

ccl_request* ccl_reduce_scatter_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl_coll_attr& attr,
                                     ccl_comm* comm,
                                     const ccl_stream* stream) {
    return nullptr;
}

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
                                       const ccl_stream* stream) {
    return nullptr;
}
