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

#include "ccl.hpp"
#include "coll/algorithms/algorithms.hpp"
#include "coll/algorithms/sparse.hpp"
#include "coll/coll.hpp"
#include "coll/selection/selection.hpp"
#include "common/global/global.hpp"
#include "common/request/request.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "unordered_coll/unordered_coll.hpp"

ccl_coll_attr::ccl_coll_attr(const ccl_coll_attr_t* attr)
{
    *this = attr ?: global_data.default_coll_attr.get();
}

ccl_coll_attr& ccl_coll_attr::operator= (const ccl_coll_attr_t* attr)
{
    prologue_fn = attr->prologue_fn;
    epilogue_fn = attr->epilogue_fn;
    reduction_fn = attr->reduction_fn;
    priority = attr->priority;
    synchronous = attr->synchronous;
    to_cache = attr->to_cache && attr->match_id && attr->match_id[0];
    match_id = (attr->match_id ? attr->match_id : "");

    if (to_cache != attr->to_cache)
        LOG_INFO("collective caching is requested but no match_id is provided, disable caching");

    return *this;
}

const char* ccl_coll_type_to_str(ccl_coll_type type)
{
    switch (type)
    {
        case ccl_coll_allgatherv:
            return "allgatherv";
        case ccl_coll_allreduce:
            return "allreduce";
        case ccl_coll_alltoall:
            return "alltoall";
        case ccl_coll_barrier:
            return "barrier";
        case ccl_coll_bcast:
            return "bcast";
        case ccl_coll_reduce:
            return "reduce";
        case ccl_coll_sparse_allreduce:
            return "sparse_allreduce";
        case ccl_coll_internal:
            return "internal";
        default:
            CCL_FATAL("unexpected coll_type ", type);
            return "unknown";
    }
}

/* param is not const because param.comm can be updated for unordered colls */
static ccl_request* ccl_coll_create(ccl_coll_param& param,
                                    const ccl_coll_attr& attr)
{
    /* 1. decide whether schedule should be postponed (this includes caching and staring) */
    bool postpone_schedule = false;
    if (env_data.enable_unordered_coll)
    {
        if (!attr.match_id.empty())
        {
            auto comm = global_data.unordered_coll_manager->get_comm(std::string(attr.match_id)).get();
            if (!comm)
            {
                if (attr.synchronous)
                {
                    CCL_THROW("unsupported collective (synchronous && unordered && !communicator)");
                }
                LOG_DEBUG("didn't find comm for match_id ", attr.match_id, ", postpone schedule");
                postpone_schedule = true;
            }
            else
            {
                LOG_DEBUG("found comm ", comm->id(), " for match_id ", attr.match_id);
                param.comm = comm;
            }
        }
        else
        {
            /* use comm provided by user, it is ordered collective */
        }
    }

    /* 2. create or get schedule */
    ccl_master_sched* sched = ccl_master_sched::create(param, attr, postpone_schedule);

    /* 3. fuse schedule */
    if (!postpone_schedule && env_data.enable_fusion)
    {
        if (global_data.fusion_manager->add(sched))
        {
            LOG_DEBUG("sched ", sched, ", ctype ",
                      ccl_coll_type_to_str(sched->coll_param.ctype), " will be fused");
            return sched;
        }
    }

    /* 4. parallelize schedule */
    sched->commit(global_data.parallelizer.get());

    /* 5. postpone unordered coll schedule */
    if (postpone_schedule)
    {
        /* 
            user has provided match_id that has not been resolved yet.
            schedule will be postponed until comm resolution
        */
        return global_data.unordered_coll_manager->postpone(sched);
    }

    /* 6. regular schedule execution */
    ccl_request* request = sched->start(global_data.executor.get());
    if (sched->coll_attr.synchronous)
    {
        ccl_wait_impl<ccl_master_sched>(global_data.executor.get(), request);
        request = nullptr;
    }

    return request;
}

ccl_status_t ccl_coll_build_allgatherv(
    ccl_sched* sched,
    ccl_buffer send_buf,
    size_t send_count,
    ccl_buffer recv_buf,
    const size_t* recv_counts,
    ccl_datatype_internal_t dtype)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_allgatherv;
    sched->coll_param.send_count = send_count;
    sched->coll_param.recv_counts = recv_counts;

    auto algo = global_data.algorithm_selector->get<ccl_coll_allgatherv>(sched->coll_param);

    switch (algo)
    {
        case ccl_coll_allgatherv_direct:
            CCL_CALL(ccl_coll_build_direct_allgatherv(sched, send_buf, send_count, recv_buf, recv_counts,
                                                      dtype));
            break;
        case ccl_coll_allgatherv_naive:
            CCL_CALL(ccl_coll_build_naive_allgatherv(sched, send_buf, send_count, recv_buf, recv_counts,
                                                     dtype));
            break;
        default:
            CCL_FATAL("unexpected allgatherv_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl_status_invalid_arguments;
    }
    return status;
}

ccl_status_t ccl_coll_build_allreduce(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t count,
    ccl_datatype_internal_t dtype,
    ccl_reduction_t reduction)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_allreduce;
    sched->coll_param.count = count;

    auto algo = global_data.algorithm_selector->get<ccl_coll_allreduce>(sched->coll_param);

    switch (algo)
    {
        case ccl_coll_allreduce_direct:
            CCL_CALL(ccl_coll_build_direct_allreduce(sched, send_buf, recv_buf, count, dtype, reduction));
            break;
        case ccl_coll_allreduce_rabenseifner:
            CCL_CALL(ccl_coll_build_rabenseifner_allreduce(sched, send_buf, recv_buf, count, dtype, reduction));
            break;
        case ccl_coll_allreduce_starlike:
            CCL_CALL(ccl_coll_build_starlike_allreduce(sched, send_buf, recv_buf, count, dtype, reduction));
            break;
        case ccl_coll_allreduce_ring:
            CCL_CALL(ccl_coll_build_ring_allreduce(sched, send_buf, recv_buf, count, dtype, reduction));
            break;
        case ccl_coll_allreduce_ring_rma:
            CCL_CALL(ccl_coll_build_ring_rma_allreduce(sched, send_buf, recv_buf, count, dtype, reduction));
            break;
        case ccl_coll_allreduce_double_tree:
            CCL_CALL(
                ccl_coll_build_double_tree_op(sched, ccl_coll_allreduce, send_buf, recv_buf,
                                              count, dtype, reduction, sched->coll_param.comm->dtree()));
            break;
        case ccl_coll_allreduce_recursive_doubling:
            CCL_CALL(ccl_coll_build_recursive_doubling_allreduce(sched, send_buf, recv_buf, count, dtype, reduction));
            break;
        default:
            CCL_FATAL("unexpected allreduce_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl_status_invalid_arguments;
    }

    return status;
}

ccl_status_t ccl_coll_build_alltoall(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t count,
    ccl_datatype_internal_t dtype)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_alltoall;
    sched->coll_param.count = count;

    auto algo = global_data.algorithm_selector->get<ccl_coll_alltoall>(sched->coll_param);

    switch (algo)
    {
        case ccl_coll_alltoall_direct:
            CCL_CALL(ccl_coll_build_direct_alltoall(sched, send_buf, recv_buf, count, dtype));
            break;
#if 0
        case ccl_coll_alltoall_scatter:
            CCL_CALL(ccl_coll_build_scatter_alltoall(sched, send_buf, recv_buf, count, dtype));
            break;
#endif
        default:
            CCL_FATAL("unexpected allreduce_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl_status_invalid_arguments;
    }

    return status;
}

ccl_status_t ccl_coll_build_barrier(ccl_sched* sched)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_barrier;

    auto algo = global_data.algorithm_selector->get<ccl_coll_barrier>(sched->coll_param);

    switch (algo)
    {
        case ccl_coll_barrier_direct:
            CCL_CALL(ccl_coll_build_direct_barrier(sched));
            break;
        case ccl_coll_barrier_ring:
            CCL_CALL(ccl_coll_build_dissemination_barrier(sched));
            break;
        default:
            CCL_FATAL("unexpected barrier_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl_status_invalid_arguments;
    }

    return status;
}

ccl_status_t ccl_coll_build_bcast(ccl_sched* sched,
                                  ccl_buffer buf,
                                  size_t count,
                                  ccl_datatype_internal_t dtype,
                                  size_t root)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_bcast;
    sched->coll_param.count = count;

    auto algo = global_data.algorithm_selector->get<ccl_coll_bcast>(sched->coll_param);

    switch (algo)
    {
        case ccl_coll_bcast_direct:
            CCL_CALL(ccl_coll_build_direct_bcast(sched, buf, count, dtype, root));
            break;
        case ccl_coll_bcast_ring:
            CCL_CALL(ccl_coll_build_scatter_ring_allgather_bcast(sched, buf, count, dtype, root));
            break;
        case ccl_coll_bcast_double_tree:
            CCL_CALL(ccl_coll_build_double_tree_op(sched, ccl_coll_bcast, ccl_buffer(), buf, count, dtype,
                                                   ccl_reduction_custom,
                                                   root == 0 ? sched->coll_param.comm->dtree() :
                                                   sched->coll_param.comm->dtree().copy_with_new_root(root)));
            break;
        case ccl_coll_bcast_naive:
            CCL_CALL(ccl_coll_build_naive_bcast(sched, buf, count, dtype, root));
            break;
        default:
            CCL_FATAL("unexpected bcast_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl_status_invalid_arguments;
    }
    return status;
}

ccl_status_t ccl_coll_build_reduce(ccl_sched* sched,
                                   ccl_buffer send_buf,
                                   ccl_buffer recv_buf,
                                   size_t count,
                                   ccl_datatype_internal_t dtype,
                                   ccl_reduction_t reduction,
                                   size_t root)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_reduce;
    sched->coll_param.count = count;

    auto algo = global_data.algorithm_selector->get<ccl_coll_reduce>(sched->coll_param);

    switch (algo)
    {
        case ccl_coll_reduce_direct:
            CCL_CALL(ccl_coll_build_direct_reduce(sched, send_buf, recv_buf, count, dtype, reduction, root));
            break;
        case ccl_coll_reduce_rabenseifner:
            CCL_CALL(ccl_coll_build_rabenseifner_reduce(sched, send_buf, recv_buf, count, dtype, reduction, root));
            break;
        case ccl_coll_reduce_tree:
            CCL_CALL(ccl_coll_build_binomial_reduce(sched, send_buf, recv_buf, count, dtype, reduction, root));
            break;
        case ccl_coll_reduce_double_tree:
            CCL_CALL(ccl_coll_build_double_tree_op(sched, ccl_coll_reduce, send_buf, recv_buf, count, dtype,
                                                   reduction,
                                                   root == 0 ? sched->coll_param.comm->dtree() :
                                                   sched->coll_param.comm->dtree().copy_with_new_root(root)));
            break;
        default:
            CCL_FATAL("unexpected reduce_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl_status_invalid_arguments;
    }

    return status;
}


ccl_status_t ccl_coll_build_sparse_allreduce(
    ccl_sched* sched,
    ccl_buffer send_ind_buf, size_t send_ind_count,
    ccl_buffer send_val_buf, size_t send_val_count,
    ccl_buffer recv_ind_buf, size_t* recv_ind_count,
    ccl_buffer recv_val_buf, size_t* recv_val_count,
    ccl_datatype_internal_t index_dtype,
    ccl_datatype_internal_t value_dtype,
    ccl_reduction_t reduction)
{
    ccl_status_t status = ccl_status_success;

    sched->coll_param.ctype = ccl_coll_sparse_allreduce;
    sched->coll_param.sparse_param.send_val_count = send_val_count;

    auto algo = global_data.algorithm_selector->get<ccl_coll_sparse_allreduce>(sched->coll_param);

    LOG_DEBUG("build sparse allreduce, param:",
              "\nsend_ind_buf ", send_ind_buf,
              "\nsend_ind_count ", send_ind_count,
              "\nsend_val_buf ", send_val_buf,
              "\nsend_val_count ", send_val_count,
              "\nrecv_ind_buf ", recv_ind_buf,
              "\nrecv_ind_count ", recv_ind_count,
              "\nrecv_val_buf ", recv_val_buf,
              "\nrecv_val_count ", recv_val_count,
              "\nindex_dtype ", ccl_datatype_get_name(index_dtype),
              "\nvalue_dtype ", ccl_datatype_get_name(value_dtype),
              "\nop ", ccl_reduction_to_str(reduction));

    switch (index_dtype->type)
    {
        case ccl_dtype_char:
            CCL_DEFINE_VALUE(char);
            break;
        case ccl_dtype_int:
            CCL_DEFINE_VALUE(int);
            break;
        case ccl_dtype_int64:
            CCL_DEFINE_VALUE(int64_t);
            break;
        case ccl_dtype_uint64:
            CCL_DEFINE_VALUE(uint64_t);
            break;
        default:
            CCL_FATAL("index data type ", ccl_datatype_get_name(index_dtype), " is not supported yet");
            return ccl_status_invalid_arguments;
    }

    return status;
}

ccl_request* ccl_allgatherv_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const size_t* recv_counts,
                                 ccl_datatype_t dtype,
                                 const ccl_coll_attr_t* attr,
                                 ccl_comm* comm,
                                 const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_allgatherv;
    param.send_buf = send_buf;
    param.recv_buf = recv_buf;
    param.send_count = send_count;
    param.recv_counts = recv_counts;
    param.dtype = ccl_datatype_get(dtype);
    param.stream = stream;
    param.comm = comm;

    auto req = ccl_coll_create(param, ccl_coll_attr(attr));
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_allreduce_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl_datatype_t dtype,
                                ccl_reduction_t reduction,
                                const ccl_coll_attr_t* attr,
                                ccl_comm* comm,
                                const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_allreduce;
    param.send_buf = send_buf;
    param.recv_buf = recv_buf;
    param.count = count;
    param.dtype = ccl_datatype_get(dtype);
    param.reduction = reduction;
    param.stream = stream;
    param.comm = comm;

    auto req = ccl_coll_create(param, ccl_coll_attr(attr));
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req, " count ", count);
    return req;
}

ccl_request* ccl_alltoall_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl_datatype_t dtype,
                               const ccl_coll_attr_t* attr,
                               ccl_comm* comm,
                               const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_alltoall;
    param.send_buf = send_buf;
    param.recv_buf = recv_buf;
    param.count = count;
    param.dtype = ccl_datatype_get(dtype);
    param.stream = stream;
    param.comm = comm;

    auto req = ccl_coll_create(param, ccl_coll_attr(attr));
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req, " count ", count);
    return req;
}

void ccl_barrier_impl(ccl_comm* comm, const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_barrier;
    param.dtype = ccl_dtype_internal_char;
    param.stream = stream;
    param.comm = comm;

    ccl_coll_attr_t attr{};
    attr.synchronous = 1;

    ccl_coll_create(param, ccl_coll_attr(&attr));
}

ccl_request* ccl_bcast_impl(void* buf,
                            size_t count,
                            ccl_datatype_t dtype,
                            size_t root,
                            const ccl_coll_attr_t* attr,
                            ccl_comm* comm,
                            const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_bcast;
    param.buf = buf;
    param.count = count;
    param.dtype = ccl_datatype_get(dtype);
    param.root = root;
    param.stream = stream;
    param.comm = comm;

    auto req = ccl_coll_create(param, ccl_coll_attr(attr));
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_reduce_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl_datatype_t dtype,
                             ccl_reduction_t reduction,
                             size_t root,
                             const ccl_coll_attr_t* attr,
                             ccl_comm* comm,
                             const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_reduce;
    param.send_buf = send_buf;
    param.recv_buf = recv_buf;
    param.count = count;
    param.dtype = ccl_datatype_get(dtype);
    param.reduction = reduction;
    param.root = root;
    param.stream = stream;
    param.comm = comm;

    auto req = ccl_coll_create(param, ccl_coll_attr(attr));
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_sparse_allreduce_impl(const void* send_ind_buf, size_t send_ind_count,
                                       const void* send_val_buf, size_t send_val_count,
                                       void** recv_ind_buf, size_t* recv_ind_count,
                                       void** recv_val_buf, size_t* recv_val_count,
                                       ccl_datatype_t index_dtype, ccl_datatype_t dtype,
                                       ccl_reduction_t reduction, const ccl_coll_attr_t* attr,
                                       ccl_comm* comm, const ccl_stream* stream)
{
    ccl_coll_param param{};

    param.ctype = ccl_coll_sparse_allreduce;
    param.sparse_param.send_ind_buf = send_ind_buf;
    param.sparse_param.send_ind_count = send_ind_count;
    param.sparse_param.send_val_buf = send_val_buf;
    param.sparse_param.send_val_count = send_val_count;
    param.sparse_param.recv_ind_buf = recv_ind_buf;
    param.sparse_param.recv_ind_count = recv_ind_count;
    param.sparse_param.recv_val_buf = recv_val_buf;
    param.sparse_param.recv_val_count = recv_val_count;
    param.dtype = ccl_datatype_get(dtype);
    param.sparse_param.itype = ccl_datatype_get(index_dtype);
    param.reduction = reduction;
    param.stream = stream;
    param.comm = comm;

    auto req = ccl_coll_create(param, ccl_coll_attr(attr));
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}
