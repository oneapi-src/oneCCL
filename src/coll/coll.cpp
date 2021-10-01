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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"

#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"

#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "oneapi/ccl/coll_attr.hpp"

#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/stream.hpp"

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
#include "coll/coll_check.hpp"
#include "coll/coll_param.hpp"

#include "common/global/global.hpp"

#include "coll/algorithms/algorithms.hpp"
#include "coll/algorithms/algorithms_enum.hpp"
#include "coll/algorithms/allreduce/allreduce_2d.hpp"
#include "coll/algorithms/sparse_allreduce/sparse_allreduce.hpp"
#include "coll/selection/selection.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "unordered_coll/unordered_coll.hpp"

/* param is not const because param.comm can be updated for unordered colls */
static ccl_request* ccl_coll_create(ccl_coll_param& param, const ccl_coll_attr& in_attr) {
    ccl_coll_attr& attr = const_cast<ccl_coll_attr&>(in_attr);

#ifdef CCL_ENABLE_SYCL
    if (ccl::global_data::env().enable_op_sync)
        attr.synchronous = 1;
#endif // CCL_ENABLE_SYCL

    LOG_DEBUG("\n{\n",
              "  param: ",
              param.to_string(),
              "\n"
              "  attr: ",
              attr.to_string(),
              "\n"
              "}");

    ccl_coll_validate_user_input(param, attr);

    ccl::global_data& data = ccl::global_data::get();

    /* 1. decide whether schedule should be postponed (this includes caching and starting) */
    bool postpone_schedule = false;
    if (ccl::global_data::env().enable_unordered_coll) {
        if (!attr.match_id.empty()) {
            auto comm =
                param.comm->unordered_coll_manager->get_comm(std::string(attr.match_id)).get();
            if (!comm) {
                if (attr.synchronous) {
                    CCL_THROW("unsupported collective (synchronous && unordered && !communicator)");
                }
                LOG_DEBUG("didn't find comm for match_id ", attr.match_id, ", postpone schedule");
                postpone_schedule = true;
            }
            else {
                LOG_DEBUG("found comm ", comm->id(), " for match_id ", attr.match_id);
                param.comm = comm;
            }
        }
        else {
            /* use comm provided by user, it is ordered collective */
        }
    }

    /* 2. create or get schedule */
    ccl_master_sched* sched = ccl_master_sched::create(param, attr);

    /* 3. fuse schedule */
    if (!postpone_schedule && ccl::global_data::env().enable_fusion) {
        if (data.fusion_manager->add(sched)) {
            LOG_DEBUG("sched ",
                      sched,
                      ", coll ",
                      ccl_coll_type_to_str(sched->coll_param.ctype),
                      " will be fused");
            return sched;
        }
    }

    /* 4. parallelize schedule */
    sched->commit(data.parallelizer.get());

    /* 5. postpone unordered coll schedule */
    if (postpone_schedule) {
        /*
            user has provided match_id that has not been resolved yet.
            schedule will be postponed until comm resolution
        */
        return param.comm->unordered_coll_manager->postpone(sched);
    }

    /* 6. regular schedule execution */
    ccl_request* request = sched->start(data.executor.get());
    if (sched->coll_attr.synchronous) {
        ccl_wait_impl<ccl_master_sched>(data.executor.get(), request);
        request = nullptr;
    }

    return request;
}

ccl::status ccl_coll_build_allgatherv(ccl_sched* sched,
                                      ccl_buffer send_buf,
                                      size_t send_count,
                                      ccl_buffer recv_buf,
                                      const size_t* recv_counts,
                                      const ccl_datatype& dtype,
                                      ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_allgatherv;
    param.recv_counts = recv_counts;
    param.dtype = dtype;
    param.comm = comm;
    param.is_vector_buf = sched->coll_attr.is_vector_buf;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_allgatherv>(param);

    switch (algo) {
        case ccl_coll_allgatherv_direct:
            CCL_CALL(ccl_coll_build_direct_allgatherv(
                sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm));
            break;
        case ccl_coll_allgatherv_naive:
            CCL_CALL(ccl_coll_build_naive_allgatherv(
                sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm));
            break;
        case ccl_coll_allgatherv_ring:
            CCL_CALL(ccl_coll_build_ring_allgatherv(
                sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm));
            break;
        default:
            CCL_FATAL("unexpected allgatherv_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_allreduce(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     ccl_buffer recv_buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     ccl::reduction reduction,
                                     ccl_comm* comm) {
    CCL_ASSERT(sched != nullptr && comm != nullptr);
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_allreduce;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
    param.buf = send_buf.get_ptr();
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_allreduce>(param);

    switch (algo) {
        case ccl_coll_allreduce_direct:
            CCL_CALL(ccl_coll_build_direct_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
        case ccl_coll_allreduce_rabenseifner:
            CCL_CALL(ccl_coll_build_rabenseifner_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
        case ccl_coll_allreduce_starlike:
            CCL_CALL(ccl_coll_build_starlike_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
        case ccl_coll_allreduce_ring:
            CCL_CALL(ccl_coll_build_ring_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
        case ccl_coll_allreduce_ring_rma:
            CCL_CALL(ccl_coll_build_ring_rma_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
        case ccl_coll_allreduce_double_tree:
            CCL_CALL(ccl_coll_build_double_tree_op(sched,
                                                   ccl_coll_allreduce,
                                                   send_buf,
                                                   recv_buf,
                                                   count,
                                                   dtype,
                                                   reduction,
                                                   comm->dtree(),
                                                   comm));
            break;
        case ccl_coll_allreduce_recursive_doubling:
            CCL_CALL(ccl_coll_build_recursive_doubling_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
        case ccl_coll_allreduce_2d:
            CCL_CALL(comm->allreduce_2d_builder->build(
                sched, send_buf, recv_buf, count, dtype, reduction));
            break;
#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
        case ccl_coll_allreduce_topo_ring:
            CCL_CALL(ccl_coll_build_gpu_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
        default:
            CCL_FATAL("unexpected allreduce_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_alltoall(ccl_sched* sched,
                                    ccl_buffer send_buf,
                                    ccl_buffer recv_buf,
                                    size_t count,
                                    const ccl_datatype& dtype,
                                    ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_alltoall;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_alltoall>(param);

    switch (algo) {
        case ccl_coll_alltoall_direct:
            CCL_CALL(ccl_coll_build_direct_alltoall(sched, send_buf, recv_buf, count, dtype, comm));
            break;
        default:
            CCL_FATAL("unexpected alltoall_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_alltoallv(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     const size_t* send_counts,
                                     ccl_buffer recv_buf,
                                     const size_t* recv_counts,
                                     const ccl_datatype& dtype,
                                     ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_alltoallv;
    param.dtype = dtype;
    param.comm = comm;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_alltoallv>(param);

    switch (algo) {
        case ccl_coll_alltoallv_direct:
            CCL_CALL(ccl_coll_build_direct_alltoallv(
                sched, send_buf, send_counts, recv_buf, recv_counts, dtype, comm));
            break;
        default:
            CCL_FATAL("unexpected alltoallv_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_barrier(ccl_sched* sched, ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_barrier;
    param.count = 0;
    param.dtype = ccl_datatype_int8;
    param.comm = comm;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_barrier>(param);

    switch (algo) {
        case ccl_coll_barrier_direct: CCL_CALL(ccl_coll_build_direct_barrier(sched, comm)); break;
        case ccl_coll_barrier_ring:
            CCL_CALL(ccl_coll_build_dissemination_barrier(sched, comm));
            break;
        default:
            CCL_FATAL("unexpected barrier_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_bcast(ccl_sched* sched,
                                 ccl_buffer buf,
                                 size_t count,
                                 const ccl_datatype& dtype,
                                 int root,
                                 ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_bcast;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_bcast>(param);

    switch (algo) {
        case ccl_coll_bcast_direct:
            CCL_CALL(ccl_coll_build_direct_bcast(sched, buf, count, dtype, root, comm));
            break;
        case ccl_coll_bcast_ring:
            CCL_CALL(
                ccl_coll_build_scatter_ring_allgather_bcast(sched, buf, count, dtype, root, comm));
            break;
        case ccl_coll_bcast_double_tree:
            CCL_CALL(ccl_coll_build_double_tree_op(
                sched,
                ccl_coll_bcast,
                ccl_buffer(),
                buf,
                count,
                dtype,
                ccl::reduction::custom,
                root == 0 ? comm->dtree() : comm->dtree().copy_with_new_root(root),
                comm));
            break;
        case ccl_coll_bcast_naive:
            CCL_CALL(ccl_coll_build_naive_bcast(sched, buf, count, dtype, root, comm));
            break;
#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
        case ccl_coll_bcast_topo_ring:
            CCL_CALL(ccl_coll_build_gpu_bcast(sched, buf, count, dtype, root, comm));
            break;
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
        default:
            CCL_FATAL("unexpected bcast_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }
    return status;
}

ccl::status ccl_coll_build_reduce(ccl_sched* sched,
                                  ccl_buffer send_buf,
                                  ccl_buffer recv_buf,
                                  size_t count,
                                  const ccl_datatype& dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_reduce;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_reduce>(param);

    switch (algo) {
        case ccl_coll_reduce_direct:
            CCL_CALL(ccl_coll_build_direct_reduce(
                sched, send_buf, recv_buf, count, dtype, reduction, root, comm));
            break;
        case ccl_coll_reduce_rabenseifner:
            CCL_CALL(ccl_coll_build_rabenseifner_reduce(
                sched, send_buf, recv_buf, count, dtype, reduction, root, comm));
            break;
        case ccl_coll_reduce_tree:
            CCL_CALL(ccl_coll_build_binomial_reduce(
                sched, send_buf, recv_buf, count, dtype, reduction, root, comm));
            break;
        case ccl_coll_reduce_double_tree:
            CCL_CALL(ccl_coll_build_double_tree_op(
                sched,
                ccl_coll_reduce,
                send_buf,
                recv_buf,
                count,
                dtype,
                reduction,
                root == 0 ? comm->dtree() : comm->dtree().copy_with_new_root(root),
                comm));
            break;
#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
        case ccl_coll_reduce_topo_ring:
            CCL_CALL(ccl_coll_build_gpu_reduce(
                sched, send_buf, recv_buf, count, dtype, reduction, root, comm));
            break;
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
        default:
            CCL_FATAL("unexpected reduce_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_reduce_scatter(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction reduction,
                                          ccl_comm* comm,
                                          bool from_allreduce) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_reduce_scatter;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_reduce_scatter>(param);

    switch (algo) {
        case ccl_coll_reduce_scatter_direct:
            if (!from_allreduce) {
                CCL_CALL(ccl_coll_build_direct_reduce_scatter(
                    sched, send_buf, recv_buf, count, dtype, reduction, comm));
                break;
            }
        case ccl_coll_reduce_scatter_ring:
            if (from_allreduce) {
                CCL_CALL(ccl_coll_build_ring_reduce_scatter(
                    sched, send_buf, recv_buf, count, dtype, reduction, comm));
            }
            else {
                CCL_CALL(ccl_coll_build_ring_reduce_scatter_block(
                    sched, send_buf, recv_buf, count, dtype, reduction, comm));
            }
            break;
        default:
            CCL_FATAL("unexpected reduce_scatter_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

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
                                            ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_sparse_allreduce;
    param.count = 0;
    param.dtype = ccl_datatype_int8;
    param.comm = comm;
    param.sparse_coalesce_mode = sched->coll_attr.sparse_coalesce_mode;
    param.sparse_allreduce_alloc_fn = sched->coll_attr.sparse_allreduce_alloc_fn;

    if (!send_ind_buf.get_ptr() || !send_val_buf.get_ptr()) {
        LOG_ERROR(
            "sparse_allreduce send buffers for indices and values should not be NULL, but got "
            "indices buffer = ",
            send_ind_buf.get_ptr(),
            ", values buffer = ",
            send_val_buf.get_ptr());
        assert(send_ind_buf.get_ptr() && send_val_buf.get_ptr());

        throw ccl::exception(
            std::string(__FUNCTION__) + "sparse_allreduce send buffers for indices and values \
            should not be NULL, but got indices buffer = " +
            std::to_string((uintptr_t)send_ind_buf.get_ptr()) +
            ", values buffer = " + std::to_string((uintptr_t)send_val_buf.get_ptr()));
    }

    if (!send_ind_count || !send_val_count) {
        LOG_ERROR("sparse_allreduce send buffer count should be greater than zero, but got "
                  "indices count = ",
                  send_ind_count,
                  ", values count = ",
                  send_val_count);
        assert(send_ind_count && send_val_count);

        throw ccl::exception(
            std::string(__FUNCTION__) + "sparse_allreduce send buffer count should be \
            greater than zero, but got indices count = " +
            std::to_string(send_ind_count) + ", values count = " + std::to_string(send_val_count));
    }

    if (send_ind_count > send_val_count) {
        CCL_FATAL("sparse collective algorithms now support only 1-D indices and \
                  multi-dimensional values format\n got indices count = ",
                  send_ind_count,
                  ", values count = ",
                  send_val_count);
        return ccl::status::invalid_arguments;
    }

    if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        /*
            for now all sparse_allreduce algorithms
            may contains direct collective entries (allreduce/allgatherv)
            which should be executed in strict_order mode
        */
        sched->strict_order = true;
    }

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_sparse_allreduce>(param);

    LOG_DEBUG("build sparse allreduce, param:",
              "\nsend_ind_buf ",
              send_ind_buf,
              "\nsend_ind_count ",
              send_ind_count,
              "\nsend_val_buf ",
              send_val_buf,
              "\nsend_val_count ",
              send_val_count,
              "\nrecv_ind_buf ",
              recv_ind_buf,
              "\nrecv_ind_count ",
              recv_ind_count,
              "\nrecv_val_buf ",
              recv_val_buf,
              "\nrecv_val_count ",
              recv_val_count,
              "\nindex_dtype ",
              ccl::global_data::get().dtypes->name(index_dtype),
              "\nvalue_dtype ",
              ccl::global_data::get().dtypes->name(value_dtype),
              "\nop ",
              ccl_reduction_to_str(reduction));

    switch (index_dtype.idx()) {
        case ccl::datatype::int32:
            CCL_SPARSE_ALLREDUCE_SELECT_V_DTYPE(int32_t, value_dtype, algo);
            break;
        case ccl::datatype::int64:
            CCL_SPARSE_ALLREDUCE_SELECT_V_DTYPE(int64_t, value_dtype, algo);
            break;
        default:
            CCL_FATAL("index datatype ",
                      ccl::global_data::get().dtypes->name(index_dtype),
                      " is not supported yet");
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl_request* ccl_allgatherv_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const size_t* recv_counts,
                                 ccl::datatype dtype,
                                 const ccl_coll_attr& attr,
                                 ccl_comm* comm,
                                 const ccl_stream* stream,
                                 const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_allgatherv_param(
        send_buf, send_count, recv_buf, recv_counts, dtype, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_allreduce_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_allreduce_param(
        send_buf, recv_buf, count, dtype, reduction, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req, " count ", count);
    return req;
}

ccl_request* ccl_alltoall_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl::datatype dtype,
                               const ccl_coll_attr& attr,
                               ccl_comm* comm,
                               const ccl_stream* stream,
                               const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_alltoall_param(
        send_buf, recv_buf, count, dtype, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req, " count ", count);
    return req;
}

ccl_request* ccl_alltoallv_impl(const void* send_buf,
                                const size_t* send_counts,
                                void* recv_buf,
                                const size_t* recv_counts,
                                ccl::datatype dtype,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_alltoallv_param(
        send_buf, send_counts, recv_buf, recv_counts, dtype, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

void ccl_barrier_impl(ccl_comm* comm,
                      const ccl_stream* stream,
                      const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_barrier_param(comm, stream, deps);

    ccl_coll_attr attr{};
    attr.synchronous = 1;

    ccl_coll_create(param, attr);

    if (ccl::global_data::get().sched_cache->try_flush()) {
        LOG_DEBUG("flushed cache in barrier");
    }
    else {
        LOG_DEBUG("didn't flush cache in barrier");
    }
}

ccl_request* ccl_broadcast_impl(void* buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps) {
    ccl_coll_param param =
        ccl_coll_param::create_broadcast_param(buf, count, dtype, root, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_reduce_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::datatype dtype,
                             ccl::reduction reduction,
                             int root,
                             const ccl_coll_attr& attr,
                             ccl_comm* comm,
                             const ccl_stream* stream,
                             const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_reduce_param(
        send_buf, recv_buf, count, dtype, reduction, root, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_reduce_scatter_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl_coll_attr& attr,
                                     ccl_comm* comm,
                                     const ccl_stream* stream,
                                     const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_reduce_scatter_param(
        send_buf, recv_buf, recv_count, dtype, reduction, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
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
                                       ccl::datatype value_dtype,
                                       ccl::reduction reduction,
                                       const ccl_coll_attr& attr,
                                       ccl_comm* comm,
                                       const ccl_stream* stream,
                                       const std::vector<ccl::event>& deps) {
    CCL_THROW("unsupported path");

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
    param.dtype = ccl::global_data::get().dtypes->get(value_dtype);
    param.sparse_param.itype = ccl::global_data::get().dtypes->get(index_dtype);
    param.reduction = reduction;
    param.stream = (ccl_stream*)stream;
    param.comm = comm;
    param.copy_deps(deps);

    ccl_coll_attr internal_attr(attr);
    internal_attr.to_cache = 0; /* skip to_cache flag, unsupported yet */

    auto req = ccl_coll_create(param, internal_attr);
    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}
