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

#include "comm/comm.hpp"
#include "coll/coll.hpp"
#include "coll/attr/ccl_common_op_attrs.hpp"
#include "coll/attr/ccl_allgather_op_attr.hpp"
#include "coll/attr/ccl_allreduce_op_attr.hpp"
#include "coll/attr/ccl_alltoall_op_attr.hpp"
#include "coll/attr/ccl_alltoallv_op_attr.hpp"
#include "coll/attr/ccl_barrier_op_attr.hpp"
#include "coll/attr/ccl_bcast_op_attr.hpp"
#include "coll/attr/ccl_pt2pt_op_attr.hpp"
#include "coll/attr/ccl_reduce_op_attr.hpp"
#include "coll/attr/ccl_reduce_scatter_op_attr.hpp"
#include "coll/coll_check.hpp"
#include "coll/coll_param.hpp"
#include "coll/coll_util.hpp"

#include "common/global/global.hpp"

#include "coll/algorithms/algorithm_utils.hpp"
#include "coll/algorithms/algorithms.hpp"
#include "coll/selection/selection.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched_timer.hpp"
#include "unordered_coll/unordered_coll.hpp"

ccl_request* allgatherv_case(ccl_coll_param& param) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::vector<sycl::event> events{};
    for (size_t idx = 0; idx < param.deps.size(); idx++) {
        events.push_back(param.deps[idx].get_native());
    }
    sycl::event ev;
#if ICPX_VERSION >= 140000
    ev = param.stream->get_native_stream().ext_oneapi_submit_barrier(events);
#elif ICPX_VERSION < 140000
    ev = param.stream->get_native_stream().submit_barrier(events);
#endif // ICPX_VERSION
    if (ccl::utils::should_use_sycl_output_event(param.stream)) {
        ccl_coll_param dummy_param{};
        dummy_param.comm = param.comm;
        auto dummy_sched = ccl_sched::create(dummy_param, {});
        auto req = dummy_sched->get_request();
        req->set_native_event(ev);
        return req;
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    return nullptr;
}

/* param is not const because param.comm can be updated for unordered colls */
static ccl_request* ccl_coll_create(ccl_coll_param& param, const ccl_coll_attr& in_attr) {
    ccl_coll_attr& attr = const_cast<ccl_coll_attr&>(in_attr);

#ifdef CCL_ENABLE_ITT
    __itt_event coll_create_itt_event =
        ccl::profile::itt::event_get(ccl_coll_type_to_str(param.ctype));
    ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT

#ifdef CCL_ENABLE_SYCL
    if (ccl::global_data::env().enable_op_sync)
        attr.synchronous = 1;

    // TODO: remove after MLSL-1915 (ring barrier is broken) is done
    // this is needed because OFI transport means we need ring barrier
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi)
        attr.synchronous = 1;

    auto is_inorder_queue = ccl::is_queue_in_order(param.stream);
    if ((param.ctype == ccl_coll_send || param.ctype == ccl_coll_recv) && is_inorder_queue)
        attr.synchronous = 1;

    auto enable_sycl_output_event_saved = ccl::global_data::env().enable_sycl_output_event;
    ccl::enable_sycl_output_barrier_in_order_queue(param.stream);

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

    if (param.ctype == ccl_coll_allgatherv && param.comm->size() == 1 && param.is_inplace()) {
        return allgatherv_case(param);
    }

    /* 1. decide whether schedule should be postponed (this includes caching and starting) */
    bool postpone_schedule = false;
    if (ccl::global_data::env().enable_unordered_coll) {
        if (!attr.match_id.empty()) {
            auto comm = param.comm->get_unordered_coll_manager()
                            ->get_comm(std::string(attr.match_id))
                            .get();
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
    ccl_sched* sched = ccl_sched::create(param, attr);

    /* 3. fuse schedule */
    if (!postpone_schedule &&
        ccl::global_data::env().enable_fusion
#ifdef CCL_ENABLE_SYCL
        // TODO: enable fusion for async case with CCL_SYCL_OUTPUT_EVENT
        && (attr.synchronous || !ccl::utils::should_use_sycl_output_event(sched->coll_param.stream))
#endif //CCL_ENABLE_SYCL
    ) {
        if (data.fusion_manager->add(sched)) {
            LOG_DEBUG("sched ",
                      sched,
                      ", coll ",
                      ccl_coll_type_to_str(sched->coll_param.ctype),
                      " will be fused");
            return sched->get_request();
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
        return param.comm->get_unordered_coll_manager()->postpone(sched);
    }

    sched->set_submitted_to_gpu(false);

    /* 6. regular schedule execution */
    ccl_request* request = sched->start(data.executor.get());
    if (sched->coll_attr.synchronous) {
        request->synchronous = true;
        // request->synchronous is true,
        // so ccl_wait_impl should not release the `request`
        auto wait_result = ccl_wait_impl<ccl_sched>(data.executor.get(), request);
        CCL_THROW_IF_NOT(wait_result != ccl_wait_result_completed_released,
                         "internal error, valid request was released");
    }
#ifdef CCL_ENABLE_SYCL
    else if (ccl::utils::should_use_sycl_output_event(sched->coll_param.stream)) {
        LOG_DEBUG("waiting for sched ", sched, " to be submitted_to_gpu");
        while (!sched->is_submitted_to_gpu() && !request->is_completed()) {
            data.executor.get()->do_work();
        }
        LOG_DEBUG("setting sycl_barrier on sched ", sched);
        if (!request->is_completed()) {
            if (is_inorder_queue) {
                request->set_native_event(ccl::utils::submit_barrier(
                    sched->coll_param.stream->get_native_stream(), request->get_sync_event()));
            }
            else {
                request->set_native_event(request->get_sync_event());
            }
        }
        else {
            request->set_native_event(
                ccl::utils::submit_barrier(sched->coll_param.stream->get_native_stream()));
        }
    }
    ccl::global_data::env().enable_sycl_output_event = enable_sycl_output_event_saved;
    LOG_DEBUG("CCL_SYCL_OUTPUT_EVENT is restored: ", enable_sycl_output_event_saved);
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT

    return request;
}

ccl::status ccl_coll_build_allgatherv(ccl_sched* sched,
                                      ccl_buffer send_buf,
                                      size_t send_count,
                                      ccl_buffer recv_buf,
                                      const size_t* recv_counts,
                                      const ccl_datatype& dtype,
                                      ccl_comm* comm,
                                      bool is_scaleout) {
    ccl::status status = ccl::status::success;

    ccl_selector_param selector_param;
    selector_param.ctype = ccl_coll_allgatherv;
    selector_param.recv_counts = recv_counts;
    selector_param.dtype = dtype;
    selector_param.comm = comm;
    selector_param.stream = sched->coll_param.stream;
    selector_param.buf = send_buf.get_ptr();
    selector_param.is_vector_buf = false;
#ifdef CCL_ENABLE_SYCL
    selector_param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    selector_param.hint_algo = sched->hint_algo;
    selector_param.is_scaleout = is_scaleout;

    auto algo =
        ccl::global_data::get().algorithm_selector->get<ccl_coll_allgatherv>(selector_param);

    auto get_coll_param = [&]() {
        ccl_coll_param coll_param{};
        coll_param.ctype = ccl_coll_allgatherv, coll_param.send_bufs.push_back(send_buf.get_ptr()),
        coll_param.send_counts.push_back(send_count),
        coll_param.recv_bufs.push_back(recv_buf.get_ptr()),
        coll_param.recv_counts.reserve(comm->size()),
        coll_param.recv_counts.insert(
            coll_param.recv_counts.end(), recv_counts, recv_counts + comm->size()),
        coll_param.dtype = dtype, coll_param.comm = comm,
        coll_param.stream = sched->coll_param.stream;
        return coll_param;
    };
    std::vector<ccl_sched*> part_scheds = { sched };

    switch (algo) {
        case ccl_coll_allgatherv_direct:
            CCL_CALL(ccl_coll_build_direct_allgatherv(
                sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm));
            break;
        case ccl_coll_allgatherv_flat:
            CCL_CALL(ccl_coll_build_flat_allgatherv(nullptr, part_scheds, get_coll_param()));
            break;
        case ccl_coll_allgatherv_multi_bcast:
            CCL_CALL(
                ccl_coll_build_multi_bcast_allgatherv(nullptr, part_scheds, get_coll_param(), 1));
            break;
        case ccl_coll_allgatherv_naive:
            CCL_CALL(ccl_coll_build_naive_allgatherv(
                sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm));
            break;
        case ccl_coll_allgatherv_ring:
            CCL_CALL(ccl_coll_build_ring_allgatherv(
                nullptr, part_scheds, send_buf, send_count, recv_buf, recv_counts, dtype, comm));
            break;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_allgatherv_topo:
            CCL_CALL(ccl_coll_build_topo_allgatherv(nullptr, part_scheds, get_coll_param()));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
                                     ccl_comm* comm,
                                     bool is_scaleout) {
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
    param.is_scaleout = is_scaleout;

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
        case ccl_coll_allreduce_nreduce:
            CCL_CALL(ccl_coll_build_nreduce_allreduce(
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
            CCL_CALL(ccl_coll_build_2d_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_allreduce_topo:
            CCL_CALL(ccl_coll_build_topo_allreduce(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
                                    ccl_comm* comm,
                                    bool is_scaleout) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_alltoall;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.hint_algo = sched->hint_algo;
    param.is_scaleout = is_scaleout;

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
                                     ccl_comm* comm,
                                     bool is_scaleout) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_alltoallv;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.hint_algo = sched->hint_algo;
    param.is_scaleout = is_scaleout;

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
    param.buf = buf.get_ptr();
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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_bcast_topo:
            CCL_CALL(ccl_coll_build_topo_bcast(sched, buf, count, dtype, root, comm));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
                                  ccl_comm* comm,
                                  bool is_scaleout) {
    ccl::status status = ccl::status::success;
    CCL_THROW_IF_NOT(root >= 0 && root < comm->size(), "wrong root");

    ccl_selector_param param;
    param.ctype = ccl_coll_reduce;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
    param.buf = send_buf.get_ptr();
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.hint_algo = sched->hint_algo;
    param.is_scaleout = is_scaleout;

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
        case ccl_coll_reduce_ring:
            CCL_CALL(ccl_coll_build_ring_reduce(
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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_reduce_topo:
            CCL_CALL(ccl_coll_build_topo_reduce(
                sched, send_buf, recv_buf, count, dtype, reduction, root, comm));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
                                          bool is_scaleout,
                                          bool from_allreduce) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    param.ctype = ccl_coll_reduce_scatter;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
    param.buf = send_buf.get_ptr();
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.hint_algo = sched->hint_algo;
    param.is_scaleout = is_scaleout;

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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_reduce_scatter_topo:
            CCL_CALL(ccl_coll_build_topo_reduce_scatter(
                sched, send_buf, recv_buf, count, dtype, reduction, comm));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
        default:
            CCL_FATAL("unexpected reduce_scatter_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_recv(ccl_sched* sched,
                                ccl_buffer buf,
                                size_t count,
                                const ccl_datatype& dtype,
                                int peer_rank,
                                ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    sched->coll_param.ctype = ccl_coll_recv;
    param.ctype = sched->coll_param.ctype;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
    param.buf = buf.get_ptr();
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.peer_rank = peer_rank;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_recv>(param);

    switch (algo) {
        case ccl_coll_recv_direct:
        case ccl_coll_recv_offload:
            CCL_CALL(ccl_coll_build_direct_recv(sched, buf, count, dtype, peer_rank, comm));
            break;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_recv_topo:
            CCL_CALL(ccl_coll_build_topo_recv(sched, buf, count, dtype, peer_rank, comm));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
        default:
            CCL_FATAL("unexpected recv_algo ", ccl_coll_algorithm_to_str(algo));
            return ccl::status::invalid_arguments;
    }

    return status;
}

ccl::status ccl_coll_build_send(ccl_sched* sched,
                                ccl_buffer buf,
                                size_t count,
                                const ccl_datatype& dtype,
                                int peer_rank,
                                ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_selector_param param;
    sched->coll_param.ctype = ccl_coll_send;
    param.ctype = sched->coll_param.ctype;
    param.count = count;
    param.dtype = dtype;
    param.comm = comm;
    param.stream = sched->coll_param.stream;
    param.buf = buf.get_ptr();
#ifdef CCL_ENABLE_SYCL
    param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    param.peer_rank = peer_rank;
    param.hint_algo = sched->hint_algo;

    auto algo = ccl::global_data::get().algorithm_selector->get<ccl_coll_send>(param);

    switch (algo) {
        case ccl_coll_send_direct:
        case ccl_coll_send_offload:
            CCL_CALL(ccl_coll_build_direct_send(sched, buf, count, dtype, peer_rank, comm));
            break;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case ccl_coll_send_topo:
            CCL_CALL(ccl_coll_build_topo_send(sched, buf, count, dtype, peer_rank, comm));
            break;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
        default:
            CCL_FATAL("unexpected send_algo ", ccl_coll_algorithm_to_str(algo));
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

ccl_request* ccl_barrier_impl(ccl_comm* comm,
                              const ccl_stream* stream,
                              const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_barrier_param(comm, stream, deps);

    ccl_coll_attr attr{};
    attr.synchronous = 1;

#ifdef CCL_ENABLE_SYCL
    if (!ccl::global_data::env().sync_barrier && ccl::is_queue_in_order(stream)) {
        attr.synchronous = 0;
    }
#endif // CCL_ENABLE_SYCL

    auto req = ccl_coll_create(param, attr);

    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), " created, req ", req)

    if (ccl::global_data::get().sched_cache->try_flush()) {
        LOG_DEBUG("flushed cache in barrier");
    }
    else {
        LOG_DEBUG("didn't flush cache in barrier");
    }

    return req;
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

ccl_request* ccl_recv_impl(void* recv_buf,
                           size_t recv_count,
                           ccl::datatype dtype,
                           int peer_rank,
                           const ccl_coll_attr& attr,
                           ccl_comm* comm,
                           const ccl_stream* stream,
                           const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_recv_param(
        recv_buf, recv_count, dtype, peer_rank, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);

    LOG_DEBUG("op ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}

ccl_request* ccl_send_impl(const void* send_buf,
                           size_t send_count,
                           ccl::datatype dtype,
                           int peer_rank,
                           const ccl_coll_attr& attr,
                           ccl_comm* comm,
                           const ccl_stream* stream,
                           const std::vector<ccl::event>& deps) {
    ccl_coll_param param = ccl_coll_param::create_send_param(
        send_buf, send_count, dtype, peer_rank, attr, comm, stream, deps);

    auto req = ccl_coll_create(param, attr);

    LOG_DEBUG("op ", ccl_coll_type_to_str(param.ctype), " created, req ", req);
    return req;
}
