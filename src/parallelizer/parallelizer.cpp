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
#include <algorithm>
#include <numeric>

#include "coll/selection/selection.hpp"
#include "common/global/global.hpp"
#include "common/utils/sycl_utils.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/entry/coll/coll_entry_helper.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_ATL_LARGE_MSG_SIZE (1024 * 1024 * 1024)

typedef struct {
    /* keep these 3 fields on the top of structure */
    void* buf;
    size_t count;
    ccl::datatype dt_idx;
    /*---*/

    size_t part_idx;
    size_t part_count;
} ccl_parallelizer_prologue_ctx;

typedef struct {
    void* buf;
    size_t count;
    size_t dtype_size;
} ccl_parallelizer_sparse_callback_ctx;

ccl::status ccl_parallelizer_sparse_callback_get_buf(const void* ctx, void* field_ptr) {
    ccl_parallelizer_sparse_callback_ctx* cctx = (ccl_parallelizer_sparse_callback_ctx*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(cctx->buf, cctx->count * cctx->dtype_size, 0);
    return ccl::status::success;
}

ccl::status ccl_parallelizer_sparse_callback_get_count(const void* ctx, void* field_ptr) {
    ccl_parallelizer_sparse_callback_ctx* cctx = (ccl_parallelizer_sparse_callback_ctx*)ctx;
    size_t* count_ptr = (size_t*)field_ptr;
    *count_ptr = cctx->count;
    return ccl::status::success;
}

ccl::status ccl_parallelizer_prologue_get_buf(const void* ctx, void* field_ptr) {
    ccl_parallelizer_prologue_ctx* pctx = (ccl_parallelizer_prologue_ctx*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    size_t dtype_size = ccl::global_data::get().dtypes->get(pctx->dt_idx).size();
    buf_ptr->set(pctx->buf,
                 pctx->count * dtype_size,
                 pctx->part_idx * (pctx->count / pctx->part_count) * dtype_size);
    return ccl::status::success;
}

ccl::status ccl_parallelizer_prologue_get_count(const void* ctx, void* field_ptr) {
    ccl_parallelizer_prologue_ctx* pctx = (ccl_parallelizer_prologue_ctx*)ctx;
    size_t count = pctx->count / pctx->part_count;
    if (pctx->part_idx == (pctx->part_count - 1))
        count += pctx->count % pctx->part_count;
    size_t* count_ptr = (size_t*)field_ptr;
    *count_ptr = count;
    return ccl::status::success;
}

ccl::status ccl_parallelizer_prologue_get_dtype(const void* ctx, void* field_ptr) {
    ccl_parallelizer_prologue_ctx* pctx = (ccl_parallelizer_prologue_ctx*)ctx;
    ccl_datatype* dtype_ptr = (ccl_datatype*)field_ptr;
    *dtype_ptr = ccl::global_data::get().dtypes->get(pctx->dt_idx);
    return ccl::status::success;
}

ccl::status ccl_parallelizer::process(ccl_master_sched* sched) {
    process_base(sched);

#ifdef CCL_ENABLE_SYCL
    ccl_coll_param& param = sched->coll_param;
    if (param.stream && param.stream->is_sycl_device_stream() &&
        (!param.device_send_bufs.empty() || !param.device_recv_bufs.empty())) {
        process_pre_post_copies(sched);
    }
    process_output_event(sched);
#endif // CCL_ENABLE_SYCL

    /* should be the last call in the sequence of process_* calls
       because it sets dependencies for all partial schedules
       which already should be filled */
    process_deps(sched);

    return ccl::status::success;
}

ccl::status ccl_parallelizer::process_deps(ccl_master_sched* sched) {
    auto& part_scheds = sched->partial_scheds;
    ccl_sched* deps_sched = part_scheds[0].get();
    size_t sched_count = part_scheds.size();

    for (size_t idx = 0; idx < sched_count; idx++) {
        part_scheds[idx]->set_add_mode(ccl_sched_add_front);
    }
    sched->sync_partial_scheds();

    entry_factory::make_entry<deps_entry>(deps_sched);
    deps_sched->add_barrier();

    return ccl::status::success;
}

#ifdef CCL_ENABLE_SYCL
ccl::status ccl_parallelizer::process_pre_post_copies(ccl_master_sched* sched) {
    auto& part_scheds = sched->partial_scheds;
    size_t sched_count = part_scheds.size();
    ccl_coll_param& coll_param = sched->coll_param;
    ccl_comm* comm = coll_param.comm;
    int my_rank = comm->rank();
    const ccl_datatype& dtype = coll_param.dtype;
    size_t dtype_size = dtype.size();
    ccl_coll_type coll_type = coll_param.ctype;
    size_t device_in_buf_offset = 0;

    std::vector<size_t> d2h_counts;
    std::vector<size_t> h2d_counts;
    bool reuse_buffers;
    sched->get_pre_post_copy_counts(d2h_counts, h2d_counts, reuse_buffers);

    if ((coll_type == ccl_coll_allgatherv) &&
        coll_param.is_inplace(ccl_coll_param::buf_type::device)) {
        CCL_THROW_IF_NOT(coll_param.device_recv_bufs.size() == 1,
                         "unexpected device_recv_bufs.size ",
                         coll_param.device_recv_bufs.size());
        device_in_buf_offset = std::accumulate(
            coll_param.recv_counts.begin(), coll_param.recv_counts.begin() + my_rank, 0);
        LOG_TRACE("device_in_buf_offset = ", device_in_buf_offset);
    }

    size_t total_d2h_count = std::accumulate(d2h_counts.begin(), d2h_counts.end(), 0);
    size_t total_h2d_count = std::accumulate(h2d_counts.begin(), h2d_counts.end(), 0);

    if (total_d2h_count) {
        for (size_t idx = 0; idx < sched_count; idx++) {
            part_scheds[idx]->set_add_mode(ccl_sched_add_front);
        }
        sched->sync_partial_scheds();

        for (size_t idx = 0; idx < d2h_counts.size(); idx++) {
            size_t sched_idx = idx % sched_count;
            size_t count = d2h_counts[idx];
            size_t bytes = count * dtype_size;

            entry_factory::make_entry<copy_entry>(
                part_scheds[sched_idx].get(),
                ccl_buffer(coll_param.get_send_buf_ptr(idx, ccl_coll_param::buf_type::device),
                           bytes,
                           ccl_buffer_type::INDIRECT),
                ccl_buffer(coll_param.get_send_buf(idx), bytes),
                count,
                dtype,
                copy_attr(copy_direction::d2h, device_in_buf_offset));
        }
    }

    if (total_h2d_count) {
        for (size_t idx = 0; idx < sched_count; idx++) {
            part_scheds[idx]->set_add_mode(ccl_sched_add_back);
        }
        sched->sync_partial_scheds();

        for (size_t idx = 0; idx < h2d_counts.size(); idx++) {
            size_t sched_idx = idx % sched_count;
            size_t count = h2d_counts[idx];
            size_t bytes = count * dtype_size;

            entry_factory::make_entry<copy_entry>(
                part_scheds[sched_idx].get(),
                ccl_buffer(coll_param.get_recv_buf(idx), bytes),
                ccl_buffer(coll_param.get_recv_buf_ptr(idx, ccl_coll_param::buf_type::device),
                           bytes,
                           ccl_buffer_type::INDIRECT),
                count,
                dtype,
                copy_attr(copy_direction::h2d, 0));
        }

        sched->sync_partial_scheds();
    }

    return ccl::status::success;
}

ccl::status ccl_parallelizer::process_output_event(ccl_master_sched* sched) {
    if (!ccl::utils::should_use_sycl_output_event(sched->coll_param.stream)) {
        return ccl::status::success;
    }

    auto& part_scheds = sched->partial_scheds;
    size_t sched_count = part_scheds.size();

    for (size_t idx = 0; idx < sched_count; idx++) {
        part_scheds[idx]->set_add_mode(ccl_sched_add_back);
    }
    sched->sync_partial_scheds();

    entry_factory::make_entry<ze_event_signal_entry>(part_scheds[0].get(), sched);

    return ccl::status::success;
}
#endif // CCL_ENABLE_SYCL

ccl::status ccl_parallelizer::process_base(ccl_master_sched* sched) {
    /* TODO: split on per-collective classes */

    CCL_ASSERT(sched);

    ccl::global_data& data = ccl::global_data::get();

    ccl::status status = ccl::status::success;
    size_t part_count = 1, idx, base_count, dtype_size, comm_size;

    ccl_coll_param& coll_param = sched->coll_param;
    ccl_coll_attr& coll_attr = sched->coll_attr;

    ccl_comm* comm = coll_param.comm;

    const ccl_datatype& dtype = coll_param.dtype;
    dtype_size = dtype.size();
    comm_size = comm->size();

    ccl_coll_type coll_type = coll_param.ctype;

    std::vector<size_t> counts;
    std::vector<size_t> offsets;
    auto& part_scheds = sched->partial_scheds;
    std::vector<ccl_sched*> part_scheds_vector;

    std::vector<ccl_buffer> ag_recv_bufs;
    size_t ag_recv_bytes = 0, ag_recv_count = 0;
    size_t a2av_send_bytes = 0, a2av_recv_bytes = 0;
    size_t a2av_send_count = 0, a2av_recv_count = 0;

    ccl_coll_algo algo;
    ccl_coll_algo internal_algo;

    std::vector<ccl_parallelizer_prologue_ctx*> part_ctxs;

    ccl_selector_param selector_param;
    selector_param.ctype = coll_type;
    selector_param.count = coll_param.get_send_count();
    selector_param.dtype = dtype;
    selector_param.comm = comm;
    selector_param.stream = coll_param.stream;
    selector_param.is_vector_buf = coll_attr.is_vector_buf;
#ifdef CCL_ENABLE_SYCL
    selector_param.is_sycl_buf = coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL

    switch (coll_type) {
        case ccl_coll_barrier: part_count = max_data_partition_count; break;
        case ccl_coll_bcast:
            if (ccl::global_data::env().bcast_part_count != CCL_ENV_SIZET_NOT_SPECIFIED) {
                part_count = ccl::global_data::env().bcast_part_count;
                break;
            }
        case ccl_coll_reduce:
        case ccl_coll_allreduce:
            if ((coll_param.get_send_count() * dtype_size <=
                 ccl::global_data::env().max_short_size) ||
                (coll_param.get_send_count() < max_data_partition_count)) {
                part_count = 1;
            }
            else {
                /* to workaround lack of large msg protocol on ATL level */
                part_count = (coll_param.get_send_count() * dtype_size) / CCL_ATL_LARGE_MSG_SIZE;
                if (part_count < max_data_partition_count)
                    part_count = max_data_partition_count;
            }
            if (ccl_is_topo_ring_algo(selector_param)) {
                part_count = 1;
            }
            break;
        case ccl_coll_alltoall:
            algo.alltoall = data.algorithm_selector->get<ccl_coll_alltoall>(selector_param);
            if (algo.alltoall == ccl_coll_alltoall_direct) {
                part_count = 1;
            }
            else {
                part_count = std::min(comm_size, max_data_partition_count);
            }
            break;
        case ccl_coll_alltoallv:
            algo.alltoallv = data.algorithm_selector->get<ccl_coll_alltoallv>(selector_param);
            if (algo.alltoallv == ccl_coll_alltoallv_direct) {
                part_count = 1;
            }
            else {
                part_count = std::min(comm_size, max_data_partition_count);
            }
            break;
        case ccl_coll_allgatherv:
            selector_param.recv_counts = coll_param.recv_counts.data();
            algo.allgatherv = data.algorithm_selector->get<ccl_coll_allgatherv>(selector_param);
            if (algo.allgatherv == ccl_coll_allgatherv_direct ||
                algo.allgatherv == ccl_coll_allgatherv_naive ||
                algo.allgatherv == ccl_coll_allgatherv_ring) {
                part_count = 1;
            }
            else if (algo.allgatherv == ccl_coll_allgatherv_multi_bcast ||
                     algo.allgatherv == ccl_coll_allgatherv_flat) {
                part_count = comm_size;
                ag_recv_bufs.resize(comm_size);
                if (algo.allgatherv == ccl_coll_allgatherv_multi_bcast) {
                    selector_param.ctype = ccl_coll_bcast;
                    selector_param.count = sched->coll_param.get_send_count();
                    selector_param.dtype = dtype;
                    internal_algo.bcast =
                        data.algorithm_selector->get<ccl_coll_bcast>(selector_param);
                    if (internal_algo.bcast == ccl_coll_bcast_direct) {
                        /*
                            group all direct bcasts for specific worker together into single schedule w/o
                            any barriers to get all required MPI level tags by single do_progress call
                        */
                        part_count = max_data_partition_count;
                        LOG_DEBUG("allgatherv over direct multi_bcast, set part_count ",
                                  part_count);
                    }
                }
            }
            else {
                CCL_FATAL("unexpected allgatherv_algo ", algo.allgatherv);
            }
            break;
        case ccl_coll_reduce_scatter: part_count = 1; break;
        case ccl_coll_sparse_allreduce: part_count = 1; break;
        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }

    LOG_DEBUG("sched ",
              sched,
              ", coll_type ",
              ccl_coll_type_to_str(coll_type),
              ", part_count ",
              part_count);

    if (coll_type == ccl_coll_allgatherv && algo.allgatherv == ccl_coll_allgatherv_multi_bcast &&
        internal_algo.bcast == ccl_coll_bcast_direct) {
        counts.resize(comm_size, 0);
        offsets.resize(comm_size, 0);
    }
    else {
        counts.resize(part_count, 0);
        offsets.resize(part_count, 0);
    }

    for (idx = 0; idx < part_count; idx++) {
        ccl_coll_param part_coll_param{};
        part_coll_param.ctype = ccl_coll_partial;
        part_coll_param.stream = sched->coll_param.stream;
        part_coll_param.comm = comm;
        sched->add_partial_sched(part_coll_param);
    }

    part_scheds_vector.resize(part_count);

    for (idx = 0; idx < part_scheds.size(); idx++) {
        /* in this place all coll attributes for partial schedules
         * are taken from master schedule, including priority */
        part_scheds[idx]->coll_attr = coll_attr;
        part_scheds_vector[idx] = part_scheds[idx].get();
    }

    switch (coll_type) {
        case ccl_coll_barrier: break;
        case ccl_coll_bcast:
        case ccl_coll_reduce:
        case ccl_coll_allreduce:
        case ccl_coll_reduce_scatter:
        case ccl_coll_sparse_allreduce:
            base_count = coll_param.get_recv_count() / part_count;
            for (idx = 0; idx < counts.size(); idx++) {
                counts[idx] = base_count;
                offsets[idx] = idx * counts[idx] * dtype_size;
            }
            counts[counts.size() - 1] += coll_param.get_recv_count() % counts.size();
            // for (idx = 0; idx < part_scheds.size(); idx++) {
            //     part_scheds[idx]->coll_param.recv_counts.push_back(counts[idx]);
            // }
            break;
        case ccl_coll_alltoall:
        case ccl_coll_alltoallv:
            if (coll_type == ccl_coll_alltoallv) {
                a2av_send_count = std::accumulate(
                    coll_param.send_counts.begin(), coll_param.send_counts.end(), 0);
                a2av_recv_count = std::accumulate(
                    coll_param.recv_counts.begin(), coll_param.recv_counts.end(), 0);
            }
            else {
                a2av_send_count = coll_param.get_send_count() * comm_size;
                a2av_recv_count = coll_param.get_recv_count() * comm_size;
            }
            a2av_send_bytes = a2av_send_count * dtype_size;
            a2av_recv_bytes = a2av_recv_count * dtype_size;
            break;
        case ccl_coll_allgatherv:
            counts[0] = coll_param.get_recv_count(0);
            offsets[0] = 0;
            if (algo.allgatherv == ccl_coll_allgatherv_direct ||
                algo.allgatherv == ccl_coll_allgatherv_naive ||
                algo.allgatherv == ccl_coll_allgatherv_ring) {
            }
            else {
                for (idx = 1; idx < comm_size; idx++) {
                    counts[idx] = coll_param.get_recv_count(idx);
                    offsets[idx] = offsets[idx - 1] + counts[idx - 1] * dtype_size;
                }
            }
            ag_recv_count =
                std::accumulate(coll_param.recv_counts.begin(), coll_param.recv_counts.end(), 0);
            ag_recv_bytes = ag_recv_count * dtype_size;
            break;
        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }

    switch (coll_type) {
        case ccl_coll_barrier:
            sched->sync_partial_scheds();
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_barrier;
                param.dtype = ccl_datatype_int8;
                param.comm = comm;
                coll_entry_helper::add_coll_entry<ccl_coll_barrier>(part_scheds[idx].get(), param);
            }
            sched->sync_partial_scheds();
            break;
        case ccl_coll_bcast:
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_bcast;
                param.recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                            coll_param.get_recv_count() * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.count = counts[idx];
                param.dtype = dtype;
                param.root = coll_param.root;
                param.comm = comm;
                coll_entry_helper::add_coll_entry<ccl_coll_bcast>(part_scheds[idx].get(), param);
            }
            break;

        case ccl_coll_reduce:
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_reduce;
                param.send_buf = ccl_buffer(coll_param.get_send_buf_ptr(),
                                            coll_param.get_send_count() * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                            coll_param.get_recv_count() * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.count = counts[idx];
                param.dtype = dtype;
                param.reduction = coll_param.reduction;
                param.root = coll_param.root;
                param.comm = comm;
                param.stream = coll_param.stream;
                coll_entry_helper::add_coll_entry<ccl_coll_reduce>(part_scheds[idx].get(), param);
            }
            break;

        case ccl_coll_reduce_scatter:
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_reduce_scatter;

                bool inplace = coll_param.is_inplace();
                size_t recv_buf_size = coll_param.get_recv_count() * dtype_size;
                if (inplace)
                    recv_buf_size *= comm_size;

                param.send_buf = ccl_buffer(coll_param.get_send_buf_ptr(),
                                            coll_param.get_send_count() * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                            recv_buf_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.count = counts[idx];
                param.dtype = dtype;
                param.reduction = coll_param.reduction;
                param.comm = comm;
                coll_entry_helper::add_coll_entry<ccl_coll_reduce_scatter>(part_scheds[idx].get(),
                                                                           param);
            }
            break;

        case ccl_coll_allreduce: {
            ccl_parallelizer_prologue_ctx* main_ctx = nullptr;
            if (coll_attr.prologue_fn) {
                part_ctxs.reserve(part_count);

                main_ctx = (ccl_parallelizer_prologue_ctx*)part_scheds[0]
                               ->alloc_buffer(sizeof(ccl_parallelizer_prologue_ctx))
                               .get_ptr();
                main_ctx->part_idx = 0;
                main_ctx->part_count = 1;
                entry_factory::make_entry<prologue_entry>(
                    part_scheds[0].get(),
                    coll_attr.prologue_fn,
                    ccl_buffer(coll_param.get_send_buf_ptr(),
                               coll_param.get_send_count() * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    coll_param.get_send_count(),
                    dtype,
                    &(main_ctx->buf),
                    &(main_ctx->count),
                    &(main_ctx->dt_idx));

                sched->sync_partial_scheds();

                for (idx = 0; idx < part_count; idx++) {
                    ccl_parallelizer_prologue_ctx* part_ctx =
                        (ccl_parallelizer_prologue_ctx*)part_scheds[idx]
                            ->alloc_buffer(sizeof(ccl_parallelizer_prologue_ctx))
                            .get_ptr();
                    part_ctxs.emplace_back(part_ctx);

                    part_ctx->part_idx = idx;
                    part_ctx->part_count = part_count;

                    entry_factory::make_entry<copy_entry>(
                        part_scheds[idx].get(),
                        ccl_buffer(main_ctx, sizeof(ccl_parallelizer_prologue_ctx)),
                        ccl_buffer(part_ctx, sizeof(ccl_parallelizer_prologue_ctx)),
                        sizeof(void*) + sizeof(size_t) + sizeof(ccl::datatype),
                        ccl_datatype_int8);
                }
            }

            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_allreduce;
                if (!coll_attr.prologue_fn) {
                    param.send_buf = ccl_buffer(coll_param.get_send_buf_ptr(),
                                                coll_param.get_send_count() * dtype_size,
                                                offsets[idx],
                                                ccl_buffer_type::INDIRECT);
                    param.recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                coll_param.get_recv_count() * dtype_size,
                                                offsets[idx],
                                                ccl_buffer_type::INDIRECT);
                    param.count = counts[idx];
                    param.dtype = dtype;
                }
                else {
                    param.send_buf = ccl_buffer();
                    param.recv_buf = ccl_buffer();
                    param.count = 0;
                    param.dtype = ccl_datatype_int8;
                }
                param.reduction = coll_param.reduction;
                param.comm = comm;
                param.stream = coll_param.stream;

                auto entry = coll_entry_helper::add_coll_entry<ccl_coll_allreduce>(
                    part_scheds[idx].get(), param);

                if (coll_attr.prologue_fn) {
                    auto part_ctx = part_ctxs[idx];
                    entry->set_field_fn<ccl_sched_entry_field_send_buf>(
                        ccl_parallelizer_prologue_get_buf, part_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_recv_buf>(
                        ccl_parallelizer_prologue_get_buf, part_ctx, false); // in-place
                    entry->set_field_fn<ccl_sched_entry_field_cnt>(
                        ccl_parallelizer_prologue_get_count, part_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_dtype>(
                        ccl_parallelizer_prologue_get_dtype, part_ctx, false);
                }
            }

            if (coll_attr.prologue_fn && !coll_attr.epilogue_fn) {
                sched->sync_partial_scheds();

                auto entry = entry_factory::make_entry<copy_entry>(
                    part_scheds[0].get(),
                    ccl_buffer(), /* in_buf */
                    ccl_buffer(coll_param.get_recv_buf_ptr(),
                               coll_param.get_recv_count() * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    0, /* count */
                    ccl_datatype_int8);
                entry->set_field_fn<ccl_sched_entry_field_in_buf>(
                    ccl_parallelizer_prologue_get_buf, main_ctx, false);
                entry->set_field_fn<ccl_sched_entry_field_cnt>(
                    ccl_parallelizer_prologue_get_count, main_ctx, false);
                entry->set_field_fn<ccl_sched_entry_field_dtype>(
                    ccl_parallelizer_prologue_get_dtype, main_ctx, false);
            }

            if (coll_attr.epilogue_fn) {
                sched->sync_partial_scheds();
                auto entry = entry_factory::make_entry<epilogue_entry>(
                    part_scheds[0].get(),
                    coll_attr.epilogue_fn,
                    ccl_buffer(coll_param.get_recv_buf_ptr(),
                               coll_param.get_recv_count() * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    coll_param.get_recv_count(),
                    dtype,
                    ccl_buffer(coll_param.get_recv_buf_ptr(),
                               coll_param.get_recv_count() * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    coll_param.get_recv_count(),
                    dtype);
                if (coll_attr.prologue_fn) {
                    entry->set_field_fn<ccl_sched_entry_field_in_buf>(
                        ccl_parallelizer_prologue_get_buf, main_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_in_cnt>(
                        ccl_parallelizer_prologue_get_count, main_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_in_dtype>(
                        ccl_parallelizer_prologue_get_dtype, main_ctx, false);
                }
            }
            break;
        }

        case ccl_coll_allgatherv: {
            if (algo.allgatherv == ccl_coll_allgatherv_direct ||
                algo.allgatherv == ccl_coll_allgatherv_naive ||
                algo.allgatherv == ccl_coll_allgatherv_ring) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_allgatherv;
                param.send_buf = ccl_buffer(coll_param.get_send_buf_ptr(),
                                            coll_param.get_send_count() * dtype_size,
                                            ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(
                    coll_param.get_recv_buf_ptr(), ag_recv_bytes, ccl_buffer_type::INDIRECT);
                param.send_count = coll_param.get_send_count();
                param.recv_counts = coll_param.recv_counts.data();
                param.dtype = dtype;
                param.comm = comm;
                coll_entry_helper::add_coll_entry<ccl_coll_allgatherv>(part_scheds[0].get(), param);
            }
            else {
                CCL_ASSERT(algo.allgatherv == ccl_coll_allgatherv_flat ||
                               algo.allgatherv == ccl_coll_allgatherv_multi_bcast,
                           "unexpected allgatherv algorithm");

                if (algo.allgatherv == ccl_coll_allgatherv_flat) {
                    ccl_coll_build_flat_allgatherv(sched, part_scheds_vector, coll_param);
                }
                else {
                    ccl_coll_build_multi_bcast_allgatherv(
                        sched, part_scheds_vector, coll_param, max_data_partition_count);
                }
            }
            break;
        }

        case ccl_coll_alltoall:
        case ccl_coll_alltoallv: {
            if (algo.alltoall == ccl_coll_alltoall_naive ||
                algo.alltoallv == ccl_coll_alltoallv_naive) {
                ccl_coll_build_naive_alltoallv(sched, part_scheds_vector, coll_param);
            }
            else if (algo.alltoall == ccl_coll_alltoall_scatter ||
                     algo.alltoallv == ccl_coll_alltoallv_scatter) {
                ccl_coll_build_scatter_alltoallv(sched, part_scheds_vector, coll_param);
            }
            else if (algo.alltoall == ccl_coll_alltoall_scatter_barrier ||
                     algo.alltoallv == ccl_coll_alltoallv_scatter_barrier) {
                ccl_coll_build_scatter_barrier_alltoallv(sched, part_scheds_vector, coll_param);
            }
            else {
                ccl_coll_entry_param param{};
                param.ctype = coll_type;
                param.send_buf = ccl_buffer(
                    coll_param.get_send_buf_ptr(), a2av_send_bytes, ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(
                    coll_param.get_recv_buf_ptr(), a2av_recv_bytes, ccl_buffer_type::INDIRECT);
                param.dtype = dtype;
                param.comm = comm;

                if (coll_type == ccl_coll_alltoall) {
                    param.count = coll_param.get_send_count();
                    coll_entry_helper::add_coll_entry<ccl_coll_alltoall>(part_scheds[0].get(),
                                                                         param);
                }
                else {
                    param.send_counts = coll_param.send_counts.data();
                    param.recv_counts = coll_param.recv_counts.data();
                    coll_entry_helper::add_coll_entry<ccl_coll_alltoallv>(part_scheds[0].get(),
                                                                          param);
                }
            }
            break;
        }

        case ccl_coll_sparse_allreduce: {
            ccl_parallelizer_sparse_callback_ctx* i_ctx =
                (ccl_parallelizer_sparse_callback_ctx*)part_scheds[0]
                    ->alloc_buffer(sizeof(ccl_parallelizer_sparse_callback_ctx))
                    .get_ptr();

            i_ctx->buf = coll_param.sparse_param.recv_ind_buf;
            i_ctx->dtype_size = coll_param.sparse_param.itype.size();

            ccl_parallelizer_sparse_callback_ctx* v_ctx =
                (ccl_parallelizer_sparse_callback_ctx*)part_scheds[0]
                    ->alloc_buffer(sizeof(ccl_parallelizer_sparse_callback_ctx))
                    .get_ptr();

            v_ctx->buf = coll_param.sparse_param.recv_val_buf;
            v_ctx->dtype_size = dtype.size();

            for (idx = 0; idx < part_count; idx++) {
                CCL_CALL(ccl_coll_build_sparse_allreduce(
                    part_scheds[idx].get(),
                    ccl_buffer(&(coll_param.sparse_param.send_ind_buf),
                               coll_param.sparse_param.send_ind_count *
                                   coll_param.sparse_param.itype.size(),
                               offsets[idx],
                               ccl_buffer_type::INDIRECT),
                    coll_param.sparse_param.send_ind_count,
                    ccl_buffer(&(coll_param.sparse_param.send_val_buf),
                               coll_param.sparse_param.send_val_count * dtype_size,
                               offsets[idx],
                               ccl_buffer_type::INDIRECT),
                    coll_param.sparse_param.send_val_count,
                    &(i_ctx->buf),
                    &(i_ctx->count),
                    &(v_ctx->buf),
                    &(v_ctx->count),
                    coll_param.sparse_param.itype,
                    dtype,
                    coll_param.reduction,
                    comm));
            }

            if (coll_attr.sparse_allreduce_completion_fn) {
                CCL_THROW_IF_NOT(!coll_attr.sparse_allreduce_alloc_fn);

                sched->sync_partial_scheds();

                auto entry = entry_factory::make_entry<sparse_allreduce_completion_entry>(
                    part_scheds[0].get(),
                    coll_attr.sparse_allreduce_completion_fn,
                    coll_attr.sparse_allreduce_fn_ctx,
                    ccl_buffer(),
                    0,
                    coll_param.sparse_param.itype,
                    ccl_buffer(),
                    0,
                    dtype);

                entry->set_field_fn<ccl_sched_entry_field_idx_buf>(
                    ccl_parallelizer_sparse_callback_get_buf, i_ctx, false);

                entry->set_field_fn<ccl_sched_entry_field_idx_cnt>(
                    ccl_parallelizer_sparse_callback_get_count, i_ctx, false);

                entry->set_field_fn<ccl_sched_entry_field_val_buf>(
                    ccl_parallelizer_sparse_callback_get_buf, v_ctx, false);

                entry->set_field_fn<ccl_sched_entry_field_val_cnt>(
                    ccl_parallelizer_sparse_callback_get_count, v_ctx, false);
            }
            break;
        }

        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }
    return status;
}
