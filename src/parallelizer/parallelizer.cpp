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
    ccl_coll_param& coll_param = sched->coll_param;
    if (coll_param.stream && coll_param.stream->is_sycl_device_stream() &&
        (coll_param.device_send_buf || coll_param.device_recv_buf)) {
        process_pre_post_copies(sched);
    }
#endif /* CCL_ENABLE_SYCL */

    /* should be the last call in the sequence of process_* calls
       because it sets dependencies for all partial schedules
       which already should be filled */
    process_deps(sched);

    return ccl::status::success;
}

ccl::status ccl_parallelizer::process_deps(ccl_master_sched* sched) {
    auto& part_scheds = sched->partial_scheds;
    ccl_sched* deps_sched = part_scheds[0].get();
    size_t part_count = part_scheds.size();

    for (size_t idx = 0; idx < part_count; idx++) {
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
    ccl_sched* copy_sched = part_scheds[0].get();
    size_t part_count = part_scheds.size();

    ccl_coll_param& coll_param = sched->coll_param;
    ccl_comm* comm = coll_param.comm;
    int comm_size = comm->size();
    int my_rank = comm->rank();

    const ccl_datatype& dtype = coll_param.dtype;
    size_t dtype_size = dtype.size();

    ccl_coll_type coll_type = coll_param.ctype;

    size_t d2h_bytes = 0, h2d_bytes = 0;
    size_t d2h_count = 0, h2d_count = 0;

    void* device_in_buf = nullptr;
    void* device_out_buf = nullptr;
    void* host_in_buf = nullptr;
    void* host_out_buf = nullptr;

    size_t device_in_buf_offset = 0;

    switch (coll_type) {
        case ccl_coll_bcast:
            if (comm->rank() == coll_param.root)
                d2h_count = coll_param.count;
            else
                d2h_count = 0;
            h2d_count = coll_param.count;
            break;

        case ccl_coll_reduce:
            d2h_count = coll_param.count;
            if (my_rank == coll_param.root)
                h2d_count = coll_param.count;
            else
                h2d_count = 0;
            break;

        case ccl_coll_reduce_scatter:
            d2h_count = coll_param.count * comm_size;
            h2d_count = coll_param.count;
            break;

        case ccl_coll_allreduce: d2h_count = h2d_count = coll_param.count; break;

        case ccl_coll_allgatherv:
            if (coll_param.device_send_buf == coll_param.device_recv_buf) {
                device_in_buf_offset =
                    std::accumulate(coll_param.recv_counts, coll_param.recv_counts + my_rank, 0);
                LOG_TRACE("device_in_buf_offset = ", device_in_buf_offset);
            }
            d2h_count = coll_param.send_count;
            h2d_count =
                std::accumulate(coll_param.recv_counts, coll_param.recv_counts + comm_size, 0);
            break;

        case ccl_coll_alltoall: d2h_count = h2d_count = coll_param.count * comm_size; break;
        case ccl_coll_alltoallv:
            d2h_count =
                std::accumulate(coll_param.send_counts, coll_param.send_counts + comm_size, 0);
            h2d_count =
                std::accumulate(coll_param.recv_counts, coll_param.recv_counts + comm_size, 0);
            break;

        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }

    device_in_buf = &(coll_param.device_send_buf);
    host_in_buf = (void*)coll_param.send_buf;
    d2h_bytes = d2h_count * dtype_size;

    host_out_buf = coll_param.recv_buf;
    device_out_buf = &(coll_param.device_recv_buf);
    h2d_bytes = h2d_count * dtype_size;

    if (d2h_bytes) {
        for (size_t idx = 0; idx < part_count; idx++) {
            part_scheds[idx]->set_add_mode(ccl_sched_add_front);
        }
        sched->sync_partial_scheds();

        entry_factory::make_entry<sycl_copy_entry>(
            copy_sched,
            copy_direction::d2h,
            ccl_buffer(device_in_buf, d2h_bytes, ccl_buffer_type::INDIRECT),
            ccl_buffer(host_in_buf, d2h_bytes),
            d2h_count,
            dtype,
            coll_param.stream,
            device_in_buf_offset);
    }

    if (h2d_bytes) {
        for (size_t idx = 0; idx < part_count; idx++) {
            part_scheds[idx]->set_add_mode(ccl_sched_add_back);
        }
        sched->sync_partial_scheds();

        entry_factory::make_entry<sycl_copy_entry>(
            copy_sched,
            copy_direction::h2d,
            ccl_buffer(host_out_buf, h2d_bytes),
            ccl_buffer(device_out_buf, h2d_bytes, ccl_buffer_type::INDIRECT),
            h2d_count,
            dtype,
            coll_param.stream);
        part_scheds[0]->add_barrier();
    }

    return ccl::status::success;
}
#endif /* CCL_ENABLE_SYCL */

ccl::status ccl_parallelizer::process_base(ccl_master_sched* sched) {
    /* TODO: split on per-collective classes */

    CCL_ASSERT(sched);

    ccl::global_data& data = ccl::global_data::get();

    ccl::status status = ccl::status::success;
    size_t part_count = 1, idx, base_count, dtype_size, comm_size, my_rank;

    ccl_coll_param& coll_param = sched->coll_param;
    ccl_coll_param_copy* coll_param_copy = &(sched->coll_param_copy);
    ccl_coll_attr* coll_attr = &(sched->coll_attr);
    ccl_comm* comm = coll_param.comm;

    const ccl_datatype& dtype = coll_param.dtype;
    dtype_size = dtype.size();
    comm_size = comm->size();
    my_rank = comm->rank();

    ccl_coll_type coll_type = coll_param.ctype;

    std::vector<size_t> counts;
    std::vector<size_t> offsets;
    auto& part_scheds = sched->partial_scheds;
    std::vector<ccl_sched*> part_scheds_vector;

    const size_t* recv_counts = nullptr;

    std::vector<ccl_buffer> ag_recv_bufs;
    size_t ag_recv_bytes = 0, ag_recv_count = 0;
    size_t a2av_send_bytes = 0, a2av_recv_bytes = 0;
    size_t a2av_send_count = 0, a2av_recv_count = 0;

    ccl_coll_allgatherv_algo ag_algo = ccl_coll_allgatherv_naive;
    ccl_coll_alltoall_algo a2a_algo = ccl_coll_alltoall_direct;
    ccl_coll_alltoallv_algo a2av_algo = ccl_coll_alltoallv_direct;
    ccl_coll_bcast_algo ag_mbcast_algo = ccl_coll_bcast_naive;

    std::vector<ccl_parallelizer_prologue_ctx*> part_ctxs;

    ccl_selector_param selector_param;
    selector_param.ctype = coll_type;
    selector_param.count = coll_param.count;
    selector_param.recv_counts = coll_param.recv_counts;
    selector_param.dtype = dtype;
    selector_param.comm = comm;

    switch (coll_type) {
        case ccl_coll_barrier: part_count = max_data_partition_count; break;
        case ccl_coll_bcast:
            if (ccl::global_data::env().bcast_part_count != CCL_ENV_SIZET_NOT_SPECIFIED) {
                part_count = ccl::global_data::env().bcast_part_count;
                break;
            }
        case ccl_coll_reduce:
        case ccl_coll_allreduce:
            if ((coll_param.count * dtype_size <= ccl::global_data::env().max_short_size) ||
                (coll_param.count < max_data_partition_count)) {
                part_count = 1;
            }
            else {
                /* to workaround lack of large msg protocol on ATL level */
                part_count = (coll_param.count * dtype_size) / CCL_ATL_LARGE_MSG_SIZE;
                if (part_count < max_data_partition_count)
                    part_count = max_data_partition_count;
            }
            break;
        case ccl_coll_alltoall:
            a2a_algo = data.algorithm_selector->get<ccl_coll_alltoall>(selector_param);
            if (a2a_algo == ccl_coll_alltoall_direct) {
                part_count = 1;
            }
            else {
                part_count = std::min(comm_size, max_data_partition_count);
            }
            break;
        case ccl_coll_alltoallv:
            a2av_algo = data.algorithm_selector->get<ccl_coll_alltoallv>(selector_param);
            coll_param_copy->a2av_send_counts.assign((size_t*)coll_param.send_counts,
                                                     (size_t*)coll_param.send_counts + comm_size);
            coll_param_copy->a2av_recv_counts.assign((size_t*)coll_param.recv_counts,
                                                     (size_t*)coll_param.recv_counts + comm_size);

            if (a2av_algo == ccl_coll_alltoallv_direct) {
                part_count = 1;
            }
            else {
                part_count = std::min(comm_size, max_data_partition_count);
            }
            break;
        case ccl_coll_allgatherv:
            selector_param.vector_buf = coll_attr->vector_buf;
            ag_algo = data.algorithm_selector->get<ccl_coll_allgatherv>(selector_param);
            coll_param_copy->ag_recv_counts.assign((size_t*)coll_param.recv_counts,
                                                   (size_t*)coll_param.recv_counts + comm_size);

            if (ag_algo == ccl_coll_allgatherv_direct || ag_algo == ccl_coll_allgatherv_naive ||
                ag_algo == ccl_coll_allgatherv_ring) {
                part_count = 1;
            }
            else if (ag_algo == ccl_coll_allgatherv_multi_bcast ||
                     ag_algo == ccl_coll_allgatherv_flat) {
                part_count = comm_size;
                ag_recv_bufs.resize(comm_size);

                if (coll_attr->vector_buf) {
                    coll_param_copy->ag_recv_bufs.assign((void**)coll_param.recv_buf,
                                                         (void**)coll_param.recv_buf + comm_size);
                }

                if (ag_algo == ccl_coll_allgatherv_multi_bcast) {
                    selector_param.ctype = ccl_coll_bcast;
                    selector_param.count = sched->coll_param.send_count;
                    selector_param.dtype = dtype;
                    ag_mbcast_algo = data.algorithm_selector->get<ccl_coll_bcast>(selector_param);
                    if (ag_mbcast_algo == ccl_coll_bcast_direct) {
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
                CCL_FATAL("unexpected allgatherv_algo ", ag_algo);
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

    if (coll_type == ccl_coll_allgatherv && ag_algo == ccl_coll_allgatherv_multi_bcast &&
        ag_mbcast_algo == ccl_coll_bcast_direct) {
        counts.resize(comm_size, 0);
        offsets.resize(comm_size, 0);
    }
    else {
        counts.resize(part_count, 0);
        offsets.resize(part_count, 0);
    }

    for (idx = 0; idx < part_count; idx++) {
        ccl_coll_param part_coll_param{};
        part_coll_param.ctype = sched->coll_param.ctype;
        part_coll_param.dtype = sched->coll_param.dtype;
        part_coll_param.stream = sched->coll_param.stream;
        part_coll_param.comm = comm;
        sched->add_partial_sched(part_coll_param);
    }

    part_scheds_vector.resize(part_count);

    for (idx = 0; idx < part_scheds.size(); idx++) {
        /* in this place all coll attributes for partial schedules
         * are taken from master schedule, including priority */
        part_scheds[idx]->coll_attr = *coll_attr;
        part_scheds_vector[idx] = part_scheds[idx].get();
    }

    switch (coll_type) {
        case ccl_coll_barrier: break;
        case ccl_coll_bcast:
        case ccl_coll_reduce:
        case ccl_coll_allreduce:
        case ccl_coll_reduce_scatter:
        case ccl_coll_sparse_allreduce:
            base_count = coll_param.count / part_count;
            for (idx = 0; idx < counts.size(); idx++) {
                counts[idx] = base_count;
                offsets[idx] = idx * counts[idx] * dtype_size;
            }
            counts[counts.size() - 1] += coll_param.count % counts.size();
            for (idx = 0; idx < part_scheds.size(); idx++) {
                part_scheds[idx]->coll_param.count = counts[idx];
            }
            break;
        case ccl_coll_alltoall:
        case ccl_coll_alltoallv:
            if (coll_type == ccl_coll_alltoallv) {
                CCL_ASSERT(coll_param.send_counts);
                CCL_ASSERT(coll_param.recv_counts);
                a2av_send_count =
                    std::accumulate(coll_param.send_counts, coll_param.send_counts + comm_size, 0);
                a2av_recv_count =
                    std::accumulate(coll_param.recv_counts, coll_param.recv_counts + comm_size, 0);
            }
            else {
                a2av_send_count = coll_param.count * comm_size;
                a2av_recv_count = coll_param.count * comm_size;
            }
            a2av_send_bytes = a2av_send_count * dtype_size;
            a2av_recv_bytes = a2av_recv_count * dtype_size;
            break;
        case ccl_coll_allgatherv:
            recv_counts = coll_param.recv_counts;
            CCL_ASSERT(recv_counts);
            counts[0] = recv_counts[0];
            offsets[0] = 0;
            if (ag_algo == ccl_coll_allgatherv_direct || ag_algo == ccl_coll_allgatherv_naive ||
                ag_algo == ccl_coll_allgatherv_ring) {
            }
            else {
                for (idx = 1; idx < comm_size; idx++) {
                    counts[idx] = recv_counts[idx];
                    offsets[idx] = offsets[idx - 1] + counts[idx - 1] * dtype_size;
                }
            }
            ag_recv_count =
                std::accumulate(coll_param.recv_counts, coll_param.recv_counts + comm_size, 0);
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
                param.recv_buf = ccl_buffer(&(coll_param.recv_buf),
                                            coll_param.count * dtype_size,
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
                param.send_buf = ccl_buffer(&(coll_param.send_buf),
                                            coll_param.count * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(&(coll_param.recv_buf),
                                            coll_param.count * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.count = counts[idx];
                param.dtype = dtype;
                param.reduction = coll_param.reduction;
                param.root = coll_param.root;
                param.comm = comm;
                coll_entry_helper::add_coll_entry<ccl_coll_reduce>(part_scheds[idx].get(), param);
            }
            break;

        case ccl_coll_reduce_scatter:
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_reduce_scatter;

                bool inplace = (coll_param.send_buf == coll_param.recv_buf) ? true : false;
                size_t recv_buf_size = coll_param.count * dtype_size;
                if (inplace)
                    recv_buf_size *= comm_size;

                param.send_buf = ccl_buffer(&(coll_param.send_buf),
                                            coll_param.count * comm_size * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(
                    &(coll_param.recv_buf), recv_buf_size, offsets[idx], ccl_buffer_type::INDIRECT);
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
            if (coll_attr->prologue_fn) {
                part_ctxs.reserve(part_count);

                main_ctx = (ccl_parallelizer_prologue_ctx*)part_scheds[0]
                               ->alloc_buffer(sizeof(ccl_parallelizer_prologue_ctx))
                               .get_ptr();
                main_ctx->part_idx = 0;
                main_ctx->part_count = 1;
                entry_factory::make_entry<prologue_entry>(part_scheds[0].get(),
                                                          coll_attr->prologue_fn,
                                                          ccl_buffer(&(coll_param.send_buf),
                                                                     coll_param.count * dtype_size,
                                                                     ccl_buffer_type::INDIRECT),
                                                          coll_param.count,
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
                if (!coll_attr->prologue_fn) {
                    param.send_buf = ccl_buffer(&(coll_param.send_buf),
                                                coll_param.count * dtype_size,
                                                offsets[idx],
                                                ccl_buffer_type::INDIRECT);
                    param.recv_buf = ccl_buffer(&(coll_param.recv_buf),
                                                coll_param.count * dtype_size,
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

                auto entry = coll_entry_helper::add_coll_entry<ccl_coll_allreduce>(
                    part_scheds[idx].get(), param);

                if (coll_attr->prologue_fn) {
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

            if (coll_attr->prologue_fn && !coll_attr->epilogue_fn) {
                sched->sync_partial_scheds();

                auto entry =
                    entry_factory::make_entry<copy_entry>(part_scheds[0].get(),
                                                          ccl_buffer(), /* in_buf */
                                                          ccl_buffer(&(coll_param.recv_buf),
                                                                     coll_param.count * dtype_size,
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

            if (coll_attr->epilogue_fn) {
                sched->sync_partial_scheds();
                auto entry = entry_factory::make_entry<epilogue_entry>(
                    part_scheds[0].get(),
                    coll_attr->epilogue_fn,
                    ccl_buffer(&(coll_param.recv_buf),
                               coll_param.count * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    coll_param.count,
                    dtype,
                    ccl_buffer(&(coll_param.recv_buf),
                               coll_param.count * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    coll_param.count,
                    dtype);
                if (coll_attr->prologue_fn) {
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
            if (ag_algo == ccl_coll_allgatherv_direct || ag_algo == ccl_coll_allgatherv_naive ||
                ag_algo == ccl_coll_allgatherv_ring) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_allgatherv;
                param.send_buf = ccl_buffer(&(coll_param.send_buf),
                                            coll_param.send_count * dtype_size,
                                            ccl_buffer_type::INDIRECT);
                param.recv_buf =
                    ccl_buffer(&(coll_param.recv_buf), ag_recv_bytes, ccl_buffer_type::INDIRECT);
                param.send_count = coll_param.send_count;
                param.recv_counts = coll_param_copy->ag_recv_counts.data();
                param.dtype = dtype;
                param.comm = comm;
                coll_entry_helper::add_coll_entry<ccl_coll_allgatherv>(part_scheds[0].get(), param);
            }
            else {
                CCL_ASSERT(ag_algo == ccl_coll_allgatherv_flat ||
                               ag_algo == ccl_coll_allgatherv_multi_bcast,
                           "unexpected allgatherv algorithm");

                for (idx = 0; idx < comm_size; idx++) {
                    if (coll_attr->vector_buf) {
                        ag_recv_bufs[idx].set(&(coll_param_copy->ag_recv_bufs[idx]),
                                              counts[idx] * dtype_size,
                                              ccl_buffer_type::INDIRECT);
                    }
                    else {
                        ag_recv_bufs[idx].set(&(coll_param.recv_buf),
                                              ag_recv_bytes,
                                              offsets[idx],
                                              ccl_buffer_type::INDIRECT);
                    }
                }

                if (ag_algo == ccl_coll_allgatherv_flat) {
                    auto send_seg = ccl_buffer(&(coll_param.send_buf),
                                               coll_param.send_count * dtype_size,
                                               ccl_buffer_type::INDIRECT);

                    if (coll_param.send_buf != coll_param.recv_buf) {
                        entry_factory::make_entry<copy_entry>(
                            part_scheds[2 * my_rank % part_count].get(),
                            ccl_buffer(&(coll_param.send_buf),
                                       coll_param.send_count * dtype_size,
                                       ccl_buffer_type::INDIRECT),
                            ag_recv_bufs[my_rank],
                            counts[my_rank],
                            dtype);
                    }
                    else {
                        send_seg = ccl_buffer(&(coll_param.send_buf),
                                              ag_recv_bytes,
                                              offsets[my_rank],
                                              ccl_buffer_type::INDIRECT);
                    }

                    CCL_ASSERT(part_count == comm_size);

                    for (idx = 0; idx < part_count; idx++) {
                        if (idx == my_rank)
                            continue;

                        entry_factory::make_entry<recv_entry>(
                            part_scheds[(my_rank + idx) % part_count].get(),
                            ag_recv_bufs[idx],
                            counts[idx],
                            dtype,
                            idx,
                            comm);
                        entry_factory::make_entry<send_entry>(
                            part_scheds[(my_rank + idx) % part_count].get(),
                            send_seg,
                            counts[my_rank],
                            dtype,
                            idx,
                            comm);
                    }
                    sched->sync_partial_scheds();
                }
                else {
                    CCL_ASSERT(ag_algo == ccl_coll_allgatherv_multi_bcast);

                    if (coll_param.send_buf != coll_param.recv_buf) {
                        std::vector<size_t> copy_counts(max_data_partition_count);
                        std::vector<size_t> copy_offsets(max_data_partition_count);
                        for (idx = 0; idx < max_data_partition_count; idx++) {
                            copy_counts[idx] = counts[comm->rank()] / max_data_partition_count;
                            copy_offsets[idx] = idx * copy_counts[idx] * dtype_size;
                        }
                        copy_counts[max_data_partition_count - 1] +=
                            counts[comm->rank()] % max_data_partition_count;

                        CCL_ASSERT(part_scheds.size() >= max_data_partition_count);

                        for (idx = 0; idx < max_data_partition_count; idx++) {
                            entry_factory::make_entry<copy_entry>(
                                part_scheds[idx].get(),
                                ccl_buffer(&(coll_param.send_buf),
                                           coll_param.send_count * dtype_size,
                                           copy_offsets[idx],
                                           ccl_buffer_type::INDIRECT),
                                ag_recv_bufs[comm->rank()] + copy_offsets[idx],
                                copy_counts[idx],
                                dtype);
                        }
                        sched->sync_partial_scheds();
                    }

                    for (idx = 0; idx < comm_size; idx++) {
                        ccl_coll_entry_param param{};
                        param.ctype = ccl_coll_bcast;
                        param.recv_buf = ag_recv_bufs[idx];
                        param.count = counts[idx];
                        param.dtype = dtype;
                        param.root = idx;
                        param.comm = comm;
                        coll_entry_helper::add_coll_entry<ccl_coll_bcast>(
                            part_scheds[idx % part_count].get(), param);
                    }
                }
            }
            break;
        }

        case ccl_coll_alltoall:
        case ccl_coll_alltoallv: {
            if (a2a_algo == ccl_coll_alltoall_naive || a2av_algo == ccl_coll_alltoallv_naive) {
                ccl_coll_build_naive_alltoallv(sched, part_scheds_vector, coll_param);
            }
            else if (a2a_algo == ccl_coll_alltoall_scatter ||
                     a2av_algo == ccl_coll_alltoallv_scatter) {
                ccl_coll_build_scatter_alltoallv(sched, part_scheds_vector, coll_param);
            }
            else if (a2a_algo == ccl_coll_alltoall_scatter_barrier ||
                     a2av_algo == ccl_coll_alltoallv_scatter_barrier) {
                ccl_coll_build_scatter_barrier_alltoallv(sched, part_scheds_vector, coll_param);
            }
            else {
                ccl_coll_entry_param param{};
                param.ctype = coll_type;
                param.send_buf =
                    ccl_buffer(&(coll_param.send_buf), a2av_send_bytes, ccl_buffer_type::INDIRECT);
                param.recv_buf =
                    ccl_buffer(&(coll_param.recv_buf), a2av_recv_bytes, ccl_buffer_type::INDIRECT);
                param.dtype = dtype;
                param.comm = comm;

                if (coll_type == ccl_coll_alltoall) {
                    param.count = coll_param.count;
                    coll_entry_helper::add_coll_entry<ccl_coll_alltoall>(part_scheds[0].get(),
                                                                         param);
                }
                else {
                    param.send_counts = coll_param_copy->a2av_send_counts.data();
                    param.recv_counts = coll_param_copy->a2av_recv_counts.data();
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

            if (coll_attr->sparse_allreduce_completion_fn) {
                CCL_THROW_IF_NOT(!coll_attr->sparse_allreduce_alloc_fn);

                sched->sync_partial_scheds();

                auto entry = entry_factory::make_entry<sparse_allreduce_completion_entry>(
                    part_scheds[0].get(),
                    coll_attr->sparse_allreduce_completion_fn,
                    coll_attr->sparse_allreduce_fn_ctx,
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
