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

#include "coll/selection/selection.hpp"
#include "common/env/env.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/entry/factory/entry_factory.hpp"

typedef struct
{
    /* keep these 4 fields on the top of structure */
    void* buf;
    size_t count;
    ccl_datatype_t dtype;
    size_t dtype_size;
    /*---*/

    ccl_datatype_internal dtype_internal;
    size_t part_idx;
    size_t part_count;
} ccl_parallelizer_prologue_ctx;

ccl_status_t ccl_parallelizer_prologue_get_buf(const void* ctx, void* field_ptr)
{
    ccl_parallelizer_prologue_ctx* pctx = (ccl_parallelizer_prologue_ctx*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(pctx->buf, pctx->count * pctx->dtype_size,
                 pctx->part_idx * (pctx->count / pctx->part_count) * pctx->dtype_size);
    return ccl_status_success;
}

ccl_status_t ccl_parallelizer_prologue_get_count(const void* ctx, void* field_ptr)
{
    ccl_parallelizer_prologue_ctx* pctx = (ccl_parallelizer_prologue_ctx*)ctx;
    size_t count = pctx->count / pctx->part_count;
    if (pctx->part_idx == (pctx->part_count - 1)) count += pctx->count % pctx->part_count;
    size_t* count_ptr = (size_t*)field_ptr;
    *count_ptr = count;
    return ccl_status_success;
}

ccl_status_t ccl_parallelizer_prologue_get_dtype(const void* ctx, void* field_ptr)
{
    ccl_parallelizer_prologue_ctx* pctx = (ccl_parallelizer_prologue_ctx*)ctx;
    ccl_datatype_internal_t* dtype_ptr = (ccl_datatype_internal_t*)field_ptr;
    pctx->dtype_internal.type = pctx->dtype;
    pctx->dtype_internal.size = pctx->dtype_size;
    pctx->dtype_internal.name = "UNKNOWN";
    *dtype_ptr = &pctx->dtype_internal;
    return ccl_status_success;
}

ccl_status_t ccl_parallelizer::process(ccl_master_sched* sched)
{
    CCL_ASSERT(sched);

    ccl_status_t status = ccl_status_success;
    size_t part_count = 1, idx, base_count, dtype_size;
    ccl_coll_param* coll_param = &(sched->coll_param);
    ccl_datatype_internal_t dtype = coll_param->dtype;
    dtype_size = ccl_datatype_get_size(dtype);
    ccl_coll_type coll_type = coll_param->ctype;

    std::vector<size_t> counts;
    std::vector<size_t> offsets;
    auto& part_scheds = sched->partial_scheds;
    const size_t* recv_counts = nullptr;

    std::vector<ccl_buffer> ag_recv_bufs;
    size_t ag_recv_bytes = 0, ag_recv_count = 0;

    ccl_datatype_internal_t itype = ccl_dtype_internal_none;
    size_t itype_size = 0;

    ccl_coll_allgatherv_algo ag_algo = ccl_coll_allgatherv_naive;
    ccl_coll_alltoall_algo a2a_algo = ccl_coll_alltoall_direct;
    ccl_coll_bcast_algo ag_mbcast_algo = ccl_coll_bcast_naive;

    switch (coll_type)
    {
        case ccl_coll_barrier:
            part_count = max_data_partition_count;
            break;
        case ccl_coll_bcast:
        case ccl_coll_reduce:
        case ccl_coll_allreduce:
            if (coll_param->count * dtype_size <= env_data.max_short_size)
            {
                part_count = 1;
            }
            else
            {
                part_count = max_data_partition_count;
            }
            break;
        case ccl_coll_alltoall:
            a2a_algo = global_data.algorithm_selector->get<ccl_coll_alltoall>(sched->coll_param);
            if (a2a_algo == ccl_coll_alltoall_scatter)
            {
                part_count = coll_param->comm->size();
            }
            else if (a2a_algo == ccl_coll_alltoall_scatter_message)
            {
                if (coll_param->count * dtype_size <= env_data.max_short_size)
                {
                    part_count = 1;
                }
                else
                {
                    part_count = max_data_partition_count;
                }
            }
            else
            {
                part_count = 1;
            }
            break;
        case ccl_coll_allgatherv:
            ag_algo = global_data.algorithm_selector->get<ccl_coll_allgatherv>(sched->coll_param);
            if (ag_algo == ccl_coll_allgatherv_direct ||
                ag_algo == ccl_coll_allgatherv_naive)
            {
                part_count = 1;
            }
            else if (ag_algo == ccl_coll_allgatherv_multi_bcast ||
                     ag_algo == ccl_coll_allgatherv_flat)
            {
                part_count = coll_param->comm->size();
                ag_recv_bufs.resize(coll_param->comm->size());

                if (env_data.enable_allgatherv_iov)
                {
                    coll_param->ag_recv_bufs.assign((void**)coll_param->recv_buf,
                                                    (void**)coll_param->recv_buf + coll_param->comm->size());
                }

                if (ag_algo == ccl_coll_allgatherv_multi_bcast)
                {
                    ccl_coll_param bcast_param;
                    bcast_param.ctype = ccl_coll_bcast;
                    bcast_param.count = sched->coll_param.send_count;
                    bcast_param.dtype = dtype;
                    ag_mbcast_algo = global_data.algorithm_selector->get<ccl_coll_bcast>(bcast_param);
                    if (ag_mbcast_algo == ccl_coll_bcast_direct)
                    {
                        /*
                            group all direct bcasts for specific worker together into single schedule w/o
                            any barriers to get all required MPI level tags by single do_progress call
                        */
                        part_count = max_data_partition_count;
                        LOG_DEBUG("allgatherv over direct multi_bcast, set part_count ", part_count);
                    }
                }
            }
            else
            {
                CCL_FATAL("unexpected allgatherv_algo ", ag_algo);
            }
            break;
        case ccl_coll_sparse_allreduce:
            part_count = 1;
            break;
        default:
            CCL_FATAL("unexpected coll_type ", coll_type);
            break;
    }

    LOG_DEBUG("sched ", sched, ", coll_type ", ccl_coll_type_to_str(coll_type), ", part_count ", part_count);

    if (coll_type == ccl_coll_allgatherv &&
        ag_algo == ccl_coll_allgatherv_multi_bcast &&
        ag_mbcast_algo == ccl_coll_bcast_direct)
    {
        counts.resize(coll_param->comm->size(), 0);
        offsets.resize(coll_param->comm->size(), 0);
    }
    else
    {
        counts.resize(part_count, 0);
        offsets.resize(part_count, 0);
    }

    for (idx = 0; idx < part_count; idx++)
    {
        ccl_coll_param part_coll_param{};
        part_coll_param.comm = sched->coll_param.comm;
        part_coll_param.ctype = sched->coll_param.ctype;
        part_coll_param.dtype = sched->coll_param.dtype;
        sched->add_partial_sched(part_coll_param);
    }

    for (idx = 0; idx < part_count; idx++)
    {
        /* in this place all coll attributes for partial schedules
         * are taken from master schedule, including priority */
        part_scheds[idx]->coll_attr = sched->coll_attr;
    }

    switch (coll_type)
    {
        case ccl_coll_barrier:
            break;
        case ccl_coll_bcast:
        case ccl_coll_reduce:
        case ccl_coll_allreduce:
        case ccl_coll_alltoall:
            if (a2a_algo == ccl_coll_alltoall_scatter)
            {
                base_count = coll_param->count;
                for (idx = 0; idx < part_count; idx++)
                {
                    counts[idx] = base_count;
                    offsets[idx] = idx * counts[idx] * dtype_size;
                    part_scheds[idx]->coll_param.count = counts[idx];
                }
                break;
            }
        case ccl_coll_sparse_allreduce:
            base_count = coll_param->count / part_count;
            for (idx = 0; idx < part_count; idx++)
            {
                counts[idx] = base_count;
                offsets[idx] = idx * counts[idx] * dtype_size;
            }
            counts[part_count - 1] += coll_param->count % part_count;
            for (idx = 0; idx < part_count; idx++)
            {
                part_scheds[idx]->coll_param.count = counts[idx];
            }
            break;
        case ccl_coll_allgatherv:
            recv_counts = coll_param->recv_counts;
            CCL_ASSERT(recv_counts);
            counts[0] = recv_counts[0];
            offsets[0] = 0;
            if (ag_algo == ccl_coll_allgatherv_direct ||
                ag_algo == ccl_coll_allgatherv_naive)
            {
                for (idx = 0; idx < coll_param->comm->size(); idx++)
                    ag_recv_count += recv_counts[idx];
                ag_recv_bytes = ag_recv_count * dtype_size;
            }
            else
            {
                ag_recv_count = counts[0];
                for (idx = 1; idx < coll_param->comm->size(); idx++)
                {
                    counts[idx] = recv_counts[idx];
                    offsets[idx] = offsets[idx - 1] + counts[idx - 1] * dtype_size;
                    ag_recv_count += counts[idx];
                }
                ag_recv_bytes = ag_recv_count * dtype_size;
            }
            break;
        default:
            CCL_FATAL("unexpected coll_type ", coll_type);
            break;
    }

    switch (coll_type)
    {
        case ccl_coll_barrier:
            sched->sync_partial_scheds();
            for (idx = 0; idx < part_count; idx++)
            {
                CCL_CALL(ccl_coll_build_barrier(part_scheds[idx].get()));
            }
            sched->sync_partial_scheds();
            break;
        case ccl_coll_bcast:
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                if (coll_param->comm->rank() == coll_param->root)
                {
                    entry_factory::make_entry<sycl_copy_device_to_host_entry>(part_scheds[0].get(),
                                                                              ccl_buffer(&(coll_param->sycl_buf),
                                                                                         coll_param->count * dtype_size,
                                                                                         ccl_buffer_type::INDIRECT),
                                                                              ccl_buffer(coll_param->buf,
                                                                                         coll_param->count * dtype_size),
                                                                              coll_param->count,
                                                                              dtype, coll_param->stream);
                }
                sched->sync_partial_scheds();
            }
#endif /* CCL_ENABLE_SYCL */
            for (idx = 0; idx < part_count; idx++)
            {
                CCL_CALL(ccl_coll_build_bcast(part_scheds[idx].get(),
                                              ccl_buffer(&(coll_param->buf),
                                                         coll_param->count * dtype_size,
                                                         offsets[idx],
                                                         ccl_buffer_type::INDIRECT),
                                              counts[idx],
                                              dtype,
                                              coll_param->root));
            }
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                sched->sync_partial_scheds();
                entry_factory::make_entry<sycl_copy_host_to_device_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(coll_param->buf,
                                                                                     coll_param->count * dtype_size),
                                                                          ccl_buffer(&(coll_param->sycl_buf),
                                                                                     coll_param->count * dtype_size,
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          coll_param->count,
                                                                          dtype, coll_param->stream);
            }
#endif /* CCL_ENABLE_SYCL */
            break;
        case ccl_coll_reduce:
            for (idx = 0; idx < part_count; idx++)
            {
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                entry_factory::make_entry<sycl_copy_device_to_host_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(&(coll_param->sycl_send_buf),
                                                                                     coll_param->count * dtype_size,
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          ccl_buffer((void*)coll_param->send_buf,
                                                                                     coll_param->count * dtype_size),
                                                                          coll_param->count,
                                                                          dtype, coll_param->stream);
                sched->sync_partial_scheds();
            }
#endif /* CCL_ENABLE_SYCL */
                CCL_CALL(ccl_coll_build_reduce(part_scheds[idx].get(),
                                               ccl_buffer(&(coll_param->send_buf),
                                                          coll_param->count * dtype_size,
                                                          offsets[idx],
                                                          ccl_buffer_type::INDIRECT),
                                               ccl_buffer(&(coll_param->recv_buf),
                                                          coll_param->count * dtype_size,
                                                          offsets[idx],
                                                          ccl_buffer_type::INDIRECT),
                                               counts[idx],
                                               dtype,
                                               coll_param->reduction,
                                               coll_param->root));
            }
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                sched->sync_partial_scheds();
                if (coll_param->comm->rank() == coll_param->root)
                {
                    entry_factory::make_entry<sycl_copy_host_to_device_entry>(part_scheds[0].get(),
                                                                              ccl_buffer(coll_param->recv_buf,
                                                                                         coll_param->count * dtype_size),
                                                                              ccl_buffer(&(coll_param->sycl_recv_buf),
                                                                                         coll_param->count * dtype_size,
                                                                                         ccl_buffer_type::INDIRECT),
                                                                              coll_param->count,
                                                                              dtype, coll_param->stream);
                }
            }
#endif /* CCL_ENABLE_SYCL */

            break;
        case ccl_coll_allreduce:
        {
            ccl_parallelizer_prologue_ctx* main_ctx = nullptr;

#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                entry_factory::make_entry<sycl_copy_device_to_host_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(&(coll_param->sycl_send_buf),
                                                                                     coll_param->count * dtype_size,
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          ccl_buffer((void*)coll_param->send_buf,
                                                                                     coll_param->count * dtype_size),
                                                                          coll_param->count,
                                                                          dtype, coll_param->stream);
                sched->sync_partial_scheds();
            }
#endif /* CCL_ENABLE_SYCL */

            if (sched->coll_attr.prologue_fn)
            {
                main_ctx =
                    (ccl_parallelizer_prologue_ctx*)part_scheds[0]->alloc_buffer(sizeof(ccl_parallelizer_prologue_ctx)).get_ptr();
                main_ctx->part_idx = 0;
                main_ctx->part_count = 1;
                entry_factory::make_entry<prologue_entry>(part_scheds[0].get(),
                                                          sched->coll_attr.prologue_fn,
                                                          ccl_buffer(&(coll_param->send_buf),
                                                                     coll_param->count * dtype_size,
                                                                     ccl_buffer_type::INDIRECT),
                                                          coll_param->count,
                                                          dtype,
                                                          &(main_ctx->buf),
                                                          &(main_ctx->count),
                                                          &(main_ctx->dtype),
                                                          &(main_ctx->dtype_size));

                sched->sync_partial_scheds();

                for (idx = 0; idx < part_count; idx++)
                {
                    ccl_parallelizer_prologue_ctx* part_ctx =
                        (ccl_parallelizer_prologue_ctx*)part_scheds[idx]->alloc_buffer(sizeof(ccl_parallelizer_prologue_ctx)).get_ptr();
                    part_ctx->part_idx = idx;
                    part_ctx->part_count = part_count;

                    entry_factory::make_entry<copy_entry>(part_scheds[idx].get(),
                                                          ccl_buffer(main_ctx, sizeof(ccl_parallelizer_prologue_ctx)),
                                                          ccl_buffer(part_ctx, sizeof(ccl_parallelizer_prologue_ctx)),
                                                          sizeof(void*) + sizeof(size_t) +
                                                          sizeof(ccl_datatype_t) + sizeof(size_t),
                                                          ccl_dtype_internal_char);

                    ccl_coll_entry_param param{};
                    param.ctype = ccl_coll_allreduce;
                    param.send_buf = ccl_buffer();
                    param.recv_buf = ccl_buffer();
                    param.count = 0;
                    param.dtype = ccl_dtype_internal_none;
                    param.reduction = coll_param->reduction;
                    coll_entry* entry = entry_factory::make_entry<coll_entry>(part_scheds[idx].get(), param);

                    entry->set_field_fn<ccl_sched_entry_field_send_buf>(ccl_parallelizer_prologue_get_buf,
                                                                        part_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_recv_buf>(ccl_parallelizer_prologue_get_buf,
                                                                        part_ctx, false); // in-place
                    entry->set_field_fn<ccl_sched_entry_field_cnt>(ccl_parallelizer_prologue_get_count,
                                                                   part_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_dtype>(ccl_parallelizer_prologue_get_dtype,
                                                                     part_ctx, false);
                }

                if (!sched->coll_attr.epilogue_fn)
                {
                    sched->sync_partial_scheds();

                    copy_entry* entry = entry_factory::make_entry<copy_entry>(part_scheds[0].get(),
                                                                              ccl_buffer(), /* in_buf */
                                                                              ccl_buffer(&(coll_param->recv_buf),
                                                                                         coll_param->count * dtype_size,
                                                                                         ccl_buffer_type::INDIRECT),
                                                                              0, /* count */
                                                                              ccl_dtype_internal_none);
                    entry->set_field_fn<ccl_sched_entry_field_in_buf>(ccl_parallelizer_prologue_get_buf,
                                                                      main_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_cnt>(ccl_parallelizer_prologue_get_count,
                                                                   main_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_dtype>(ccl_parallelizer_prologue_get_dtype,
                                                                     main_ctx, false);
                }
            }
            else
            {
                for (idx = 0; idx < part_count; idx++)
                {
                    CCL_CALL(ccl_coll_build_allreduce(part_scheds[idx].get(),
                                                      ccl_buffer(&(coll_param->send_buf),
                                                                 coll_param->count * dtype_size,
                                                                 offsets[idx],
                                                                 ccl_buffer_type::INDIRECT),
                                                      ccl_buffer(&(coll_param->recv_buf),
                                                                 coll_param->count * dtype_size,
                                                                 offsets[idx],
                                                                 ccl_buffer_type::INDIRECT),
                                                      counts[idx],
                                                      dtype,
                                                      coll_param->reduction));
                }
            }

            if (sched->coll_attr.epilogue_fn)
            {
                sched->sync_partial_scheds();
                epilogue_entry* entry = entry_factory::make_entry<epilogue_entry>(part_scheds[0].get(),
                                                              sched->coll_attr.epilogue_fn,
                                                              ccl_buffer(&(coll_param->recv_buf),
                                                                         coll_param->count * dtype_size,
                                                                         ccl_buffer_type::INDIRECT),
                                                              coll_param->count,
                                                              dtype,
                                                              ccl_buffer(&(coll_param->recv_buf),
                                                                         coll_param->count * dtype_size,
                                                                         ccl_buffer_type::INDIRECT),
                                                              coll_param->count,
                                                              dtype);
                if (sched->coll_attr.prologue_fn)
                {
                    entry->set_field_fn<ccl_sched_entry_field_in_buf>(ccl_parallelizer_prologue_get_buf,
                                                                      main_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_in_cnt>(ccl_parallelizer_prologue_get_count,
                                                                      main_ctx, false);
                    entry->set_field_fn<ccl_sched_entry_field_in_dtype>(ccl_parallelizer_prologue_get_dtype,
                                                                        main_ctx, false);
                }
            }

#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                sched->sync_partial_scheds();
                entry_factory::make_entry<sycl_copy_host_to_device_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(coll_param->recv_buf,
                                                                                     coll_param->count * dtype_size),
                                                                          ccl_buffer(&(coll_param->sycl_recv_buf),
                                                                                     coll_param->count * dtype_size,
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          coll_param->count,
                                                                          dtype, coll_param->stream);
            }
#endif /* CCL_ENABLE_SYCL */

            break;
        }
        case ccl_coll_allgatherv:
        {
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                entry_factory::make_entry<sycl_copy_device_to_host_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(&(coll_param->sycl_send_buf),
                                                                                     coll_param->send_count * dtype_size,
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          ccl_buffer((void*)coll_param->send_buf,
                                                                                     coll_param->send_count * dtype_size),
                                                                          coll_param->send_count,
                                                                          dtype, coll_param->stream);
                sched->sync_partial_scheds();
            }
#endif /* CCL_ENABLE_SYCL */
            if (ag_algo == ccl_coll_allgatherv_direct ||
                ag_algo == ccl_coll_allgatherv_naive)
            {
                CCL_CALL(ccl_coll_build_allgatherv(part_scheds[0].get(),
                                                   ccl_buffer(&(coll_param->send_buf),
                                                              coll_param->send_count * dtype_size,
                                                              ccl_buffer_type::INDIRECT),
                                                   coll_param->send_count,
                                                   ccl_buffer(&(coll_param->recv_buf),
                                                              ag_recv_bytes,
                                                              ccl_buffer_type::INDIRECT),
                                                   coll_param->recv_counts,
                                                   dtype));
            }
            else
            {
                CCL_ASSERT(ag_algo == ccl_coll_allgatherv_flat ||
                           ag_algo == ccl_coll_allgatherv_multi_bcast,
                           "unexpected allgatherv algorithm");

                for (idx = 0; idx < coll_param->comm->size(); idx++)
                {
                    if (env_data.enable_allgatherv_iov)
                    {
                        ag_recv_bufs[idx].set(&(coll_param->ag_recv_bufs[idx]),
                                              counts[idx] * dtype_size,
                                              ccl_buffer_type::INDIRECT);
                    }
                    else
                    {
                        ag_recv_bufs[idx].set(&(coll_param->recv_buf),
                                              ag_recv_bytes,
                                              offsets[idx],
                                              ccl_buffer_type::INDIRECT);
                    }
                }

                if (ag_algo == ccl_coll_allgatherv_flat)
                {
                    auto send_seg = ccl_buffer(&(coll_param->send_buf),
                                               coll_param->send_count * dtype_size,
                                               ccl_buffer_type::INDIRECT);

                    size_t my_rank = coll_param->comm->rank();
                    if (coll_param->send_buf != coll_param->recv_buf)
                    {
                        entry_factory::make_entry<copy_entry>(part_scheds[2 * my_rank % part_count].get(),
                                                              ccl_buffer(&(coll_param->send_buf),
                                                                         coll_param->send_count * dtype_size,
                                                                         ccl_buffer_type::INDIRECT),
                                                              ag_recv_bufs[my_rank],
                                                              counts[my_rank], dtype);

                    }
                    else
                    {
                        send_seg = ccl_buffer(&(coll_param->send_buf),
                                              ag_recv_bytes,
                                              offsets[my_rank],
                                              ccl_buffer_type::INDIRECT);
                    }

                    CCL_ASSERT(part_count == coll_param->comm->size());

                    for (idx = 0; idx < part_count; idx++)
                    {
                         if (idx == my_rank) continue;

                         entry_factory::make_entry<recv_entry>(part_scheds[(my_rank+idx) % part_count].get(),
                                                               ag_recv_bufs[idx],
                                                               counts[idx],
                                                               dtype,
                                                               idx);
                         entry_factory::make_entry<send_entry>(part_scheds[(my_rank+idx) % part_count].get(),
                                                               send_seg,
                                                               counts[my_rank],
                                                               dtype,
                                                               idx);
                    }
                    sched->sync_partial_scheds();
                }
                else
                {
                    CCL_ASSERT(ag_algo == ccl_coll_allgatherv_multi_bcast);

                    if (coll_param->send_buf != coll_param->recv_buf)
                    {
                        std::vector<size_t> copy_counts(max_data_partition_count);
                        std::vector<size_t> copy_offsets(max_data_partition_count);
                        for (idx = 0; idx < max_data_partition_count; idx++)
                        {
                            copy_counts[idx] = counts[coll_param->comm->rank()] / max_data_partition_count;
                            copy_offsets[idx] = idx * copy_counts[idx] * dtype_size;
                        }
                        copy_counts[max_data_partition_count - 1] +=
                            counts[coll_param->comm->rank()] % max_data_partition_count;

                        CCL_ASSERT(part_scheds.size() >= max_data_partition_count);

                        for (idx = 0; idx < max_data_partition_count; idx++)
                        {
                            entry_factory::make_entry<copy_entry>(part_scheds[idx].get(),
                                                                  ccl_buffer(&(coll_param->send_buf),
                                                                             coll_param->send_count * dtype_size,
                                                                             copy_offsets[idx],
                                                                             ccl_buffer_type::INDIRECT),
                                                                  ag_recv_bufs[coll_param->comm->rank()] + copy_offsets[idx],
                                                                  copy_counts[idx], dtype);
                        }
                        sched->sync_partial_scheds();
                    }

                    for (idx = 0; idx < coll_param->comm->size(); idx++)
                    {
                        CCL_CALL(ccl_coll_build_bcast(part_scheds[idx % part_count].get(),
                                                      ag_recv_bufs[idx],
                                                      counts[idx],
                                                      dtype,
                                                      idx));
                    }
                }
            }
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                sched->sync_partial_scheds();
                entry_factory::make_entry<sycl_copy_host_to_device_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(coll_param->recv_buf,
                                                                                     ag_recv_bytes),
                                                                          ccl_buffer(&(coll_param->sycl_recv_buf),
                                                                                     ag_recv_bytes,
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          ag_recv_count,
                                                                          dtype, coll_param->stream);
            }
#endif /* CCL_ENABLE_SYCL */
            break;
        }
        case ccl_coll_alltoall:
#ifdef CCL_ENABLE_SYCL
            /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                entry_factory::make_entry<sycl_copy_device_to_host_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(&(coll_param->sycl_send_buf),
                                                                                     coll_param->count * dtype_size * coll_param->comm->size(),
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          ccl_buffer((void*)coll_param->send_buf,
                                                                                     coll_param->count * dtype_size * coll_param->comm->size()),
                                                                          coll_param->count * coll_param->comm->size(),
                                                                          dtype, coll_param->stream);
                sched->sync_partial_scheds();
            }
#endif /* CCL_ENABLE_SYCL */
            if (a2a_algo == ccl_coll_alltoall_scatter)
            {
                size_t my_rank = coll_param->comm->rank();
                for (idx = 0; idx < part_count; idx++)
                {
                    ccl_buffer recv_buf = (coll_param->send_buf == coll_param->recv_buf) ?
                                          part_scheds[(my_rank+idx) % part_count].get()->alloc_buffer(counts[idx] * dtype_size) :
                                          ccl_buffer(&(coll_param->recv_buf),
                                                coll_param->count * dtype_size * coll_param->comm->size(),
                                                offsets[idx],
                                                ccl_buffer_type::INDIRECT);
                    entry_factory::make_entry<recv_entry>(part_scheds[(my_rank+idx) % part_count].get(),
                                                          recv_buf,
                                                          counts[idx],
                                                          dtype,
                                                          idx);
                    entry_factory::make_entry<send_entry>(part_scheds[(my_rank+idx) % part_count].get(),
                                                          ccl_buffer(&(coll_param->send_buf),
                                                                     coll_param->count * dtype_size * coll_param->comm->size(),
                                                                     offsets[idx],
                                                                     ccl_buffer_type::INDIRECT),
                                                          counts[idx],
                                                          dtype,
                                                          idx);
                    if  (coll_param->send_buf == coll_param->recv_buf)
                    {
                        part_scheds[(my_rank + idx) % part_count].get()->add_barrier();
                        entry_factory::make_entry<copy_entry>(part_scheds[(my_rank + idx) % part_count].get(),
                                                              recv_buf,
                                                              ccl_buffer(&(coll_param->recv_buf),
                                                                         coll_param->count * dtype_size * coll_param->comm->size(),
                                                                         offsets[idx],
                                                                         ccl_buffer_type::INDIRECT),
                                                              counts[idx],
                                                              dtype);
                        part_scheds[(my_rank + idx) % part_count].get()->add_barrier();
                    }
                }
            }
            else if (a2a_algo == ccl_coll_alltoall_scatter_message)
            {
                for (idx = 0; idx < part_count; idx++)
                {
                    for (size_t rank = 0; rank < coll_param->comm->size(); rank++)
                    {
                        ccl_buffer recv_buf = (coll_param->send_buf == coll_param->recv_buf) ?
                                              part_scheds[idx].get()->alloc_buffer(counts[idx] * dtype_size) :
                                              ccl_buffer(&(coll_param->recv_buf),
                                                         coll_param->count * dtype_size * coll_param->comm->size(),
                                                         offsets[idx] + coll_param->count * rank * dtype_size,
                                                         ccl_buffer_type::INDIRECT);
                        entry_factory::make_entry<recv_entry>(part_scheds[idx].get(),
                                                              recv_buf,
                                                              counts[idx],
                                                              dtype,
                                                              rank);
                        entry_factory::make_entry<send_entry>(part_scheds[idx].get(),
                                                              ccl_buffer(&(coll_param->send_buf),
                                                                         coll_param->count * dtype_size * coll_param->comm->size(),
                                                                         offsets[idx] + coll_param->count * rank * dtype_size,
                                                                         ccl_buffer_type::INDIRECT),
                                                              counts[idx],
                                                              dtype,
                                                              rank);
                        if  (coll_param->send_buf == coll_param->recv_buf)
                        {
                            part_scheds[idx].get()->add_barrier();
                            entry_factory::make_entry<copy_entry>(part_scheds[idx].get(),
                                                                  recv_buf,
                                                                  ccl_buffer(&(coll_param->recv_buf),
                                                                             coll_param->count * dtype_size * coll_param->comm->size(),
                                                                             offsets[idx] + coll_param->count * rank * dtype_size,
                                                                             ccl_buffer_type::INDIRECT),
                                                                  counts[idx],
                                                                  dtype);
                            part_scheds[idx].get()->add_barrier();
                        }
                    }
                }
            }
            else
            {
                CCL_CALL(ccl_coll_build_alltoall(part_scheds[0].get(),
                                                 ccl_buffer(&(coll_param->send_buf),
                                                            coll_param->count * dtype_size * coll_param->comm->size(),
                                                            offsets[0],
                                                            ccl_buffer_type::INDIRECT),
                                                 ccl_buffer(&(coll_param->recv_buf),
                                                            coll_param->count * dtype_size * coll_param->comm->size(),
                                                            offsets[0],
                                                            ccl_buffer_type::INDIRECT),
                                                 counts[0],
                                                 dtype));
            }
#ifdef CCL_ENABLE_SYCL
        /* convert sycl buffer */
            if (coll_param->stream && (ccl_stream_type_t)(coll_param->stream->get_type()) == ccl_stream_sycl)
            {
                sched->sync_partial_scheds();
                entry_factory::make_entry<sycl_copy_host_to_device_entry>(part_scheds[0].get(),
                                                                          ccl_buffer(coll_param->recv_buf,
                                                                                     coll_param->count * dtype_size * coll_param->comm->size()),
                                                                          ccl_buffer(&(coll_param->sycl_recv_buf),
                                                                                     coll_param->count * dtype_size * coll_param->comm->size(),
                                                                                     ccl_buffer_type::INDIRECT),
                                                                          coll_param->count * coll_param->comm->size(),
                                                                          dtype, coll_param->stream);
            }
#endif /* CCL_ENABLE_SYCL */

            break;
        case ccl_coll_sparse_allreduce:
            itype = coll_param->sparse_param.itype;
            itype_size = ccl_datatype_get_size(itype);
            for (idx = 0; idx < part_count; idx++)
            {
                CCL_CALL(ccl_coll_build_sparse_allreduce(part_scheds[idx].get(),
                                                         ccl_buffer(&(coll_param->sparse_param.send_ind_buf),
                                                                    coll_param->sparse_param.send_ind_count * itype_size,
                                                                    offsets[idx],
                                                                    ccl_buffer_type::INDIRECT),
                                                         coll_param->sparse_param.send_ind_count,
                                                         ccl_buffer(&(coll_param->sparse_param.send_val_buf),
                                                                    coll_param->sparse_param.send_val_count * dtype_size,
                                                                    offsets[idx],
                                                                    ccl_buffer_type::INDIRECT),
                                                         coll_param->sparse_param.send_val_count,
                                                         ccl_buffer(&(coll_param->sparse_param.recv_ind_buf),
                                                                    -1, /* unknown size */
                                                                    offsets[idx],
                                                                    ccl_buffer_type::INDIRECT),
                                                         coll_param->sparse_param.recv_ind_count,
                                                         ccl_buffer(&(coll_param->sparse_param.recv_val_buf),
                                                                    -1, /* unknown size */
                                                                    offsets[idx],
                                                                    ccl_buffer_type::INDIRECT),
                                                         coll_param->sparse_param.recv_val_count,
                                                         coll_param->sparse_param.itype,
                                                         dtype,
                                                         coll_param->reduction));
            }
            break;

        default:
            CCL_FATAL("unexpected coll_type ", coll_type);
            break;
    }
    return status;
}
