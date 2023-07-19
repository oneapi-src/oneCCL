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

#include "coll/coll_util.hpp"
#include "coll/selection/selection.hpp"
#include "common/global/global.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

#define CCL_ATL_LARGE_MSG_SIZE (1024 * 1024 * 1024)

ccl::status ccl_parallelizer::process(ccl_sched* sched, bool update_sched_id) {
    process_base(sched, update_sched_id);

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

ccl::status ccl_parallelizer::process_deps(ccl_sched* sched) {
    auto& part_scheds = sched->get_subscheds();
    ccl_sched* deps_sched = part_scheds[0].get();
    size_t sched_count = part_scheds.size();

    for (size_t idx = 0; idx < sched_count; idx++) {
        part_scheds[idx]->set_add_mode(ccl_sched_add_front);
    }
    sched->sync_subscheds();

    entry_factory::create<deps_entry>(deps_sched);
    deps_sched->add_barrier();

    return ccl::status::success;
}

#ifdef CCL_ENABLE_SYCL
ccl::status ccl_parallelizer::process_pre_post_copies(ccl_sched* sched) {
    auto& part_scheds = sched->get_subscheds();
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
        coll_param.is_inplace(ccl_coll_param::buf_type::device) &&
        (coll_param.device_recv_bufs.size() == 1)) {
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
        sched->sync_subscheds();

        for (size_t idx = 0; idx < d2h_counts.size(); idx++) {
            size_t sched_idx = idx % sched_count;
            size_t count = d2h_counts[idx];
            if (count == 0) {
                continue;
            }

            size_t bytes = count * dtype_size;

            entry_factory::create<copy_entry>(
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
        sched->sync_subscheds();

        for (size_t idx = 0; idx < h2d_counts.size(); idx++) {
            size_t sched_idx = idx % sched_count;
            size_t count = h2d_counts[idx];
            if (count == 0) {
                continue;
            }

            size_t bytes = count * dtype_size;
            if (bytes == 0) {
                continue;
            }

            entry_factory::create<copy_entry>(
                part_scheds[sched_idx].get(),
                ccl_buffer(coll_param.get_recv_buf(idx), bytes),
                ccl_buffer(coll_param.get_recv_buf_ptr(idx, ccl_coll_param::buf_type::device),
                           bytes,
                           ccl_buffer_type::INDIRECT),
                count,
                dtype,
                copy_attr(copy_direction::h2d, 0));
        }

        sched->sync_subscheds();
    }

    return ccl::status::success;
}

ccl::status ccl_parallelizer::process_output_event(ccl_sched* sched) {
    if (!ccl::utils::should_use_sycl_output_event(sched->coll_param.stream)) {
        return ccl::status::success;
    }

    auto& part_scheds = sched->get_subscheds();
    size_t sched_count = part_scheds.size();

    for (size_t idx = 0; idx < sched_count; idx++) {
        part_scheds[idx]->set_add_mode(ccl_sched_add_back);
    }
    sched->sync_subscheds();

    entry_factory::create<ze_event_signal_entry>(part_scheds[0].get(), sched);

    return ccl::status::success;
}
#endif // CCL_ENABLE_SYCL

ccl::status ccl_parallelizer::process_base(ccl_sched* sched, bool update_sched_id) {
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
    auto& part_scheds = sched->get_subscheds();
    std::vector<ccl_sched*> part_scheds_vector;

    std::vector<ccl_buffer> ag_recv_bufs;
    size_t ag_recv_bytes = 0, ag_recv_count = 0;
    size_t a2av_send_bytes = 0, a2av_recv_bytes = 0;
    size_t a2av_send_count = 0, a2av_recv_count = 0;

    ccl_coll_algo algo;
    ccl_coll_algo internal_algo;

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
            if (ccl_is_device_side_algo(selector_param)) {
                part_count = 1;
            }
            break;
        case ccl_coll_alltoall:
            selector_param.is_scaleout = coll_param.is_scaleout;
            algo.alltoall = data.algorithm_selector->get<ccl_coll_alltoall>(selector_param);
            if (algo.alltoall == ccl_coll_alltoall_direct) {
                part_count = 1;
            }
            else {
                part_count = std::min(comm_size, max_data_partition_count);
            }
            break;
        case ccl_coll_alltoallv:
            selector_param.is_scaleout = coll_param.is_scaleout;
            algo.alltoallv = data.algorithm_selector->get<ccl_coll_alltoallv>(selector_param);
            if (algo.alltoallv == ccl_coll_alltoallv_direct) {
                part_count = 1;
            }
            else {
                part_count = std::min(comm_size, max_data_partition_count);
            }
            break;
        case ccl_coll_allgatherv:
            selector_param.is_scaleout = coll_param.is_scaleout;
            selector_param.recv_counts = coll_param.recv_counts.data();
            algo.allgatherv = data.algorithm_selector->get<ccl_coll_allgatherv>(selector_param);

            if (algo.allgatherv == ccl_coll_allgatherv_direct ||
                algo.allgatherv == ccl_coll_allgatherv_naive ||
                ccl_is_device_side_algo(selector_param)) {
                part_count = 1;
            }
            else if (algo.allgatherv == ccl_coll_allgatherv_ring) {
                // limit the data partitioning for small data, because the performance drops
                // significantly for small data cases if parallelism is enabled
                if ((coll_param.get_send_count() * dtype_size <=
                     ccl::global_data::env().max_short_size) ||
                    (coll_param.get_send_count() < max_data_partition_count)) {
                    part_count = 1;
                }
                else {
                    part_count = max_data_partition_count;
                }
            }
            else if (algo.allgatherv == ccl_coll_allgatherv_multi_bcast ||
                     algo.allgatherv == ccl_coll_allgatherv_flat) {
                part_count = comm_size;
                ag_recv_bufs.resize(comm_size);
                if (algo.allgatherv == ccl_coll_allgatherv_multi_bcast) {
                    ccl_selector_param bcast_selector_param;
                    bcast_selector_param.ctype = ccl_coll_bcast;
                    bcast_selector_param.count = selector_param.count;
                    bcast_selector_param.dtype = selector_param.dtype;
                    bcast_selector_param.comm = selector_param.comm;
                    internal_algo.bcast =
                        data.algorithm_selector->get<ccl_coll_bcast>(bcast_selector_param);
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
        case ccl_coll_recv:
        case ccl_coll_send:
            part_count = (coll_param.get_send_count() * dtype_size) / CCL_ATL_LARGE_MSG_SIZE;
            if (part_count < max_data_partition_count)
                part_count = max_data_partition_count;
            LOG_DEBUG("send-recv operation, set part_count ", part_count);
            break;
        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }

    LOG_DEBUG(
        "sched ", sched, ", coll ", ccl_coll_type_to_str(coll_type), ", part_count ", part_count);

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
        sched->add_subsched(part_coll_param, update_sched_id);
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
            base_count = coll_param.get_recv_count() / part_count;
            for (idx = 0; idx < counts.size(); idx++) {
                counts[idx] = base_count;
                offsets[idx] = idx * counts[idx] * dtype_size;
            }
            counts[counts.size() - 1] += coll_param.get_recv_count() % counts.size();
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
            selector_param.recv_counts = coll_param.recv_counts.data();
            if (algo.allgatherv == ccl_coll_allgatherv_direct ||
                algo.allgatherv == ccl_coll_allgatherv_naive ||
                algo.allgatherv == ccl_coll_allgatherv_ring ||
                ccl_is_device_side_algo(selector_param)) {
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
        case ccl_coll_recv:
            base_count = coll_param.get_recv_count() / part_count;
            for (idx = 0; idx < counts.size(); idx++) {
                counts[idx] = base_count;
                offsets[idx] = idx * counts[idx] * dtype_size;
            }
            counts[counts.size() - 1] += coll_param.get_recv_count() % counts.size();
            break;
        case ccl_coll_send:
            base_count = coll_param.get_send_count() / part_count;
            for (idx = 0; idx < counts.size(); idx++) {
                counts[idx] = base_count;
                offsets[idx] = idx * counts[idx] * dtype_size;
            }
            counts[counts.size() - 1] += coll_param.get_send_count() % counts.size();
            break;
        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }

    switch (coll_type) {
        case ccl_coll_barrier:
            sched->sync_subscheds();
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_barrier;
                param.dtype = ccl_datatype_int8;
                param.comm = comm;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
            }
            sched->sync_subscheds();
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
                param.stream = coll_param.stream;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
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
                param.is_scaleout = coll_param.is_scaleout;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
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
                param.stream = coll_param.stream;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
            }
            break;

        case ccl_coll_allreduce: {
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_allreduce;
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
                param.comm = comm;
                param.stream = coll_param.stream;
                param.is_scaleout = coll_param.is_scaleout;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
            }
            break;
        }

        case ccl_coll_allgatherv: {
            if (algo.allgatherv == ccl_coll_allgatherv_direct ||
                algo.allgatherv == ccl_coll_allgatherv_naive) {
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
                param.stream = coll_param.stream;
                param.is_scaleout = coll_param.is_scaleout;
                ccl::add_coll_entry(part_scheds[0].get(), param);
            }
            else {
                if (algo.allgatherv == ccl_coll_allgatherv_ring) {
                    ccl_buffer send_buf = ccl_buffer(coll_param.get_send_buf_ptr(),
                                                     coll_param.get_send_count() * dtype_size,
                                                     ccl_buffer_type::INDIRECT);
                    ccl_buffer recv_buf = ccl_buffer(
                        coll_param.get_recv_buf_ptr(), ag_recv_bytes, ccl_buffer_type::INDIRECT);
                    ccl_coll_build_ring_allgatherv(sched,
                                                   part_scheds_vector,
                                                   send_buf,
                                                   coll_param.get_send_count(),
                                                   recv_buf,
                                                   coll_param.recv_counts.data(),
                                                   dtype,
                                                   comm);
                }
                else if (algo.allgatherv == ccl_coll_allgatherv_flat) {
                    ccl_coll_build_flat_allgatherv(sched, part_scheds_vector, coll_param);
                }
                else if (algo.allgatherv == ccl_coll_allgatherv_multi_bcast) {
                    ccl_coll_build_multi_bcast_allgatherv(
                        sched, part_scheds_vector, coll_param, max_data_partition_count);
                }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
                else if (algo.allgatherv == ccl_coll_allgatherv_topo) {
                    ccl_coll_build_topo_allgatherv(sched, part_scheds_vector, coll_param);
                }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
                else {
                    CCL_THROW("unexpected allgatherv algorithm");
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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
            else if (algo.alltoall == ccl_coll_alltoall_topo ||
                     algo.alltoallv == ccl_coll_alltoallv_topo) {
                ccl_coll_build_topo_alltoallv(sched, part_scheds_vector, coll_param);
            }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
            else {
                ccl_coll_entry_param param{};
                param.ctype = coll_type;
                param.send_buf = ccl_buffer(
                    coll_param.get_send_buf_ptr(), a2av_send_bytes, ccl_buffer_type::INDIRECT);
                param.recv_buf = ccl_buffer(
                    coll_param.get_recv_buf_ptr(), a2av_recv_bytes, ccl_buffer_type::INDIRECT);
                param.dtype = dtype;
                param.comm = comm;
                param.is_scaleout = coll_param.is_scaleout;

                if (coll_type == ccl_coll_alltoall) {
                    param.count = coll_param.get_send_count();
                    ccl::add_coll_entry(part_scheds[0].get(), param);
                }
                else {
                    param.send_counts = coll_param.send_counts.data();
                    param.recv_counts = coll_param.recv_counts.data();
                    ccl::add_coll_entry(part_scheds[0].get(), param);
                }
            }
            break;
        }

        case ccl_coll_recv:
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_recv;
                param.recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                            coll_param.get_recv_count() * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.count = counts[idx];
                param.dtype = dtype;
                param.peer_rank = coll_param.peer_rank;
                param.comm = comm;
                param.stream = coll_param.stream;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
            }
            break;

        case ccl_coll_send:
            for (idx = 0; idx < part_count; idx++) {
                ccl_coll_entry_param param{};
                param.ctype = ccl_coll_send;
                param.send_buf = ccl_buffer(coll_param.get_send_buf_ptr(),
                                            coll_param.get_send_count() * dtype_size,
                                            offsets[idx],
                                            ccl_buffer_type::INDIRECT);
                param.count = counts[idx];
                param.dtype = dtype;
                param.peer_rank = coll_param.peer_rank;
                param.comm = comm;
                param.stream = coll_param.stream;
                ccl::add_coll_entry(part_scheds[idx].get(), param);
            }
            break;

        default: CCL_FATAL("unexpected coll_type ", coll_type); break;
    }
    return status;
}
