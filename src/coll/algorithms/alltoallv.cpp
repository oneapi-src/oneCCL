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

/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <numeric>

#include "coll/algorithms/algorithms.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

ccl_status_t ccl_coll_build_direct_alltoallv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             const size_t* send_counts,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm) {
    LOG_DEBUG("build direct alltoallv");

    entry_factory::make_entry<alltoallv_entry>(
        sched, send_buf, send_counts, recv_buf, recv_counts, dtype, comm);
    return ccl_status_success;
}

ccl_status_t ccl_coll_add_scatter_alltoallv_barriers(std::vector<ccl_sched*>& scheds,
                                                     size_t sched_idx) {
    ssize_t max_ops = ccl::global_data::env().alltoall_scatter_max_ops;

    if (max_ops != CCL_ENV_SIZET_NOT_SPECIFIED) {
        if (scheds[sched_idx]->entries_count() % max_ops == 0)
            scheds[sched_idx]->add_barrier();

        if (ccl::global_data::env().alltoall_scatter_plain) {
            for (auto s : scheds) {
                if (s->entries_count() % max_ops == 0)
                    s->add_barrier();
            }
        }
    }

    return ccl_status_success;
}

ccl_status_t ccl_coll_calculate_alltoallv_counts(const ccl_coll_param& coll_param,
                                                 std::vector<size_t>& send_counts,
                                                 std::vector<size_t>& recv_counts,
                                                 std::vector<size_t>& send_offsets,
                                                 std::vector<size_t>& recv_offsets,
                                                 size_t& total_send_count,
                                                 size_t& total_recv_count,
                                                 size_t& total_send_bytes,
                                                 size_t& total_recv_bytes) {
    ccl_coll_type coll_type = coll_param.ctype;
    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    size_t comm_size = comm->size();
    size_t dtype_size = dtype.size();

    if (coll_type == ccl_coll_alltoall) {
        send_counts.resize(comm_size, coll_param.count);
        recv_counts.resize(comm_size, coll_param.count);
    }
    else if (coll_type == ccl_coll_alltoallv) {
        CCL_ASSERT(coll_param.send_counts);
        CCL_ASSERT(coll_param.recv_counts);
        send_counts.assign((size_t*)coll_param.send_counts,
                           (size_t*)coll_param.send_counts + comm_size);
        recv_counts.assign((size_t*)coll_param.recv_counts,
                           (size_t*)coll_param.recv_counts + comm_size);
    }

    send_offsets.resize(comm_size, 0);
    recv_offsets.resize(comm_size, 0);

    for (size_t idx = 1; idx < comm_size; idx++) {
        send_offsets[idx] = send_offsets[idx - 1] + send_counts[idx - 1] * dtype_size;
        recv_offsets[idx] = recv_offsets[idx - 1] + recv_counts[idx - 1] * dtype_size;
    }

    total_send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0);
    total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    total_send_bytes = total_send_count * dtype_size;
    total_recv_bytes = total_recv_count * dtype_size;

    LOG_DEBUG("total_send_count ",
              total_send_count,
              ", total_recv_count ",
              total_recv_count,
              ", total_send_bytes ",
              total_send_bytes,
              ", total_recv_bytes ",
              total_recv_bytes);

    return ccl_status_success;
}

ccl_status_t ccl_coll_build_naive_alltoallv(ccl_master_sched* main_sched,
                                            std::vector<ccl_sched*>& scheds,
                                            const ccl_coll_param& coll_param) {
    LOG_DEBUG("build naive alltoallv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    size_t comm_rank = comm->rank();
    size_t comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    bool inplace =
        (coll_param.send_buf && (coll_param.send_buf == coll_param.recv_buf)) ? true : false;

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::make_entry<copy_entry>(scheds[sched_idx],
                                              ccl_buffer((void*)(&(coll_param.send_buf)),
                                                         total_send_bytes,
                                                         send_offsets[comm_rank],
                                                         ccl_buffer_type::INDIRECT),
                                              ccl_buffer((void*)(&(coll_param.recv_buf)),
                                                         total_recv_bytes,
                                                         recv_offsets[comm_rank],
                                                         ccl_buffer_type::INDIRECT),
                                              send_counts[comm_rank],
                                              dtype);
    }

    for (size_t idx = 0; idx < comm_size; idx++) {
        if (idx == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + idx) % sched_count;

        ccl_buffer recv_buf;

        if (inplace)
            recv_buf = scheds[sched_idx]->alloc_buffer(recv_counts[idx] * dtype_size);
        else
            recv_buf = ccl_buffer((void*)(&(coll_param.recv_buf)),
                                  total_recv_bytes,
                                  recv_offsets[idx],
                                  ccl_buffer_type::INDIRECT);

        entry_factory::make_chunked_recv_entry(
            scheds, sched_idx, recv_buf, recv_counts[idx], dtype, idx, comm);

        entry_factory::make_chunked_send_entry(scheds,
                                               sched_idx,
                                               ccl_buffer((void*)(&(coll_param.send_buf)),
                                                          total_send_bytes,
                                                          send_offsets[idx],
                                                          ccl_buffer_type::INDIRECT),
                                               send_counts[idx],
                                               dtype,
                                               idx,
                                               comm);

        if (inplace) {
            scheds[sched_idx]->add_barrier();
            entry_factory::make_entry<copy_entry>(scheds[sched_idx],
                                                  recv_buf,
                                                  ccl_buffer((void*)(&(coll_param.recv_buf)),
                                                             total_recv_bytes,
                                                             recv_offsets[idx],
                                                             ccl_buffer_type::INDIRECT),
                                                  recv_counts[idx],
                                                  dtype);
            scheds[sched_idx]->add_barrier();
        }
    }

    return ccl_status_success;
}

ccl_status_t ccl_coll_build_scatter_alltoallv(ccl_master_sched* main_sched,
                                              std::vector<ccl_sched*>& scheds,
                                              const ccl_coll_param& coll_param) {
    LOG_DEBUG("build scatter alltoall");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    size_t comm_rank = comm->rank();
    size_t comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    bool inplace =
        (coll_param.send_buf && (coll_param.send_buf == coll_param.recv_buf)) ? true : false;

    std::vector<ccl_buffer> recv_bufs;
    if (inplace)
        recv_bufs.resize(comm_size);

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::make_entry<copy_entry>(scheds[sched_idx],
                                              ccl_buffer((void*)(&(coll_param.send_buf)),
                                                         total_send_bytes,
                                                         send_offsets[comm_rank],
                                                         ccl_buffer_type::INDIRECT),
                                              ccl_buffer((void*)(&(coll_param.recv_buf)),
                                                         total_recv_bytes,
                                                         recv_offsets[comm_rank],
                                                         ccl_buffer_type::INDIRECT),
                                              send_counts[comm_rank],
                                              dtype);
    }

    for (size_t idx = 0; idx < comm_size; idx++) {
        size_t src = (comm_rank + idx) % comm_size;

        if (src == comm_rank)
            continue;

        size_t sched_idx =
            (ccl::global_data::env().alltoall_scatter_plain) ? 0 : (comm_rank + src) % sched_count;
        ccl_buffer recv_buf;

        if (inplace) {
            recv_buf = scheds[sched_idx]->alloc_buffer(recv_counts[src] * dtype_size);
            recv_bufs[src] = recv_buf;
        }
        else
            recv_buf = ccl_buffer((void*)(&(coll_param.recv_buf)),
                                  total_recv_bytes,
                                  recv_offsets[src],
                                  ccl_buffer_type::INDIRECT);

        entry_factory::make_chunked_recv_entry(
            scheds, sched_idx, recv_buf, recv_counts[src], dtype, src, comm);

        ccl_coll_add_scatter_alltoallv_barriers(scheds, sched_idx);
    }

    for (size_t idx = 0; idx < comm_size; idx++) {
        size_t dst = (comm_rank - idx + comm_size) % comm_size;

        if (dst == comm_rank)
            continue;

        size_t sched_idx =
            (ccl::global_data::env().alltoall_scatter_plain) ? 0 : (comm_rank + dst) % sched_count;

        entry_factory::make_chunked_send_entry(scheds,
                                               sched_idx,
                                               ccl_buffer((void*)(&(coll_param.send_buf)),
                                                          total_send_bytes,
                                                          send_offsets[dst],
                                                          ccl_buffer_type::INDIRECT),
                                               send_counts[dst],
                                               dtype,
                                               dst,
                                               comm);

        ccl_coll_add_scatter_alltoallv_barriers(scheds, sched_idx);
    }

    if (!inplace)
        return ccl_status_success;

    main_sched->sync_partial_scheds();

    for (size_t idx = 0; idx < comm_size; idx++) {
        if (idx == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + idx) % sched_count;

        entry_factory::make_entry<copy_entry>(scheds[sched_idx],
                                              recv_bufs[idx],
                                              ccl_buffer((void*)(&(coll_param.recv_buf)),
                                                         total_recv_bytes,
                                                         recv_offsets[idx],
                                                         ccl_buffer_type::INDIRECT),
                                              recv_counts[idx],
                                              dtype);
    }

    return ccl_status_success;
}

ccl_status_t ccl_coll_build_scatter_barrier_alltoallv(ccl_master_sched* main_sched,
                                                      std::vector<ccl_sched*>& scheds,
                                                      const ccl_coll_param& coll_param) {
    LOG_DEBUG("build scatter_barrier alltoallv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    size_t comm_rank = comm->rank();
    size_t comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    bool inplace =
        (coll_param.send_buf && (coll_param.send_buf == coll_param.recv_buf)) ? true : false;

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    std::vector<ccl_buffer> recv_bufs;
    if (inplace)
        recv_bufs.resize(comm_size);

    std::vector<ccl_sched*> recv_scheds(sched_count);
    std::vector<ccl_sched*> send_scheds(sched_count);

    for (size_t idx = 0; idx < sched_count; idx++) {
        auto recv_sched = entry_factory::make_entry<subsched_entry>(
                              scheds[idx], 0, [](ccl_sched* s) {}, "A2AV_RECV")
                              ->get_subsched();

        recv_scheds[idx] = recv_sched;

        auto send_sched = entry_factory::make_entry<subsched_entry>(
                              scheds[idx], 0, [](ccl_sched* s) {}, "A2AV_SEND")
                              ->get_subsched();

        send_scheds[idx] = send_sched;
    }

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::make_entry<copy_entry>(recv_scheds[sched_idx],
                                              ccl_buffer((void*)(&(coll_param.send_buf)),
                                                         total_send_bytes,
                                                         send_offsets[comm_rank],
                                                         ccl_buffer_type::INDIRECT),
                                              ccl_buffer((void*)(&(coll_param.recv_buf)),
                                                         total_recv_bytes,
                                                         recv_offsets[comm_rank],
                                                         ccl_buffer_type::INDIRECT),
                                              send_counts[comm_rank],
                                              dtype);
    }

    for (size_t idx = 0; idx < comm_size; idx++) {
        size_t src = (comm_rank + idx) % comm_size;

        if (src == comm_rank)
            continue;

        size_t sched_idx =
            (ccl::global_data::env().alltoall_scatter_plain) ? 0 : (comm_rank + src) % sched_count;

        auto sched = recv_scheds[sched_idx];

        ccl_buffer recv_buf;

        if (inplace) {
            recv_buf = sched->alloc_buffer(recv_counts[src] * dtype_size);
            recv_bufs[src] = recv_buf;
        }
        else
            recv_buf = ccl_buffer((void*)(&(coll_param.recv_buf)),
                                  total_recv_bytes,
                                  recv_offsets[src],
                                  ccl_buffer_type::INDIRECT);

        entry_factory::make_chunked_recv_entry(
            recv_scheds, sched_idx, recv_buf, recv_counts[src], dtype, src, comm);

        ccl_coll_add_scatter_alltoallv_barriers(recv_scheds, sched_idx);
    }

    for (size_t idx = 0; idx < comm_size; idx++) {
        size_t dst = (comm_rank - idx + comm_size) % comm_size;

        if (dst == comm_rank)
            continue;

        size_t sched_idx =
            (ccl::global_data::env().alltoall_scatter_plain) ? 0 : (comm_rank + dst) % sched_count;

        entry_factory::make_chunked_send_entry(send_scheds,
                                               sched_idx,
                                               ccl_buffer((void*)(&(coll_param.send_buf)),
                                                          total_send_bytes,
                                                          send_offsets[dst],
                                                          ccl_buffer_type::INDIRECT),
                                               send_counts[dst],
                                               dtype,
                                               dst,
                                               comm);

        ccl_coll_add_scatter_alltoallv_barriers(send_scheds, sched_idx);
    }

    if (!inplace)
        return ccl_status_success;

    main_sched->sync_partial_scheds();

    for (size_t idx = 0; idx < comm_size; idx++) {
        if (idx == comm_rank)
            continue;

        size_t sched_idx = (comm_rank + idx) % sched_count;

        entry_factory::make_entry<copy_entry>(scheds[sched_idx],
                                              recv_bufs[idx],
                                              ccl_buffer((void*)(&(coll_param.recv_buf)),
                                                         total_recv_bytes,
                                                         recv_offsets[idx],
                                                         ccl_buffer_type::INDIRECT),
                                              recv_counts[idx],
                                              dtype);
    }

    return ccl_status_success;
}
