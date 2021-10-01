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
#include <numeric>

#include "coll/algorithms/algorithms.hpp"
#include "sched/entry/coll/coll_entry_helper.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

ccl::status ccl_coll_build_direct_allgatherv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             size_t send_count,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm) {
    LOG_DEBUG("build direct allgatherv");

    entry_factory::make_entry<allgatherv_entry>(
        sched, send_buf, send_count, recv_buf, recv_counts, dtype, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_build_naive_allgatherv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            size_t send_count,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm) {
    LOG_DEBUG("build naive allgatherv");

    int comm_size = comm->size();
    int this_rank = comm->rank();
    size_t dtype_size = dtype.size();
    size_t* offsets = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    ccl::status status = ccl::status::success;

    offsets[0] = 0;
    for (int rank_idx = 1; rank_idx < comm_size; ++rank_idx) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * dtype_size;
    }

    if (send_buf != recv_buf) {
        // out-of-place case
        entry_factory::make_entry<copy_entry>(
            sched, send_buf, recv_buf + offsets[this_rank], send_count, dtype);
    }

    for (int rank_idx = 0; rank_idx < comm_size; ++rank_idx) {
        if (rank_idx != this_rank) {
            // send own buffer to other ranks
            entry_factory::make_chunked_send_entry(
                sched, recv_buf + offsets[this_rank], send_count, dtype, rank_idx, comm);
            // recv other's rank buffer
            entry_factory::make_chunked_recv_entry(
                sched, recv_buf + offsets[rank_idx], recv_counts[rank_idx], dtype, rank_idx, comm);
        }
    }

    CCL_FREE(offsets);
    return status;
}

ccl::status ccl_coll_build_ring_allgatherv(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           size_t send_count,
                                           ccl_buffer recv_buf,
                                           const size_t* recv_counts,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm) {
    LOG_DEBUG("build ring allgatherv, send_count ", send_count);

    ccl::status status = ccl::status::success;
    int comm_size, rank;
    size_t dtype_size = dtype.size();
    int src, dst;

    comm_size = comm->size();
    rank = comm->rank();

    size_t* offsets = static_cast<size_t*>(CCL_MALLOC(comm_size * sizeof(size_t), "offsets"));
    offsets[0] = 0;
    for (int rank_idx = 1; rank_idx < comm_size; ++rank_idx) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * dtype_size;
    }

    if (send_buf != recv_buf) {
        entry_factory::make_entry<copy_entry>(
            sched, send_buf, recv_buf + offsets[rank], send_count, dtype);
    }

    ccl_buffer sbuf, rbuf;

    src = (comm_size + rank - 1) % comm_size;
    dst = (comm_size + rank + 1) % comm_size;

    size_t block_idx =
        rank; // start send with 'rank' block and recv with 'rank-1' block and move blocks left
    size_t send_block_idx, recv_block_idx;
    size_t send_block_count, recv_block_count;
    size_t send_block_offset, recv_block_offset;

    for (int idx = 0; idx < (comm_size - 1); idx++) {
        send_block_idx = block_idx;
        recv_block_idx = (comm_size + block_idx - 1) % comm_size;
        send_block_count = recv_counts[send_block_idx];
        recv_block_count = recv_counts[recv_block_idx];
        send_block_offset = offsets[send_block_idx];
        recv_block_offset = offsets[recv_block_idx];
        sbuf = recv_buf + send_block_offset;
        rbuf = recv_buf + recv_block_offset;

        entry_factory::make_entry<send_entry>(sched, sbuf, send_block_count, dtype, dst, comm);
        entry_factory::make_entry<recv_entry>(sched, rbuf, recv_block_count, dtype, src, comm);
        sched->add_barrier();

        block_idx = (comm_size + block_idx - 1) % comm_size; // move left
    }

    CCL_FREE(offsets);
    return status;
}

ccl::status ccl_coll_get_allgatherv_bufs_and_offsets(const ccl_coll_param& coll_param,
                                                     std::vector<ccl_buffer>& recv_bufs,
                                                     std::vector<size_t>& recv_offsets) {
    int comm_size = coll_param.comm->size();
    size_t dtype_size = coll_param.dtype.size();

    recv_bufs.resize(comm_size);
    recv_offsets.resize(comm_size);

    if (coll_param.recv_bufs.size() > 1) {
        CCL_THROW_IF_NOT((int)coll_param.recv_bufs.size() == comm_size,
                         "unexpected recv_bufs.size ",
                         coll_param.recv_bufs.size(),
                         ", expected ",
                         comm_size);

        for (int idx = 0; idx < comm_size; idx++) {
            recv_bufs[idx].set(coll_param.get_recv_buf_ptr(idx),
                               coll_param.get_recv_count(idx) * dtype_size,
                               ccl_buffer_type::INDIRECT);
            recv_offsets[idx] = 0;
        }
    }
    else {
        size_t offset = 0;
        size_t dtype_size = coll_param.dtype.size();
        for (int idx = 0; idx < comm_size; idx++) {
            size_t bytes = coll_param.get_recv_count(idx) * dtype_size;
            recv_bufs[idx].set(
                coll_param.get_recv_buf_ptr(), offset + bytes, offset, ccl_buffer_type::INDIRECT);
            recv_offsets[idx] = offset;
            offset += bytes;
        }
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_flat_allgatherv(ccl_master_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param) {
    LOG_DEBUG("build flat allgatherv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    bool inplace = coll_param.is_inplace();

    std::vector<ccl_buffer> recv_bufs;
    std::vector<size_t> recv_offsets;
    ccl_coll_get_allgatherv_bufs_and_offsets(coll_param, recv_bufs, recv_offsets);

    auto send_seg = ccl_buffer(coll_param.get_send_buf_ptr(),
                               coll_param.get_send_count() * dtype_size,
                               ccl_buffer_type::INDIRECT);

    if (!inplace) {
        entry_factory::make_entry<copy_entry>(scheds[2 * comm_rank % sched_count],
                                              ccl_buffer(coll_param.get_send_buf_ptr(),
                                                         coll_param.get_send_count() * dtype_size,
                                                         ccl_buffer_type::INDIRECT),
                                              recv_bufs[comm_rank],
                                              coll_param.get_recv_count(comm_rank),
                                              dtype);
    }
    else {
        size_t total_recv_bytes =
            std::accumulate(coll_param.recv_counts.begin(), coll_param.recv_counts.end(), 0) *
            dtype_size;
        send_seg = ccl_buffer(coll_param.get_send_buf_ptr(),
                              total_recv_bytes,
                              recv_offsets[comm_rank],
                              ccl_buffer_type::INDIRECT);
    }

    CCL_THROW_IF_NOT(static_cast<int>(sched_count) == comm_size,
                     "unexpected sched_count ",
                     sched_count,
                     ", expected ",
                     comm_size);

    for (size_t idx = 0; idx < sched_count; idx++) {
        if (static_cast<int>(idx) == comm_rank)
            continue;

        entry_factory::make_entry<recv_entry>(scheds[(comm_rank + idx) % sched_count],
                                              recv_bufs[idx],
                                              coll_param.get_recv_count(idx),
                                              dtype,
                                              idx,
                                              comm);

        entry_factory::make_entry<send_entry>(scheds[(comm_rank + idx) % sched_count],
                                              send_seg,
                                              coll_param.get_recv_count(comm_rank),
                                              dtype,
                                              idx,
                                              comm);
    }
    main_sched->sync_partial_scheds();

    return ccl::status::success;
}

ccl::status ccl_coll_build_multi_bcast_allgatherv(ccl_master_sched* main_sched,
                                                  std::vector<ccl_sched*>& scheds,
                                                  const ccl_coll_param& coll_param,
                                                  size_t data_partition_count) {
    LOG_DEBUG("build multi_bcast allgatherv");

    CCL_THROW_IF_NOT(data_partition_count > 0, "data_partition_count should be > 0 ");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    bool inplace = coll_param.is_inplace();

    std::vector<ccl_buffer> recv_bufs;
    std::vector<size_t> recv_offsets;
    ccl_coll_get_allgatherv_bufs_and_offsets(coll_param, recv_bufs, recv_offsets);

    if (!inplace) {
        std::vector<size_t> copy_counts(data_partition_count);
        std::vector<size_t> copy_offsets(data_partition_count);
        for (size_t idx = 0; idx < data_partition_count; idx++) {
            copy_counts[idx] = coll_param.get_recv_count(comm_rank) / data_partition_count;
            copy_offsets[idx] = idx * copy_counts[idx] * dtype_size;
        }
        copy_counts[data_partition_count - 1] +=
            coll_param.get_recv_count(comm_rank) % data_partition_count;

        CCL_ASSERT(scheds.size() >= data_partition_count);

        for (size_t idx = 0; idx < data_partition_count; idx++) {
            entry_factory::make_entry<copy_entry>(
                scheds[idx],
                ccl_buffer(coll_param.get_send_buf_ptr(),
                           coll_param.get_send_count() * dtype_size,
                           copy_offsets[idx],
                           ccl_buffer_type::INDIRECT),
                recv_bufs[comm_rank] + copy_offsets[idx],
                copy_counts[idx],
                dtype);
        }
        main_sched->sync_partial_scheds();
    }

    for (int idx = 0; idx < comm_size; idx++) {
        ccl_coll_entry_param param{};
        param.ctype = ccl_coll_bcast;
        param.recv_buf = recv_bufs[idx];
        param.count = coll_param.get_recv_count(idx);
        param.dtype = dtype;
        param.root = idx;
        param.comm = comm;
        param.stream = coll_param.stream;
        coll_entry_helper::add_coll_entry<ccl_coll_bcast>(scheds[idx % sched_count], param);
    }

    return ccl::status::success;
}
