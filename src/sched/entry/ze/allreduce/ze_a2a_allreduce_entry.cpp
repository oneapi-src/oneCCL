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
#include "common/stream/stream.hpp"
#include "sched/entry/ze/allreduce/ze_a2a_allreduce_entry.hpp"
#include "sched/entry/ze/ze_a2a_allgatherv_entry.hpp"
#include "sched/entry/ze/ze_a2a_reduce_scatter_entry.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/queue/queue.hpp"

#include <algorithm>
#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_a2a_allreduce_entry::ze_a2a_allreduce_entry(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t cnt,
                                               const ccl_datatype& dtype,
                                               reduction op,
                                               ccl_comm* comm,
                                               std::vector<ze_event_handle_t> wait_events,
                                               size_t send_buf_idx,
                                               size_t recv_buf_idx)
        : ze_base_entry(sched,
                        (init_mode::compute | init_mode::copy),
                        comm,
                        comm->size() * event_group_count,
                        wait_events),
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          send_buf_idx(send_buf_idx),
          recv_buf_idx(recv_buf_idx),
          peer_count(comm->size() - 1) {
    size_t segment_count = cnt / comm->size();
    bool count_check =
        (segment_count > 0) || (segment_count == 0 && static_cast<size_t>(comm->rank()) < cnt);
    skip_entry = !count_check || ((comm->size() == 1) && (send_buf == recv_buf));
    if (skip_entry) {
        // skip entry init and finalize
        sched->get_memory().ze_entries.pop_back();
    }
}

void ze_a2a_allreduce_entry::init_ze_hook() {
    /* get peer buffers */
    std::vector<ccl_buffer> peer_send_bufs(peer_count);
    std::vector<ccl_buffer> peer_recv_bufs(peer_count);

    for (int i = 0; i < peer_count; ++i) {
        int peer_rank = (comm_rank + i + 1) % comm->size();
        sched->get_memory().handle_manager.get(peer_rank, send_buf_idx, peer_send_bufs[i], comm);
        CCL_THROW_IF_NOT(peer_send_bufs[i].get_ptr(), "null IPC buffer is received");
        sched->get_memory().handle_manager.get(peer_rank, recv_buf_idx, peer_recv_bufs[i], comm);
        CCL_THROW_IF_NOT(peer_recv_bufs[i].get_ptr(), "null IPC buffer is received");
    }

    size_t main_block_count = cnt / comm_size;
    if (main_block_count == 0 && static_cast<size_t>(comm_rank) < cnt) {
        main_block_count = 1;
    }

    size_t block_count = main_block_count;
    if (comm_rank == comm_size - 1) {
        block_count += cnt - main_block_count * comm_size;
    }

    CCL_THROW_IF_NOT(main_block_count > 0, "wrong segment count");

    /* alloc temp buffer */
    size_t tmp_buf_bytes = peer_count * block_count * dtype.size();
    ccl::alloc_param alloc_param(tmp_buf_bytes, buffer_type::ze, buffer_place::device);
    void* tmp_buf = sched->alloc_buffer(alloc_param).get_ptr();

    LOG_DEBUG("rank ",
              comm_size,
              ", main_block_count: ",
              main_block_count,
              ", block_count: ",
              block_count,
              ", tmp buf size: ",
              tmp_buf_bytes,
              ", cnt: ",
              cnt);

    /* copy peer segments to temp buffer */
    size_t main_block_bytes = main_block_count * dtype.size();
    size_t block_bytes = block_count * dtype.size();

    pre_copy_events.resize(peer_count);
    for (auto& event : pre_copy_events) {
        event = ze_base_entry::create_event();
    }

    kernel_events.resize(peer_count);
    for (auto& event : kernel_events) {
        event = ze_base_entry::create_event();
    }

    barrier_event = ze_base_entry::create_event();

    ze_a2a_reduce_scatter_entry::fill_list(ze_base_entry::get_copy_list(),
                                           ze_base_entry::get_comp_list(),
                                           send_buf.get_ptr(),
                                           tmp_buf,
                                           peer_send_bufs,
                                           peer_count,
                                           comm_rank,
                                           block_count,
                                           comm_rank * main_block_bytes,
                                           pre_copy_events,
                                           kernels,
                                           kernel_events,
                                           barrier_event,
                                           dtype,
                                           module,
                                           device,
                                           context,
                                           op,
                                           worker_idx);

    post_copy_events.resize(comm_size);
    for (auto& event : post_copy_events) {
        event = ze_base_entry::create_event();
    }

    ze_a2a_allgatherv_entry::fill_list(ze_base_entry::get_copy_list(),
                                       tmp_buf,
                                       recv_buf.get_ptr(),
                                       peer_recv_bufs,
                                       peer_count,
                                       block_bytes,
                                       comm_rank * main_block_bytes,
                                       false,
                                       post_copy_events,
                                       kernel_events.back());
}

void ze_a2a_allreduce_entry::start() {
    if (skip_entry) {
        ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
        status = ccl_sched_entry_status_complete;
        return;
    }

    ze_base_entry::start();
}

void ze_a2a_allreduce_entry::update() {
    for (const auto& event : post_copy_events) {
        if (!ze_base_entry::is_event_completed(event)) {
            return;
        }
    }

    ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
    ze_base_entry::update();
}
