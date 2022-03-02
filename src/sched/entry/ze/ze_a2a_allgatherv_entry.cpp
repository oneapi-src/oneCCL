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
#include "sched/entry/ze/ze_a2a_allgatherv_entry.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <numeric>

using namespace ccl;
using namespace ccl::ze;

ze_a2a_allgatherv_entry::ze_a2a_allgatherv_entry(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 size_t send_count,
                                                 ccl_buffer recv_buf,
                                                 const size_t* recv_counts,
                                                 const ccl_datatype& dtype,
                                                 ccl_comm* comm,
                                                 std::vector<ze_event_handle_t> wait_events,
                                                 size_t peer_buf_idx,
                                                 size_t peer_buf_offset)
        : ze_base_entry(sched, comm, comm->size() * event_group_count, wait_events),
          send_buf(send_buf),
          send_count(send_count),
          recv_buf(recv_buf),
          recv_counts(recv_counts, recv_counts + comm->size()),
          dtype(dtype),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1) {}

void ze_a2a_allgatherv_entry::fill_list(const ze_base_entry* entry,
                                        int comm_rank,
                                        void* send_buf,
                                        void* recv_buf,
                                        const std::vector<ccl_buffer>& peer_recv_bufs,
                                        int peer_count,
                                        size_t copy_bytes,
                                        size_t offset_bytes,
                                        bool is_inplace,
                                        std::vector<ze_event_handle_t>& copy_events,
                                        ze_event_handle_t wait_event) {
    /* copy send_buf to peer buffers */
    for (int i = 0; i < peer_count; ++i) {
        void* src = send_buf;
        if (is_inplace) {
            src = static_cast<char*>(recv_buf) + offset_bytes;
        }
        void* dst = static_cast<char*>(peer_recv_bufs[i].get_ptr()) + offset_bytes;
        auto list = entry->get_copy_list(i, true);
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (list, dst, src, copy_bytes, copy_events.at(i), (wait_event) ? 1 : 0, &wait_event));
    }

    if (!is_inplace) {
        /* copy send_buf to my buffer */
        void* src = send_buf;
        void* dst = static_cast<char*>(recv_buf) + offset_bytes;
        auto list = entry->get_copy_list();
        ZE_CALL(
            zeCommandListAppendMemoryCopy,
            (list, dst, src, copy_bytes, copy_events.back(), (wait_event) ? 1 : 0, &wait_event));
    }
}

void ze_a2a_allgatherv_entry::init_ze_hook() {
    /* get peer recv buffers */
    std::vector<ccl_buffer> peer_recv_bufs(peer_count);

    for (int i = 0; i < peer_count; ++i) {
        int peer_rank = (comm_rank + i + 1) % comm->size();
        ccl_buffer buf{};
        sched->get_memory().handle_manager.get(peer_rank, peer_buf_idx, buf, comm);
        CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
        peer_recv_bufs[i] = buf + peer_buf_offset * dtype.size();
    }

    bool is_inplace{};
    if (send_buf == recv_buf) {
        is_inplace = true;
    }

    size_t offset_count = std::accumulate(recv_counts.begin(), recv_counts.begin() + comm_rank, 0);
    size_t offset_bytes = offset_count * dtype.size();
    size_t block_bytes =
        (!is_inplace) ? (send_count * dtype.size()) : recv_counts[comm_rank] * dtype.size();
    LOG_DEBUG("rank: ", comm_rank, ", block_bytes: ", block_bytes);

    copy_events.resize((!is_inplace) ? comm_size : peer_count);
    for (auto& event : copy_events) {
        event = ze_base_entry::create_event();
    }

    fill_list(this,
              comm_rank,
              send_buf.get_ptr(),
              recv_buf.get_ptr(),
              peer_recv_bufs,
              peer_count,
              block_bytes,
              offset_bytes,
              is_inplace,
              copy_events);
}

void ze_a2a_allgatherv_entry::update() {
    for (const auto& event : copy_events) {
        if (!ze_base_entry::is_event_completed(event)) {
            return;
        }
    }

    ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
    ze_base_entry::update();
}
