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
#include "sched/entry/ze/ze_a2a_gatherv_entry.hpp"

#include <numeric>

using namespace ccl;
using namespace ccl::ze;

ze_a2a_gatherv_entry::ze_a2a_gatherv_entry(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           size_t send_count,
                                           ccl_buffer recv_buf,
                                           const size_t* recv_counts,
                                           const ccl_datatype& dtype,
                                           int root,
                                           ccl_comm* comm,
                                           size_t peer_buf_idx)
        : ze_base_entry(sched, comm),
          send_buf(send_buf),
          send_bytes(send_count * dtype.size()),
          recv_buf(recv_buf),
          recv_counts(recv_counts, recv_counts + comm->size()),
          dtype(dtype),
          root(root),
          peer_buf_idx(peer_buf_idx) {}

void ze_a2a_gatherv_entry::init_ze_hook() {
    /* get peer recv buffers */
    ccl_buffer peer_recv_buf{};

    bool is_root = comm_rank == root;
    if (!is_root) {
        sched->get_memory().handle_manager.get(root, peer_buf_idx, peer_recv_buf, comm);
        CCL_THROW_IF_NOT(peer_recv_buf.get_ptr(), "null IPC buffer is received");
    }
    else {
        peer_recv_buf = recv_buf;
    }

    bool is_inplace = send_buf == recv_buf;

    size_t offset_count = std::accumulate(recv_counts.begin(), recv_counts.begin() + comm_rank, 0);
    size_t offset_bytes = offset_count * dtype.size();
    size_t block_bytes = (!is_inplace) ? send_bytes : recv_counts[comm_rank] * dtype.size();
    LOG_DEBUG("rank: ", comm_rank, ", block_bytes: ", block_bytes);

    if (!is_root || (is_root && !is_inplace)) {
        void* src = send_buf.get_ptr();
        if (is_inplace) {
            src = static_cast<char*>(recv_buf.get_ptr()) + offset_bytes;
        }
        void* dst = static_cast<char*>(peer_recv_buf.get_ptr()) + offset_bytes;
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 dst,
                 src,
                 block_bytes,
                 ze_base_entry::entry_event,
                 0,
                 nullptr));
    }
}

std::string ze_a2a_gatherv_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << send_bytes;
    return out.str();
}

void ze_a2a_gatherv_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", send_buf ",
                       send_buf,
                       ", recv_buf ",
                       recv_buf,
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}
