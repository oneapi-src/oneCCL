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
#include "sched/entry/ze/ze_copy_entry.hpp"

using namespace ccl;

ze_copy_entry::ze_copy_entry(ccl_sched* sched,
                             ccl_buffer in_buf,
                             ccl_buffer out_buf,
                             size_t count,
                             const ccl_datatype& dtype,
                             const copy_attr& attr,
                             std::vector<ze_event_handle_t> wait_events)
        : ze_base_entry(sched, nullptr, 1, wait_events),
          sched(sched),
          in_buf(in_buf),
          out_buf(out_buf),
          dtype(dtype),
          attr(attr),
          count(count) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

void ze_copy_entry::init_ze_hook() {
    if (attr.peer_rank != ccl_comm::invalid_rank) {
        if (!out_buf) {
            sched->get_memory().handle_manager.get(
                attr.peer_rank, attr.peer_buf_idx, out_buf, attr.map_comm);
        }

        if (!in_buf) {
            sched->get_memory().handle_manager.get(
                attr.peer_rank, attr.peer_buf_idx, in_buf, attr.map_comm);
        }
    }

    void* dst = static_cast<char*>(out_buf.get_ptr()) + attr.out_buf_offset * dtype.size();
    void* src = static_cast<char*>(in_buf.get_ptr()) + attr.in_buf_offset * dtype.size();
    ze_command_list_handle_t list =
        ze_base_entry::get_copy_list(attr.hint_queue_index, attr.is_peer_card_copy);

    ZE_CALL(zeCommandListAppendMemoryCopy,
            (list, dst, src, dtype.size() * count, ze_base_entry::entry_event, 0, nullptr));
}
