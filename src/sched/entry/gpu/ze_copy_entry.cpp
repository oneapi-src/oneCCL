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
#include "sched/entry/gpu/ze_copy_entry.hpp"

#include <ze_api.h>

using namespace ccl;

ze_copy_entry::ze_copy_entry(ccl_sched* sched,
                             ccl_buffer in_buf,
                             ccl_buffer out_buf,
                             size_t count,
                             const ccl_datatype& dtype,
                             copy_attr attr)
        : ze_base_entry(sched),
          sched(sched),
          in_buf(in_buf),
          out_buf(out_buf),
          dtype(dtype),
          attr(attr),
          buf_size_bytes(dtype.size() * count) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

ze_copy_entry::~ze_copy_entry() {
    finalize();
}

void ze_copy_entry::init() {
    if (ze_base_entry::is_initialized) {
        return;
    }

    LOG_DEBUG("initialization");

    ze_base_entry::init(init_mode::copy);

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

    void* dst = out_buf.get_ptr();
    void* src = static_cast<char*>(in_buf.get_ptr()) + attr.in_buf_offset * dtype.size();
    ze_command_list_handle_t list = ze_base_entry::get_copy_list();

    ZE_CALL(zeCommandListAppendMemoryCopy,
            (list, dst, src, buf_size_bytes, ze_base_entry::entry_event, 0, nullptr));
    ZE_CALL(zeCommandListClose, (list));

    LOG_DEBUG("initialization complete");
}

void ze_copy_entry::start() {
    init();

    ze_base_entry::start();

    status = ccl_sched_entry_status_started;
}

void ze_copy_entry::update() {
    ze_base_entry::update();
    if (status == ccl_sched_entry_status_complete && !sched->coll_attr.to_cache) {
        finalize();
    }
}

void ze_copy_entry::finalize() {
    if (!ze_base_entry::is_initialized) {
        return;
    }

    LOG_DEBUG("finalization");

    ze_base_entry::finalize();

    LOG_DEBUG("finalization complete");
}
