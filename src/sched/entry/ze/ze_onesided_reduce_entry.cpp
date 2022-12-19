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
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_onesided_reduce_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/queue/queue.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_onesided_reduce_entry::ze_onesided_reduce_entry(ccl_sched* sched,
                                                   ccl_buffer send_buf,
                                                   ccl_buffer recv_buf,
                                                   size_t cnt,
                                                   const ccl_datatype& dtype,
                                                   reduction op,
                                                   int root,
                                                   ccl_comm* comm,
                                                   std::vector<ze_event_handle_t> wait_events,
                                                   size_t peer_buf_offset)
        : ze_base_entry(sched, comm, 2 /* request additional events */, wait_events),
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          root(root),
          buf_size_bytes(dtype.size() * cnt),
          peer_buf_offset_bytes(dtype.size() * peer_buf_offset),
          empty_kernel_event(nullptr),
          empty_kernel(nullptr),
          empty_kernel_name("empty_kernel") {
    skip_entry = !cnt || ((comm->size() == 1) && (send_buf == recv_buf));
    if (skip_entry) {
        // skip entry init and finalize
        sched->ze_entries.pop_back();
    }
}

void ze_onesided_reduce_entry::init_ze_hook() {
    CCL_THROW_IF_NOT(comm_rank == root, "unexpected comm_rank ", comm_rank, ", expected ", root);

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();

    if (comm->size() == 1) {
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 recv_buf_ptr,
                 send_buf_ptr,
                 buf_size_bytes,
                 ze_base_entry::entry_event,
                 0,
                 nullptr));
        return;
    }

    /* create kernels */
    ccl_buffer right_send_buf;
    int peer_rank = (comm_rank + 1) % comm_size;
    sched->get_memory().handle_manager.get(peer_rank, 0, right_send_buf, comm);
    LOG_DEBUG(
        "get IPC pointers from ", peer_rank, " by ", root, ", right_send_buf: ", right_send_buf);

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();

    // TODO: in place case check! diff idx for handle_mngr

    right_send_buf_ptr = static_cast<char*>(right_send_buf.get_ptr()) + peer_buf_offset_bytes;

    void* kernel_input_buf2 = right_send_buf_ptr;
    if (global_data::env().enable_kernel_1s_copy_ops) {
        ccl::alloc_param alloc_param(buf_size_bytes, buffer_type::ze, buffer_place::device);
        void* tmp_buf_ptr = sched->alloc_buffer(alloc_param).get_ptr();
        kernel_input_buf2 = tmp_buf_ptr;
    }

    ze_kernel_args_t reduce_local_kernel_args{ &comm_rank,    &comm_size,         &cnt,
                                               &send_buf_ptr, &kernel_input_buf2, &recv_buf_ptr };

    ccl::global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    main_kernel_name =
        "reduce_local_outofplace_kernel_" + to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    LOG_DEBUG("get kernel: name: ", main_kernel_name);
    ccl::global_data::get().ze_data->cache->get(worker_idx, module, main_kernel_name, &main_kernel);

    LOG_DEBUG("kernel ", main_kernel, " args:\n", to_string(reduce_local_kernel_args));
    set_kernel_args(main_kernel, reduce_local_kernel_args);

    ze_group_size_t group_size;
    get_suggested_group_size(main_kernel, cnt, &group_size);
    LOG_DEBUG("suggested group size: ", to_string(group_size));

    get_suggested_group_count(group_size, cnt, &group_count);
    LOG_DEBUG("suggested group count: ", to_string(group_count));

    ZE_CALL(zeKernelSetGroupSize,
            (main_kernel, group_size.groupSizeX, group_size.groupSizeY, group_size.groupSizeZ));

    if (ccl::global_data::env().enable_kernel_1s_ipc_wa) {
        LOG_DEBUG("get kernel: name: ", empty_kernel_name);
        ccl::global_data::get().ze_data->cache->get(
            worker_idx, module, empty_kernel_name, &empty_kernel);
        CCL_THROW_IF_NOT(empty_kernel, "null empty_kernel");
        /* use allreduce_kernel_args since they have pointers to peer mem */
        set_kernel_args(empty_kernel, reduce_local_kernel_args);
    }

    if (empty_kernel) {
        empty_kernel_event = ze_base_entry::create_event();
    }

    ze_event_handle_t kernel_wait_event = nullptr;
    /* do appends */
    if (empty_kernel) {
        LOG_DEBUG("append empty kernel");
        ze_group_count_t empty_group_count = { 1, 1, 1 };
        ZE_CALL(
            zeCommandListAppendLaunchKernel,
            (get_comp_list(), empty_kernel, &empty_group_count, empty_kernel_event, 0, nullptr));
        kernel_wait_event = empty_kernel_event;
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        LOG_DEBUG("one-sided multi-phase algorithm");

        copy_from_peer_event = ze_base_entry::create_event();
        // copy to tmp buf
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 kernel_input_buf2,
                 right_send_buf_ptr,
                 buf_size_bytes,
                 copy_from_peer_event,
                 (empty_kernel_event) ? 1 : 0,
                 &empty_kernel_event));
        kernel_wait_event = copy_from_peer_event;
    }
    else {
        LOG_DEBUG("one-sided monolithic algorithm");
    }

    ZE_CALL(zeCommandListAppendLaunchKernel,
            (get_comp_list(),
             main_kernel,
             &group_count,
             entry_event,
             (kernel_wait_event) ? 1 : 0,
             &kernel_wait_event));
}

void ze_onesided_reduce_entry::finalize_ze_hook() {
    if (comm->size() == 1) {
        return;
    }
    if (empty_kernel_event) {
        ccl::global_data::get().ze_data->cache->push(
            worker_idx, module, empty_kernel_name, empty_kernel);
    }
    ccl::global_data::get().ze_data->cache->push(worker_idx, module, main_kernel_name, main_kernel);
}

void ze_onesided_reduce_entry::start() {
    if (skip_entry) {
        ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
        status = ccl_sched_entry_status_complete;
        return;
    }

    size_t kernel_counter = 0;
    if (ccl::global_data::env().enable_kernel_sync) {
        kernel_counter = ccl::global_data::get().ze_data->kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
    }
    else {
        ccl::global_data::get().ze_data->kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_onesided_reduce_entry::update() {
    ze_base_entry::update();

    if (ccl::global_data::env().enable_kernel_sync &&
        ccl::global_data::get().ze_data->kernel_counter > 0) {
        ccl::global_data::get().ze_data->kernel_counter--;
    }
}

std::string ze_onesided_reduce_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << cnt * dtype.size();
    return out.str();
}

void ze_onesided_reduce_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", cnt ",
                       cnt,
                       ", send_buf ",
                       send_buf,
                       ", recv_buf ",
                       recv_buf,
                       ", op ",
                       ccl_reduction_to_str(op),
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}
