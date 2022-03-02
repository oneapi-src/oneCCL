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
#include "sched/entry/ze/allreduce/ze_onesided_allreduce_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/queue/queue.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_onesided_allreduce_entry::ze_onesided_allreduce_entry(ccl_sched* sched,
                                                         ccl_buffer send_buf,
                                                         ccl_buffer recv_buf,
                                                         size_t cnt,
                                                         const ccl_datatype& dtype,
                                                         reduction op,
                                                         ccl_comm* comm,
                                                         std::vector<ze_event_handle_t> wait_events,
                                                         const size_t buf_offset_cnt)
        : ze_base_entry(sched, comm, 3 /* request additional events */, wait_events),
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          buf_size_bytes(dtype.size() * cnt),
          buf_offset_bytes(dtype.size() * buf_offset_cnt) {}

void ze_onesided_allreduce_entry::init_ze_hook() {
    /* create kernels */
    ccl_buffer right_send_buf;
    ccl_buffer right_recv_buf;
    int peer_rank = (comm_rank + 1) % comm_size;

    send_buf_ptr = static_cast<char*>(send_buf.get_ptr()) + buf_offset_bytes;
    recv_buf_ptr = static_cast<char*>(recv_buf.get_ptr()) + buf_offset_bytes;
    if (send_buf_ptr == recv_buf_ptr) {
        sched->get_memory().handle_manager.get(peer_rank, 1, right_send_buf, comm);
        sched->get_memory().handle_manager.get(peer_rank, 1, right_recv_buf, comm);
    }
    else {
        sched->get_memory().handle_manager.get(peer_rank, 0, right_send_buf, comm);
        sched->get_memory().handle_manager.get(peer_rank, 1, right_recv_buf, comm);
    }

    right_send_buf_ptr = static_cast<char*>(right_send_buf.get_ptr()) + buf_offset_bytes;
    right_recv_buf_ptr = static_cast<char*>(right_recv_buf.get_ptr()) + buf_offset_bytes;

    void* tmp_buf_ptr{};

    if (global_data::env().enable_kernel_1s_copy_ops) {
        main_kernel_name = "reduce_local_outofplace_kernel_";
        ccl::alloc_param alloc_param(buf_size_bytes, buffer_type::ze, buffer_place::device);
        tmp_buf_ptr = sched->alloc_buffer(alloc_param).get_ptr();
    }
    else {
        main_kernel_name = "allreduce_kernel_";
    }

    main_kernel_name += to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    LOG_DEBUG("get kernel: name: ", main_kernel_name);

    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
    global_data::get().ze_data->cache->get(worker_idx, module, main_kernel_name, &main_kernel);

    ze_kernel_args_t allreduce_kernel_args{ &comm_rank,         &comm_size,    &cnt,
                                            &send_buf_ptr,      &recv_buf_ptr, &right_send_buf_ptr,
                                            &right_recv_buf_ptr };
    ze_kernel_args_t reduce_local_kernel_args{ &comm_rank,    &comm_size,   &cnt,
                                               &send_buf_ptr, &tmp_buf_ptr, &recv_buf_ptr };

    auto& main_kernel_args = (global_data::env().enable_kernel_1s_copy_ops)
                                 ? reduce_local_kernel_args
                                 : allreduce_kernel_args;
    LOG_DEBUG("kernel ", main_kernel, " args:\n", to_string(main_kernel_args));
    set_kernel_args(main_kernel, main_kernel_args);

    ze_group_size_t group_size;
    get_suggested_group_size(main_kernel, cnt, &group_size);
    LOG_DEBUG("suggested group size: ", to_string(group_size));

    get_suggested_group_count(group_size, cnt, &group_count);
    LOG_DEBUG("suggested group count: ", to_string(group_count));

    ZE_CALL(zeKernelSetGroupSize,
            (main_kernel, group_size.groupSizeX, group_size.groupSizeY, group_size.groupSizeZ));

    if (global_data::env().enable_kernel_1s_ipc_wa) {
        LOG_DEBUG("get kernel: name: ", empty_kernel_name);
        global_data::get().ze_data->cache->get(
            worker_idx, module, empty_kernel_name, &empty_kernel);
        CCL_THROW_IF_NOT(empty_kernel, "null empty_kernel");
        /* use allreduce_kernel_args since they have pointers to peer mem */
        set_kernel_args(empty_kernel, allreduce_kernel_args);
    }

    if (empty_kernel) {
        empty_kernel_event = ze_base_entry::create_event();
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        copy_from_peer_event = ze_base_entry::create_event();
        reduce_local_kernel_event = ze_base_entry::create_event();
    }

    /* do appends */
    if (empty_kernel) {
        LOG_DEBUG("append empty kernel");
        ze_group_count_t empty_group_count = { 1, 1, 1 };
        ZE_CALL(zeCommandListAppendLaunchKernel,
                (ze_base_entry::get_comp_list(),
                 empty_kernel,
                 &empty_group_count,
                 empty_kernel_event,
                 0,
                 nullptr));
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        LOG_DEBUG("one-sided multi-phase algorithm");

        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 tmp_buf_ptr,
                 right_send_buf_ptr,
                 buf_size_bytes,
                 copy_from_peer_event,
                 (empty_kernel_event) ? 1 : 0,
                 &empty_kernel_event));

        ZE_CALL(zeCommandListAppendLaunchKernel,
                (ze_base_entry::get_comp_list(),
                 main_kernel,
                 &group_count,
                 reduce_local_kernel_event,
                 1,
                 &copy_from_peer_event));

        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 right_recv_buf_ptr,
                 recv_buf_ptr,
                 buf_size_bytes,
                 ze_base_entry::entry_event,
                 1,
                 &reduce_local_kernel_event));
    }
    else {
        LOG_DEBUG("one-sided monolithic algorithm");
        ZE_CALL(zeCommandListAppendLaunchKernel,
                (ze_base_entry::get_comp_list(),
                 main_kernel,
                 &group_count,
                 ze_base_entry::entry_event,
                 (empty_kernel_event) ? 1 : 0,
                 &empty_kernel_event));
    }
}

void ze_onesided_allreduce_entry::finalize_ze_hook() {
    if (empty_kernel_event) {
        global_data::get().ze_data->cache->push(
            worker_idx, module, empty_kernel_name, empty_kernel);
    }
    global_data::get().ze_data->cache->push(worker_idx, module, main_kernel_name, main_kernel);
}

void ze_onesided_allreduce_entry::start() {
    size_t kernel_counter = 0;
    if (global_data::env().enable_kernel_sync) {
        kernel_counter = global_data::get().ze_data->kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
    }
    else {
        global_data::get().ze_data->kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_onesided_allreduce_entry::update() {
    ze_base_entry::update();

    if (global_data::env().enable_kernel_sync && global_data::get().ze_data->kernel_counter > 0) {
        global_data::get().ze_data->kernel_counter--;
    }
}
