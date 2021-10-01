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
#include "common/comm/l0/modules/kernel_utils.hpp"
#include "common/stream/stream.hpp"
#include "sched/entry/gpu/ze_primitives.hpp"
#include "sched/entry/gpu/ze_cache.hpp"
#include "sched/entry/gpu/ze_allreduce_entry.hpp"
#include "sched/queue/queue.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_allreduce_entry::ze_allreduce_entry(ccl_sched* sched,
                                       ccl_buffer send_buf,
                                       ccl_buffer recv_buf,
                                       size_t cnt,
                                       const ccl_datatype& dtype,
                                       reduction op,
                                       ccl_comm* comm)
        : ze_base_entry(sched, comm, local_events_count /* request additional events */),
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          buf_size_bytes(dtype.size() * cnt) {}

ze_allreduce_entry::~ze_allreduce_entry() {
    finalize();
}

void ze_allreduce_entry::init() {
    if (ze_base_entry::is_initialized) {
        return;
    }

    LOG_DEBUG("initialization");

    init_mode init_mode_type;
    if (global_data::env().enable_kernel_1s_copy_ops) {
        init_mode_type = (init_mode::copy | init_mode::compute);
    }
    else {
        init_mode_type = init_mode::compute;
    }

    ze_base_entry::init(init_mode_type);

    /* create kernels */
    ccl_buffer right_send_buf;
    ccl_buffer right_recv_buf;
    int peer_rank = (comm_rank + 1) % comm_size;

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();
    if (send_buf_ptr == recv_buf_ptr) {
        sched->get_memory().handle_manager.get(peer_rank, 1, right_send_buf, comm);
        sched->get_memory().handle_manager.get(peer_rank, 1, right_recv_buf, comm);
    }
    else {
        sched->get_memory().handle_manager.get(peer_rank, 0, right_send_buf, comm);
        sched->get_memory().handle_manager.get(peer_rank, 1, right_recv_buf, comm);
    }
    right_send_buf_ptr = right_send_buf.get_ptr();
    right_recv_buf_ptr = right_recv_buf.get_ptr();

    ze_kernel_args_t allreduce_kernel_args = { { sizeof(comm_rank), &comm_rank },
                                               { sizeof(comm_size), &comm_size },
                                               { sizeof(cnt), &cnt },
                                               { sizeof(send_buf_ptr), &send_buf_ptr },
                                               { sizeof(recv_buf_ptr), &recv_buf_ptr },
                                               { sizeof(right_send_buf_ptr), &right_send_buf_ptr },
                                               { sizeof(right_recv_buf_ptr),
                                                 &right_recv_buf_ptr } };

    ze_kernel_args_t reduce_local_kernel_args = { { sizeof(comm_rank), &comm_rank },
                                                  { sizeof(comm_size), &comm_size },
                                                  { sizeof(cnt), &cnt },
                                                  { sizeof(send_buf_ptr), &send_buf_ptr },
                                                  { sizeof(tmp_buf_ptr), &tmp_buf_ptr },
                                                  { sizeof(recv_buf_ptr), &recv_buf_ptr } };

    global_data::get().ze_cache->get(context, device, "kernels.spv", &module);

    if (global_data::env().enable_kernel_1s_copy_ops) {
        main_kernel_name = "reduce_local_outofplace_kernel_";
        device_mem_alloc_desc = default_device_mem_alloc_desc;
        global_data::get().ze_cache->get(worker_idx,
                                         context,
                                         device,
                                         device_mem_alloc_desc,
                                         buf_size_bytes,
                                         0, /*alignment*/
                                         &tmp_buf_ptr);
    }
    else {
        main_kernel_name = "allreduce_kernel_";
    }
    main_kernel_name += to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    LOG_DEBUG("get kernel: name: ", main_kernel_name);
    global_data::get().ze_cache->get(worker_idx, module, main_kernel_name, &main_kernel);

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
        global_data::get().ze_cache->get(worker_idx, module, empty_kernel_name, &empty_kernel);
        CCL_THROW_IF_NOT(empty_kernel, "null empty_kernel");
        /* use allreduce_kernel_args since they have pointers to peer mem */
        set_kernel_args(empty_kernel, allreduce_kernel_args);
    }

    ze_event_desc_t event_desc = default_event_desc;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_SUBDEVICE;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_SUBDEVICE;

    uint32_t last_event_idx = 1; // 0 is used to track entry progress

    if (empty_kernel) {
        LOG_DEBUG("create event for empty kernel");
        event_desc.index = last_event_idx++;
        ZE_CALL(zeEventCreate, (event_pool, &event_desc, &empty_kernel_event));
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        event_desc.index = last_event_idx++;
        ZE_CALL(zeEventCreate, (event_pool, &event_desc, &copy_from_peer_event));
        event_desc.index = last_event_idx++;
        ZE_CALL(zeEventCreate, (event_pool, &event_desc, &reduce_local_kernel_event));
    }

    LOG_DEBUG("real event count: ", last_event_idx);

    /* do appends */
    if (empty_kernel) {
        LOG_DEBUG("append empty kernel");
        ze_group_count_t empty_group_count = { 1, 1, 1 };
        ZE_CALL(zeCommandListAppendLaunchKernel,
                (ze_base_entry::comp_primitives.list,
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
                (ze_base_entry::comp_primitives.list,
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
                (ze_base_entry::comp_primitives.list,
                 main_kernel,
                 &group_count,
                 ze_base_entry::entry_event,
                 (empty_kernel_event) ? 1 : 0,
                 &empty_kernel_event));
    }

    ZE_CALL(zeCommandListClose, (ze_base_entry::comp_primitives.list));
    if (global_data::env().enable_kernel_1s_copy_ops) {
        ZE_CALL(zeCommandListClose, (ze_base_entry::copy_primitives.list));
    }
    LOG_DEBUG("initialization complete");
}

void ze_allreduce_entry::start() {
    init();

    if (ze_base_entry::is_initialized && status == ccl_sched_entry_status_not_started) {
        reset_sync_objects();
    }

    size_t kernel_counter = 0;
    if (global_data::env().enable_kernel_sync) {
        kernel_counter = global_data::get().kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
        status = ccl_sched_entry_status_started;
    }
    else {
        global_data::get().kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_allreduce_entry::update() {
    ze_base_entry::update();
    if (status == ccl_sched_entry_status_complete && !sched->coll_attr.to_cache) {
        finalize();
    }

    if (global_data::env().enable_kernel_sync && global_data::get().kernel_counter > 0) {
        global_data::get().kernel_counter--;
    }
}

void ze_allreduce_entry::finalize() {
    if (!ze_base_entry::is_initialized) {
        return;
    }

    LOG_DEBUG("finalization");

    /* events */
    if (global_data::env().enable_kernel_1s_copy_ops) {
        LOG_DEBUG("copy ops finalization");
        ZE_CALL(zeEventDestroy, (copy_from_peer_event));
        ZE_CALL(zeEventDestroy, (reduce_local_kernel_event));
        /* device mem */
        global_data::get().ze_cache->push(worker_idx,
                                          context,
                                          device,
                                          device_mem_alloc_desc,
                                          buf_size_bytes,
                                          0, /*alignment*/
                                          tmp_buf_ptr);
    }

    /* kernels */
    if (empty_kernel_event) {
        ZE_CALL(zeEventDestroy, (empty_kernel_event));
        global_data::get().ze_cache->push(worker_idx, module, empty_kernel_name, empty_kernel);
    }
    global_data::get().ze_cache->push(worker_idx, module, main_kernel_name, main_kernel);

    ze_base_entry::finalize();

    LOG_DEBUG("finalization complete");
}

void ze_allreduce_entry::reset_sync_objects() {
    if (empty_kernel_event) {
        ZE_CALL(zeEventHostReset, (empty_kernel_event));
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        ZE_CALL(zeEventHostReset, (copy_from_peer_event));
        ZE_CALL(zeEventHostReset, (reduce_local_kernel_event));
    }
}
