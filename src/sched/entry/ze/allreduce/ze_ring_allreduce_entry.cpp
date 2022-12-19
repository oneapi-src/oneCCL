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
#include "comp/comp.hpp"
#include "common/stream/stream.hpp"
#include "sched/entry/ze/allreduce/ze_ring_allreduce_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/queue/queue.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_ring_allreduce_entry::ze_ring_allreduce_entry(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 ccl_buffer recv_buf,
                                                 ccl_buffer tmp_buf,
                                                 size_t cnt,
                                                 const ccl_datatype& dtype,
                                                 reduction op,
                                                 ccl_comm* comm,
                                                 size_t recv_buf_idx,
                                                 size_t tmp_buf_idx)
        : ze_base_entry(sched, comm, (comm->size() - 1) * event_group_count),
          send_buf(send_buf),
          recv_buf(recv_buf),
          tmp_buf(tmp_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          recv_buf_idx(recv_buf_idx),
          tmp_buf_idx(tmp_buf_idx),
          stage_iter_count(comm->size() - 1),
          total_iter_count(stage_iter_count * 2) {
    skip_entry = (comm->size() == 1) && (send_buf == recv_buf);
    if (skip_entry) {
        // skip entry init and finalize
        sched->ze_entries.pop_back();
    }
    else {
        atl_ops_init();
    }
}

void ze_ring_allreduce_entry::atl_ops_init() {
    left_peer = (comm_size + comm_rank - 1) % comm_size;
    right_peer = (comm_rank + 1) % comm_size;
    recv_tags.resize(total_iter_count);
    send_tags.resize(total_iter_count);
    sync_send_flags.resize(total_iter_count, comm_rank);

    for (int i = 0; i < total_iter_count; ++i) {
        send_tags[i] =
            comm->get_atl_comm()->tag_creator->create(right_peer,
                                                      comm->get_comm_id(),
                                                      sched->sched_id,
                                                      sched->get_op_id() + i + op_id_offset);
        recv_tags[i] = comm->get_atl_comm()->tag_creator->create(
            comm_rank, comm->get_comm_id(), sched->sched_id, sched->get_op_id() + i + op_id_offset);
    }

    LOG_DEBUG("atl_ops_init completed");
}

void ze_ring_allreduce_entry::recv_sync_flag(int idx) {
    auto buf = &sync_recv_flags[idx];
    auto bytes = sizeof(sync_recv_flags[idx]);
    auto src = left_peer;
    auto tag = recv_tags.at(idx);
    atl_req_t& req = recv_reqs[idx];

    CCL_THROW_IF_NOT((left_peer != comm_rank) && (left_peer < comm_size),
                     "unexpected src ",
                     src,
                     ", my rank ",
                     comm_rank,
                     ", left peer ",
                     left_peer);

    LOG_DEBUG("start recv: { src: ", src, ", tag: ", tag, ", bytes: ", bytes, "}");
    auto status = comm->get_atl_comm()->recv(sched->bin->get_atl_ep(), buf, bytes, src, tag, req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "atl status: ", atl_status_to_str(status));
}

void ze_ring_allreduce_entry::send_sync_flag(int idx) {
    auto buf = &sync_send_flags[idx];
    auto bytes = sizeof(sync_send_flags[idx]);
    auto dst = right_peer;
    auto tag = send_tags.at(idx);
    atl_req_t& req = send_reqs[idx];

    CCL_THROW_IF_NOT((right_peer != comm_rank) && (right_peer < comm_size),
                     "unexpected dst ",
                     dst,
                     ", my rank ",
                     comm_rank,
                     ", right peer ",
                     right_peer);

    LOG_DEBUG("start send: { dst: ",
              dst,
              ", tag: ",
              tag,
              ", bytes: ",
              bytes,
              ", value: ",
              sync_send_flags[idx],
              "}");
    auto status = comm->get_atl_comm()->send(sched->bin->get_atl_ep(), buf, bytes, dst, tag, req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "atl status: ", atl_status_to_str(status));
}

bool ze_ring_allreduce_entry::check_atl_req(atl_req_t& req) {
    if (!req.is_completed) {
        auto status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);
        CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "atl status: ", atl_status_to_str(status));
    }
    return req.is_completed;
}

void ze_ring_allreduce_entry::validate_sync_flags(int limit) {
    for (int i = 0; i < total_iter_count; ++i) {
        int value = sync_send_flags[i];
        CCL_THROW_IF_NOT(value == comm_rank);
        value = sync_recv_flags[i];
        if (i < limit)
            CCL_THROW_IF_NOT(value == left_peer);
    }
}

void ze_ring_allreduce_entry::init_ze_hook() {
    size_t dtype_size = dtype.size();
    bool inplace = (send_buf == recv_buf);

    if (comm_size == 1) {
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 recv_buf.get_ptr(),
                 send_buf.get_ptr(),
                 cnt * dtype_size,
                 ze_base_entry::entry_event,
                 0,
                 nullptr));
        return;
    }

    rs_copy_signal_events.resize(stage_iter_count);
    rs_copy_wait_events.resize(stage_iter_count);
    rs_reduce_signal_events.resize(stage_iter_count);
    rs_reduce_wait_events.resize(stage_iter_count);
    ag_copy_signal_events.resize(stage_iter_count);
    ag_copy_wait_events.resize(stage_iter_count);

    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
    std::string kernel_name =
        "reduce_local_inplace_kernel_" + to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    kernels.reserve(stage_iter_count);

    for (int i = 0; i < stage_iter_count; ++i) {
        rs_copy_signal_events[i] = ze_base_entry::create_event();
        rs_copy_wait_events[i] = ze_base_entry::create_event();
        rs_reduce_signal_events[i] = ze_base_entry::create_event();
        rs_reduce_wait_events[i] = ze_base_entry::create_event();
        ag_copy_signal_events[i] = ze_base_entry::create_event();
        ag_copy_wait_events[i] = ze_base_entry::create_event();
        kernels.emplace_back(module, kernel_name, worker_idx);
    }

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();
    tmp_buf_ptr = tmp_buf.get_ptr();

    ccl_buffer right_recv_buf;
    int peer_rank = (comm_rank + 1) % comm_size;
    sched->get_memory().handle_manager.get(peer_rank, recv_buf_idx, right_recv_buf, comm);
    right_recv_buf_ptr = right_recv_buf.get_ptr();

    if (inplace) {
        ccl_buffer right_tmp_buf;
        sched->get_memory().handle_manager.get(peer_rank, tmp_buf_idx, right_tmp_buf, comm);
        right_tmp_buf_ptr = right_tmp_buf.get_ptr();
    }

    // reduce_scatter stage

    size_t main_block_count = cnt / comm_size;
    int block_idx = (comm_size + comm_rank - 1) % comm_size;

    for (int i = 0; i < stage_iter_count; ++i) {
        size_t block_count = main_block_count;
        if (block_idx == (comm_size - 1))
            block_count += cnt % comm_size;
        int copy_offset = main_block_count * dtype_size * block_idx;

        LOG_DEBUG("reduce_scatter: { my rank: ",
                  comm->rank(),
                  ", iter: ",
                  i,
                  ", copy_offset: ",
                  copy_offset,
                  ", block_count: ",
                  block_count,
                  " }");

        void* src = nullptr;
        void* dst = nullptr;
        if (inplace) {
            src = recv_buf_ptr;
            dst = right_tmp_buf_ptr;
        }
        else {
            src = (i == 0) ? send_buf_ptr : recv_buf_ptr;
            dst = right_recv_buf_ptr;
        }
        src = (char*)src + copy_offset;
        dst = (char*)dst + copy_offset;

        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 dst,
                 src,
                 block_count * dtype_size,
                 rs_copy_signal_events[i],
                 1,
                 &rs_copy_wait_events[i]));

        block_idx = (block_idx + comm_size - 1) % comm_size;
        block_count = main_block_count;
        if (block_idx == (comm_size - 1))
            block_count += cnt % comm_size;
        int kernel_offset = main_block_count * dtype_size * block_idx;

        LOG_DEBUG("reduce_scatter: { my rank: ",
                  comm->rank(),
                  ", iter: ",
                  i,
                  ", kernel_offset: ",
                  copy_offset,
                  ", block_count: ",
                  block_count,
                  " }");

        void* input_buf = (inplace) ? tmp_buf_ptr : send_buf_ptr;
        input_buf = (char*)input_buf + kernel_offset;
        void* output_buf = (char*)recv_buf_ptr + kernel_offset;

        kernels[i].set_args({ &block_count, &input_buf, &output_buf });
        kernels[i].calculate_group_size(block_count);

        ZE_CALL(zeCommandListAppendLaunchKernel,
                (ze_base_entry::get_comp_list(),
                 kernels[i].get_kernel(),
                 kernels[i].get_group_count(),
                 rs_reduce_signal_events[i],
                 1,
                 &rs_reduce_wait_events[i]));
    }

    // allgather stage

    for (int i = 0; i < stage_iter_count; ++i) {
        size_t block_count = main_block_count;
        if (block_idx == (comm_size - 1))
            block_count += cnt % comm_size;

        int copy_offset = main_block_count * dtype_size * block_idx;

        LOG_DEBUG("allgather: { my rank: ",
                  comm->rank(),
                  ", iter: ",
                  i,
                  ", copy offset: ",
                  copy_offset,
                  ", block_count: ",
                  block_count,
                  " }");
        void* src = (char*)recv_buf_ptr + copy_offset;
        void* dst = (char*)right_recv_buf_ptr + copy_offset;

        ZE_CALL(zeCommandListAppendMemoryCopy,
                (ze_base_entry::get_copy_list(),
                 dst,
                 src,
                 block_count * dtype_size,
                 ag_copy_signal_events[i],
                 1,
                 &ag_copy_wait_events[i]));

        block_idx = (block_idx + comm_size - 1) % comm_size;
    }
}

void ze_ring_allreduce_entry::finalize_ze_hook() {
    kernels.clear();
}

void ze_ring_allreduce_entry::start() {
    if (skip_entry) {
        ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
        status = ccl_sched_entry_status_complete;
        return;
    }

    reset_fields();

    for (int i = 0; i < total_iter_count; ++i) {
        recv_sync_flag(i);
    }

    ze_base_entry::start();

    for (int i = 0; i < total_iter_count; ++i) {
        CCL_THROW_IF_NOT(!send_reqs[i].is_completed);
        CCL_THROW_IF_NOT(!recv_reqs[i].is_completed);
    }

    for (int i = 0; i < stage_iter_count; ++i) {
        CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_copy_signal_events[i]));
        CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_copy_wait_events[i]));
        CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_reduce_signal_events[i]));
        CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_reduce_wait_events[i]));
        CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_signal_events[i]));
        CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_wait_events[i]));
    }
}

void ze_ring_allreduce_entry::update() {
    if (comm_size == 1) {
        ze_base_entry::update();
        return;
    }

    if (iter_idx > 0) {
        validate_sync_flags(iter_idx - 1);
    }

    while (!is_rs_completed && (iter_idx < stage_iter_count)) {
        for (int i = iter_idx + 1; i < stage_iter_count; ++i) {
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_copy_signal_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_copy_wait_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_reduce_signal_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_reduce_wait_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_signal_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_wait_events[i]));
            CCL_THROW_IF_NOT(!send_reqs[i].is_completed);
            CCL_THROW_IF_NOT(!recv_reqs[i].is_completed);
        }

        if (!rs_copy_started[iter_idx]) {
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_copy_wait_events[iter_idx]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(rs_copy_signal_events[iter_idx]));

            if (iter_idx > 0) {
                CCL_THROW_IF_NOT(
                    ze_base_entry::is_event_completed(rs_reduce_signal_events[iter_idx - 1]));
                CCL_THROW_IF_NOT(
                    ze_base_entry::is_event_completed(rs_reduce_wait_events[iter_idx - 1]));
                CCL_THROW_IF_NOT(recv_reqs[iter_idx - 1].is_completed);
            }

            ZE_CALL(zeEventHostSignal, (rs_copy_wait_events[iter_idx]));
            rs_copy_started[iter_idx] = true;

            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_wait_events[iter_idx]));
        }

        if (!rs_sync_sent[iter_idx] &&
            (ze_base_entry::is_event_completed(rs_copy_signal_events[iter_idx]))) {
            send_sync_flag(iter_idx);
            rs_sync_sent[iter_idx] = true;
        }

        if (!rs_reduce_started[iter_idx]) {
            auto is_recv_completed = check_atl_req(recv_reqs[iter_idx]);
            if (is_recv_completed) {
                CCL_THROW_IF_NOT(sync_recv_flags[iter_idx] == left_peer,
                                 "iter ",
                                 iter_idx,
                                 ", expected ",
                                 left_peer,
                                 ", got ",
                                 sync_recv_flags[iter_idx]);
                CCL_THROW_IF_NOT(
                    !ze_base_entry::is_event_completed(rs_reduce_wait_events[iter_idx]));
                CCL_THROW_IF_NOT(
                    !ze_base_entry::is_event_completed(rs_reduce_signal_events[iter_idx]));

                ZE_CALL(zeEventHostSignal, (rs_reduce_wait_events[iter_idx]));
                rs_reduce_started[iter_idx] = true;

                CCL_THROW_IF_NOT(
                    ze_base_entry::is_event_completed(rs_reduce_wait_events[iter_idx]));
            }
            else {
                return;
            }
        }

        if ((ze_base_entry::is_event_completed(rs_reduce_signal_events[iter_idx])) &&
            rs_sync_sent[iter_idx] && check_atl_req(send_reqs[iter_idx])) {
            LOG_DEBUG("completed reduce_scatter iter ", iter_idx);

            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_signal_events[iter_idx]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_wait_events[iter_idx]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_reduce_signal_events[iter_idx]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_reduce_wait_events[iter_idx]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_signal_events[iter_idx]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_wait_events[iter_idx]));
            CCL_THROW_IF_NOT(send_reqs[iter_idx].is_completed);
            CCL_THROW_IF_NOT(recv_reqs[iter_idx].is_completed);

            validate_sync_flags(iter_idx);

            iter_idx++;
        }
        else {
            return;
        }
    }

    if (!is_rs_completed) {
        is_rs_completed = true;
        iter_idx = 0;

        for (int i = 0; i < stage_iter_count; ++i) {
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_signal_events[i]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_wait_events[i]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_reduce_signal_events[i]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_reduce_wait_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_signal_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_wait_events[i]));
            CCL_THROW_IF_NOT(send_reqs[i].is_completed);
            CCL_THROW_IF_NOT(recv_reqs[i].is_completed);
        }
    }

    validate_sync_flags(stage_iter_count);

    while (!is_ag_completed && (iter_idx < stage_iter_count)) {
        for (int i = iter_idx + 1; i < stage_iter_count; ++i) {
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_signal_events[i]));
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_wait_events[i]));
            CCL_THROW_IF_NOT(!send_reqs[i + stage_iter_count].is_completed);
            CCL_THROW_IF_NOT(!recv_reqs[i + stage_iter_count].is_completed);
        }

        if (!ag_copy_started[iter_idx]) {
            CCL_THROW_IF_NOT(!ze_base_entry::is_event_completed(ag_copy_wait_events[iter_idx]));
            if (iter_idx > 0) {
                CCL_THROW_IF_NOT(
                    ze_base_entry::is_event_completed(ag_copy_signal_events[iter_idx - 1]));
                CCL_THROW_IF_NOT(
                    ze_base_entry::is_event_completed(ag_copy_wait_events[iter_idx - 1]));
                CCL_THROW_IF_NOT(recv_reqs[stage_iter_count + iter_idx - 1].is_completed);
            }

            ZE_CALL(zeEventHostSignal, (ag_copy_wait_events[iter_idx]));
            ag_copy_started[iter_idx] = true;

            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(ag_copy_wait_events[iter_idx]));
        }

        if (!ag_sync_sent[iter_idx] &&
            (ze_base_entry::is_event_completed(ag_copy_signal_events[iter_idx]))) {
            send_sync_flag(iter_idx + stage_iter_count);
            ag_sync_sent[iter_idx] = true;
        }

        auto is_send_completed =
            ag_sync_sent[iter_idx] && check_atl_req(send_reqs[iter_idx + stage_iter_count]);
        auto is_recv_completed = check_atl_req(recv_reqs[iter_idx + stage_iter_count]);
        if (is_send_completed && is_recv_completed) {
            LOG_DEBUG("completed allgatherv iter ", iter_idx);

            CCL_THROW_IF_NOT(sync_recv_flags[iter_idx + stage_iter_count] == left_peer);
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(ag_copy_signal_events[iter_idx]));
            CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(ag_copy_wait_events[iter_idx]));
            CCL_THROW_IF_NOT(send_reqs[iter_idx + stage_iter_count].is_completed);
            CCL_THROW_IF_NOT(recv_reqs[iter_idx + stage_iter_count].is_completed);

            validate_sync_flags(iter_idx);

            ++iter_idx;
        }
        else {
            return;
        }
    }

    is_ag_completed = true;

    for (int i = 0; i < stage_iter_count; ++i) {
        CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_signal_events[i]));
        CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_copy_wait_events[i]));
        CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_reduce_signal_events[i]));
        CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(rs_reduce_wait_events[i]));
        CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(ag_copy_signal_events[i]));
        CCL_THROW_IF_NOT(ze_base_entry::is_event_completed(ag_copy_wait_events[i]));

        CCL_THROW_IF_NOT(send_reqs[i].is_completed);
        CCL_THROW_IF_NOT(recv_reqs[i].is_completed);
        CCL_THROW_IF_NOT(send_reqs[i + stage_iter_count].is_completed);
        CCL_THROW_IF_NOT(recv_reqs[i + stage_iter_count].is_completed);
    }
    validate_sync_flags(total_iter_count);

    ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
    ze_base_entry::update();
}

void ze_ring_allreduce_entry::reset_fields() {
    if (comm_size == 1) {
        return;
    }

    iter_idx = 0;
    is_rs_completed = is_ag_completed = false;

    send_reqs.clear();
    send_reqs.resize(total_iter_count);
    recv_reqs.clear();
    recv_reqs.resize(total_iter_count);

    if (sync_recv_flags.empty()) {
        sync_recv_flags.resize(total_iter_count, ccl_comm::invalid_rank);

        rs_sync_sent.resize(stage_iter_count, false);
        ag_sync_sent.resize(stage_iter_count, false);

        rs_copy_started.resize(stage_iter_count, false);
        rs_reduce_started.resize(stage_iter_count, false);
        ag_copy_started.resize(stage_iter_count, false);
    }
    else {
        std::fill(sync_recv_flags.begin(), sync_recv_flags.end(), ccl_comm::invalid_rank);

        std::fill(rs_sync_sent.begin(), rs_sync_sent.end(), false);
        std::fill(ag_sync_sent.begin(), ag_sync_sent.end(), false);

        std::fill(rs_copy_started.begin(), rs_copy_started.end(), false);
        std::fill(rs_reduce_started.begin(), rs_reduce_started.end(), false);
        std::fill(ag_copy_started.begin(), ag_copy_started.end(), false);
    }
}

std::string ze_ring_allreduce_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << cnt * dtype.size();
    return out.str();
}

void ze_ring_allreduce_entry::dump_detail(std::stringstream& str) const {
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
