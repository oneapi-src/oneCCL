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

// ze_a2a_allgatherv_entry
ze_a2a_allgatherv_entry::ze_a2a_allgatherv_entry(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 size_t send_count,
                                                 std::vector<ccl_buffer> recv_bufs,
                                                 std::vector<size_t> recv_counts,
                                                 const ccl_datatype& dtype,
                                                 ccl_comm* comm,
                                                 const std::vector<ze_event_handle_t>& wait_events,
                                                 size_t peer_buf_idx,
                                                 size_t peer_buf_offset,
                                                 bool is_monolithic_pipeline,
                                                 ccl_comm* pipeline_comm,
                                                 bool is_separate_block_handles)
        : ze_base_entry(sched, wait_events, comm, comm->size() * event_group_count),
          send_buf(send_buf),
          send_count(send_count),
          recv_bufs(recv_bufs),
          recv_counts(recv_counts),
          dtype(dtype),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1),
          is_monolithic_pipeline(is_monolithic_pipeline),
          pipeline_comm(pipeline_comm),
          is_separate_block_handles(is_separate_block_handles) {}

void ze_a2a_allgatherv_entry::init_ze_hook() {
    /* get peer recv buffers */
    std::vector<ccl_buffer> peer_recv_bufs(comm->size());
    std::vector<ccl_buffer> pair_peer_recv_bufs(comm->size());

    for (int i = 0; i < peer_count; ++i) {
        const int peer_rank = (comm_rank + i + 1) % comm->size();
        ccl_buffer buf{};
        sched->get_memory().handle_manager.get(peer_rank, peer_buf_idx, buf, comm);
        CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
        peer_recv_bufs[peer_rank] = buf;

        if (is_monolithic_pipeline && pipeline_comm != nullptr) {
            // get peer buffer handles with pair_comm peer when pair_comm size > 1
            if (pipeline_comm->size() > 1) {
                ccl_buffer buf_pair{};
                const int peer_global_rank = comm->get_global_rank(peer_rank);
                // currently pipeline_comm is pair_comm and it can have a maximum of 2 ranks
                CCL_THROW_IF_NOT(pipeline_comm->size() == 2,
                                 "algorithm only supports pipeline_comm of size 2");
                const int pair_peer_rank = (pipeline_comm->rank() + 1) % pipeline_comm->size();
                // when separate handles are used, there is a different handle for each peer rank.
                // position 0 is for send buffer, therefore add 1 for recv_buffer index.
                // when separate handles are not there, use the idx parameter for all peers.
                const size_t pair_peer_buf_idx =
                    (is_separate_block_handles) ? 1 + peer_global_rank : peer_buf_idx;
                sched->get_memory().handle_manager.get(
                    pair_peer_rank, pair_peer_buf_idx, buf_pair, pipeline_comm);
                pair_peer_recv_bufs[peer_rank] = buf_pair;
            }
            else {
                pair_peer_recv_bufs[peer_rank] = buf;
            }
        }
    }

    bool is_inplace{};
    if (send_buf == recv_bufs.at(comm_rank)) {
        is_inplace = true;
    }
    std::vector<size_t> rank_buf_offsets(comm_size);
    for (int i = 1; i < comm_size; i++) {
        rank_buf_offsets[i] = rank_buf_offsets[i - 1] + recv_counts[i - 1];
    }

    CCL_THROW_IF_NOT(send_count == recv_counts[comm_rank],
                     "allgatherv send_count :",
                     send_count,
                     " and recv_count :",
                     recv_counts[comm_rank],
                     " does not match");

    std::vector<size_t> block_bytes(comm_size);
    for (int i = 0; i < comm_size; i++) {
        block_bytes[i] = recv_counts[i] * dtype.size();
    }

    LOG_DEBUG("rank: ", comm_rank, ", block_bytes: ", block_bytes.at(comm_rank));

    bool is_monolithic = ccl::global_data::env().allgatherv_monolithic_kernel;
    bool is_read = ccl::global_data::env().allgatherv_topo_read;
    // TODO: MLSL-1651 make int8 work with allgatherv write monolithic kernel
    if (dtype == ccl::datatype::int8) {
        is_monolithic = false;
    }
    // when monolithic pipelined kernel is used, two events
    // are needed for the unaligned and aligned kernels.
    // otherwise we use an event for copying from each peer.
    size_t copy_events_size = (is_monolithic_pipeline) ? 2 : peer_count;
    // write requires two kernels, unaligned and aligned kernel
    if (is_monolithic && !is_read) {
        copy_events_size = (int)ccl::utils::align_kernels::count;
    }
    // need additional memcpy for non inplace data
    if (!is_inplace) {
        copy_events_size++;
    }
    copy_events.resize(copy_events_size);
    for (auto& event : copy_events) {
        event = ze_base_entry::create_event();
    }

    ze_a2a_allgatherv_op init_params(sched,
                                     this,
                                     comm,
                                     pipeline_comm,
                                     dtype,
                                     send_buf,
                                     recv_bufs,
                                     peer_recv_bufs,
                                     pair_peer_recv_bufs,
                                     block_bytes,
                                     recv_counts,
                                     peer_count,
                                     rank_buf_offsets,
                                     peer_buf_offset,
                                     copy_events,
                                     wait_events,
                                     is_monolithic,
                                     is_monolithic_pipeline,
                                     is_inplace,
                                     is_separate_block_handles);
    ze_a2a_allgatherv_op::select(init_params, kernels);
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

std::string ze_a2a_allgatherv_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << send_count * dtype.size();
    return out.str();
}

void ze_a2a_allgatherv_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str, "comm ", comm->to_string(), "\n");
}

// ze_a2a_allgatherv_op
ze_a2a_allgatherv_op::ze_a2a_allgatherv_op(ccl_sched* sched,
                                           const ze_base_entry* entry,
                                           ccl_comm* comm,
                                           ccl_comm* pipeline_comm,
                                           const ccl_datatype& dtype,
                                           ccl_buffer send_buf,
                                           const std::vector<ccl_buffer>& recv_bufs,
                                           const std::vector<ccl_buffer>& peer_bufs,
                                           const std::vector<ccl_buffer>& pair_peer_bufs,
                                           const std::vector<size_t>& copy_bytes,
                                           const std::vector<size_t>& recv_counts,
                                           int peer_count,
                                           const std::vector<size_t>& rank_buf_offsets,
                                           size_t peer_buf_offset,
                                           std::vector<ze_event_handle_t>& copy_events,
                                           std::vector<ze_event_handle_t>& wait_events,
                                           bool is_monolithic,
                                           bool is_monolithic_pipeline,
                                           bool is_inplace,
                                           bool is_separate_block_handles)
        : sched(sched),
          entry(entry),
          comm(comm),
          pipeline_comm(pipeline_comm),
          dtype(dtype),
          send_buf(send_buf),
          recv_bufs(recv_bufs),
          peer_bufs(peer_bufs),
          pair_peer_bufs(pair_peer_bufs),
          copy_bytes(copy_bytes),
          recv_counts(recv_counts),
          peer_count(peer_count),
          rank_buf_offsets(rank_buf_offsets),
          peer_buf_offset(peer_buf_offset),
          copy_events(copy_events),
          wait_events(wait_events),
          is_monolithic(is_monolithic),
          is_monolithic_pipeline(is_monolithic_pipeline),
          is_inplace(is_inplace),
          is_separate_block_handles(is_separate_block_handles) {}

// main function to choose read/write operation for a2a_allgatherv
void ze_a2a_allgatherv_op::select(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    if (args.is_monolithic_pipeline) {
        // read data using xelink and then write that data through mdfi
        // input events: wait_events
        // output event: copy_events[0]
        ze_a2a_allgatherv_op::read_write(args, kernels);
    }
    else if (ccl::global_data::env().allgatherv_topo_read) {
        // input events: wait_events
        // output event(s): copy_events[0..peer_count-1]
        ze_a2a_allgatherv_op::read(args);
    }
    else {
        // input events: wait_events
        // output event(s): copy_events[0..peer_count-1]
        ze_a2a_allgatherv_op::write(args, kernels);
    }

    if (!args.is_inplace) {
        // args.wait_events must be updated to copy_events[0 .. copy_events.size()-1]
        args.wait_events.clear();
        args.wait_events.reserve(args.copy_events.size() - 1);
        std::copy(
            args.copy_events.begin(), args.copy_events.end() - 1, back_inserter(args.wait_events));

        // copy send_buf to my buffer
        void* dst = args.recv_bufs.at(args.comm->rank()).get_ptr();
        if (args.is_monolithic_pipeline) {
            const int my_global_rank = args.comm->get_global_rank(args.comm->rank());
            dst = args.recv_bufs.at(my_global_rank).get_ptr();
        }
        ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                ze_cmd_memory_copy,
                                args.entry->get_copy_list(copy_direction::t2t),
                                dst,
                                args.send_buf.get_ptr(), // src
                                args.copy_bytes.at(args.comm->rank()),
                                args.copy_events.back(),
                                args.wait_events);
    }
}

// prepare params for pipeline to peers via monolithic kernels
// monolithic kernel reads data using xelink and then write that data through mdfi
void ze_a2a_allgatherv_op::read_write(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    auto& a = args;
    auto device = a.sched->coll_param.stream->get_ze_device();
    auto context = a.sched->coll_param.stream->get_ze_context();
    auto worker_idx = a.sched->queue->get_idx();

    ze_module_handle_t module = nullptr;
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
    std::string monolithic_kernel_name =
        "read_write_monolithic_kernel_" + std::to_string(a.peer_count) + "_" +
        to_string(a.dtype.idx()) + "_" + ccl_reduction_to_str(ccl::reduction::custom);
    LOG_DEBUG("Allgatherv monolithic pipeline kernel name: ", monolithic_kernel_name);

    constexpr int unaligned_kernel_idx = (int)ccl::utils::align_kernels::unaligned; // 0
    constexpr int aligned_kernel_idx = (int)ccl::utils::align_kernels::aligned; // 1
    constexpr int kernel_count = (int)ccl::utils::align_kernels::count; // 2
    std::vector<size_t> counts[kernel_count];

    // kernel reads from peer_even_bufs using xelnik and write to local_bufs
    // and then copies from local_bufs to peer_pair_bufs using MDFI
    std::vector<const void*> peer_even_bufs[kernel_count], local_bufs[kernel_count],
        peer_pair_bufs[kernel_count];

    for (int i = 0; i < a.peer_count; i++) {
        const int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();

        size_t base_index_local = 0, offset = 0;
        if (a.is_separate_block_handles) {
            // when using separate handles use the global rank
            // since user passes the count array with global index
            base_index_local = a.comm->get_global_rank(peer_rank);
        }
        else {
            // when using non-separate handles we fill the data
            // using the peer_rank as index
            base_index_local = peer_rank;
            // when only single handle to the base of buffer is there,
            // we add the offset for each peer to the base address
            offset = (a.rank_buf_offsets.at(peer_rank) + a.peer_buf_offset) * a.dtype.size();
        }

        const size_t count = a.recv_counts[base_index_local];

        const void* peer_even_buf = (a.peer_bufs[peer_rank] + offset).get_ptr();
        const void* local_buf = a.recv_bufs[base_index_local].get_ptr();
        const void* peer_pair_buf = (a.pair_peer_bufs[peer_rank] + offset).get_ptr();

        // based on the starting address of local recv buffer,
        // create the unaligned and aligned kernels, based on the assumption
        // local and remote buffers have same alignment.
        size_t offset_byte = ccl::utils::get_aligned_offset_byte(
            local_buf, count * a.dtype.size(), ccl::global_data::env().kernel_mem_align);
        const size_t offset_count = offset_byte / a.dtype.size();
        counts[unaligned_kernel_idx].emplace_back(offset_count);
        counts[aligned_kernel_idx].emplace_back(count - offset_count);

        peer_even_bufs[unaligned_kernel_idx].push_back(peer_even_buf);
        local_bufs[unaligned_kernel_idx].push_back(local_buf);
        peer_pair_bufs[unaligned_kernel_idx].push_back(peer_pair_buf);

        // when aligned kernel is of zero size, adding offset_byte
        // will go outside allocated memory and therefore pass the
        // original pointer as dummy value
        if (counts[aligned_kernel_idx].back() == 0) {
            offset_byte = 0;
        }

        peer_even_bufs[aligned_kernel_idx].push_back((char*)peer_even_buf + offset_byte);
        local_bufs[aligned_kernel_idx].push_back((char*)local_buf + offset_byte);
        peer_pair_bufs[aligned_kernel_idx].push_back((char*)peer_pair_buf + offset_byte);
    }

    for (int i = 0; i < kernel_count; i++) {
        int pipeline_comm_size = a.pipeline_comm->size();

        ze_kernel_args_t kernel_args{
            &pipeline_comm_size, peer_even_bufs[i], local_bufs[i], peer_pair_bufs[i], counts[i]
        };

        auto this_count = *std::max_element(counts[i].begin(), counts[i].end());
        ze_kernel kernel(module, monolithic_kernel_name, kernel_args, this_count, worker_idx);

        ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                ze_cmd_launch_kernel,
                                a.entry->get_comp_list(),
                                std::move(kernel),
                                a.copy_events.at(i),
                                a.wait_events);
    }
}

// prepare params for read from peers via ze copy
void ze_a2a_allgatherv_op::read(ze_a2a_allgatherv_op& args) {
    auto& a = args;

    if (a.is_monolithic) {
        LOG_DEBUG("allgatherv read is not supported by monolithic kernels");
    }

    for (int i = 0; i < a.peer_count; ++i) {
        const int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();
        void* src = a.peer_bufs[peer_rank].get_ptr();
        if (a.is_inplace) {
            // TODO: use peer_bufs directly without adding offset
            src = (a.peer_bufs[peer_rank] +
                   (a.rank_buf_offsets.at(peer_rank) + a.peer_buf_offset) * a.dtype.size())
                      .get_ptr();
        }

        ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                ze_cmd_memory_copy,
                                a.entry->get_copy_list(copy_direction::c2c, i),
                                a.recv_bufs[peer_rank].get_ptr(),
                                src,
                                a.copy_bytes.at(peer_rank),
                                a.copy_events.at(i),
                                a.wait_events);
    }
}

// prepare params for write to peers via monolithic kernels
void ze_a2a_allgatherv_op::write(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    auto& a = args;
    auto device = a.sched->coll_param.stream->get_ze_device();
    auto context = a.sched->coll_param.stream->get_ze_context();
    auto worker_idx = a.sched->queue->get_idx();

    // copy send_buf to peer buffers
    std::vector<ccl_buffer> peer_dst_bufs;
    ccl_buffer src_buf = a.send_buf;
    if (a.is_inplace) {
        src_buf = a.recv_bufs.at(a.comm->rank());
    }
    for (int i = 0; i < a.peer_count; ++i) {
        const int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();
        // TODO: use peer_bufs directly without adding offset
        ccl_buffer dst_buf =
            a.peer_bufs[peer_rank] +
            (a.rank_buf_offsets.at(a.comm->rank()) + a.peer_buf_offset) * a.dtype.size();

        if (a.is_monolithic) {
            peer_dst_bufs.push_back(dst_buf);
        }
        else {
            // request copy engine at even index, can be helpful in certain situation
            // TODO: if we on the same device, then use t2t direction
            auto list = a.entry->get_copy_list(copy_direction::c2c, (i + 1) * 2);
            ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                    ze_cmd_memory_copy,
                                    list,
                                    dst_buf.get_ptr(),
                                    src_buf.get_ptr(),
                                    a.copy_bytes.at(a.comm->rank()),
                                    a.copy_events.at(i),
                                    a.wait_events);
        }
    }
    if (a.is_monolithic) {
        ze_module_handle_t module = nullptr;
        //TODO: add fallback path for peer_count > max_peer_count
        CCL_THROW_IF_NOT(size_t(a.peer_count) <= ccl::ze::max_peer_count,
                         "monolithic kernel not supported for peer_count ",
                         a.peer_count,
                         " > ",
                         ccl::ze::max_peer_count);
        global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
        std::string monolithic_kernel_name =
            "write_monolithic_kernel_" + std::to_string(a.peer_count) + "_" +
            to_string(a.dtype.idx()) + "_" + ccl_reduction_to_str(ccl::reduction::custom);
        LOG_DEBUG("allgatherv monolithic kernel name: ", monolithic_kernel_name);

        // use src instead of peer dst since we cannot make sure
        // peer dst got using ipc has the same alignment as remote buffer
        // we assume local buffer and remote buffer has the same alignment
        size_t buf_size_bytes = a.copy_bytes.at(a.comm->rank());
        unsigned long pre_align_offset_byte =
            ccl::utils::get_aligned_offset_byte(a.recv_bufs.at(a.comm->rank()).get_ptr(),
                                                buf_size_bytes,
                                                ccl::global_data::env().kernel_mem_align);

        // First kernel starts from location 0 to pre_align_offset_byte
        // and the second kernel starts from location pre_align_offset_byte to the rest
        const size_t copy_count = a.copy_bytes.at(a.comm->rank()) / a.dtype.size();
        constexpr int kernel_count = (int)ccl::utils::align_kernels::count;
        const unsigned long offsets[kernel_count] = { 0, pre_align_offset_byte };
        const unsigned long counts[kernel_count] = { pre_align_offset_byte / a.dtype.size(),
                                                     copy_count -
                                                         pre_align_offset_byte / a.dtype.size() };

        // start two kernels, first kernel for the part of the array before the aligned start offset
        // and second kernel for the rest of the array from the aligned start offset to the end
        for (int i = 0; i < kernel_count; i++) {
            unsigned long count_local = counts[i];
            // data count is small and there is no need to execute the second aligned kernel
            if (i == (int)ccl::utils::align_kernels::aligned && count_local == 0) {
                a.copy_events.at(i) = a.copy_events.at(i - 1);
                break;
            }
            void* src = (src_buf + offsets[i]).get_ptr();
            std::vector<void*> dsts;
            for (auto& peer_dst_buf : peer_dst_bufs) {
                dsts.push_back((peer_dst_buf + offsets[i]).get_ptr());
            }

            ze_kernel_args_t kernel_args{
                &count_local,
                &src,
                dsts // peer_bufs_ze_arg
            };
            kernels.emplace_back(
                module, monolithic_kernel_name, kernel_args, count_local, worker_idx);

            ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                    ze_cmd_launch_kernel,
                                    a.entry->get_comp_list(),
                                    std::move(kernels.back()),
                                    a.copy_events.at(i),
                                    a.wait_events);
        }
    }
}
