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
                                                 std::vector<ze_event_handle_t> wait_events,
                                                 size_t peer_buf_idx,
                                                 size_t peer_buf_offset,
                                                 bool is_monolithic_pipeline,
                                                 ccl_comm* pipeline_comm)
        : ze_base_entry(sched, comm, comm->size() * event_group_count, wait_events),
          send_buf(send_buf),
          send_count(send_count),
          recv_bufs(recv_bufs),
          recv_counts(recv_counts),
          dtype(dtype),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1),
          is_monolithic_pipeline(is_monolithic_pipeline),
          pipeline_comm(pipeline_comm) {}

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
                // currently pipepine_comm is pair_comm and it can have a maximum of 2 ranks
                const int pair_peer_rank = (pipeline_comm->rank() + 1) % pipeline_comm->size();
                sched->get_memory().handle_manager.get(
                    pair_peer_rank, 1 + peer_global_rank, buf_pair, pipeline_comm);
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
    // when monolithic pipelined kernel is used only one event is needed for the kernel
    // otherwise we use an event for copying from each peer.
    size_t copy_events_size = (is_monolithic_pipeline) ? 1 : peer_count;
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

    std::vector<ze_event_handle_t> empty_wait_events;

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
                                     empty_wait_events,
                                     is_monolithic,
                                     is_monolithic_pipeline,
                                     is_inplace);
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
                                           bool is_inplace)
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
          is_inplace(is_inplace) {}

// main function to choose read/write operation for a2a_allgatherv
void ze_a2a_allgatherv_op::select(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    if (args.is_monolithic_pipeline) {
        // read data using xelink and then write that data through mdfi
        ze_a2a_allgatherv_op::read_write(args, kernels);
    }
    else if (ccl::global_data::env().allgatherv_topo_read) {
        ze_a2a_allgatherv_op::read(args);
    }
    else {
        ze_a2a_allgatherv_op::write(args, kernels);
    }

    if (!args.is_inplace) {
        // copy send_buf to my buffer
        void* dst = args.recv_bufs.at(args.comm->rank()).get_ptr();
        if (args.is_monolithic_pipeline) {
            const int my_global_rank = args.comm->get_global_rank(args.comm->rank());
            dst = args.recv_bufs.at(my_global_rank).get_ptr();
        }
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (args.entry->get_copy_list(copy_direction::t2t),
                 dst,
                 args.send_buf.get_ptr(), // src
                 args.copy_bytes.at(args.comm->rank()),
                 args.copy_events.back(),
                 args.wait_events.size(),
                 args.wait_events.data()));
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

    std::vector<void*> peer_even_bufs, local_bufs, peer_pair_bufs;
    std::vector<size_t> counts;
    for (int i = 0; i < a.peer_count; ++i) {
        int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();
        peer_even_bufs.push_back(a.peer_bufs[peer_rank].get_ptr());
        local_bufs.push_back(a.recv_bufs[a.comm->get_global_rank(peer_rank)].get_ptr());
        counts.push_back(a.recv_counts[a.comm->get_global_rank(peer_rank)]);
        peer_pair_bufs.push_back(a.pair_peer_bufs[peer_rank].get_ptr());
    }

    kernels.emplace_back(module, monolithic_kernel_name, worker_idx);
    ze_kernel_arg_t peer_send_bufs_ze_arg(peer_even_bufs.data(), peer_even_bufs.size());
    ze_kernel_arg_t recv_bufs_ze_arg(local_bufs.data(), local_bufs.size());
    ze_kernel_arg_t peer_pair_bufs_ze_arg(peer_pair_bufs.data(), peer_pair_bufs.size());
    ze_kernel_arg_t counts_ze_arg(counts.data(), counts.size());
    int pipeline_comm_size = a.pipeline_comm->size();

    kernels.back().set_args({ &pipeline_comm_size,
                              peer_send_bufs_ze_arg,
                              recv_bufs_ze_arg,
                              peer_pair_bufs_ze_arg,
                              counts_ze_arg });
    kernels.back().calculate_group_size(*std::max_element(counts.begin(), counts.end()));

    ZE_CALL(zeCommandListAppendLaunchKernel,
            (a.entry->get_comp_list(),
             kernels.back().get_kernel(),
             kernels.back().get_group_count(),
             a.copy_events.at(0),
             a.wait_events.size(),
             a.wait_events.data()));
}

// prepare params for read from peers via ze copy
void ze_a2a_allgatherv_op::read(ze_a2a_allgatherv_op& args) {
    auto& a = args;

    if (a.is_monolithic) {
        LOG_INFO("allgatherv read is not supported by monolithic kernels");
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

        ZE_CALL(zeCommandListAppendMemoryCopy,
                (a.entry->get_copy_list(copy_direction::c2c, i),
                 a.recv_bufs[peer_rank].get_ptr(),
                 src,
                 a.copy_bytes.at(peer_rank),
                 a.copy_events.at(i),
                 a.wait_events.size(),
                 a.wait_events.data()));
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
            // TODO: if we on the same device, then use t2t direction
            auto list = a.entry->get_copy_list(copy_direction::c2c, i);
            ZE_CALL(zeCommandListAppendMemoryCopy,
                    (list,
                     dst_buf.get_ptr(),
                     src_buf.get_ptr(),
                     a.copy_bytes.at(a.comm->rank()),
                     a.copy_events.at(i),
                     a.wait_events.size(),
                     a.wait_events.data()));
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
            kernels.emplace_back(module, monolithic_kernel_name, worker_idx);
            ze_kernel_arg_t peer_bufs_ze_arg(dsts.data(), dsts.size());
            kernels.back().set_args({ &count_local, &src, peer_bufs_ze_arg });
            kernels.back().calculate_group_size(count_local);

            ZE_CALL(zeCommandListAppendLaunchKernel,
                    (a.entry->get_comp_list(),
                     kernels.back().get_kernel(),
                     kernels.back().get_group_count(),
                     a.copy_events.at(i),
                     a.wait_events.size(),
                     a.wait_events.data()));
        }
    }
}
