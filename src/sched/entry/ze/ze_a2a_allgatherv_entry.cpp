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
                                                 std::vector<ccl_buffer> recv_bufs,
                                                 std::vector<size_t> recv_counts,
                                                 const ccl_datatype& dtype,
                                                 ccl_comm* comm,
                                                 std::vector<ze_event_handle_t> wait_events,
                                                 size_t peer_buf_idx,
                                                 size_t peer_buf_offset)
        : ze_base_entry(sched, comm, comm->size() * event_group_count, wait_events),
          send_buf(send_buf),
          send_count(send_count),
          recv_bufs(recv_bufs),
          recv_counts(recv_counts),
          dtype(dtype),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1) {}

void ze_a2a_allgatherv_entry::fill_list_read(const ze_base_entry* entry,
                                             int comm_rank,
                                             ccl_buffer send_buf,
                                             const std::vector<ccl_buffer>& recv_bufs,
                                             const std::vector<ccl_buffer>& peer_send_bufs,
                                             int peer_count,
                                             const std::vector<size_t>& copy_bytes,
                                             const ccl_datatype& dtype,
                                             const std::vector<size_t>& rank_buf_offsets,
                                             bool is_inplace,
                                             std::vector<ze_event_handle_t>& copy_events,
                                             std::vector<ze_event_handle_t>& wait_events,
                                             size_t peer_buf_offset,
                                             bool is_monolithic) {
    if (is_monolithic) {
        LOG_INFO("allgatherv read not allowed with monolithic kernel");
    }
    const size_t comm_size = peer_count + 1;
    for (int i = 0; i < peer_count; ++i) {
        const int peer_rank = (comm_rank + i + 1) % comm_size;
        void* src = peer_send_bufs[peer_rank].get_ptr();
        if (is_inplace) {
            // TODO: use peer_send_bufs directly without adding offset
            src = (peer_send_bufs[peer_rank] +
                   (rank_buf_offsets.at(peer_rank) + peer_buf_offset) * dtype.size())
                      .get_ptr();
        }

        void* dst = recv_bufs[peer_rank].get_ptr();

        auto list = entry->get_copy_list(copy_direction::c2c, i);
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (list,
                 dst,
                 src,
                 copy_bytes.at(peer_rank),
                 copy_events.at(i),
                 wait_events.size(),
                 wait_events.data()));
    }
}

void ze_a2a_allgatherv_entry::fill_list_write(const ze_base_entry* entry,
                                              int comm_rank,
                                              ccl_buffer send_buf,
                                              const std::vector<ccl_buffer>& recv_bufs,
                                              const std::vector<ccl_buffer>& peer_recv_bufs,
                                              int peer_count,
                                              const std::vector<size_t>& copy_bytes,
                                              const ccl_datatype& dtype,
                                              const std::vector<size_t>& rank_buf_offsets,
                                              bool is_inplace,
                                              std::vector<ze_event_handle_t>& copy_events,
                                              std::vector<ze_event_handle_t>& wait_events,
                                              std::vector<ze_kernel>& kernels,
                                              ze_module_handle_t module,
                                              ze_device_handle_t device,
                                              ze_context_handle_t context,
                                              size_t worker_idx,
                                              size_t peer_buf_offset,
                                              bool is_monolithic) {
    /* copy send_buf to peer buffers */
    const size_t comm_size = peer_count + 1;

    std::vector<ccl_buffer> peer_bufs;
    ccl_buffer src_buf = send_buf;
    if (is_inplace) {
        src_buf = recv_bufs.at(comm_rank);
    }
    for (int i = 0; i < peer_count; ++i) {
        const int peer_rank = (comm_rank + i + 1) % comm_size;
        // TODO: use peer_recv_bufs directly without adding offset
        ccl_buffer dst_buf = peer_recv_bufs[peer_rank] +
                             (rank_buf_offsets.at(comm_rank) + peer_buf_offset) * dtype.size();

        if (is_monolithic) {
            peer_bufs.push_back(dst_buf);
        }
        else {
            // TODO: if we on the same device, then use t2t direction
            auto list = entry->get_copy_list(copy_direction::c2c, i);
            ZE_CALL(zeCommandListAppendMemoryCopy,
                    (list,
                     dst_buf.get_ptr(),
                     src_buf.get_ptr(),
                     copy_bytes.at(comm_rank),
                     copy_events.at(i),
                     wait_events.size(),
                     wait_events.data()));
        }
    }
    if (is_monolithic) {
        //TODO: add fallback path for peer_count > max_peer_count
        CCL_THROW_IF_NOT(size_t(peer_count) <= ccl::ze::max_peer_count,
                         "monolithic kernel not supported for peer_count ",
                         peer_count,
                         " > ",
                         ccl::ze::max_peer_count);
        global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
        std::string monolithic_kernel_name =
            "write_monolithic_kernel_" + std::to_string(peer_count) + "_" + to_string(dtype.idx()) +
            "_" + ccl_reduction_to_str(ccl::reduction::custom);
        LOG_DEBUG("allgatherv monolithic kernel name: ", monolithic_kernel_name);

        // use src instead of peer dst since we cannot make sure
        // peer dst got using ipc has the same alignment as remote buffer
        // we assume local buffer and remote buffer has the same alignment
        size_t buf_size_bytes = copy_bytes.at(comm_rank);
        unsigned long pre_align_offset_byte =
            ccl::utils::get_aligned_offset_byte(recv_bufs.at(comm_rank).get_ptr(),
                                                buf_size_bytes,
                                                ccl::global_data::env().kernel_mem_align);

        // First kernel starts from location 0 to pre_align_offset_byte
        // and the second kernel starts from location pre_align_offset_byte to the rest
        const size_t copy_count = copy_bytes.at(comm_rank) / dtype.size();
        constexpr int kernel_count = (int)ccl::utils::align_kernels::count;
        const unsigned long offsets[kernel_count] = { 0, pre_align_offset_byte };
        const unsigned long counts[kernel_count] = {
            pre_align_offset_byte / dtype.size(), copy_count - pre_align_offset_byte / dtype.size()
        };

        // Start two kernels, first kernel for the part of the array before the aligned start offset
        // and second kernel for the rest of the array from the aligned start offset to the end
        for (int i = 0; i < kernel_count; i++) {
            unsigned long count_local = counts[i];
            // data count is small and there is no need to execute the second aligned kernel
            if (i == (int)ccl::utils::align_kernels::aligned && count_local == 0) {
                copy_events.at(i) = copy_events.at(i - 1);
                break;
            }
            void* src = (src_buf + offsets[i]).get_ptr();
            std::vector<void*> dsts;
            for (auto& peer_buf : peer_bufs) {
                dsts.push_back((peer_buf + offsets[i]).get_ptr());
            }
            kernels.emplace_back(module, monolithic_kernel_name, worker_idx);
            ze_kernel_arg_t peer_bufs_ze_arg(dsts.data(), dsts.size());
            kernels.back().set_args({ &count_local, &src, peer_bufs_ze_arg });
            kernels.back().calculate_group_size(count_local);

            ZE_CALL(zeCommandListAppendLaunchKernel,
                    (entry->get_comp_list(),
                     kernels.back().get_kernel(),
                     kernels.back().get_group_count(),
                     copy_events.at(i),
                     wait_events.size(),
                     wait_events.data()));
        }
    }
}

void ze_a2a_allgatherv_entry::fill_list(const ze_base_entry* entry,
                                        int comm_rank,
                                        ccl_buffer send_buf,
                                        const std::vector<ccl_buffer>& recv_bufs,
                                        const std::vector<ccl_buffer>& peer_bufs,
                                        int peer_count,
                                        const std::vector<size_t>& copy_bytes,
                                        const ccl_datatype& dtype,
                                        const std::vector<size_t>& rank_buf_offsets,
                                        bool is_inplace,
                                        std::vector<ze_event_handle_t>& copy_events,
                                        std::vector<ze_event_handle_t>& wait_events,
                                        std::vector<ze_kernel>& kernels,
                                        ze_module_handle_t module,
                                        ze_device_handle_t device,
                                        ze_context_handle_t context,
                                        size_t worker_idx,
                                        size_t peer_buf_offset,
                                        bool is_read,
                                        bool is_monolithic) {
    if (is_read) {
        fill_list_read(entry,
                       comm_rank,
                       send_buf,
                       recv_bufs,
                       peer_bufs,
                       peer_count,
                       copy_bytes,
                       dtype,
                       rank_buf_offsets,
                       is_inplace,
                       copy_events,
                       wait_events,
                       peer_buf_offset,
                       is_monolithic);
    }
    else {
        fill_list_write(entry,
                        comm_rank,
                        send_buf,
                        recv_bufs,
                        peer_bufs,
                        peer_count,
                        copy_bytes,
                        dtype,
                        rank_buf_offsets,
                        is_inplace,
                        copy_events,
                        wait_events,
                        kernels,
                        module,
                        device,
                        context,
                        worker_idx,
                        peer_buf_offset,
                        is_monolithic);
    }

    if (!is_inplace) {
        /* copy send_buf to my buffer */
        void* src = send_buf.get_ptr();
        void* dst = recv_bufs.at(comm_rank).get_ptr();
        auto list = entry->get_copy_list(copy_direction::t2t);
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (list,
                 dst,
                 src,
                 copy_bytes.at(comm_rank),
                 copy_events.back(),
                 wait_events.size(),
                 wait_events.data()));
    }
}

void ze_a2a_allgatherv_entry::init_ze_hook() {
    /* get peer recv buffers */
    std::vector<ccl_buffer> peer_recv_bufs(comm->size());

    for (int i = 0; i < peer_count; ++i) {
        const int peer_rank = (comm_rank + i + 1) % comm->size();
        ccl_buffer buf{};
        sched->get_memory().handle_manager.get(peer_rank, peer_buf_idx, buf, comm);
        CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
        peer_recv_bufs[peer_rank] = buf;
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
    size_t copy_events_size = peer_count;
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
    fill_list(this,
              comm_rank,
              send_buf,
              recv_bufs,
              peer_recv_bufs,
              peer_count,
              block_bytes,
              dtype,
              rank_buf_offsets,
              is_inplace,
              copy_events,
              empty_wait_events,
              kernels,
              module,
              device,
              context,
              worker_idx,
              peer_buf_offset,
              is_read,
              is_monolithic);
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
