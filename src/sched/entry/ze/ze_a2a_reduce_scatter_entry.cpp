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
#include "sched/entry/ze/ze_a2a_reduce_scatter_entry.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <numeric>

using namespace ccl;
using namespace ccl::ze;

ze_a2a_reduce_scatter_entry::ze_a2a_reduce_scatter_entry(ccl_sched* sched,
                                                         ccl_buffer send_buf,
                                                         ccl_buffer recv_buf,
                                                         const size_t* recv_counts,
                                                         const ccl_datatype& dtype,
                                                         reduction op,
                                                         ccl_comm* comm,
                                                         std::vector<ze_event_handle_t> wait_events,
                                                         size_t peer_buf_idx,
                                                         size_t peer_buf_offset)
        : ze_base_entry(sched, comm, comm->size() * event_group_count, wait_events),
          send_buf(send_buf),
          recv_buf(recv_buf),
          dtype(dtype),
          op(op),
          recv_counts(recv_counts, recv_counts + comm->size()),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1) {}

void ze_a2a_reduce_scatter_entry::kernel_init(size_t rank_buf_offset,
                                              size_t block_count,
                                              void* send_buf,
                                              void* base_ptr,
                                              const std::vector<ccl_buffer>& peer_send_bufs,
                                              int peer_count,
                                              const ccl_datatype& dtype,
                                              int comm_rank,
                                              std::vector<ze_kernel>& kernels,
                                              ze_module_handle_t module,
                                              ze_device_handle_t device,
                                              ze_context_handle_t context,
                                              ccl::reduction op,
                                              size_t worker_idx,
                                              size_t peer_buf_offset,
                                              bool is_monolithic,
                                              bool is_single_kernel) {
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    unsigned long count = block_count;
    if (is_monolithic && peer_count <= (int)ccl::ze::max_peer_count) {
        std::string monolithic_kernel_name =
            "reduce_monolithic_kernel_" + std::to_string(peer_count) + "_" +
            to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
        LOG_DEBUG("Reduce scatter monolithic kernel name: ", monolithic_kernel_name);

        // use send_buf instead of peer_send_buf_ptr since we cannot make sure
        // peer_send_buf_ptr got using ipc has the same alignment as remote send_buf=.
        // we assume local send_buf and remote send_buf has the same alignment
        void* send_buf_ptr = static_cast<char*>(send_buf) + rank_buf_offset * dtype.size();
        size_t buf_size_bytes = count * dtype.size();
        unsigned long pre_align_offset_byte = ccl::utils::get_aligned_offset_byte(
            send_buf_ptr, buf_size_bytes, ccl::global_data::env().kernel_mem_align);

        // First kernel starts from location 0 to pre_align_offset_byte
        // and the second kernel starts from location pre_align_offset_byte to the rest
        constexpr int kernel_count = (int)ccl::utils::align_kernels::count;
        kernels.reserve(kernel_count);

        const unsigned long offsets[kernel_count] = { 0, pre_align_offset_byte };
        const unsigned long counts[kernel_count] = { pre_align_offset_byte / dtype.size(),
                                                     count - pre_align_offset_byte / dtype.size() };

        // Start two kernels, first kernel for the part of the array before the aligned start offset
        // and second kernel for the rest of the array from the aligned start offset to the end
        for (int i = 0; i < kernel_count; i++) {
            unsigned long count_local = counts[i];
            // data count is small and there is no need to excute the second aligned kernel
            if (i == (int)ccl::utils::align_kernels::aligned && count_local == 0) {
                break;
            }
            void* input_buf = static_cast<char*>(send_buf_ptr) + offsets[i];
            std::vector<void*> peer_bufs;
            peer_bufs.reserve(peer_count);
            for (auto& peer_send_buf : peer_send_bufs) {
                void* peer_buf = static_cast<char*>(peer_send_buf.get_ptr()) +
                                 (rank_buf_offset + peer_buf_offset) * dtype.size() + offsets[i];
                peer_bufs.push_back(peer_buf);
            }
            ze_kernel_arg_t peer_bufs_ze_arg(peer_bufs.data(), peer_bufs.size());
            void* output_buf = static_cast<char*>(base_ptr) + offsets[i];
            kernels.emplace_back(module, monolithic_kernel_name, worker_idx);
            kernels.back().set_args({ &count_local, &input_buf, peer_bufs_ze_arg, &output_buf });
            kernels.back().calculate_group_size(count_local);
        }
    }
    else if (is_single_kernel || (is_monolithic && peer_count > (int)ccl::ze::max_peer_count)) {
        // fallback path for monolithic kernel if peer_count > max_peer_count
        if (is_monolithic) {
            LOG_WARN("monolithic kernel not supported for peer_count ",
                     peer_count,
                     " > ",
                     ccl::ze::max_peer_count);
        }
        std::string kernel_name = "reduce_single_local_inplace_kernel_" + to_string(dtype.idx()) +
                                  "_" + ccl_reduction_to_str(op);

        // reduce peer values in tmp_buf and own values in send_buf into tmp_buf
        kernels.reserve(1);
        void* input_buf = static_cast<char*>(send_buf) + rank_buf_offset * dtype.size();
        void* inoutput_buf = base_ptr;
        kernels.emplace_back(module, kernel_name, worker_idx);
        kernels.back().set_args({ &count, &peer_count, &input_buf, &inoutput_buf });
        kernels.back().calculate_group_size(count);
    }
    else {
        std::string kernel_name = "reduce_local_inplace_kernel_" + to_string(dtype.idx()) + "_" +
                                  ccl_reduction_to_str(op);

        // reduce peer values in tmp_buf only
        kernels.reserve(peer_count);
        for (int i = 1; i < peer_count; ++i) {
            void* input_buf = static_cast<char*>(base_ptr) + i * block_count * dtype.size();
            void* inoutput_buf = base_ptr;
            kernels.emplace_back(module, kernel_name, worker_idx);
            kernels.back().set_args({ &count, &input_buf, &inoutput_buf });
            kernels.back().calculate_group_size(count);
        }

        // reduce send_buf + tmp_buf
        void* input_buf = static_cast<char*>(send_buf) + rank_buf_offset * dtype.size();
        void* inoutput_buf = base_ptr;
        kernels.emplace_back(module, kernel_name, worker_idx);
        kernels.back().set_args({ &count, &input_buf, &inoutput_buf });
        kernels.back().calculate_group_size(count);
    }
}

void ze_a2a_reduce_scatter_entry::fill_list(const ze_base_entry* entry,
                                            void* send_buf,
                                            void* tmp_buf,
                                            const std::vector<ccl_buffer>& peer_send_bufs,
                                            int peer_count,
                                            int comm_rank,
                                            size_t block_count,
                                            size_t rank_buf_offset,
                                            std::vector<ze_event_handle_t>& copy_events,
                                            std::vector<ze_kernel>& kernels,
                                            std::vector<ze_event_handle_t>& kernel_events,
                                            ze_event_handle_t& barrier_event,
                                            const ccl_datatype& dtype,
                                            ze_module_handle_t module,
                                            ze_device_handle_t device,
                                            ze_context_handle_t context,
                                            ccl::reduction op,
                                            size_t worker_idx,
                                            size_t peer_buf_offset,
                                            bool is_monolithic,
                                            bool is_single_kernel) {
    kernel_init(rank_buf_offset,
                block_count,
                send_buf,
                tmp_buf,
                peer_send_bufs,
                peer_count,
                dtype,
                comm_rank,
                kernels,
                module,
                device,
                context,
                op,
                worker_idx,
                peer_buf_offset,
                is_monolithic,
                is_single_kernel);

    if (is_monolithic && peer_count <= (int)ccl::ze::max_peer_count) {
        // reduce stage
        for (size_t i = 0; i < kernels.size(); ++i) {
            ZE_CALL(zeCommandListAppendLaunchKernel,
                    (entry->get_comp_list(),
                     kernels[i].get_kernel(),
                     kernels[i].get_group_count(),
                     kernel_events.at(i),
                     0,
                     nullptr));
        }
        // if only unaligned kernel is executed, then fill the event for
        // aligned kernel also since calling function expect two events
        if (kernels.size() < (int)ccl::utils::align_kernels::count) {
            CCL_THROW_IF_NOT(kernel_events.size() == (int)ccl::utils::align_kernels::count,
                             "monolithic kernel event count ",
                             kernel_events.size(),
                             " != ",
                             (int)ccl::utils::align_kernels::count);
            // assign kernel_events[1] = kernel_events[0]
            kernel_events.back() = kernel_events.front();
        }
    }
    else {
        size_t copy_bytes = block_count * dtype.size();
        /* copy peer segments to temp buffer */
        for (int i = 0; i < peer_count; i++) {
            void* src = static_cast<char*>(peer_send_bufs[i].get_ptr()) +
                        (rank_buf_offset + peer_buf_offset) * dtype.size();
            void* dst = static_cast<char*>(tmp_buf) + i * copy_bytes;
            // TODO: if we on the same device, then use t2t direction
            auto list = entry->get_copy_list(copy_direction::c2c, i);
            ZE_CALL(zeCommandListAppendMemoryCopy,
                    (list, dst, src, copy_bytes, copy_events.at(i), 0, nullptr));
        }

        ZE_CALL(zeCommandListAppendBarrier,
                (entry->get_comp_list(), barrier_event, copy_events.size(), copy_events.data()));

        /* reduce stage */
        for (size_t i = 0; i < kernels.size(); ++i) {
            ZE_CALL(zeCommandListAppendLaunchKernel,
                    (entry->get_comp_list(),
                     kernels[i].get_kernel(),
                     kernels[i].get_group_count(),
                     kernel_events.at(i),
                     1,
                     (i == 0) ? &barrier_event : &kernel_events.at(i - 1)));
        }
    }
}

void ze_a2a_reduce_scatter_entry::init_ze_hook() {
    /* get peer buffers */
    std::vector<ccl_buffer> peer_send_bufs(peer_count);

    for (int i = 0; i < peer_count; ++i) {
        int peer_rank = (comm_rank + i + 1) % comm->size();
        sched->get_memory().handle_manager.get(peer_rank, peer_buf_idx, peer_send_bufs[i], comm);
        CCL_THROW_IF_NOT(peer_send_bufs[i].get_ptr(), "null IPC buffer is received");
    }

    /* alloc temp buffer */
    size_t buf_bytes = dtype.size() * recv_counts[comm_rank];
    size_t tmp_buf_bytes = peer_count * buf_bytes;
    if (tmp_buf_bytes == 0) {
        return;
    }
    ccl::alloc_param alloc_param(tmp_buf_bytes, buffer_type::ze, buffer_place::device);
    void* tmp_buf = sched->alloc_buffer(alloc_param).get_ptr();

    LOG_DEBUG("rank ",
              comm_size,
              ", tmp buf size: ",
              tmp_buf_bytes,
              ", buf_count: ",
              recv_counts[comm_rank]);

    /* copy peer segments to temp buffer */

    // do no need separate memcpys when using monolithic kernel
    if (!ccl::global_data::env().reduce_scatter_monolithic_kernel) {
        pre_copy_events.resize(peer_count);
        for (auto& event : pre_copy_events) {
            event = ze_base_entry::create_event();
        }
    }

    if (ccl::global_data::env().reduce_scatter_monolithic_kernel) {
        // leftover kernel and aligned kernel
        kernel_events.resize((int)ccl::utils::align_kernels::count);
    }
    else if (ccl::global_data::env().enable_kernel_single_reduce_peers) {
        // when kernel merge is used only one kernel is required
        kernel_events.resize(1);
    }
    else {
        kernel_events.resize(peer_count);
    }
    for (auto& event : kernel_events) {
        event = ze_base_entry::create_event();
    }

    size_t rank_buf_offset =
        std::accumulate(recv_counts.begin(), recv_counts.begin() + comm_rank, 0);

    barrier_event = ze_base_entry::create_event();

    fill_list(this,
              send_buf.get_ptr(),
              tmp_buf,
              peer_send_bufs,
              peer_count,
              comm_rank,
              recv_counts[comm_rank],
              rank_buf_offset,
              pre_copy_events,
              kernels,
              kernel_events,
              barrier_event,
              dtype,
              module,
              device,
              context,
              op,
              worker_idx,
              peer_buf_offset,
              ccl::global_data::env().reduce_scatter_monolithic_kernel,
              ccl::global_data::env().enable_kernel_single_reduce_peers);
    post_copy_events.resize(1);
    for (auto& event : post_copy_events) {
        event = ze_base_entry::create_event();
    }
    ZE_CALL(zeCommandListAppendMemoryCopy,
            (ze_base_entry::get_copy_list(),
             recv_buf.get_ptr(),
             tmp_buf,
             buf_bytes,
             post_copy_events.back(),
             kernel_events.size(),
             kernel_events.data()));
}

void ze_a2a_reduce_scatter_entry::update() {
    for (const auto& event : post_copy_events) {
        if (!ze_base_entry::is_event_completed(event)) {
            return;
        }
    }
    ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
    ze_base_entry::update();
}

std::string ze_a2a_reduce_scatter_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":"
        << std::accumulate(recv_counts.begin(), recv_counts.end(), 0) * dtype.size();
    return out.str();
}
