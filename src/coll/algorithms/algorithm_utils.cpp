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
#include <algorithm>
#include <numeric>
#include <sstream>

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/log/log.hpp"
#include "sched/entry/factory/entry_factory.hpp"

const char* ccl_coll_type_to_str(ccl_coll_type type) {
    auto type_str = "undefined";
    switch (type) {
        case ccl_coll_allgatherv: return "allgatherv";
        case ccl_coll_allreduce: return "allreduce";
        case ccl_coll_alltoall: return "alltoall";
        case ccl_coll_alltoallv: return "alltoallv";
        case ccl_coll_barrier: return "barrier";
        case ccl_coll_bcast: return "bcast";
        case ccl_coll_recv: return "recv";
        case ccl_coll_reduce: return "reduce";
        case ccl_coll_reduce_scatter: return "reduce_scatter";
        case ccl_coll_send: return "send";
        case ccl_coll_partial: return "partial";
        case ccl_coll_undefined: return type_str;
        default: type_str = "unknown";
    }
    return type_str;
}

void ccl_get_segment_sizes(size_t dtype_size,
                           size_t elem_count,
                           size_t requested_seg_size,
                           std::vector<size_t>& seg_sizes) {
    seg_sizes.clear();

    if (dtype_size * elem_count == 0) {
        return;
    }
    else if (dtype_size >= requested_seg_size) {
        seg_sizes.resize(elem_count, 1);
    }
    else {
        size_t seg_size = (requested_seg_size + dtype_size - 1) / dtype_size;
        size_t total_seg_count = std::max((elem_count + seg_size - 1) / seg_size, 1UL);
        size_t regular_seg_size = elem_count / total_seg_count;
        size_t large_seg_size = regular_seg_size + ((elem_count % total_seg_count) != 0);
        size_t regular_seg_count = total_seg_count * large_seg_size - elem_count;

        seg_sizes.resize(total_seg_count, regular_seg_size);
        std::fill(seg_sizes.begin() + regular_seg_count, seg_sizes.end(), large_seg_size);

        size_t sum =
            std::accumulate(seg_sizes.begin(), seg_sizes.end(), ccl::utils::initial_count_value);
        if (sum != elem_count) {
            std::stringstream ss;
            for (size_t idx = 0; idx < seg_sizes.size(); idx++) {
                ss << seg_sizes[idx] << " ";
            }
            CCL_THROW_IF_NOT(false,
                             "unexpected sum of seg_sizes ",
                             sum,
                             ", expected ",
                             elem_count,
                             ", total_seg_count ",
                             total_seg_count,
                             ", regular_seg_count ",
                             regular_seg_count,
                             ", regular_seg_size ",
                             regular_seg_size,
                             ", large_seg_size ",
                             large_seg_size,
                             ", all seg_sizes: ",
                             ss.str());
        }
    }
}

bool ccl_is_ptr_aligned(uintptr_t ptr, size_t alignment) {
    CCL_THROW_IF_NOT(alignment != 0, "memory alignment cannot be 0 by definition");
    return (ptr % alignment) == 0;
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
static bool is_reorderable_algo(const char* algo_name) {
    const char* reordable_algo_prefixes[] = {
        "ALLREDUCE_PIPE",
        "ALLGATHERV_PIPE",
        "REDUCE_PIPE",
        "REDUCE_SCATTER_PIPE",
    };
    for (auto& reordable_algo_prefix : reordable_algo_prefixes) {
        if (0 == strncmp(algo_name, reordable_algo_prefix, strlen(reordable_algo_prefix))) {
            return true;
        }
    }
    return false;
}

// Reorders commands to minimize wait time
//
// Sample case before submission, 3-stage command split into 2 chunks:
//      chunk_0.0 chunk_0.1 chunk_0.2 chunk_1.0 chunk_1.1 chunk_1.2
//
// Problem: each command within chunk depends on its predecessor, serial execution
//
// Submission order within this function:
//      chunk_0.0 chunk_1.0 chunk_0.1 chunk_1.1 chunk_0.2 chunk_1.2
// allows parallel execution of e.g. chunk_0.0 and chunk_1.1
//
// Reordering only happens for entries that have a name that matches `is_reorderable_algo`
//
// Opportunity for micro-optimization and simplification: compare enum instead of strings
uint32_t ccl_submit_ze_commands_in_subsched_entries(ccl_sched* sched) {
    std::vector<subsched_entry*> subsched_chunks;
    for (auto& entry : sched->entries) {
        if (is_reorderable_algo(entry->name())) {
            subsched_chunks.push_back(static_cast<subsched_entry*>(entry.get()));
        }
        else {
            LOG_DEBUG("entry: ", entry->name(), "is NOT reorderable algo")
        }
    }

    auto chunk_count = subsched_chunks.size();
    LOG_DEBUG("chunk_count ", chunk_count);

    std::vector<size_t> next_entry(chunk_count, 0);
    bool done = false;
    uint32_t command_count = 0;
    int cmd_idx = 0;
    // iterate over each stage
    while (!done) {
        done = true;
        // iterate over each chunk
        for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
            LOG_DEBUG("cmd_idx=",
                      cmd_idx,
                      ", chunk_idx=",
                      chunk_idx,
                      ", | ",
                      subsched_chunks[chunk_idx]->name(),
                      ", entries.size=",
                      subsched_chunks[chunk_idx]->get_subsched()->entries.size(),
                      ", next_entry=",
                      next_entry[chunk_idx]);
            if (next_entry[chunk_idx] <
                subsched_chunks[chunk_idx]->get_subsched()->entries.size()) {
                LOG_DEBUG("cmd_idx=", cmd_idx, ", chunk_idx=", chunk_idx, ", submitting commands");
                command_count += subsched_chunks[chunk_idx]
                                     ->get_subsched()
                                     ->entries[next_entry[chunk_idx]++]
                                     ->ze_commands_submit();
                done = false;
            }
            LOG_DEBUG("cmd_idx=", cmd_idx, ", chunk_idx=", chunk_idx, ", done=", done);
        }
        ++cmd_idx;
    }

    return command_count;
}

// checks if pipelining is enabled and returns pipelining parameters
// output params:
//  - main_chunk_count: number of elements in the "standard" chunk, i.e. not counting the remainder
//  - mem_align: alignment required for each chunk; cannot be zero
bool ccl_is_pipe_enabled(const size_t count,
                         const size_t dtype_size,
                         const size_t chunk_count,
                         size_t& main_chunk_count,
                         size_t& mem_align) {
    bool ret = true;
    // Note about cache lines and pipelining: The same cache line must contain
    // a single chunk only.
    //
    // If the same cache line contains two chunks (or more), and we parallelize
    // the instructions required for both chunks, a conflict (race condition)
    // may appear between the copy-out for the scaleout portion and the
    // reduce_scatter phase.
    //
    // The easiest way to avoid that race condition is to require that each
    // cache line contains a single entry. If that is not the case, we must not
    // parallelize the instructions for different chunks.

    bool is_pipe = chunk_count > 1 && ccl::global_data::env().enable_ze_single_list;

    // TODO: why does oneCCL have CACHELINE_SIZE *and* CCL_KERNEL_MEM_ALIGN?
    mem_align = ccl::global_data::env().kernel_mem_align;
    CCL_THROW_IF_NOT(mem_align != 0, "memory alignment cannot be zero by definition");
    size_t buf_size_bytes = count * dtype_size;

    // First, determine if we need to fallback to non-pipelined algorightm.
    // Such a fallback may happen in cases such as (1) the user requests it,
    // (2) message fits into a cache line, or (3) the cache line size is not
    // divisible by the data type size.

    size_t number_of_cache_lines_per_chunk =
        is_pipe ? std::max(mem_align, buf_size_bytes / chunk_count) / mem_align : 1;
    size_t main_chunk_size_bytes = mem_align * number_of_cache_lines_per_chunk;
    main_chunk_count = main_chunk_size_bytes / dtype_size;

    bool is_dtype_divisible = ((main_chunk_size_bytes % dtype_size) == 0);
    bool is_msg_bigger_than_cache_line = buf_size_bytes > main_chunk_size_bytes;

    bool is_singleworker =
        !ccl::global_data::env().ze_multi_workers || (ccl::global_data::env().worker_count <= 1);

    if (!is_pipe) {
        LOG_DEBUG("Pipelining code disabled");
        ret = false;
    }
    else {
        if (!is_dtype_divisible) {
            LOG_INFO("Running without pipelining because datatype size (",
                     dtype_size,
                     ") is not divisible by cache line size (",
                     mem_align,
                     ")");
            ret = false;
        }
        if (!is_msg_bigger_than_cache_line) {
            LOG_INFO("Running without pipelining because message size (",
                     buf_size_bytes,
                     ") is smaller than a cache line (",
                     mem_align,
                     ") or than main_chunk_size_bytes (",
                     main_chunk_size_bytes,
                     ")");
            ret = false;
        }
        if (!is_singleworker) {
            LOG_INFO("Running without pipelining because ze_multi_workers was requested with ",
                     ccl::global_data::env().worker_count,
                     " workers, which is more than one worker ");
            ret = false;
        }
    }

    return ret;
}

ccl::status ccl_build_topo_uniform_buff_size_op(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t count,
    size_t dtype_size,
    size_t pipe_nof_chunks,
    const std::string& op_name,
    ccl::profile::metrics_counter& metrics,
    std::function<
        ccl::status(ccl_sched* sched, ccl_buffer send_buf, ccl_buffer recv_buf, size_t count)>
        fill_op_lambda) {
    size_t mem_align = 0;
    size_t main_chunk_count = 0;
    if (!ccl_is_pipe_enabled(count, dtype_size, pipe_nof_chunks, main_chunk_count, mem_align)) {
        // Fall back to topo algorithm without pipelining
        fill_op_lambda(sched, send_buf, recv_buf, count);
        entry_factory::create<ze_execute_cmdlists_on_init_entry>(sched);

        return ccl::status::success;
    }

    LOG_DEBUG("build pipe ", op_name);

    // Need to re-calculate chunk_count after main_chunk_size_bytes calculation
    // with cache alignment in mind.
    size_t chunk_count = count / main_chunk_count;
    size_t last_chunk_count = main_chunk_count + (count % main_chunk_count);

    sched->try_enable_ze_single_list();
    auto sync_obj = std::make_shared<sync_object>(chunk_count);
    bool is_parallelizable_chunks = true;

    for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        size_t chunk_offset = chunk_idx * main_chunk_count * dtype_size;
        ccl_buffer sbuf = send_buf + chunk_offset;
        ccl_buffer rbuf = recv_buf + chunk_offset;
        size_t this_chunk_count =
            (chunk_idx == (chunk_count - 1)) ? last_chunk_count : main_chunk_count;

        if (this_chunk_count || (count == 0 && chunk_idx == 0)) {
            entry_factory::create<subsched_entry>(
                sched,
                chunk_idx,
                [sched, sbuf, rbuf, this_chunk_count, sync_obj, fill_op_lambda](ccl_sched* s) {
                    s->inherit_ze_managers_from(sched);
                    s->set_init_ze_hook_sync_obj(sync_obj);
                    s->set_ze_commands_bypass_flag(false);

                    fill_op_lambda(s, sbuf, rbuf, this_chunk_count);
                },
                (op_name + "_PIPE" + std::to_string(chunk_idx)).c_str());
        }
        // WARNING: previous chunk has part of this chunk's first cache
        // line. Cannot use pipelining. However, since this is a
        // "local" decision (i.e., other ranks may decide differently),
        // we still need to apply chunking. However, we will run one
        // chunk at a time, without parallelizing them.
        // Another way to have implemented this would be to link the
        // last task of the prev chunk with the first of this chunk
        // with an event.
        is_parallelizable_chunks &=
            ccl_is_ptr_aligned(reinterpret_cast<uintptr_t>(rbuf.get_ptr()), mem_align);
    }

    static bool is_chunk_mem_align_warning_printed{};
    if (!is_parallelizable_chunks && !is_chunk_mem_align_warning_printed) {
        is_chunk_mem_align_warning_printed = true;
        LOG_WARN(
            "[",
            op_name,
            " pipelining]: For best performance, (i) chunk size should be a multiple of a cache line (",
            mem_align,
            " bytes), and (ii) buffers in all ranks should be aligned to ",
            mem_align);
    }

    if (!is_parallelizable_chunks) {
        metrics.nonparallel_calls_per_count[count]++;
    }
    else {
        metrics.parallel_calls_per_count[count]++;
    }

    entry_factory::create<ze_execute_cmdlists_on_start_entry>(
        sched,
        sync_obj,
        is_parallelizable_chunks ? ccl_submit_ze_commands_in_subsched_entries : nullptr);

    return ccl::status::success;
}

#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL
