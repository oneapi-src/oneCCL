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
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "sched/cache/cache.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_FUSION_CHECK_SCHEDS_ITERS (1024)

ccl_status_t complete_user_request(const void* ctx) {
    ccl_master_sched* sched = (ccl_master_sched*)ctx;
    LOG_DEBUG("complete fusion request: ", static_cast<ccl_request*>(sched));
    sched->complete();
    return ccl_status_success;
}

ccl_status_t release_fusion_buf(const void* ctx) {
    void* buf = (void*)ctx;

    if (ccl::global_data::get().fusion_manager)
        ccl::global_data::get().fusion_manager->release_buffer(buf);

    return ccl_status_success;
}

ccl_status_t release_fusion_buf_for_cached_sched(ccl_sched* sched, const void* ctx) {
    return release_fusion_buf(ctx);
}

ccl_fusion_buffer_cache::ccl_fusion_buffer_cache(size_t buf_size) : buf_size(buf_size) {
    void* buf;
    for (size_t idx = 0; idx < CCL_FUSION_BUFFER_CACHE_PREALLOC; idx++) {
        buf = CCL_MALLOC(buf_size, "buffer");
        free_buffers.push_back(buf);
        all_buffers.push_back(buf);
    }
    LOG_INFO("created buffer_cache: buf_size ", buf_size);
}

ccl_fusion_buffer_cache::~ccl_fusion_buffer_cache() {
    std::lock_guard<ccl_fusion_lock_t> lock{ guard };

    if (free_buffers.size() > all_buffers.size()) {
        CCL_FATAL("unexpected buffer count - free_buffers: ",
                  free_buffers.size(),
                  ", all_buffers: ",
                  all_buffers.size());
    }

    for (size_t idx = 0; idx < all_buffers.size(); idx++) {
        CCL_FREE(all_buffers[idx]);
    }

    all_buffers.clear();
    free_buffers.clear();
}

void* ccl_fusion_buffer_cache::get() {
    std::lock_guard<ccl_fusion_lock_t> lock{ guard };

    void* buf;
    if (!free_buffers.empty()) {
        buf = free_buffers.front();
        free_buffers.pop_front();
    }
    else {
        buf = CCL_MALLOC(buf_size, "buffer");
        LOG_DEBUG("get buf from extra allocation ", buf);
        all_buffers.push_back(buf);
    }
    CCL_THROW_IF_NOT(buf, "empty buf");

    return buf;
}

void ccl_fusion_buffer_cache::release(void* buf) {
    std::lock_guard<ccl_fusion_lock_t> lock{ guard };
    CCL_THROW_IF_NOT(buf, "empty buf");
    free_buffers.push_back(buf);
}

ccl_fusion_manager::ccl_fusion_manager()
        : bytes_threshold(ccl::global_data::env().fusion_bytes_threshold),
          count_threshold(ccl::global_data::env().fusion_count_threshold),
          buf_cache(ccl::global_data::env().fusion_bytes_threshold *
                    ccl::global_data::env().fusion_count_threshold) {
    CCL_ASSERT(bytes_threshold >= 1, "unexpected fusion_bytes_threshold ", bytes_threshold);
    CCL_ASSERT(count_threshold >= 1, "unexpected fusion_count_threshold ", count_threshold);

    long cycle_usec = long(ccl::global_data::env().fusion_cycle_ms * 1000.0);
    cycle = std::chrono::microseconds(cycle_usec);
    last_exec_time = std::chrono::steady_clock::now();

    LOG_INFO("created fusion manager, cycle_usec ",
             cycle_usec,
             ", bytes_threshold ",
             bytes_threshold,
             ", count_threshold ",
             count_threshold);
}

ccl_fusion_manager::~ccl_fusion_manager() {
    LOG_INFO("fused_bytes ",
             stat_fused_bytes,
             ", fused_ops ",
             stat_fused_ops,
             ", empty_exec_calls ",
             stat_empty_exec_calls,
             ", overlapped_exec_calls ",
             stat_overlapped_exec_calls);

    while (!tracked_scheds.empty())
        check_tracked_scheds(true);

    CCL_ASSERT(postponed_queue.empty() && exec_queue.empty() && tracked_scheds.empty(),
               "queues are not empty, ",
               postponed_queue.size(),
               " ",
               exec_queue.size(),
               " ",
               tracked_scheds.size());
}

bool ccl_fusion_manager::can_fuse(ccl_master_sched* sched) {
    size_t bytes = sched->coll_param.count * sched->coll_param.dtype.size();
    if (bytes >= bytes_threshold) {
        LOG_DEBUG("can't fuse due to size ", bytes, ", max ", bytes_threshold);
        return false;
    }

    if (sched->coll_param.ctype != ccl_coll_allreduce) {
        LOG_DEBUG("can't fuse due to coll_type ", ccl_coll_type_to_str(sched->coll_param.ctype));
        return false;
    }

    if (sched->coll_attr.prologue_fn || sched->coll_attr.epilogue_fn ||
        sched->coll_attr.reduction_fn || sched->coll_attr.synchronous) {
        LOG_DEBUG("can't fuse due to unexpected fields in coll_attr");
        return false;
    }

    LOG_DEBUG("can fuse, bytes ", bytes);
    return true;
}

bool ccl_fusion_manager::add(ccl_master_sched* sched) {
    if (!can_fuse(sched)) {
        return false;
    }

    CCL_THROW_IF_NOT(sched->is_completed(), "incorrect completion counter");
    sched->set_counter(1);

    std::lock_guard<ccl_fusion_lock_t> lock{ guard };
    postponed_queue.push_back(sched);
    return true;
}

ccl_master_sched* ccl_fusion_manager::build_sched() {
    size_t sum_count = 0, sum_bytes = 0, dtype_size;
    size_t max_priority = 0;
    bool use_cache = true;
    ccl_comm* comm;
    ccl_reduction_t reduction;
    ccl_coll_type ctype;
    const ccl_stream* stream __attribute__((unused)) = nullptr;
    void* fusion_buf = nullptr;
    bool fill_sched = true;

    CCL_THROW_IF_NOT(exec_queue.size(), "empty queue");

    auto first_sched = exec_queue.front();
    auto last_sched = exec_queue.back();
    const ccl_datatype& dtype = first_sched->coll_param.dtype;
    dtype_size = dtype.size();
    reduction = first_sched->coll_param.reduction;
    comm = first_sched->coll_param.comm;
    ctype = first_sched->coll_param.ctype;
    stream = first_sched->coll_param.stream;
    max_priority = first_sched->coll_attr.priority;

    for (const auto& s : exec_queue) {
        sum_count += s->coll_param.count;
        if (!s->coll_attr.to_cache) {
            use_cache = false;
        }
        max_priority = std::max(s->coll_attr.priority, max_priority);
    }
    sum_bytes = sum_count * dtype_size;

    LOG_DEBUG("build fused_sched for sum_count ",
              sum_count,
              ", sum_bytes ",
              sum_bytes,
              ", sched_count ",
              exec_queue.size());

    ccl_master_sched* sched = nullptr;
    auto create_fn = [this, ctype, &fusion_buf, sum_count, dtype, reduction, comm]() {
        ccl_master_sched* sched = nullptr;
        switch (ctype) {
            case ccl_coll_allreduce: {
                ccl_coll_param coll_param{};
                fusion_buf = this->buf_cache.get();
                coll_param.ctype = ctype;
                coll_param.send_buf = fusion_buf;
                coll_param.recv_buf = fusion_buf;
                coll_param.count = sum_count;
                coll_param.dtype = dtype;
                coll_param.reduction = reduction;
                coll_param.comm = comm;
                sched = new ccl_master_sched(coll_param);
                sched->internal_type = ccl_sched_internal_fusion;
            } break;
            default: CCL_FATAL("not supported"); break;
        }
        return sched;
    };

    if (use_cache) {
        ccl_sched_key key{};
        key.f.ctype = ctype;
        key.f.count1 = sum_count;
        key.f.count2 = exec_queue.size();
        key.f.dtype = dtype.idx();
        key.f.reduction = reduction;
        key.f.comm = comm;
        key.match_id = first_sched->coll_attr.match_id + last_sched->coll_attr.match_id;
        LOG_DEBUG("key.match_id ", key.match_id);
        bool is_created = false;
        std::tie(sched, is_created) =
            ccl::global_data::get().sched_cache->find_or_create(std::move(key), create_fn);

        fill_sched = is_created;

        if (!is_created) {
            LOG_DEBUG("found fused_sched in cache");
            if (!sched->is_completed()) {
                LOG_DEBUG("it is not completed sched");
                stat_overlapped_exec_calls++;
                if (ccl::global_data::get().executor->get_worker_count() > 1) {
                    LOG_DEBUG("found fused_sched in cache, which is not completed yet");
                    ccl::global_data::get().executor->wait(sched);
                }
                else {
                    CCL_THROW_IF_NOT(sched->is_completed(),
                                     "non completed fused_sched found in cache");
                }
            }
        }
    }
    else {
        sched = create_fn();
    }

    CCL_THROW_IF_NOT(sched);

    tracked_scheds.push_back(sched);
    sched->coll_attr.priority = max_priority;
    sched->coll_attr.to_cache = use_cache;

    stat_fused_bytes += sum_bytes;
    stat_fused_ops += exec_queue.size();

    if (!fill_sched) {
        clear_exec_queue();
        return sched;
    }

    sched->commit(ccl::global_data::get().parallelizer.get());

    size_t exec_queue_size = exec_queue.size();
    size_t part_count = sched->partial_scheds.size();
    std::vector<std::shared_ptr<ccl_sched>>& part_scheds = sched->partial_scheds;
    size_t copies_per_part = exec_queue_size / part_count;
    size_t copies_per_last_part = copies_per_part + exec_queue_size % part_count;

    CCL_THROW_IF_NOT(part_count > 0, "unexpected part_count");

    LOG_DEBUG("part_count ",
              part_count,
              ", sum_count ",
              sum_count,
              ", exec_queue_size ",
              exec_queue_size);

    for (size_t idx = 0; idx < part_count; idx++) {
        part_scheds[idx]->add_barrier();
        part_scheds[idx]->set_add_mode(ccl_sched_add_front);
    }
    sched->sync_partial_scheds();

    size_t offset = 0;
    for (size_t idx = 0; idx < part_count; idx++) {
        size_t copies_count = (idx < part_count - 1) ? copies_per_part : copies_per_last_part;

        for (size_t copy_idx = 0; copy_idx < copies_count; copy_idx++) {
            size_t global_copy_idx = idx * copies_per_part + copy_idx;

#ifdef CCL_ENABLE_SYCL
            if (stream && stream->is_sycl_device_stream())
                entry_factory::make_entry<sycl_copy_device_to_host_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(&(exec_queue[global_copy_idx]->coll_param.sycl_send_buf),
                               exec_queue[global_copy_idx]->coll_param.count * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    ccl_buffer(fusion_buf, buf_cache.get_buf_size(), offset),
                    exec_queue[global_copy_idx]->coll_param.count,
                    dtype,
                    stream);
            else
#endif /* CCL_ENABLE_SYCL */
                entry_factory::make_entry<copy_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(&(exec_queue[global_copy_idx]->coll_param.send_buf),
                               exec_queue[global_copy_idx]->coll_param.count * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    ccl_buffer(fusion_buf, buf_cache.get_buf_size(), offset),
                    exec_queue[global_copy_idx]->coll_param.count,
                    dtype);

            offset += exec_queue[global_copy_idx]->coll_param.count * dtype_size;
        }
    }

    for (size_t idx = 0; idx < part_count; idx++) {
        part_scheds[idx]->set_add_mode(ccl_sched_add_back);
    }
    sched->sync_partial_scheds();

    offset = 0;
    for (size_t idx = 0; idx < part_count; idx++) {
        size_t copies_count = (idx < part_count - 1) ? copies_per_part : copies_per_last_part;

        for (size_t copy_idx = 0; copy_idx < copies_count; copy_idx++) {
            size_t global_copy_idx = idx * copies_per_part + copy_idx;

#ifdef CCL_ENABLE_SYCL
            if (stream && stream->is_sycl_device_stream())
                entry_factory::make_entry<sycl_copy_host_to_device_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(fusion_buf, buf_cache.get_buf_size(), offset),
                    ccl_buffer(&(exec_queue[global_copy_idx]->coll_param.sycl_recv_buf),
                               exec_queue[global_copy_idx]->coll_param.count * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    exec_queue[global_copy_idx]->coll_param.count,
                    dtype,
                    stream);
            else
#endif /* CCL_ENABLE_SYCL */
                entry_factory::make_entry<copy_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(fusion_buf, buf_cache.get_buf_size(), offset),
                    ccl_buffer(&(exec_queue[global_copy_idx]->coll_param.recv_buf),
                               exec_queue[global_copy_idx]->coll_param.count * dtype_size,
                               ccl_buffer_type::INDIRECT),
                    exec_queue[global_copy_idx]->coll_param.count,
                    dtype);

            offset += exec_queue[global_copy_idx]->coll_param.count * dtype_size;
            entry_factory::make_entry<function_entry>(
                part_scheds[idx].get(), complete_user_request, exec_queue[global_copy_idx]);
            CCL_THROW_IF_NOT(!exec_queue[global_copy_idx]->is_completed(),
                             "incorrect completion counter");
        }
    }

    if (use_cache) {
        part_scheds[0]->set_finalize_fn(release_fusion_buf_for_cached_sched, fusion_buf);
    }
    else {
        sched->sync_partial_scheds();
        entry_factory::make_entry<function_entry>(
            part_scheds[0].get(), release_fusion_buf, fusion_buf);
    }

    clear_exec_queue();

    return sched;
}

void ccl_fusion_manager::execute() {
    auto this_time = std::chrono::steady_clock::now();
    auto diff = (last_exec_time + cycle - this_time);
    if (diff > std::chrono::steady_clock::duration::zero()) {
        /* it is too early, do nothing */
        stat_empty_exec_calls++;
        return;
    }
    last_exec_time = std::chrono::steady_clock::now();

    bool flush_exec_queue = false;
    if (ccl::global_data::env().fusion_check_urgent && !exec_queue.empty()) {
        /* recheck scheds from exec_queue, maybe some of them were marked as urgent since previous call */
        for (auto it = exec_queue.begin(); it != exec_queue.end(); ++it) {
            if ((*it)->urgent) {
                LOG_DEBUG("found urgent sched in exec_queue, flush exec_queue");
                flush_exec_queue = true;
                break;
            }
        }
    }
    /* separate block to reduce lock scope */
    {
        std::lock_guard<ccl_fusion_lock_t> lock{ guard };
        if (!postponed_queue.empty()) {
            LOG_DEBUG("postponed_queue size ", postponed_queue.size());

            ccl_master_sched* first_sched;
            if (!exec_queue.empty()) {
                first_sched = exec_queue.front();
            }
            else {
                first_sched = postponed_queue.front();
                exec_queue.push_back(first_sched);
                postponed_queue.pop_front();
                exec_queue_sum_bytes =
                    first_sched->coll_param.count * first_sched->coll_param.dtype.size();
            }

            for (auto it = postponed_queue.begin(); it != postponed_queue.end();) {
                auto s = *it;
                if (s->coll_param.dtype.idx() == first_sched->coll_param.dtype.idx() &&
                    s->coll_param.comm == first_sched->coll_param.comm &&
                    s->coll_param.ctype == first_sched->coll_param.ctype &&
                    s->coll_param.reduction == first_sched->coll_param.reduction &&
                    s->coll_param.stream == first_sched->coll_param.stream) {
                    size_t size = s->coll_param.count * s->coll_param.dtype.size();
                    if (exec_queue_sum_bytes + size > CCL_FUSION_BUFFER_SIZE) {
                        LOG_DEBUG("too much bytes in buffer, flush exec_queue");
                        flush_exec_queue = true;
                        break;
                    }
                    exec_queue_sum_bytes += size;

                    if (ccl::global_data::env().fusion_check_urgent && !flush_exec_queue &&
                        s->urgent) {
                        LOG_DEBUG(
                            "found urgent sched in postponed_queue, flush exec_queue, postponed_queue size ",
                            postponed_queue.size());
                        flush_exec_queue = true;
                    }

                    exec_queue.push_back(s);
                    it = postponed_queue.erase(it);

                    if (exec_queue.size() == count_threshold) {
                        LOG_DEBUG("too many scheds, flush exec_queue");
                        flush_exec_queue = true;
                        break;
                    }
                }
                else {
                    ++it;
                }
            }
        }
    }

    if (flush_exec_queue) {
        LOG_DEBUG("exec_queue size ", exec_queue.size(), ", bytes ", exec_queue_sum_bytes);
        ccl_master_sched* sched = build_sched();
        sched->start(ccl::global_data::get().executor.get());
    }

    if (stat_fused_ops % CCL_FUSION_CHECK_SCHEDS_ITERS == 0) {
        check_tracked_scheds();
    }
}

void ccl_fusion_manager::release_buffer(void* buf) {
    buf_cache.release(buf);
}

void ccl_fusion_manager::clear_exec_queue() {
    exec_queue.clear();
    exec_queue_sum_bytes = 0;
}

void ccl_fusion_manager::check_tracked_scheds(bool force_release) {
    for (auto it = tracked_scheds.begin(); it != tracked_scheds.end();) {
        ccl_master_sched* sched = *it;
        if (sched->is_completed() && (!sched->coll_attr.to_cache || force_release)) {
            ccl_release_sched(sched);
            it = tracked_scheds.erase(it);
        }
        else {
            ++it;
        }
    }
}
