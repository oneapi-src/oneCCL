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
#include <numeric>

#include "coll/algorithms/algorithms_enum.hpp"
#include "coll/coll_param.hpp"
#include "coll/selection/selection.hpp"
#include "common/global/global.hpp"
#include "common/comm/comm.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"
#include "sched/buffer_cache.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched_base.hpp"

ccl_sched_base::ccl_sched_base(const ccl_coll_param& coll_param) : coll_param(coll_param) {
#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
    if (coll_param.stream) {
        ccl_comm* node_comm =
            coll_param.comm->get_host_comm()->get_node_comm().get()->get_ccl_comm().get();
        memory.handle_manager.init(node_comm, coll_param.stream);
    }
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
}
std::string to_string(ccl_sched_add_mode mode) {
    switch (mode) {
        case ccl_sched_add_front: return "FRONT";
        case ccl_sched_add_back: return "BACK";
        default: return "DEFAULT";
    }
    return "DEFAULT";
}

ccl_sched_base::~ccl_sched_base() {
    free_memory();
}

void ccl_sched_base::set_coll_attr(const ccl_coll_attr& attr) {
    coll_attr = attr;
}

void ccl_sched_base::update_coll_param_and_attr(const ccl_coll_param& param,
                                                const ccl_coll_attr& attr) {
#ifdef CCL_ENABLE_SYCL
    coll_param.sync_deps(param.stream, param.deps);
#endif // CCL_ENABLE_SYCL

    bool has_pre_post_copies =
        (!coll_param.device_send_bufs.empty() || !coll_param.device_recv_bufs.empty()) ? true
                                                                                       : false;

    if (has_pre_post_copies) {
        CCL_THROW_IF_NOT(coll_param.device_send_bufs.size() == param.send_bufs.size(),
                         "send_bufs sizes mismatch");
        CCL_THROW_IF_NOT(coll_param.device_recv_bufs.size() == param.recv_bufs.size(),
                         "recv_bufs sizes mismatch");
        coll_param.device_send_bufs = param.send_bufs;
        coll_param.device_recv_bufs = param.recv_bufs;
    }
    else {
        CCL_THROW_IF_NOT(coll_param.send_bufs.size() == param.send_bufs.size(),
                         "send_bufs sizes mismatch");
        CCL_THROW_IF_NOT(coll_param.recv_bufs.size() == param.recv_bufs.size(),
                         "recv_bufs sizes mismatch");
        coll_param.send_bufs = param.send_bufs;
        coll_param.recv_bufs = param.recv_bufs;
    }

    int comm_size = coll_param.comm->size();

    if (coll_param.ctype == ccl_coll_allgatherv) {
        if (coll_attr.is_vector_buf)
            CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_bufs.size()) == comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_counts.size()) == comm_size);
    }

    if (coll_param.ctype == ccl_coll_alltoallv) {
        if (coll_attr.is_vector_buf)
            CCL_THROW_IF_NOT(static_cast<int>(coll_param.send_bufs.size()) == comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.send_counts.size()) == comm_size);

        if (coll_attr.is_vector_buf)
            CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_bufs.size()) == comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_counts.size()) == comm_size);
    }

    if (coll_param.ctype == ccl_coll_sparse_allreduce) {
        coll_param.sparse_param.send_ind_buf = param.sparse_param.send_ind_buf;
        coll_param.sparse_param.send_val_buf = param.sparse_param.send_val_buf;
        coll_param.sparse_param.recv_ind_buf = param.sparse_param.recv_ind_buf;
        coll_param.sparse_param.recv_val_buf = param.sparse_param.recv_val_buf;
    }

    if (ccl::global_data::env().priority_mode == ccl_priority_direct) {
        coll_attr.priority = attr.priority;
    }
}

size_t ccl_sched_base::get_priority() const {
    size_t priority = 0;

    switch (ccl::global_data::env().priority_mode) {
        case ccl_priority_none: priority = 0; break;
        case ccl_priority_direct:
        case ccl_priority_lifo: priority = coll_attr.priority; break;
        default:
            CCL_FATAL("unexpected priority_mode ", ccl::global_data::env().priority_mode);
            break;
    }

    LOG_DEBUG("sched, ", this, ", priority ", priority);

    return priority;
}

void* ccl_sched_base::alloc_buffer_unmanaged(size_t bytes, ccl_sched_buf_type buf_type) {
    LOG_DEBUG("try to allocate buffer size: ", bytes);
    CCL_THROW_IF_NOT(bytes > 0, "incorrect buffer size: ", bytes);

    void* ptr = nullptr;
    if (buf_type == ccl_sched_buf_system) {
        ccl::global_data::get().buffer_cache->get(sched_id, bytes, &ptr);
    }
#ifdef CCL_ENABLE_SYCL
    else if (buf_type == ccl_sched_buf_runtime) {
        CCL_THROW_IF_NOT(coll_param.stream, "null stream");
        sycl::context ctx = coll_param.stream->get_native_stream().get_context();
        ccl::global_data::get().buffer_cache->get(sched_id, bytes, ctx, &ptr);
    }
#endif // CCL_ENABLE_SYCL
    else {
        CCL_THROW("unexpected buf_type ", buf_type);
    }

    LOG_DEBUG("allocated buffer: ", ptr, ", size: ", bytes);
    return ptr;
}

void ccl_sched_base::free_buffer_unmanaged(void* ptr, size_t bytes, ccl_sched_buf_type buf_type) {
    LOG_DEBUG("free buffer: ", ptr, ", buf_type: ", buf_type);

    if (buf_type == ccl_sched_buf_system) {
        ccl::global_data::get().buffer_cache->push(sched_id, bytes, ptr);
    }
#ifdef CCL_ENABLE_SYCL
    else if (buf_type == ccl_sched_buf_runtime) {
        CCL_THROW_IF_NOT(coll_param.stream, "null stream");
        sycl::context ctx = coll_param.stream->get_native_stream().get_context();
        ccl::global_data::get().buffer_cache->push(sched_id, bytes, ctx, ptr);
    }
#endif // CCL_ENABLE_SYCL
    else {
        CCL_THROW("unexpected buf_type ", buf_type);
    }
}

ccl_buffer ccl_sched_base::alloc_buffer(size_t bytes, ccl_sched_buf_type buf_type) {
    ccl_buffer buffer =
        ccl_buffer(alloc_buffer_unmanaged(bytes, buf_type), bytes, 0, ccl_buffer_type::DIRECT);

    if (buf_type == ccl_sched_buf_system) {
        memory.buf_list.emplace_back(buffer, bytes);
    }
#ifdef CCL_ENABLE_SYCL
    else if (buf_type == ccl_sched_buf_runtime) {
        CCL_THROW_IF_NOT(coll_param.stream, "null stream");
        sycl::context ctx = coll_param.stream->get_native_stream().get_context();
        memory.sycl_buf_list.emplace_back(buffer, bytes, ctx);
        LOG_DEBUG(
            "allocated host usm buffer ptr: ", buffer.get_ptr(), ", size: ", buffer.get_size());
    }
#endif // CCL_ENABLE_SYCL

    CCL_THROW_IF_NOT(buffer.get_ptr(), "null ptr");
    return buffer;
}

#ifdef CCL_ENABLE_SYCL
ccl_buffer ccl_sched_base::alloc_staging_buffer(size_t bytes) {
    LOG_DEBUG("try to allocate usm host buffer size: ", bytes);
    CCL_THROW_IF_NOT(bytes > 0, "incorrect buffer size: ", bytes);

    ccl_sched_buf_type buf_type = ccl_sched_buf_system;
    if (ccl::global_data::env().staging_buffer == ccl_staging_usm) {
        buf_type = ccl_sched_buf_runtime;
    }
    ccl_buffer buffer = alloc_buffer(bytes, buf_type);

    CCL_THROW_IF_NOT(buffer.get_ptr(), "null ptr");

    return buffer;
}
#endif // CCL_ENABLE_SYCL

void ccl_sched_base::free_memory() {
    std::list<ccl_sched_buffer_handler>::iterator it;
    for (it = memory.buf_list.begin(); it != memory.buf_list.end(); it++) {
        free_buffer_unmanaged(it->buffer.get_ptr(), it->size, ccl_sched_buf_system);
    }
    memory.buf_list.clear();

    free_memory_regions();

#ifdef CCL_ENABLE_SYCL
    std::list<ccl_sched_sycl_buffer_handler>::iterator sycl_it;
    for (sycl_it = memory.sycl_buf_list.begin(); sycl_it != memory.sycl_buf_list.end(); sycl_it++) {
        LOG_DEBUG("free host usm ", sycl_it->buffer.get_ptr());
        ccl::global_data::get().buffer_cache->push(
            sched_id, sycl_it->size, sycl_it->ctx, sycl_it->buffer.get_ptr());
    }
    memory.sycl_buf_list.clear();

#ifdef MULTI_GPU_SUPPORT
    memory.handle_manager.clear();
#endif // MULTI_GPU_SUPPORT

#endif // CCL_ENABLE_SYCL
}

ccl_buffer ccl_sched_base::update_buffer(ccl_buffer buffer, size_t new_size) {
    LOG_DEBUG("update pointer data size: ",
              buffer.get_ptr(),
              ", from: ",
              buffer.get_size(),
              ", to: ",
              new_size);
    CCL_THROW_IF_NOT(new_size > 0, "incorrect buffer size: ", new_size);

    /* in case old_ptr will be freed */
    void* aux_ptr = buffer.get_ptr();

    ccl_buffer new_buf = ccl_buffer(
        CCL_REALLOC(
            buffer.get_ptr(), (size_t)buffer.get_size(), new_size, CACHELINE_SIZE, "sched_buffer"),
        new_size,
        0,
        ccl_buffer_type::DIRECT);
    bool updated = false;
    for (auto& it : memory.buf_list) {
        if (it.buffer.get_ptr() == aux_ptr) {
            /* assign ptr unconditionally, because realloc can return the same pointer */
            it.buffer = new_buf;
            it.size = new_size;
            updated = true;
            break;
        }
    }

    CCL_THROW_IF_NOT(updated, "Cannot update memory in buf_list for addres: ", new_buf.get_ptr());
    return new_buf;
}

ccl_buffer ccl_sched_base::find_and_realloc_buffer(void* in_ptr,
                                                   size_t new_size,
                                                   size_t expected_size) {
    LOG_DEBUG("sched: ", this, ", contains buffer objects: ", memory.buf_list.size());
    for (auto& it : memory.buf_list) {
        if (it.buffer.get_ptr() == in_ptr) {
#ifdef ENABLE_DEBUG_SPARSE
            if (expected_size != 0 && (it.buffer.get_size() < expected_size)) {
                std::stringstream ss;
                ss << "Unexpected realloc buffer by pointer: " << in_ptr
                   << ", cur size: " << it.buffer.get_size() << ", to: " << new_size
                   << ", expected: " << expected_size;
                ss << "\nbuffers:\n";
                for (const auto& it : memory.buf_list) {
                    ss << it.buffer << ", ";
                }
                LOG_ERROR(ss.str());
                CCL_ASSERT(false, ss.str());
                CCL_THROW_IF_NOT(
                    false, "Cannot fin buffer by ptr: ", in_ptr, ", available buffers: ", ss.str());
            }
#endif //ENABLE_DEBUG_SPARSE
            if ((it.buffer.get_size() < 0) ||
                (static_cast<size_t>(it.buffer.get_size()) < new_size)) {
                LOG_DEBUG("try to realloc buffer by pointer: ",
                          in_ptr,
                          ", from: ",
                          it.buffer.get_size(),
                          ", to: ",
                          new_size,
                          ", expected: ",
                          expected_size);

                it.buffer = ccl_buffer(CCL_REALLOC(in_ptr,
                                                   (size_t)it.buffer.get_size(),
                                                   new_size,
                                                   CACHELINE_SIZE,
                                                   "sched_buffer"),
                                       new_size,
                                       0,
                                       ccl_buffer_type::DIRECT);
                it.size = new_size;
            }
            return it.buffer;
        }
    }

    /* throw expection */
    std::stringstream ss;
    for (const auto& it : memory.buf_list) {
        ss << it.buffer << ", ";
    }
    CCL_THROW_IF_NOT(
        false, "cannot find buffer by ptr: ", in_ptr, ", available buffers: ", ss.str());
    return ccl_buffer();
}

void ccl_sched_base::add_memory_region(atl_mr_t* mr) {
    CCL_THROW_IF_NOT(mr);
    memory.mr_list.emplace_back(mr);
}

void ccl_sched_base::free_memory_regions() {
    if (memory.mr_list.empty()) {
        return;
    }

    /* perform deregistration in worker thread */

    ccl_coll_param param{};
    param.ctype = ccl_coll_internal;
    param.comm = coll_param.comm;
    std::unique_ptr<ccl_extra_sched> dereg_sched(new ccl_extra_sched(param, sched_id));
    entry_factory::make_entry<deregister_entry>(dereg_sched.get(), memory.mr_list, param.comm);

    if (ccl::global_data::get().is_worker_thread || !ccl::global_data::env().worker_offload) {
        dereg_sched->do_progress();
    }
    else {
        CCL_THROW("unsupported path");
        /* release ownership, because ccl_wait_impl use delete inside */
        // ccl_wait_impl<ccl_extra_sched>(
        //     ccl::global_data::get().executor.get(),
        //     start_subsched(dereg_sched.release()));
    }

    if (!memory.mr_list.empty()) {
        LOG_ERROR("memory region list is not empty after deregister_entry completion");
    }
}

void ccl_sched_base::get_pre_post_copy_counts(std::vector<size_t>& d2h_counts,
                                              std::vector<size_t>& h2d_counts,
                                              bool& reuse_buffers) {
    ccl_coll_param& param = coll_param;

    d2h_counts.clear();
    h2d_counts.clear();
    reuse_buffers = false;

    switch (param.ctype) {
        case ccl_coll_allgatherv:
            d2h_counts.push_back(param.get_send_count());
            if (param.recv_bufs.size() > 1) {
                h2d_counts.insert(
                    h2d_counts.end(), param.recv_counts.begin(), param.recv_counts.end());
            }
            else {
                h2d_counts.push_back(
                    std::accumulate(param.recv_counts.begin(), param.recv_counts.end(), 0));
            }
            break;
        case ccl_coll_allreduce:
            d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count());
            /* use in-place to avoid allocation of extra staging buffer*/
            reuse_buffers = true;
            break;
        case ccl_coll_alltoall:
            d2h_counts.push_back(param.get_send_count() * param.comm->size());
            h2d_counts.push_back(param.get_recv_count() * param.comm->size());
            break;
        case ccl_coll_alltoallv:
            if (param.recv_bufs.size() > 1) {
                /* expect that is_vector_buf is enabled for send/recv both */
                d2h_counts.insert(
                    d2h_counts.end(), param.send_counts.begin(), param.send_counts.end());
                h2d_counts.insert(
                    h2d_counts.end(), param.recv_counts.begin(), param.recv_counts.end());
            }
            else {
                d2h_counts.push_back(
                    std::accumulate(param.send_counts.begin(), param.send_counts.end(), 0));
                h2d_counts.push_back(
                    std::accumulate(param.recv_counts.begin(), param.recv_counts.end(), 0));
            }
            break;
        case ccl_coll_bcast:
            if (param.comm->rank() == param.root)
                d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count());
            reuse_buffers = true;
            break;
        case ccl_coll_reduce:
            d2h_counts.push_back(param.get_send_count());
            if (param.comm->rank() == param.root)
                h2d_counts.push_back(param.get_recv_count());
            break;
        case ccl_coll_reduce_scatter:
            d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count());
            break;
        case ccl_coll_sparse_allreduce:
            CCL_FATAL("SYCL stream is not supported for sparse_allreduce yet");
            CCL_ASSERT(0);
            break;
        default: break;
    }
}

void ccl_sched_base::alloc_buffers_for_pre_post_copy() {
#ifdef CCL_ENABLE_SYCL

    ccl_coll_param& param = coll_param;

    param.device_send_bufs.clear();
    param.device_recv_bufs.clear();

    // TODO: WA skip sycl pre_post_copy for allreduce gpu algo
    ccl_selector_param selector_param;
    selector_param.ctype = param.ctype;
    selector_param.count = param.get_send_count();
    selector_param.dtype = param.dtype;
    selector_param.comm = param.comm;
    selector_param.stream = param.stream;
    selector_param.is_sycl_buf = coll_attr.is_sycl_buf;

    if (!param.stream || !param.stream->is_sycl_device_stream() ||
        ccl_is_topo_ring_algo(selector_param)) {
        return;
    }

    bool should_alloc_buffers = true;

    if (!coll_attr.is_sycl_buf) {
        auto bufs = param.get_all_non_zero_bufs();
        if (!bufs.empty()) {
            auto usm_type =
                sycl::get_pointer_type(bufs[0], param.stream->get_native_stream().get_context());
            if ((usm_type == sycl::usm::alloc::host) || (usm_type == sycl::usm::alloc::shared) ||
                ((usm_type == sycl::usm::alloc::device) && atl_wrapper::attr.out.enable_hmem)) {
                should_alloc_buffers = false;
            }
        }
    }

    LOG_DEBUG("coll_type ", param.ctype, ", should_alloc_buffers ", should_alloc_buffers);

    if (!should_alloc_buffers) {
        return;
    }

    /*
        move user-supplied pointers into device_* fields
        they will be used further for pre-post copies
    */
    param.device_send_bufs = param.send_bufs;
    param.device_recv_bufs = param.recv_bufs;

    std::vector<size_t> d2h_counts;
    std::vector<size_t> h2d_counts;
    bool reuse_buffers;
    get_pre_post_copy_counts(d2h_counts, h2d_counts, reuse_buffers);

    LOG_DEBUG("alloc tmp buffers for D2H and H2D copies, coll_type ",
              ccl_coll_type_to_str(param.ctype),
              ", dtype_size ",
              param.dtype.size(),
              ", comm_size ",
              param.comm->size(),
              ", d2h_counts_size ",
              d2h_counts.size(),
              ", h2d_counts_size ",
              h2d_counts.size(),
              ", reuse_buffers ",
              reuse_buffers);

    if (reuse_buffers) {
        /* keep only single vector with counts */
        if (d2h_counts.size() < h2d_counts.size())
            d2h_counts = h2d_counts;
        h2d_counts.clear();
    }

    for (size_t idx = 0; idx < d2h_counts.size(); idx++) {
        if (d2h_counts[idx])
            param.send_bufs[idx] =
                alloc_staging_buffer(d2h_counts[idx] * param.dtype.size()).get_ptr();
        else
            param.send_bufs[idx] = nullptr;
    }

    for (size_t idx = 0; idx < h2d_counts.size(); idx++) {
        if (h2d_counts[idx])
            param.recv_bufs[idx] =
                alloc_staging_buffer(h2d_counts[idx] * param.dtype.size()).get_ptr();
        else
            param.recv_bufs[idx] = nullptr;
    }

    if (reuse_buffers) {
        param.recv_bufs = param.send_bufs;
    }

    CCL_THROW_IF_NOT(param.send_bufs.size() == param.device_send_bufs.size(),
                     "send_bufs.size() mismatch: ",
                     param.send_bufs.size(),
                     " vs ",
                     param.device_send_bufs.size());

    CCL_THROW_IF_NOT(param.recv_bufs.size() == param.device_recv_bufs.size(),
                     "recv_bufs.size() mismatch: ",
                     param.recv_bufs.size(),
                     " vs ",
                     param.device_recv_bufs.size());

#endif // CCL_ENABLE_SYCL
}

void ccl_sched_base::update_id() {
    sched_id = coll_param.comm->get_sched_id(internal_type != ccl_sched_internal_none);
}

void ccl_sched_base::dump(std::ostream& out, const char* name) const {
    ccl_logger::format(out, "\n-----------------", name, "---------------\n");
    ccl_logger::format(out,
                       "sched: ",
                       this,
                       ", coll ",
                       ccl_coll_type_to_str(coll_param.ctype),
                       ", comm_id ",
                       std::dec,
                       coll_param.comm->id(),
                       ", sched_id ",
                       sched_id);
}
