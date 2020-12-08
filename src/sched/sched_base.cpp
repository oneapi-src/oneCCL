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
#include "sched/sched_base.hpp"
#include "common/global/global.hpp"

std::string to_string(ccl_sched_add_mode mode) {
    switch (mode) {
        case ccl_sched_add_front: return "FRONT";
        case ccl_sched_add_back: return "BACK";
        default: return "DEFAULT";
    }
    return "DEFAULT";
}

void ccl_sched_base::set_coll_attr(const ccl_coll_attr& attr) {
    coll_attr = attr;
}

void ccl_sched_base::update_coll_param_and_attr(const ccl_coll_param& param,
                                                const ccl_coll_attr& attr) {
#ifdef CCL_ENABLE_SYCL
    if (param.stream && param.stream->is_sycl_device_stream()) {
        coll_param.sycl_buf = static_cast<ccl_sycl_buffer_t*>(param.buf);
        coll_param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)param.send_buf);
        coll_param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
    }
    else {
#endif /* CCL_ENABLE_SYCL */
        coll_param.buf = param.buf;
        coll_param.send_buf = param.send_buf;
        coll_param.recv_buf = param.recv_buf;
#ifdef CCL_ENABLE_SYCL
    }
#endif /* CCL_ENABLE_SYCL */

    if (coll_param.ctype == ccl_coll_allgatherv) {
        coll_param.recv_counts = param.recv_counts;
        CCL_THROW_IF_NOT((int)coll_param_copy.ag_recv_counts.size() == coll_param.comm->size());
        coll_param_copy.ag_recv_counts.assign((size_t*)param.recv_counts,
                                              (size_t*)param.recv_counts + coll_param.comm->size());

        if (coll_attr.vector_buf) {
            CCL_THROW_IF_NOT((int)coll_param_copy.ag_recv_bufs.size() == coll_param.comm->size());
            coll_param_copy.ag_recv_bufs.assign((void**)param.recv_buf,
                                                (void**)param.recv_buf + coll_param.comm->size());
        }
    }

    if (coll_param.ctype == ccl_coll_alltoallv) {
        coll_param.send_counts = param.send_counts;
        coll_param.recv_counts = param.recv_counts;

        CCL_THROW_IF_NOT((int)coll_param_copy.a2av_send_counts.size() == coll_param.comm->size());
        CCL_THROW_IF_NOT((int)coll_param_copy.a2av_recv_counts.size() == coll_param.comm->size());

        coll_param_copy.a2av_send_counts.assign(
            (size_t*)param.send_counts, (size_t*)param.send_counts + coll_param.comm->size());
        coll_param_copy.a2av_recv_counts.assign(
            (size_t*)param.recv_counts, (size_t*)param.recv_counts + coll_param.comm->size());
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

ccl_buffer ccl_sched_base::alloc_buffer(size_t bytes) {
    LOG_DEBUG("try to allocate buffer size: ", bytes);
    CCL_THROW_IF_NOT(bytes > 0, "incorrect buffer size: ", bytes);

    ccl_buffer buffer =
        ccl_buffer(CCL_CALLOC(bytes, "sched_buffer"), bytes, 0, ccl_buffer_type::DIRECT);
    memory.buf_list.emplace_back(buffer, bytes);
    CCL_THROW_IF_NOT(buffer.get_ptr(), "null ptr");

    LOG_DEBUG("allocated buffer ptr: ", buffer.get_ptr(), ", size: ", buffer.get_size());
    return buffer;
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

void ccl_sched_base::free_buffers() {
    std::list<ccl_sched_buffer_handler>::iterator it;
    for (it = memory.buf_list.begin(); it != memory.buf_list.end(); it++) {
        LOG_DEBUG("free ", it->buffer.get_ptr());
        CCL_FREE(it->buffer.get_ptr());
    }
    memory.buf_list.clear();
}

void ccl_sched_base::add_memory_region(atl_mr_t* mr) {
    CCL_THROW_IF_NOT(mr);
    memory.mr_list.emplace_back(mr);
}

void ccl_sched_base::alloc_buffers_for_sycl_copy() {
#ifdef CCL_ENABLE_SYCL

    ccl_coll_param& param = coll_param;
    if (!param.stream || (!param.stream->is_sycl_device_stream()))
        return;

    LOG_DEBUG("alloc tmp buffers for D2H and H2D copies, coll_type ",
              ccl_coll_type_to_str(param.ctype),
              ", dtype_size ",
              param.dtype.size(),
              ", comm_size ",
              param.comm->size(),
              ", count ",
              param.count);

    size_t idx, send_count = 0, recv_count = 0;

    switch (param.ctype) {
        case ccl_coll_allgatherv:
            param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)param.send_buf);
            param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
            param.send_buf = alloc_buffer(param.send_count * param.dtype.size()).get_ptr();
            for (idx = 0; idx < param.comm->size(); idx++)
                recv_count += param.recv_counts[idx];
            param.recv_buf = alloc_buffer(recv_count * param.dtype.size()).get_ptr();
            break;
        case ccl_coll_allreduce:
            param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)param.send_buf);
            param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
            param.send_buf = alloc_buffer(param.count * param.dtype.size()).get_ptr();
            param.recv_buf = alloc_buffer(param.count * param.dtype.size()).get_ptr();
            break;
        case ccl_coll_alltoall:
            param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)param.send_buf);
            param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
            param.send_buf =
                alloc_buffer(param.count * param.dtype.size() * param.comm->size()).get_ptr();
            param.recv_buf =
                alloc_buffer(param.count * param.dtype.size() * param.comm->size()).get_ptr();
            break;
        case ccl_coll_alltoallv:
            param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)param.send_buf);
            param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
            for (idx = 0; idx < param.comm->size(); idx++) {
                send_count += param.send_counts[idx];
                recv_count += param.recv_counts[idx];
            }
            param.send_buf = alloc_buffer(send_count * param.dtype.size()).get_ptr();
            param.recv_buf = alloc_buffer(recv_count * param.dtype.size()).get_ptr();
            break;
        case ccl_coll_bcast:
            param.sycl_buf = static_cast<ccl_sycl_buffer_t*>(param.buf);
            param.buf = alloc_buffer(param.count * param.dtype.size()).get_ptr();
            break;
        case ccl_coll_reduce:
            param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)(param.send_buf));
            param.send_buf = alloc_buffer(param.count * param.dtype.size()).get_ptr();
            if (param.comm->rank() == param.root) {
                param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
                param.recv_buf = alloc_buffer(param.count * param.dtype.size()).get_ptr();
            }
            else {
                param.recv_buf = nullptr;
            }
            break;
        case ccl_coll_reduce_scatter:
            param.sycl_send_buf = static_cast<ccl_sycl_buffer_t*>((void*)param.send_buf);
            param.sycl_recv_buf = static_cast<ccl_sycl_buffer_t*>(param.recv_buf);
            param.send_buf =
                alloc_buffer(param.count * param.comm->size() * param.dtype.size()).get_ptr();
            param.recv_buf = alloc_buffer(param.count * param.dtype.size()).get_ptr();
            break;
        case ccl_coll_sparse_allreduce:
            CCL_FATAL("SYCL stream is not supported for sparse_allreduce yet");
            CCL_ASSERT(0);
            break;
        default: break;
    }
#endif /* CCL_ENABLE_SYCL */
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
