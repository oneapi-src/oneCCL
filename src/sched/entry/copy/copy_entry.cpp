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
#include "sched/entry/copy/copy_entry.hpp"
#include "sched/queue/queue.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>
#endif // CCL_ENABLE_SYCL

copy_entry::copy_entry(ccl_sched* sched,
                       ccl_buffer in_buf,
                       ccl_buffer out_buf,
                       size_t count,
                       const ccl_datatype& dtype,
                       copy_attr attr)
        :
#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
          ze_copy_entry(sched, in_buf, out_buf, count, dtype, attr),
#else
          sched_entry(sched),
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
          sched(sched),
          in_buf(in_buf),
          out_buf(out_buf),
          count(count),
          dtype(dtype),
          attr(attr) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

void copy_entry::start() {
    //update_fields();

    LOG_DEBUG(class_name(), ": in_buf ", in_buf, ", out_buf ", out_buf, ", count ", count);

#ifdef CCL_ENABLE_SYCL
    int is_sycl_buf = sched->coll_attr.is_sycl_buf;
    sycl::queue* q = nullptr;

    sycl::usm::alloc in_ptr_type = sycl::usm::alloc::unknown;
    sycl::usm::alloc out_ptr_type = sycl::usm::alloc::unknown;

    if (sched->coll_param.stream) {
        q = sched->coll_param.stream->get_native_stream(sched->queue->get_idx());
        CCL_THROW_IF_NOT(q, "null sycl queue");
        in_ptr_type = sycl::get_pointer_type(in_buf.get_ptr(), q->get_context());
        out_ptr_type = sycl::get_pointer_type(out_buf.get_ptr(), q->get_context());

        LOG_DEBUG("in_ptr_type: ",
                  ccl::utils::usm_type_to_str(in_ptr_type),
                  ", out_ptr_type: ",
                  ccl::utils::usm_type_to_str(out_ptr_type));

        if (attr.direction == copy_direction::undefined) {
            if (in_ptr_type == sycl::usm::alloc::device &&
                out_ptr_type == sycl::usm::alloc::device) {
                attr.direction = copy_direction::d2d;
            }

            if ((in_ptr_type != sycl::usm::alloc::device) &&
                (out_ptr_type != sycl::usm::alloc::device)) {
                attr.direction = copy_direction::h2h;
            }

            if ((in_ptr_type == sycl::usm::alloc::device) &&
                (out_ptr_type != sycl::usm::alloc::device)) {
                attr.direction = copy_direction::d2h;
            }

            if ((in_ptr_type != sycl::usm::alloc::device) &&
                (out_ptr_type == sycl::usm::alloc::device)) {
                attr.direction = copy_direction::h2d;
            }

            CCL_THROW_IF_NOT(attr.direction != copy_direction::undefined);
        }
    }
#endif // CCL_ENABLE_SYCL

    LOG_DEBUG("count: ", count, ", direction: ", to_string(attr.direction));

    if (!sched->coll_param.stream || (attr.direction == copy_direction::h2h)) {
#ifdef CCL_ENABLE_SYCL
        CCL_THROW_IF_NOT(in_ptr_type != sycl::usm::alloc::device,
                         "unexpected device usm type for input buffer");
        CCL_THROW_IF_NOT(out_ptr_type != sycl::usm::alloc::device,
                         "unexpected device usm type for output buffer");
#endif // CCL_ENABLE_SYCL
        do_regular_copy();
        return;
    }

#ifdef CCL_ENABLE_SYCL
    if (q->get_backend() != cl::sycl::backend::level_zero || is_sycl_buf) {
        ctype = copy_type::sycl;
        if (!is_sycl_buf) {
            if ((in_ptr_type != sycl::usm::alloc::device) &&
                (out_ptr_type != sycl::usm::alloc::device)) {
                do_regular_copy();
                return;
            }
        }

        copier = sycl_copier(
            attr.direction, in_buf, out_buf, count, dtype, is_sycl_buf, attr.in_buf_offset);
        copier.set_queue(q);
        ccl_tuple_for_each_indexed<ccl_sycl_buffer_one_dim_types>(copier);
        status = ccl_sched_entry_status_started;
    }
#ifdef MULTI_GPU_SUPPORT
    else {
        ctype = copy_type::ze;
        ze_copy_entry::start(); // status
    }
#endif // MULTI_GPU_SUPPORT
#endif // CCL_ENABLE_SYCL
}

void copy_entry::update() {
#ifdef CCL_ENABLE_SYCL
    if (ctype == copy_type::sycl) {
        if (copier.is_completed()) {
            status = ccl_sched_entry_status_complete;
        }
    }
#ifdef MULTI_GPU_SUPPORT
    else {
        ze_copy_entry::update();
    }
#endif // MULTI_GPU_SUPPORT
#endif // CCL_ENABLE_SYCL
}

void copy_entry::do_regular_copy() {
    size_t bytes = dtype.size() * count;
    auto comp_status = ccl_comp_copy(in_buf.get_ptr(bytes), out_buf.get_ptr(bytes), count, dtype);
    CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);
    status = ccl_sched_entry_status_complete;
}

const ccl_buffer& copy_entry::get_field_ref(field_id_t<ccl_sched_entry_field_in_buf> id) {
    return in_buf;
}

const size_t& copy_entry::get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id) {
    return count;
}

const ccl_datatype& copy_entry::get_field_ref(field_id_t<ccl_sched_entry_field_dtype> id) {
    return dtype;
}
