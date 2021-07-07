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
#pragma once

#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/entry.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif /* CCL_ENABLE_SYCL */

class copy_entry : public sched_entry,
                   public postponed_fields<copy_entry,
                                           ccl_sched_entry_field_in_buf,
                                           ccl_sched_entry_field_cnt,
                                           ccl_sched_entry_field_dtype> {
public:
    static constexpr const char* class_name() noexcept {
        return "COPY";
    }

    copy_entry() = delete;
    copy_entry(ccl_sched* sched,
               const ccl_buffer in_buf,
               ccl_buffer out_buf,
               size_t count,
               const ccl_datatype& dtype,
               size_t in_buf_offset = 0)
            : sched_entry(sched),
              in_buf(in_buf),
              out_buf(out_buf),
              count(count),
              dtype(dtype),
              in_buf_offset(in_buf_offset) {}

    void start() override {
        update_fields();

#ifdef CCL_ENABLE_SYCL
        ccl_stream* stream = (ccl_stream*)sched->coll_param.stream;

        if (!stream) {
            do_regular_copy();
            return;
        }

        sycl::queue* q = stream->get_native_stream(sched->queue->get_idx());
        CCL_THROW_IF_NOT(q, "null sycl queue");
        auto in_ptr_type = sycl::get_pointer_type(in_buf.get_ptr(), q->get_context());
        auto out_ptr_type = sycl::get_pointer_type(out_buf.get_ptr(), q->get_context());

        LOG_DEBUG("in_ptr_type: ",
                  native::detail::usm_to_string(in_ptr_type),
                  ", out_ptr_type: ",
                  native::detail::usm_to_string(out_ptr_type),
                  ", native_stream: ",
                  stream->to_string(),
                  ", count: ",
                  count)

        if ((in_ptr_type != sycl::usm::alloc::device) &&
            (out_ptr_type != sycl::usm::alloc::device)) {
            do_regular_copy();
            return;
        }

        copy_direction direction;

        if ((in_ptr_type == sycl::usm::alloc::device) &&
            (out_ptr_type == sycl::usm::alloc::device)) {
            direction = copy_direction::d2d;
        }

        if ((in_ptr_type == sycl::usm::alloc::host) && (out_ptr_type == sycl::usm::alloc::device)) {
            direction = copy_direction::h2d;
        }

        if ((in_ptr_type == sycl::usm::alloc::device) && (out_ptr_type == sycl::usm::alloc::host)) {
            direction = copy_direction::d2h;
        }

        copier = sycl_copier(direction, in_buf, out_buf, count, dtype, 0);
        copier.set_queue(q);
        ccl_tuple_for_each_indexed<ccl_sycl_buffer_one_dim_types>(copier);
        status = ccl_sched_entry_status_started;
#else /* CCL_ENABLE_SYCL */
        do_regular_copy();
#endif /* CCL_ENABLE_SYCL */
    }

    void update() override {
#ifdef CCL_ENABLE_SYCL
        if (copier.is_completed()) {
            status = ccl_sched_entry_status_complete;
        }
#endif /* CCL_ENABLE_SYCL */
    }

    void do_regular_copy() {
        size_t bytes = count * dtype.size();
        auto comp_status =
            ccl_comp_copy(in_buf.get_ptr(bytes), out_buf.get_ptr(bytes), count, dtype);
        CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);
        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_in_buf> id) {
        return in_buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id) {
        return count;
    }

    ccl_datatype& get_field_ref(field_id_t<ccl_sched_entry_field_dtype> id) {
        return dtype;
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", count ",
                           count,
                           ", in_buf ",
                           in_buf,
                           ", out_buf ",
                           out_buf,
                           ", in_buf_offset ",
                           in_buf_offset,
                           "\n");
    }

private:
    ccl_buffer in_buf;
    ccl_buffer out_buf;
    size_t count;
    ccl_datatype dtype;
    size_t in_buf_offset;

#ifdef CCL_ENABLE_SYCL
    sycl_copier copier;
#endif /* CCL_ENABLE_SYCL */
};
