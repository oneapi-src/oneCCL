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

#include "sched/entry/entry.hpp"
#include "sched/entry/sycl_entry_helper.hpp"

#include <CL/sycl.hpp>

class sycl_copy_device_to_host_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "SYCL_COPY_D2H";
    }

    sycl_copy_device_to_host_entry() = delete;
    sycl_copy_device_to_host_entry(ccl_sched* sched,
                                   ccl_buffer in_buf,
                                   ccl_buffer out_buf,
                                   size_t cnt,
                                   const ccl_datatype& dtype,
                                   const ccl_stream* stream,
                                   size_t offset = 0)
            : sched_entry(sched),
              in_buf(in_buf),
              out_buf(out_buf),
              cnt(cnt),
              dtype(dtype),
              stream(stream),
              offset(offset) {}

    void start() override {

        // LOG_DEBUG(class_name(), ": in_buf ", in_buf, ", out_buf ", out_buf, ", cnt ", cnt);
        // cl::sycl::usm::alloc usm_kind = get_pointer_type(in_buf, stream.get().get_context());
        // CCL_THROW_IF_NOT(usm_kind == cl::sycl::usm::alloc::shared, "usm_kind should be shared");

        //fill visitor with actual ccl_buffer data
        auto visitor = make_reader_visitor<cl::sycl::access::mode::read>(
            dtype, cnt, offset, in_buf, stream->get_native_stream(), std::ref(out_buf), [this](char* sycl_pointer, size_t bytes) {
                (void)sycl_pointer;
                (void)bytes;
                (void)this;
                // TODO remove function callback
            });

        ccl_tuple_for_each_indexed<ccl_sycle_buffer_one_dim_types>(visitor);
        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "  dtype ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", cnt ",
                           cnt,
                           ", in_buf ",
                           in_buf,
                           ", out_buf ",
                           out_buf,
                           ", native_stream ",
                           stream->to_string(),
                           "\n");
    }

private:
    ccl_buffer in_buf;
    ccl_buffer out_buf;
    size_t cnt;
    ccl_datatype dtype;
    const ccl_stream* stream;
    size_t offset;
};
