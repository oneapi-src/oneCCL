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

class sycl_copy_host_to_device_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "SYCL_COPY_H2D";
    }

    sycl_copy_host_to_device_entry() = delete;
    sycl_copy_host_to_device_entry(ccl_sched* sched,
                                   ccl_buffer in_buf,
                                   ccl_buffer out_buf,
                                   size_t cnt,
                                   const ccl_datatype& dtype,
                                   const ccl_stream* stream)
            : sched_entry(sched),
              in_buf(in_buf),
              out_buf(out_buf),
              cnt(cnt),
              dtype(dtype),
              stream(stream) {}

    void start() override {
        //fill visitor with actual ccl_buffer data
        auto visitor = make_visitor<cl::sycl::access::mode::write>(
            dtype, cnt, 0, out_buf, [this](void* sycl_pointer, size_t bytes) {
                auto comp_status = ccl_comp_copy(in_buf.get_ptr(bytes), sycl_pointer, cnt, dtype);
                CCL_ASSERT(comp_status == ccl_status_success, "bad status ", comp_status);
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
};
