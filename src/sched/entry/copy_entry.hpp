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
               size_t cnt,
               const ccl_datatype& dtype)
            : sched_entry(sched),
              in_buf(in_buf),
              out_buf(out_buf),
              cnt(cnt),
              dtype(dtype) {}

    void start() override {
        update_fields();

        size_t bytes = cnt * dtype.size();
        auto comp_status = ccl_comp_copy(in_buf.get_ptr(bytes), out_buf.get_ptr(bytes), cnt, dtype);
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
        return cnt;
    }

    ccl_datatype& get_field_ref(field_id_t<ccl_sched_entry_field_dtype> id) {
        return dtype;
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", cnt ",
                           cnt,
                           ", in_buf ",
                           in_buf,
                           ", out_buf ",
                           out_buf,
                           "\n");
    }

private:
    ccl_buffer in_buf;
    ccl_buffer out_buf;
    size_t cnt;
    ccl_datatype dtype;
};
