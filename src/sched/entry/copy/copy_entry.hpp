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

#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
#include "sched/entry/gpu/ze_copy_entry.hpp"
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT

enum class copy_type : int { regular, sycl, ze };

#if defined(CCL_ENABLE_SYCL) && defined(MULTI_GPU_SUPPORT)
class copy_entry : public ze_copy_entry,
#else
class copy_entry : public sched_entry,
#endif // CCL_ENABLE_SYCL && MULTI_GPU_SUPPORT
                   public postponed_fields<copy_entry,
                                           ccl_sched_entry_field_in_buf,
                                           ccl_sched_entry_field_cnt,
                                           ccl_sched_entry_field_dtype> {
public:
    static constexpr const char* class_name() noexcept {
        return "COPY";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    copy_entry() = delete;
    copy_entry(ccl_sched* sched,
               ccl_buffer in_buf,
               ccl_buffer out_buf,
               size_t count,
               const ccl_datatype& dtype,
               copy_attr attr = {});

    void start() override;
    void update() override;

    const ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_in_buf> id);
    const size_t& get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id);
    const ccl_datatype& get_field_ref(field_id_t<ccl_sched_entry_field_dtype> id);

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
                           attr.in_buf_offset,
                           "\n");
    }

private:
    ccl_sched* const sched;
    ccl_buffer in_buf{};
    ccl_buffer out_buf{};
    const size_t count;
    const ccl_datatype dtype;
    copy_attr attr;
    copy_type ctype{ copy_type::regular };

#ifdef CCL_ENABLE_SYCL
    sycl_copier copier{};
#endif // CCL_ENABLE_SYCL

    void do_regular_copy();
};
