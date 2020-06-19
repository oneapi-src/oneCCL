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

class sparse_allreduce_completion_entry : public sched_entry,
                       public postponed_fields<sparse_allreduce_completion_entry,
                                               ccl_sched_entry_field_idx_buf,
                                               ccl_sched_entry_field_idx_cnt,
                                               ccl_sched_entry_field_val_buf,
                                               ccl_sched_entry_field_val_cnt>
{
public:
    static constexpr const char* class_name() noexcept
    {
        return "SPARSE_ALLREDUCE_COMPLETION";
    }

    sparse_allreduce_completion_entry() = delete;
    sparse_allreduce_completion_entry(
      ccl_sched* sched,
      ccl_sparse_allreduce_completion_fn_t fn,
      const void* fn_ctx,
      const ccl_buffer i_buf,
      size_t i_cnt,
      const ccl_datatype& i_dtype,
      const ccl_buffer v_buf,
      size_t v_cnt,
      const ccl_datatype& v_dtype) :
        sched_entry(sched), fn(fn), fn_ctx(fn_ctx),
        i_buf(i_buf), i_cnt(i_cnt), i_dtype(i_dtype),
        v_buf(v_buf), v_cnt(v_cnt), v_dtype(v_dtype)
    {}

    void start() override
    {
        update_fields();

        size_t i_bytes = i_cnt * i_dtype.size();
        size_t v_bytes = v_cnt * v_dtype.size();

        fn(i_buf.get_ptr(i_bytes), i_cnt, i_dtype.idx(),
           v_buf.get_ptr(v_bytes), v_cnt, v_dtype.idx(),
           fn_ctx);

        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override
    {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_idx_buf> id)
    {
        return i_buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_idx_cnt> id)
    {
        return i_cnt;
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_val_buf> id)
    {
        return v_buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_val_cnt> id)
    {
        return v_cnt;
    }

protected:
    void dump_detail(std::stringstream& str) const override
    {
        ccl_logger::format(str,
                           ", i_buf ", i_buf,
                           ", i_cnt ", i_cnt,
                           "i_dt ", ccl::global_data::get().dtypes->name(i_dtype),
                           ", v_buf ", v_buf,
                           ", v_cnt ", v_cnt,
                           ", v_dt ", ccl::global_data::get().dtypes->name(v_dtype),
                           ", fn ", fn,
                           ", fn_ctx ", fn_ctx,
                           "\n");
    }

private:
    ccl_sparse_allreduce_completion_fn_t fn;
    const void* fn_ctx;
    ccl_buffer i_buf;
    size_t i_cnt;
    ccl_datatype i_dtype;
    ccl_buffer v_buf;
    size_t v_cnt;
    ccl_datatype v_dtype;
};
