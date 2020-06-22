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

class sparse_callback_entry : public sched_entry,
                       public postponed_fields<sparse_callback_entry,
                                               ccl_sched_entry_field_idx_buf,
                                               ccl_sched_entry_field_idx_cnt,
                                               ccl_sched_entry_field_val_buf,
                                               ccl_sched_entry_field_val_cnt>
{
public:
    static constexpr const char* class_name() noexcept
    {
        return "SPARSE_CALLBACK";
    }

    sparse_callback_entry() = delete;
    sparse_callback_entry(ccl_sched* sched,
                          ccl_sparse_allreduce_completion_fn_t fn,
                          const ccl_buffer i_buf,
                          size_t i_cnt,
                          const ccl_datatype& i_dtype,
                          const ccl_buffer v_buf,
                          size_t v_cnt,
                          const ccl_datatype& v_dtype,
                          const void* user_ctx) :
        sched_entry(sched), fn(fn), 
        i_buf(i_buf), i_cnt(i_cnt), i_dtype(i_dtype),
        v_buf(v_buf), v_cnt(v_cnt), v_dtype(v_dtype),
        user_ctx(user_ctx)
    {
    }

    void start() override
    {
        update_fields();

        size_t i_bytes = i_cnt * i_dtype.size();
        size_t v_bytes = v_cnt * v_dtype.size();
        const ccl_fn_context_t fn_ctx = { sched->coll_attr.match_id.c_str(), 0 };
        fn(i_buf.get_ptr(i_bytes), i_cnt, i_dtype.idx(), v_buf.get_ptr(v_bytes), v_cnt, v_dtype.idx(), &fn_ctx, user_ctx);
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
                           "i_dt ", global_data.dtypes->name(i_dtype),
                           ", i_cnt ", i_cnt,
                           ", i_buf ", i_buf,
                           ", v_dt ", global_data.dtypes->name(v_dtype),
                           ", v_cnt ", v_cnt,
                           ", v_buf ", v_buf,
                           ", fn ", fn,
                           ", v_count ", v_cnt,
                           "\n");
    }

private:
    ccl_sparse_allreduce_completion_fn_t fn;
    ccl_buffer i_buf;
    size_t i_cnt;
    ccl_datatype i_dtype;
    ccl_buffer v_buf;
    size_t v_cnt;
    ccl_datatype v_dtype;
    const void* user_ctx;
};
