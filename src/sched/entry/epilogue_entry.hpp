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

class epilogue_entry : public sched_entry,
                       public postponed_fields<epilogue_entry,
                                               ccl_sched_entry_field_in_buf,
                                               ccl_sched_entry_field_in_cnt,
                                               ccl_sched_entry_field_in_dtype>
{
public:
    static constexpr const char* class_name() noexcept
    {
        return "EPILOGUE";
    }

    epilogue_entry() = delete;
    epilogue_entry(ccl_sched* sched,
                   ccl_epilogue_fn_t fn,
                   const ccl_buffer in_buf,
                   size_t in_cnt,
                   ccl_datatype_internal_t in_dtype,
                   ccl_buffer out_buf,
                   size_t expected_out_cnt,
                   ccl_datatype_internal_t out_dtype) :
        sched_entry(sched), fn(fn), in_buf(in_buf),
        in_cnt(in_cnt), in_dtype(in_dtype),
        out_buf(out_buf), expected_out_cnt(expected_out_cnt),
        out_dtype(out_dtype)
    {
    }

    void start() override
    {
        update_fields();

        size_t in_bytes = in_cnt * ccl_datatype_get_size(in_dtype);
        size_t offset = in_buf.get_offset();
        const ccl_fn_context_t context = { sched->coll_attr.match_id.c_str(), offset };
        fn(in_buf.get_ptr(in_bytes), in_cnt, in_dtype->type, out_buf.get_ptr(), &out_cnt, &context, out_dtype->type);
        CCL_ASSERT(expected_out_cnt == out_cnt, "incorrect values ", expected_out_cnt, " ", out_cnt);
        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override
    {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_in_buf> id)
    {
        return in_buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_in_cnt> id)
    {
        return in_cnt;
    }

    ccl_datatype_internal_t& get_field_ref(field_id_t<ccl_sched_entry_field_in_dtype> id)
    {
        return in_dtype;
    }

protected:
    void dump_detail(std::stringstream& str) const override
    {
        ccl_logger::format(str,
                           "in_dt ", ccl_datatype_get_name(in_dtype),
                           ", in_cnt ", in_cnt,
                           ", in_buf ", in_buf,
                           ", out_dt ", ccl_datatype_get_name(out_dtype),
                           ", out_cnt ", out_cnt,
                           ", out_buf ", out_buf,
                           ", fn ", fn,
                           ", exp_out_count ", expected_out_cnt,
                           "\n");
    }

private:
    ccl_epilogue_fn_t fn;
    ccl_buffer in_buf;
    size_t in_cnt;
    ccl_datatype_internal_t in_dtype;
    ccl_buffer out_buf;
    size_t out_cnt;
    size_t expected_out_cnt;
    ccl_datatype_internal_t out_dtype;
};
