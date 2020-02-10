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

#include "common/global/global.hpp"
#include "sched/entry/entry.hpp"
#include "sched/queue/queue.hpp"

class recv_entry : public sched_entry,
                   public postponed_fields<recv_entry,
                                           ccl_sched_entry_field_buf,
                                           ccl_sched_entry_field_cnt>
{
public:
    static constexpr const char* class_name() noexcept
    {
        return "RECV";
    }

    recv_entry() = delete;
    recv_entry(ccl_sched* sched,
               ccl_buffer buf,
               size_t cnt,
               ccl_datatype_internal_t dtype,
               size_t src,
               ccl_comm* comm) :
        sched_entry(sched), buf(buf),
        cnt(cnt), dtype(dtype),
        src(src), comm(comm)
    {
    }

    ~recv_entry()
    {
        if (status == ccl_sched_entry_status_started)
        {
            size_t bytes = cnt * ccl_datatype_get_size(dtype);
            LOG_DEBUG("cancel RECV entry src ", src, ", req ", &req, ", bytes ", bytes);
            atl_comm_cancel(sched->bin->get_comm_ctx(), &req);
        }
    }

    void start() override
    {
        update_fields();

        size_t global_src = comm->get_global_rank(src);
        atl_tag = global_data.atl_tag->create(sched->get_comm_id(), global_src,
                                              sched->sched_id, sched->get_op_id());
        size_t bytes = cnt * ccl_datatype_get_size(dtype);

        LOG_DEBUG("RECV entry src ", global_src, ", tag ", atl_tag, ", req ", &req, ", bytes ", bytes);

        atl_status_t atl_status = atl_comm_recv(sched->bin->get_comm_ctx(), buf.get_ptr(bytes),
                                                bytes, global_src, atl_tag, &req);

        update_status(atl_status);
    }

    void update() override
    {
        int req_status;
        atl_status_t atl_status = atl_comm_check(sched->bin->get_comm_ctx(), &req_status, &req);

        if (unlikely(atl_status != atl_status_success))
        {
            CCL_THROW("RECV entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req_status)
        {
            LOG_DEBUG("RECV entry done, src ", src);
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override
    {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_buf> id)
    {
        return buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id)
    {
        return cnt;
    }

protected:
    void dump_detail(std::stringstream& str) const override
    {
        ccl_logger::format(str,
                           "dt ", ccl_datatype_get_name(dtype),
                           ", cnt ", cnt,
                           ", buf ", buf,
                           ", src ", src,
                           ", atl_tag ", atl_tag,
                           ", comm_id ", sched->get_comm_id(),
                           ", req ", &req,
                           "\n");
    }

private:
    ccl_buffer buf;
    size_t cnt;
    ccl_datatype_internal_t dtype;
    size_t src;
    ccl_comm* comm;
    uint64_t atl_tag = 0;
    atl_req_t req{};
};
