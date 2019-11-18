/*
 Copyright 2016-2019 Intel Corporation
 
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

#include "sched/entry/coll/base_coll_entry.hpp"

class barrier_entry : public base_coll_entry
{
public:
    static constexpr const char* class_name() noexcept
    {
        return "BARRIER";
    }

    barrier_entry() = delete;
    barrier_entry(ccl_sched* sched) :
        base_coll_entry(sched)
    {
    }

    void start() override
    {
        LOG_DEBUG("BARRIER entry req ", &req);

        atl_status_t atl_status = atl_comm_barrier(sched->bin->get_comm_ctx(), &req);
        if (unlikely(atl_status != atl_status_success))
        {
            CCL_THROW("BARRIER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_started;
    }

    void update() override
    {
        int req_status;
        atl_status_t atl_status = atl_comm_check(sched->bin->get_comm_ctx(), &req_status, &req);

        if (unlikely(atl_status != atl_status_success))
        {
            CCL_THROW("BARRIER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req_status)
            status = ccl_sched_entry_status_complete;
    }

    const char* name() const override
    {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override
    {
        ccl_logger::format(str,
                            "comm_id ", sched->coll_param.comm->id(),
                            ", req ",&req,
                            "\n");
    }

private:
    atl_req_t req{};
};
