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
#include "sched/sched.hpp"
#include "common/env/env.hpp"


struct ccl_extra_sched : public ccl_request, public ccl_sched
{
    static constexpr const char* class_name()
    {
        return "extra_sched";
    }

    ccl_extra_sched(ccl_coll_param& coll_param, ccl_sched_id_t id)
        : ccl_request(),
          ccl_sched(coll_param, this)
    {
        sched_id = id;
#ifdef ENABLE_DEBUG
        set_dump_callback([this](std::ostream &out)
                          {
                                dump(out);
                          });
#endif
    }

    ~ccl_extra_sched() = default;
    
    void dump(std::ostream& out) const
    {
        if (!env_data.sched_dump)
        {
            return;
        }

        ccl_sched_base::dump(out, class_name());
        ccl_logger::format(out, ", start_idx: ", start_idx,
                           ", req: ", static_cast<const ccl_request*>(this),
                           ", num_entries: ", entries.size(),
                           "\n");
        std::stringstream msg;
        for (size_t i = 0; i < entries.size(); ++i)
        {
            entries[i]->dump(msg, i);
        }
        out << msg.str();
#ifdef ENABLE_TIMERS
        ccl_logger::format(out, "\nlife time [us] ", std::setw(5), std::setbase(10),
            std::chrono::duration_cast<std::chrono::microseconds>(exec_complete_time - exec_start_time).count(),
            "\n");
#endif

        ccl_logger::format(out, "--------------------------------\n");
    }
};
