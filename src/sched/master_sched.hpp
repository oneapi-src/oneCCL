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
#include "sched/sched_base.hpp"

class ccl_sched;
class ccl_sched_key;

class ccl_master_sched : public ccl_sched_base, public ccl_request
{
public:
    static constexpr const char* class_name()
    {
        return "master_sched";
    }

    ccl_master_sched(const ccl_coll_param& coll_param)
        : ccl_sched_base(coll_param),
          ccl_request(),
          partial_scheds()
    {
#ifdef ENABLE_DEBUG
    set_dump_callback([this](std::ostream &out)
                      {
                            dump(out);
                      });
#endif
    }

    ccl_master_sched(const ccl_master_sched &src) = delete;

    ~ccl_master_sched() override;


    void add_partial_sched(ccl_coll_param& param);
    void commit(ccl_parallelizer* parallelizer = nullptr);
    ccl_request* start(ccl_executor* exec,
                       bool reset_sched = true);

    /**
     * Reset completion counter of @b req
     * @return pointer to req that can be used to track completion
     */
    ccl_request* reset_request();
    /**
     * Synchronizes partial schedules on local barrier
     */
    void sync_partial_scheds();
    void dump(std::ostream& out) const;

    //TODO encapsulate it in private.
    std::vector<std::shared_ptr<ccl_sched>> partial_scheds;

    //factory method (TODO: wrap into smart-pointer)
    using ccl_master_sched_ptr = ccl_master_sched*;
    static ccl_master_sched_ptr create(const ccl_coll_param& param,
                                       const ccl_coll_attr& attr);
private:
    void prepare_partial_scheds();
};
