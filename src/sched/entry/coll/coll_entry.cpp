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
#include "common/request/request.hpp"
#include "sched/entry/coll/coll_entry_helper.hpp"
#include "sched/extra_sched.hpp"

void coll_entry::start() {
    if (update_fields()) {
        coll_sched.reset();
    }

    if (!coll_sched) {
        ccl_coll_param coll_param{};
        coll_param.ctype = sched->coll_param.ctype;
        coll_param.comm = sched->coll_param.comm;
        coll_param.stream = sched->coll_param.stream;
        coll_sched.reset(new ccl_extra_sched(coll_param, sched->sched_id));
        coll_sched->set_op_id(coll_sched_op_id);

        auto res = coll_entry_helper::build_schedule(coll_sched.get(), sched, param);
        CCL_ASSERT(res == ccl::status::success, "error during build_schedule, res ", res);
    }

    LOG_DEBUG("starting COLL entry: ", this, ", subsched: ", coll_sched.get());
    auto req = sched->start_subsched(coll_sched.get());
    LOG_DEBUG("started COLL entry: ", this, ", subsched ", coll_sched.get(), ", req ", req);

    status = ccl_sched_entry_status_started;
}

void coll_entry::update() {
    CCL_THROW_IF_NOT(coll_sched, "empty request");
    if (coll_sched->is_completed()) {
        LOG_DEBUG("COLL entry, completed: ", this, ", sched: ", coll_sched.get());
        status = ccl_sched_entry_status_complete;
    }
}
