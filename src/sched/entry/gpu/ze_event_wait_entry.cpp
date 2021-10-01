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
#include "sched/entry/gpu/ze_event_wait_entry.hpp"

#include <ze_api.h>

ze_event_wait_entry::ze_event_wait_entry(ccl_sched* sched, ze_event_handle_t event)
        : sched_entry(sched),
          event(event) {
    CCL_THROW_IF_NOT(sched, "no sched");
    CCL_THROW_IF_NOT(event, "no event");
}

void ze_event_wait_entry::check_event_status() {
    auto query_status = zeEventQueryStatus(event);
    if (query_status == ZE_RESULT_SUCCESS) {
        LOG_DEBUG("event complete");
        status = ccl_sched_entry_status_complete;
    }
    else if (query_status == ZE_RESULT_NOT_READY) {
        // just return in case if the kernel is not ready yet, will check again on the next iteration
        return;
    }
    else {
        CCL_THROW("error at zeEventQueryStatus");
    }
}

void ze_event_wait_entry::start() {
    LOG_DEBUG("start event waiting");
    status = ccl_sched_entry_status_started;
    check_event_status();
}

void ze_event_wait_entry::update() {
    check_event_status();
}
