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
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "sched/entry/entry.hpp"
#include "sched/sched.hpp"

void sched_entry::do_progress() {
    if (is_completed())
        return;

    if (status < ccl_sched_entry_status_started) {
        CCL_THROW_IF_NOT(
            status == ccl_sched_entry_status_not_started || status == ccl_sched_entry_status_again,
            "bad status ",
            status,
            "(",
            status_to_str(status),
            ")");

        bool took_credits = false;
        if (status == ccl_sched_entry_status_not_started) {
            took_credits = sched->flow_control.take_credit();
            if (took_credits && ccl::global_data::env().sched_profile) {
                timer.start();
            }
        }
        else if (status == ccl_sched_entry_status_again) {
            took_credits = true;
        }

        if (!took_credits) {
            return;
        }

        start();
        CCL_THROW_IF_NOT(status >= ccl_sched_entry_status_again,
                         "bad status ",
                         status,
                         "(",
                         status_to_str(status),
                         ")");
    }
    else if (status == ccl_sched_entry_status_started) {
        LOG_TRACE("update entry ", name());
        update();
        CCL_THROW_IF_NOT(status >= ccl_sched_entry_status_started,
                         "bad status ",
                         status,
                         "(",
                         status_to_str(status),
                         ")");
    }

    if (status == ccl_sched_entry_status_complete) {
        if (ccl::global_data::env().sched_profile) {
            timer.stop();
        }

        if (exec_mode == ccl_sched_entry_exec_once) {
            status = ccl_sched_entry_status_complete_once;
        }

        sched->flow_control.return_credit();
    }

    CCL_THROW_IF_NOT(
        status != ccl_sched_entry_status_failed && status != ccl_sched_entry_status_invalid,
        "bad status ",
        status,
        "(",
        status_to_str(status),
        ")");
}

bool sched_entry::is_completed() {
    return (status == ccl_sched_entry_status_complete ||
            status == ccl_sched_entry_status_complete_once);
}

void sched_entry::update() {
    /*
       update is required for async ops (atl, ze, sync)
       for other ops it is empty method
    */
}

void sched_entry::reset(size_t idx) {
    if (ccl::global_data::env().sched_profile) {
        timer.reset();
    }

    if (status == ccl_sched_entry_status_complete_once) {
        return;
    }

    start_idx = idx;
    status = ccl_sched_entry_status_not_started;
}

bool sched_entry::is_strict_order_satisfied() {
    return (status >= ccl_sched_entry_status_started);
}

void sched_entry::dump(std::stringstream& str, size_t idx) const {
    // update with the longest name
    const int entry_name_w = 14;

    ccl_logger::format(str,
                       "[",
                       std::left,
                       std::setw(3),
                       idx,
                       "] ",
                       std::left,
                       std::setw(entry_name_w),
                       name(),
                       " entry, status ",
                       status_to_str(status),
                       " is_barrier ",
                       std::left,
                       std::setw(5),
                       barrier ? "TRUE" : "FALSE",
                       " ");
    dump_detail(str);
}

void sched_entry::make_barrier() {
    barrier = true;
}

bool sched_entry::is_barrier() const {
    return barrier;
}

ccl_sched_entry_status sched_entry::get_status() const {
    return status;
}

void sched_entry::set_status(ccl_sched_entry_status s) {
    status = s;
}

void sched_entry::set_exec_mode(ccl_sched_entry_exec_mode mode) {
    exec_mode = mode;
}

void sched_entry::dump_detail(std::stringstream& str) const {}

void sched_entry::update_status(atl_status_t atl_status) {
    if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
        if (atl_status == ATL_STATUS_AGAIN) {
            status = ccl_sched_entry_status_again;
            return;
        }

        std::stringstream ss;
        dump_detail(ss);
        CCL_THROW("entry: ",
                  name(),
                  " failed. atl_status: ",
                  atl_status_to_str(atl_status),
                  ". Entry data:\n",
                  ss.str());
    }
    else {
        status = ccl_sched_entry_status_started;
    }
}

const char* sched_entry::status_to_str(ccl_sched_entry_status status) {
    switch (status) {
        case ccl_sched_entry_status_not_started: return "NOT_STARTED";
        case ccl_sched_entry_status_again: return "AGAIN";
        case ccl_sched_entry_status_started: return "STARTED";
        case ccl_sched_entry_status_complete: return "COMPLETE";
        case ccl_sched_entry_status_complete_once: return "COMPLETE_ONCE";
        case ccl_sched_entry_status_failed: return "FAILED";
        case ccl_sched_entry_status_invalid: return "INVALID";
        default: return "UNKNOWN";
    }
}
