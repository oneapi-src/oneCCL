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
#include "common/utils/sync_object.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/extra_sched.hpp"
#include "sched/queue/queue.hpp"
#include "sched/sched.hpp"

ccl_sched::ccl_sched(const ccl_coll_param& coll_param, ccl_request* master_request)
        : ccl_sched_base(coll_param) {
    req = master_request;
    strict_order = ccl::global_data::env().enable_strict_order;
}

ccl_sched::~ccl_sched() {
    if (in_bin_status == ccl_sched_in_bin_added)
        LOG_DEBUG("in_bin_status == ccl_sched_in_bin_added");

    if (finalize_fn) {
        finalize_fn(this, finalize_fn_ctx);
    }
}

void ccl_sched::do_progress() {
    for (auto entry_idx = start_idx; entry_idx < entries.size(); ++entry_idx) {
        auto& entry = entries[entry_idx];

        if (entry->get_status() == ccl_sched_entry_status_not_started) {
            LOG_DEBUG("starting entry: ",
                      entry.get(),
                      ", name: ",
                      entry->name(),
                      " [",
                      entry_idx,
                      "/",
                      entries.size(),
                      "]");
        }

        entry->do_progress();

        if (entry->get_status() == ccl_sched_entry_status_again) {
            LOG_DEBUG("entry ",
                      entry->name(),
                      " is in again state, stop progressing [",
                      entry_idx,
                      "/",
                      entries.size(),
                      "]");
            break;
        }

        if (entry_idx == start_idx && entry->is_completed()) {
            /* the entry has been completed, increment start_idx */
            ++start_idx;
            LOG_DEBUG("completed entry: ",
                      entry.get(),
                      ", name: ",
                      entry->name(),
                      entry->is_barrier() ? " barrier" : "",
                      " entry [",
                      entry_idx,
                      "/",
                      entries.size(),
                      "], shift start_idx to ",
                      start_idx,
                      ", sched ",
                      this);
        }
        else if (entry->is_barrier() && (!entry->is_completed() || (start_idx != entry_idx + 1))) {
            /* barrier is not completed or completed too early, skip the further progressing */
            break;
        }
    }
}

bool ccl_sched::is_strict_order_satisfied() {
    return std::all_of(entries.begin(), entries.end(), [](const sched_entry_ptr& e) {
        return e->is_strict_order_satisfied();
    });
}

void ccl_sched::complete() {
    CCL_ASSERT(req, "ccl_sched must have req");

    if (ccl::global_data::env().sched_profile) {
        timer.stop();
        if (entries.size() > 0) {
            std::stringstream ss;
            ss << "\ncoll:";

            ccl_coll_param* profile_param = &(static_cast<ccl_master_sched*>(req)->coll_param);
            ss << ccl_coll_type_to_str(profile_param->ctype);

            /* TODO: tmp check, replace ccl_coll_entry_param by ccl_coll_param */
            if (!profile_param->send_counts.empty()) {
                ss << " count:" << profile_param->get_send_count();
            }

            ss << " time(uses):\ntotal: " << timer.str() << "\n";
            for (size_t idx = 0; idx < entries.size(); ++idx) {
                ss << "[" << idx << "] " << entries[idx]->name() << ": "
                   << entries[idx]->timer.str() << "\n";
            }
            ss << "-----------------------------";
            logger.info(ss.str());
        }
    }

    if (!coll_attr.to_cache) {
        /* don't wait sched dtor to free memory */
        free_memory();
    }

    req->complete();
}

void ccl_sched::renew(bool need_update_id /* = false*/) {
    if (need_update_id) {
        update_id();
    }
    if (ccl::global_data::env().sched_profile) {
        timer.start();
    }
    start_idx = 0;
    for (size_t idx = 0; idx < entries.size(); idx++) {
        entries[idx].get()->reset(idx);
    }
}

void ccl_sched::add_barrier() {
    if (!entries.empty()) {
        if (add_mode == ccl_sched_add_back)
            entries.back()->make_barrier();
        else if (add_mode == ccl_sched_add_front)
            entries.front()->make_barrier();
        else
            CCL_FATAL("unexpected add_mode ", add_mode);
    }
}

ccl_request* ccl_sched::start_subsched(ccl_extra_sched* subsched) {
    CCL_THROW_IF_NOT(subsched);

    subsched->sched_id = sched_id;
    subsched->coll_attr.priority = coll_attr.priority;

    subsched->renew();
    subsched->set_counter(1);

    ccl::global_data::get().executor->update_wait_condition(
        queue->get_idx(), ccl_base_thread::wait_data::update_type::increment, 1);

    queue->add(subsched);

    if (ccl::global_data::env().sched_dump) {
        std::stringstream ostream;
        subsched->dump(ostream);
        logger.info(ostream.str());
    }

    return subsched->req;
}

std::vector<ccl::event>& ccl_sched::get_deps() const {
    return static_cast<ccl_master_sched*>(req)->coll_param.deps;
}

void ccl_sched::dump(std::ostream& out) const {
    if (!ccl::global_data::env().sched_dump) {
        return;
    }

    ccl_sched_base::dump(out, class_name());
    ccl_logger::format(out,
                       ", start_idx: ",
                       start_idx,
                       ", num_entries: ",
                       entries.size(),
                       ", priority: ",
                       get_priority(),
                       ", max_flow_credits: ",
                       flow_control.get_max_credits(),
                       ", flow_credits: ",
                       flow_control.get_credits(),
                       "\n");

    std::stringstream msg;
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i]->dump(msg, i);
    }
    out << msg.str();
    ccl_logger::format(out, "--------------------------------\n");
}

size_t ccl_sched::entries_count() const {
    return entries.size();
}

ccl_comm_id_t ccl_sched::get_comm_id() {
    return coll_param.comm->id();
}
