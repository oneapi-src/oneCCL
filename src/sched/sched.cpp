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
#include "coll/coll_check.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/request/request.hpp"
#include "common/utils/sync_object.hpp"
#include "common/utils/sycl_utils.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/cache/cache.hpp"
#include "sched/cache/key.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched.hpp"
#include "sched/queue/queue.hpp"
#include "sched/sched_base.hpp"
#include "sched/sched_restart_manager.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>

#ifdef CCL_ENABLE_ZE
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

ccl_sched::ccl_sched(const ccl_sched_create_param& param, ccl_sched* master_sched)
        : ccl_sched_base(param),
          parent_sched(master_sched),
          req(new ccl_request(*this)),
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
          // only type = "master" sched should use output event
          use_output_event(false),
#endif
          top_level_sched(false),
          restart_manager() {
    type = sched_type_t::extra;
    strict_order = ccl::global_data::env().enable_strict_order;
}

ccl_sched::ccl_sched(const ccl_sched_create_param& param, bool top_level_sched)
        : ccl_sched_base(param),
          subscheds(),
          req(new ccl_request(*this)),
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
          use_output_event(top_level_sched &&
                           ccl::utils::should_use_sycl_output_event(coll_param.stream)),
#endif
          top_level_sched(top_level_sched),
          restart_manager(std::unique_ptr<sched_restart_manager>(new sched_restart_manager(this))) {
    type = sched_type_t::master;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (use_output_event) {
        ccl::global_data::get().ze_data->dynamic_event_pools.try_emplace(
            coll_param.stream->get_ze_context(), coll_param.stream);
    }
#endif
}

ccl_sched::~ccl_sched() {
    if (in_bin_status == ccl_sched_in_bin_added)
        LOG_DEBUG("in_bin_status == ccl_sched_in_bin_added");

    if (finalize_fn) {
        finalize_fn(this, finalize_fn_ctx);
    }

    // delete the last associated request
    LOG_DEBUG("deleting sched ", this, " and its req ", req);
    delete req;

    if (type != sched_type_t::master)
        return;

    for (auto& part_sched : subscheds) {
        part_sched.reset();
    }
    if (!memory.mr_list.empty())
        LOG_WARN("memory region list should be empty for master sched");
}

void ccl_sched::commit(ccl_parallelizer* parallelizer, bool update_sched_id) {
    if (ccl::global_data::env().priority_mode == ccl_priority_lifo) {
        coll_attr.priority = ccl_sched_base::get_lifo_priority();
    }

    if (subscheds.empty()) {
        /* single time operations */
        if (update_sched_id) {
            update_id();
        }
        if (parallelizer) {
            parallelizer->process(this, update_sched_id);
            CCL_THROW_IF_NOT(
                !subscheds.empty(),
                "ccl_master_sched must have at least 1 partial sched after parallelized");
        }
    }
    else {
        /* repeated operations, should happen each time to reuse schedule */
        for (size_t idx = 0; idx < subscheds.size(); idx++) {
            subscheds[idx]->coll_attr.priority = coll_attr.priority;
        }
    }

    LOG_DEBUG("sched ",
              this,
              ", sched_id ",
              sched_id,
              ", req ",
              get_request(),
              ", subscheds_count ",
              subscheds.size());
}

void ccl_sched::reset_state() {
    reset_request();
}

// restart parameter indicates that start() is executed from worker thread once the previous
// sched execution is completed. we restart only if there is at least 1 delayed request
ccl_request* ccl_sched::start(ccl_executor* exec,
                              bool reset_sched,
                              bool update_sched_id,
                              bool restart) {
    if (type == sched_type_t::master) {
        auto new_req = restart_manager->preprocess(restart);
        if (new_req)
            return new_req;
    }

    /* sanity check the schedule */
    CCL_THROW_IF_NOT(coll_param.comm);

    LOG_DEBUG("starting schedule ", this, ", type ", ccl_coll_type_to_str(coll_param.ctype));

    prepare_subscheds(update_sched_id);

    // don't reset request on restart, it's already have counter set to correct value
    if (reset_sched && !restart) {
        reset_state();
    }

    if (ccl::global_data::env().sched_dump) {
        std::stringstream ostream;
        dump(ostream);
        logger.info(ostream.str());
    }

    exec->start(this);

    return get_request();
}

int ccl_sched::calculate_request_count() const {
    return std::max(1, static_cast<int>(subscheds.size()));
}

ccl_request* ccl_sched::reset_request() {
    // if we reset the request when restarting the sched, we need to ignore completion check
    // because the request must be non-complete before it starts executing
    get_request()->set_counter(calculate_request_count());
    return get_request();
}

void ccl_sched::add_subsched(const ccl_coll_param& coll_param, bool update_sched_id) {
    ccl_sched_id_t param_sched_id =
        update_sched_id ? coll_param.comm->get_sched_id(sched_type != ccl_sched_regular) : sched_id;

    ccl_sched_create_param param = { sched_type, param_sched_id, coll_param };

    subscheds.emplace_back(std::make_shared<ccl_sched>(param, this));
}

std::vector<std::shared_ptr<ccl_sched>>& ccl_sched::get_subscheds() {
    return subscheds;
}

void ccl_sched::prepare_subscheds(bool update_sched_id) {
    for (auto& sched : subscheds) {
        sched->renew(update_sched_id, true);
    }
}

void ccl_sched::sync_subscheds() {
    CCL_THROW_IF_NOT(!subscheds.empty(), "no partial schedules");

    bool add_sync_entry = false;

    /* ensure all partial schedules have the same add_mode */
    ccl_sched_add_mode add_mode = subscheds[0]->get_add_mode();
    for (auto& sched : subscheds) {
        CCL_THROW_IF_NOT(sched->get_add_mode() == add_mode,
                         "unexpected add_mode ",
                         sched->get_add_mode(),
                         ", expected ",
                         add_mode);
    }

    /* check whether all partial schedules already have sync_entry at the tail */
    for (auto& sched : subscheds) {
        if (sched->entries.empty()) {
            add_sync_entry = true;
            break;
        }

        /* TODO: add enum field into base entry to distinguish different entry types */
        const char* tail_entry_name = (add_mode == ccl_sched_add_back)
                                          ? sched->entries.back()->name()
                                          : sched->entries.front()->name();

        if (tail_entry_name && strcmp(tail_entry_name, "SYNC")) {
            add_sync_entry = true;
            break;
        }
    }

    /* if at least one partial schedule doesn't have sync entry
       then sync all partial schedules */
    if (add_sync_entry) {
        auto sync_obj = std::make_shared<sync_object>(subscheds.size());
        for (auto& sched : subscheds) {
            entry_factory::create<sync_entry>(sched.get(), sync_obj);
        }
    }
}

void ccl_sched::dump(std::ostream& out) const {
    if (!ccl::global_data::env().sched_dump) {
        return;
    }

    ccl_sched_base::dump(out, class_name());
    ccl_logger::format(out,
                       ", start_idx: ",
                       start_idx,
                       ", req: ",
                       get_request(),
                       ", num_entries: ",
                       entries.size(),
                       ", priority: ",
                       get_priority(),
                       ", max_flow_credits: ",
                       flow_control.get_max_credits(),
                       ", flow_credits: ",
                       flow_control.get_credits(),
                       ", subscheds size: ",
                       subscheds.size(),
                       "\n");

    std::stringstream msg;
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i]->dump(msg, i);
    }

    for (const auto& sched : subscheds) {
        sched->dump(out);
    }

    out << msg.str();

    ccl_logger::format(out, "--------------------------------\n");
}

ccl_sched::ccl_sched_ptr ccl_sched::create(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    ccl_sched_key key;
    ccl_sched_ptr sched;
    bool is_created = false;
    auto create_fn = [param]() -> ccl_sched_ptr {
        return new ccl_sched({ ccl_sched_regular, param.comm->get_sched_id(false), param },
                             /* top-level sched */ true);
    };

    if (attr.to_cache) {
        key.set(param, attr);
        std::tie(sched, is_created) =
            ccl::global_data::get().sched_cache->find_or_create(std::move(key), create_fn);
    }
    else {
        sched = create_fn();
        is_created = true;
    }

    if (is_created) {
        sched->set_coll_attr(attr);
        sched->alloc_buffers_for_pre_post_copy();
        LOG_DEBUG("didn't find sched, create new one ",
                  sched,
                  ", coll ",
                  ccl_coll_type_to_str(sched->coll_param.ctype));
    }
    else {
        // we can't update params of the cached sched immediately, it might still be executing using
        // old parameters, they'll be updated once it's ready to launch with them.
        sched->restart_manager->add_launch_params({ param, attr });
        LOG_DEBUG(
            "found sched, reuse ", sched, ", type ", ccl_coll_type_to_str(sched->coll_param.ctype));
    }

    return sched;
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

    // save sched type because we cannot assume that the sched with type == master
    // is not destroyed after we complete the request
    auto* parent_sched = this->parent_sched;

    // it's important to do finalization/cleanup before full completion of the request
    // because right after its completion, the request and the sched can be destroyed
    // by a user thread waiting on a corresponding event, so it's hard to do anything
    // with the object at that time. To overcome this issue, for each request we add
    // +1 to its counter, so here we can check that while worker completed the
    // requests, it still have count 1, so we can do finalization/cleanup before
    // completing it one more time setting the counter to 0.
    if (get_request()->complete_counter() == 1) {
        if (ccl::global_data::env().sched_profile) {
            timer.stop();
            if (entries.size() > 0) {
                std::stringstream ss;
                ss << "\ncoll:";

                ccl_coll_param* profile_param = &(coll_param);
                ss << ccl_coll_type_to_str(profile_param->ctype);

                /* TODO: tmp check, replace ccl_coll_entry_param by ccl_coll_param */
                if (!profile_param->send_counts.empty()) {
                    ss << " count:" << profile_param->get_send_count();
                }

                ss << " time(usec):\ntotal: " << timer.str() << "\n";
                for (size_t idx = 0; idx < entries.size(); ++idx) {
                    ss << "[" << idx << "] " << entries[idx]->name() << ": "
                       << entries[idx]->timer.str() << "\n";
                }
                ss << "-----------------------------";
                logger.info(ss.str());
            }
        }

        sched_complete_hook();

        // now we completed everything related to finalization of the current sched,
        // so we can finaly complete it
        get_request()->complete();

        if (parent_sched) {
            // after we call try_to_restart() on the parent, it's request might be changed
            // so rememeber it here to call complete on
            auto parent_req = parent_sched->get_request();
            // check for completed parent request, see comment above for how this works
            if (parent_req->complete_counter() == 1) {
                // itt tracks only top-level sched execution
                if (top_level_sched)
                    complete_itt(parent_sched->coll_param.stream);
                // if we don't use cache, it doesn't make sense to restart the sched
                // as there are never be any requests to restart
                if (parent_sched->coll_attr.to_cache) {
                    // current sched execution is completed, always check if we need to
                    // restart it again
                    parent_sched->try_to_restart();
                }
                parent_req->complete();
            }
        }
    }
}

bool ccl_sched::is_completed() const {
    return get_request()->is_completed();
}

void ccl_sched::renew(bool need_update_id, bool reset) {
    if (need_update_id) {
        update_id();
    }

    start_idx = 0;

    if (ccl::global_data::env().sched_profile) {
        timer.start();
    }

    for (size_t idx = 0; idx < entries.size(); idx++) {
        entries[idx].get()->reset(idx);
    }

    if (reset)
        reset_state();
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

std::vector<ccl::event>& ccl_sched::get_deps() const {
    // if parent is not set then we should have own deps
    if (parent_sched)
        return const_cast<ccl_sched*>(parent_sched)->coll_param.deps;
    else
        return const_cast<ccl_sched*>(this)->coll_param.deps;
}

size_t ccl_sched::entries_count() const {
    return entries.size();
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
void ccl_sched::set_output_event(ccl_request* req) {
    if (!use_output_event) {
        return;
    }

    auto q = coll_param.stream->get_native_stream();
    auto context = q.get_context();
#ifdef CCL_ENABLE_SYCL_INTEROP_EVENT
    // even when we use cache in non-async case we cannot really guarantee that
    // some ccl event is not used by a user, so we cannot reuse request for another
    // operation, otherwise we could end up with multiple events referring to the
    // same request. So every time we run a schedule, we get a new event, but
    // use the pool to save on event/pool creation.
    auto& pools = ccl::global_data::get().ze_data->dynamic_event_pools;
    auto pool_it = pools.find(coll_param.stream->get_ze_context());
    CCL_THROW_IF_NOT(pool_it != pools.end(), "pool must be initialized for the context");

    ze_event_handle_t ev = pool_it->second.get_event();
    LOG_DEBUG("convert L0 event: ", ev, "into a SYCL event and submit a barrier");

    auto sync_event = ccl::utils::make_event(context, ev);
    req->set_sync_event(sync_event);
    req->set_native_event(ccl::utils::submit_barrier(q, sync_event));
    CCL_THROW_IF_NOT(!(req->get_native_event().is_host()), "something is wrong");

#else // CCL_ENABLE_SYCL_INTEROP_EVENT
    CCL_THROW("interop event functionality is not available with current configuration, "
              "please rebuild oneCCL using ENABLE_SYCL_INTEROP_EVENT option "
              "and a DPCPP compiler that supports that feature");
#endif // CCL_ENABLE_SYCL_INTEROP_EVENT
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

ccl_request* ccl_sched::get_request() {
    return req;
}
const ccl_request* ccl_sched::get_request() const {
    return req;
}

void ccl_sched::try_to_restart() {
    // execute for all scheds to perform a proper finalization
    if (!restart_manager->check_delayed_requests()) {
        return;
    }

    CCL_THROW_IF_NOT(top_level_sched, "only top-level scheds must be restarted");

    LOG_DEBUG("Restarting schedule: ", this);

    // we have at least 1 delayed request, need to restart the schedule
    (void)start(ccl::global_data::get().executor.get(),
                /* reset state */ true,
                /* update sched id */ true,
                /* restart */ true);
}

void ccl_sched::release_sync_event(ccl_request* request) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (use_output_event) {
        // check if the event has been reset already(is_host is true for an empty one)
        if (request->get_sync_event().is_host()) {
            LOG_DEBUG("request's event has been released already, skipping");
        }
        else {
            auto& pools = ccl::global_data::get().ze_data->dynamic_event_pools;
            auto pool_it = pools.find(coll_param.stream->get_ze_context());
            CCL_THROW_IF_NOT(pool_it != pools.end(), "pool must be initialized for the context");

            pool_it->second.put_event(ccl::utils::get_native_event(request->get_sync_event()));
        }
    }
    else {
        LOG_DEBUG("skip sync event destruction");
    }
#endif
}

void ccl_sched::update_active_request(bool use_delayed) {
    // at this point we reset the active request, but it still can
    // be referenced via an event, returned previously to the user.
    // the request object will be destroyed together with the event
    auto* old_req = req;
    req = ((use_delayed) ? restart_manager->get_delayed_request() : new ccl_request(*this));
    LOG_DEBUG("updated req: ", req, ", old: ", old_req, ", use_delayed: ", use_delayed);
}

void ccl_sched::complete_itt(const ccl_stream* stream) {
#ifdef CCL_ENABLE_ITT
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // only applicable for device execution
    if (stream) {
        ccl::profile::itt::task_end(ccl::profile::itt::task_type::completion);
    }
#endif // CCL_ENABLE_SYCL
    ccl::profile::itt::task_end(ccl::profile::itt::task_type::operation);
#else
    (void)stream;
#endif // CCL_ENABLE_ITT
}
