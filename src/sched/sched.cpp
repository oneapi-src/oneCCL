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
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/extra_sched.hpp"
#include "sched/queue/queue.hpp"
#include "sched/sched.hpp"

ccl_sched::~ccl_sched()
{
    if (in_bin_status != ccl_sched_in_bin_added)
        LOG_DEBUG("in_bin_status != ccl_sched_in_bin_added");

    if (finalize_fn)
    {
        finalize_fn(this, finalize_fn_ctx);
    }

    if (!memory.mr_list.empty())
    {
        /* perform deregistration in worker thread */
        {
            ccl_coll_param param{};
            param.ctype = ccl_coll_internal;
            param.comm = coll_param.comm;
            std::unique_ptr<ccl_extra_sched> dereg_sched(new ccl_extra_sched(param, sched_id));
            entry_factory::make_entry<deregister_entry>(dereg_sched.get(), memory.mr_list);
            if (global_data.is_worker_thread || !env_data.worker_offload)
            {
                dereg_sched->do_progress();
            }
            else
            {
                /* release ownership, because ccl_wait_impl use delete inside */
                ccl_wait_impl<ccl_extra_sched>(global_data.executor.get(), start_subsched(dereg_sched.release()));
            }
        }

        if (!memory.mr_list.empty())
        {
            LOG_ERROR("memory list is not empty");
        }

        CCL_ASSERT(memory.mr_list.empty());
    }

    free_buffers();
}

void ccl_sched::do_progress()
{
    for (auto entry_idx = start_idx; entry_idx < entries.size(); ++entry_idx)
    {
        auto& entry = entries[entry_idx];

        if (entry->get_status() == ccl_sched_entry_status_not_started)
        {
            LOG_DEBUG("starting entry ", entry->name(), " [", entry_idx, "/", entries.size(), "]");
        }

        entry->do_progress();

        if (entry->get_status() == ccl_sched_entry_status_again)
        {
            LOG_DEBUG("entry ", entry->name(), " is in again state, stop progressing [",
                      entry_idx, "/", entries.size(), "]");
            break;
        }

        if (entry_idx == start_idx && entry->is_completed())
        {
            /* the entry has been completed, increment start_idx */
            ++start_idx;
            LOG_DEBUG("completed ", entry->name(), entry->is_barrier() ? " barrier" : "",
                      " entry [", entry_idx, "/", entries.size(), "], shift start_idx to ", start_idx,
                      ", sched ", this);
        }
        else if (entry->is_barrier() && (!entry->is_completed() || (start_idx != entry_idx + 1)))
        {
            /* barrier is not completed or completed too early, skip the further progressing */
            break;
        }
    }
}

bool ccl_sched::is_strict_order_satisfied()
{
    CCL_ASSERT(strict_start_order);
    return std::all_of(entries.begin(), entries.end(), [](const sched_entry_ptr& e)
        {
            return e->is_strict_order_satisfied();
        });
}

void ccl_sched::complete()
{
#ifdef ENABLE_TIMERS
    exec_complete_time = timer_type::now();
    if (env_data.sched_dump)
    {
        dump(std::cout);
    }
#endif
    CCL_ASSERT(req, "ccl_sched must have req");
    req->complete();
}

void ccl_sched::renew(bool need_update_id/* = false*/)
{
    if (need_update_id)
    {
        update_id();
    }
#ifdef ENABLE_TIMERS
    exec_start_time = timer_type::now();
    exec_complete_time = exec_start_time;
#endif
    start_idx = 0;
    for (size_t idx = 0; idx < entries.size(); idx++)
    {
        entries[idx].get()->reset(idx);
    }
}

void ccl_sched::add_barrier()
{
    if (!entries.empty())
    {
        if (add_mode == ccl_sched_add_back)
            entries.back()->make_barrier();
        else if (add_mode == ccl_sched_add_front)
            entries.front()->make_barrier();
        else
            CCL_FATAL("unexpected add_mode ", add_mode);
    }
}

ccl_request* ccl_sched::start_subsched(ccl_extra_sched* subsched)
{
    CCL_THROW_IF_NOT(subsched);

    subsched->sched_id = sched_id;
    subsched->coll_attr.priority = coll_attr.priority;

    subsched->renew();
    subsched->set_counter(1);

    queue->add(subsched);
    subsched->dump(std::cout);

    return subsched->req;
}

void ccl_sched::dump(std::ostream& out) const
{
    if (!env_data.sched_dump)
    {
        return;
    }

    ccl_sched_base::dump(out, class_name());
    ccl_logger::format(out, ", start_idx: ", start_idx,
                       ", num_entries: ", entries.size(),
                       ", priority: ", get_priority(),
                       "\n");

    std::stringstream msg;
    for (size_t i = 0; i < entries.size(); ++i)
    {
        entries[i]->dump(msg, i);
    }
    out << msg.str();
    ccl_logger::format(out, "--------------------------------\n");
}
