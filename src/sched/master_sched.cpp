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

#include "common/global/global.hpp"
#include "common/utils/sync_object.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/cache/cache.hpp"
#include "sched/cache/key.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/extra_sched.hpp"
#include "sched/master_sched.hpp"
#include "sched/queue/queue.hpp"

ccl_master_sched::~ccl_master_sched()
{
    for (auto& part_sched: partial_scheds)
    {
        part_sched.reset();
    }

    CCL_ASSERT (memory.mr_list.empty(), "memory list is not empty");
    free_buffers();
}

void ccl_master_sched::commit(ccl_parallelizer* parallelizer)
{
    if (env_data.priority_mode == ccl_priority_lifo)
    {
        coll_attr.priority = ccl_sched_base::get_lifo_priority();
    }

    if (partial_scheds.empty())
    {
        /* single time operations */
        update_id();
        if (parallelizer)
        {
            parallelizer->process(this);
            CCL_ASSERT(!partial_scheds.empty(), "ccl_master_sched must have at least 1 partial sched after parallelized");
        }
    }
    else
    {
        /* repeated operations, should happen each time to reuse schedule */
        for (size_t idx = 0; idx < partial_scheds.size(); idx++)
        {
            partial_scheds[idx]->coll_attr.priority = coll_attr.priority;
        }
    }

    LOG_DEBUG("sched ", this,
              ", sched_id ", sched_id, ", req ", static_cast<const ccl_request *>(this),
              ", partial_sched_count ", partial_scheds.size());
}

ccl_request* ccl_master_sched::start(ccl_executor* exec, bool reset_sched)
{
    /* sanity check the schedule */
    CCL_ASSERT(coll_param.comm);

    LOG_DEBUG("starting schedule ", this, ", type ", ccl_coll_type_to_str(coll_param.ctype));

    prepare_partial_scheds();

    if (reset_sched)
    {
        reset_request();
    }

    if (env_data.sched_dump)
    {
        dump(std::cout);
    }

    exec->start(this);
    return this;
}

ccl_request* ccl_master_sched::reset_request()
{
    set_counter(std::max(1, static_cast<int>(partial_scheds.size())));
    return this;
}

void ccl_master_sched::add_partial_sched(ccl_coll_param& coll_param)
{
    partial_scheds.emplace_back(std::make_shared<ccl_sched>(coll_param, this));
    partial_scheds.back()->internal_type = internal_type;
}

void ccl_master_sched::prepare_partial_scheds()
{
    for (auto& sched: partial_scheds)
    {
        sched->renew(true);
    }
}

void ccl_master_sched::sync_partial_scheds()
{
    CCL_THROW_IF_NOT(!partial_scheds.empty(), "no partial schedules");

    auto sync_obj = std::make_shared<sync_object>(partial_scheds.size());
    for (auto& sched : partial_scheds)
    {
        entry_factory::make_entry<sync_entry>(sched.get(), sync_obj);
    }
}

void ccl_master_sched::dump(std::ostream& out) const
{
    if (!env_data.sched_dump)
    {
        return;
    }

    ccl_sched_base::dump(out, class_name());
    ccl_logger::format(out,
                       ", req: ", static_cast<const ccl_request*>(this),
                       ", worker_sched count: ", partial_scheds.size());

    for (const auto& sched : partial_scheds)
    {
        sched->dump(out);
    }

#ifdef ENABLE_TIMERS
    ccl_logger::format(out, "\nlife time [us] ", std::setw(5), std::setbase(10),
        std::chrono::duration_cast<std::chrono::microseconds>(exec_complete_time - exec_start_time).count(),
        "\n");
#endif

    ccl_logger::format(out, "--------------------------------\n");
}

ccl_master_sched::ccl_master_sched_ptr ccl_master_sched::create(const ccl_coll_param& param,
                                                                const ccl_coll_attr& attr,
                                                                bool postpone_caching)
{
    /* check contract at first */
    CCL_THROW_IF_NOT(param.ctype == ccl_coll_allreduce ||
                     !(attr.prologue_fn || attr.epilogue_fn || attr.reduction_fn),
                     "for now only allreduce supports prologue/epilogue/custom_reduction functionality");

    CCL_THROW_IF_NOT(env_data.atl_transport == ccl_atl_ofi || !(attr.reduction_fn),
                     "for now only OFI transport supports custom_reduction functionality");

    ccl_sched_key key;
    ccl_master_sched_ptr sched = nullptr;

    if (attr.to_cache)
    {
        key.set(param, attr);
        sched = global_data.sched_cache->find(key);
        if (sched)
        {
            /* update some parameters and attributes in existing schedule
               as they could be changed since previous call */
            sched->update_coll_param(param);
            sched->update_coll_attr(attr);

            LOG_DEBUG("found sched, reuse ", sched, ", type ",
                      ccl_coll_type_to_str(sched->coll_param.ctype));
        }
    }

    if (!sched)
    {
        std::unique_ptr<ccl_master_sched> new_sched(new ccl_master_sched(param));
        LOG_DEBUG("didn't find sched, create new one ", new_sched.get(), ", type ",
                  ccl_coll_type_to_str(new_sched->coll_param.ctype));

        new_sched->set_coll_attr(attr);
        new_sched->alloc_buffers_for_sycl_copy();

        if (attr.to_cache && !postpone_caching)
        {
            global_data.sched_cache->add(std::move(key), new_sched.get());
            // no use 'key' anymore, because it's moved
        }

        sched = new_sched.release();
    }
    return sched;
}
