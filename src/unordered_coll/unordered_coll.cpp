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
#include "sched/entry/factory/entry_factory.hpp"
#include "unordered_coll/unordered_coll.hpp"

#include <cstring>

struct ccl_unordered_coll_ctx
{
    ccl_comm_id_t reserved_comm_id;
    size_t match_id_size;
    void* match_id_value;
    ccl_extra_sched* service_sched;
    ccl_unordered_coll_manager* manager;
};

ccl_unordered_coll_manager::ccl_unordered_coll_manager()
{
    ccl::global_data& data = ccl::global_data::get();

    coordination_comm = std::unique_ptr<ccl_comm>(new ccl_comm(data.executor->get_global_proc_idx(),
                                                               data.executor->get_global_proc_count(),
                                                               data.comm_ids->acquire(true)));
    CCL_ASSERT(coordination_comm.get(), "coordination_comm is null");

    if (data.executor->get_global_proc_idx() == 0)
        LOG_INFO("created unordered collectives manager");
}

ccl_unordered_coll_manager::~ccl_unordered_coll_manager()
{
    coordination_comm.reset();
}

std::shared_ptr<ccl_comm> ccl_unordered_coll_manager::get_comm(const std::string& match_id)
{
    /* check if there are completed service scheds and remove them */
    remove_service_scheds();

    std::lock_guard<ccl_spinlock> lock(match_id_to_comm_map_guard);
    auto comm = match_id_to_comm_map.find(match_id);
    if (comm != match_id_to_comm_map.end())
    {
        LOG_DEBUG("comm_id ", comm->second->id(), " for match_id ", match_id, " has been found");
        return comm->second;
    }
    LOG_DEBUG("no comm for match_id ", match_id, " has been found");
    return nullptr;
}

ccl_request* ccl_unordered_coll_manager::postpone(ccl_master_sched* sched)
{
    CCL_ASSERT(!sched->coll_attr.match_id.empty(), "invalid match_id");
    const std::string& match_id = sched->coll_attr.match_id;

    ccl_request* req = sched->reset_request();

    /* 1. check whether comm was created between get_comm and postpone calls on user level */
    ccl_comm* comm = nullptr;
    {
        std::lock_guard<ccl_spinlock> lock(match_id_to_comm_map_guard);
        auto m2comm = match_id_to_comm_map.find(match_id);
        if (m2comm != match_id_to_comm_map.end())
        {
            comm = m2comm->second.get();
            CCL_ASSERT(comm);
        }
        else
        {
            /* keep match_id_to_comm_map_guard to postpone before m2c table update */
            CCL_ASSERT(!comm);
            LOG_DEBUG("postpone sched because didn't find comm_id for match_id ", match_id);
            postpone_sched(sched);
            if (get_postponed_sched_count(match_id) > 1)
            {
                return req;
            }
        }
    }

    if (comm)
    {
        LOG_INFO("don't postpone sched because found comm_id ", comm->id(),
                 " for match_id ", match_id);
        run_sched(sched, comm);
        return req;
    }

    /* 2. start coordination for match_id */
    start_coordination(match_id);

    /* 3.
        for non-coordinator ranks comm_id can arrive too early before collective with corresponding match_id
        check this and start schedule if we already have the required match_id
    */
    if (coordination_comm->rank() != CCL_UNORDERED_COLL_COORDINATOR)
    {
        std::unique_lock<ccl_spinlock> lock(unresolved_comms_guard);
        auto unresolved = unresolved_comms.find(match_id);
        if (unresolved != unresolved_comms.end())
        {
            LOG_DEBUG("found comm_id ", unresolved->second.value(),
                      " for match_id ", match_id,
                      " in unresolved_comms, ");
            auto comm_id(std::move(unresolved->second));
            unresolved_comms.erase(unresolved);
            lock.unlock();

            CCL_ASSERT(sched->coll_param.comm);
            auto comm = sched->coll_param.comm->clone_with_new_id(std::move(comm_id));
            add_comm(match_id, comm);
            run_sched(sched, comm.get());
        }
    }

    return req;
}

void ccl_unordered_coll_manager::dump(std::ostream& out) const
{
    std::stringstream s;

    {
        std::lock_guard<ccl_spinlock> lock{unresolved_comms_guard};
        s << "unresolved_comms: " << std::endl;
        for (auto& comm: unresolved_comms)
        {
            s << "[" << comm.first << ", " << comm.second.value() << "] "  << std::endl;
        }
    }

    {
        std::lock_guard<ccl_spinlock> lock{match_id_to_comm_map_guard};
        s << "match_id_to_comm_map: " << std::endl;
        for (auto& m2c: match_id_to_comm_map)
        {
            s << "[" << m2c.first << ", " << m2c.second->id() << "] " << std::endl;
        }
    }

    {
        std::lock_guard<ccl_spinlock> lock{postponed_scheds_guard};
        s << "postponed_scheds: " << std::endl;
        for (auto& sched: postponed_scheds)
        {
            s << "[" << sched.first << ", " << sched.second << "] " << std::endl;
        }
    }

    out << s.str();
}

void ccl_unordered_coll_manager::start_coordination(const std::string& match_id)
{
    CCL_THROW_IF_NOT(!match_id.empty(), "match_id is empty");

    ccl_coll_param coll_param{};
    coll_param.ctype = ccl_coll_internal;
    coll_param.dtype = ccl_datatype_char;
    coll_param.comm = coordination_comm.get();

    std::unique_ptr<ccl_extra_sched> service_sched(new ccl_extra_sched(coll_param,
                                                                       coordination_comm->get_sched_id(true)));
    service_sched->internal_type = ccl_sched_internal_unordered_coll;
    if (ccl::global_data::env().priority_mode == ccl_priority_lifo)
    {
        service_sched->coll_attr.priority = ccl_sched_base::get_lifo_priority();
    }

    LOG_DEBUG("start coordination for match_id ", match_id,
              " (service_sched ", service_sched.get(), ", req ",
              static_cast<ccl_request*>(service_sched.get()), ")");

    /* 1. broadcast match_id_size */
    auto ctx = static_cast<ccl_unordered_coll_ctx*>(service_sched->alloc_buffer(sizeof(ccl_unordered_coll_ctx)).get_ptr());
    ctx->service_sched = service_sched.get();
    ctx->manager = this;

    if (coordination_comm->rank() == CCL_UNORDERED_COLL_COORDINATOR)
    {
        ctx->match_id_size = match_id.length() + 1;
        ctx->match_id_value = service_sched->alloc_buffer(ctx->match_id_size).get_ptr();
        strncpy(static_cast<char*>(ctx->match_id_value), match_id.c_str(), ctx->match_id_size);
        ctx->reserved_comm_id = ccl::global_data::get().comm_ids->acquire_id(true);
        LOG_DEBUG("coordinator bcasts match_id ", match_id,
                  ", comm_id ", ctx->reserved_comm_id,
                  ", ctx->match_id_size ", ctx->match_id_size);
    }

    ccl_coll_entry_param match_id_size_param{};
    match_id_size_param.ctype = ccl_coll_bcast;
    match_id_size_param.buf = ccl_buffer(&ctx->match_id_size, sizeof(size_t));
    match_id_size_param.count = sizeof(size_t);
    match_id_size_param.dtype = ccl_datatype_char;
    match_id_size_param.root = CCL_UNORDERED_COLL_COORDINATOR;
    match_id_size_param.comm = coll_param.comm;
    entry_factory::make_entry<coll_entry>(service_sched.get(), match_id_size_param);

    service_sched->add_barrier();

    /* 2. broadcast match_id_value */
    ccl_coll_entry_param match_id_val_param{};
    match_id_val_param.ctype = ccl_coll_bcast;
    match_id_val_param.buf = ccl_buffer();
    match_id_val_param.count = 0;
    match_id_val_param.dtype = ccl_datatype_char;
    match_id_val_param.root = CCL_UNORDERED_COLL_COORDINATOR;
    match_id_val_param.comm = coll_param.comm;
    auto entry = entry_factory::make_entry<coll_entry>(service_sched.get(), match_id_val_param);

    entry->set_field_fn<ccl_sched_entry_field_buf>([](const void* fn_ctx, void* field_ptr)
    {
        auto ctx = static_cast<ccl_unordered_coll_ctx*>(const_cast<void*>(fn_ctx));
        if (ctx->service_sched->coll_param.comm->rank() != CCL_UNORDERED_COLL_COORDINATOR)
        {
            /* coordinator allocates and fills this buffer during schedule creation */
            ctx->match_id_value = ctx->service_sched->alloc_buffer(ctx->match_id_size).get_ptr();
        }
        ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
        buf_ptr->set(ctx->match_id_value, ctx->match_id_size);
        return ccl_status_success;
    }, ctx);

    entry->set_field_fn<ccl_sched_entry_field_cnt>([](const void* fn_ctx, void* field_ptr) -> ccl_status_t
    {
        auto ctx = static_cast<ccl_unordered_coll_ctx*>(const_cast<void*>(fn_ctx));
        auto count_ptr = static_cast<size_t*>(field_ptr);
        *count_ptr = ctx->match_id_size;
        return ccl_status_success;
    }, ctx);

    service_sched->add_barrier();

    /* 3. broadcast reserved comm_id */
    ccl_coll_entry_param reserved_comm_id_param{};
    reserved_comm_id_param.ctype = ccl_coll_bcast;
    reserved_comm_id_param.buf = ccl_buffer(&ctx->reserved_comm_id,
                                            sizeof(ccl_comm_id_t));
    reserved_comm_id_param.count = sizeof(ccl_comm_id_t);
    reserved_comm_id_param.dtype = ccl_datatype_char;
    reserved_comm_id_param.root = CCL_UNORDERED_COLL_COORDINATOR;
    reserved_comm_id_param.comm = coll_param.comm;
    entry_factory::make_entry<coll_entry>(service_sched.get(), reserved_comm_id_param);

    service_sched->add_barrier();

    /* 4. start post actions (create communicator and start postponed schedules) */
    entry_factory::make_entry<function_entry>(service_sched.get(), [](const void* func_ctx) -> ccl_status_t
    {
        auto ctx = static_cast<ccl_unordered_coll_ctx*>(const_cast<void*>(func_ctx));
        ctx->manager->start_post_coordination_actions(ctx);
        return ccl_status_success;
    }, ctx);

    LOG_DEBUG("start service_sched ", service_sched.get(), " for match_id ", match_id);
    /* release ownership */
    ccl::global_data::get().executor->start(service_sched.release());
}

void ccl_unordered_coll_manager::start_post_coordination_actions(ccl_unordered_coll_ctx* ctx)
{
    if (coordination_comm->rank() != CCL_UNORDERED_COLL_COORDINATOR)
    {
        /* comm_id is not allocated yet */
        LOG_DEBUG("start_post_coordination_actions: pull_id ", ctx->reserved_comm_id);
        ccl::global_data::get().comm_ids->pull_id(ctx->reserved_comm_id);
    }

    ccl_comm_id_storage::comm_id id(*ccl::global_data::get().comm_ids, ctx->reserved_comm_id);
    std::string match_id{static_cast<const char*>(ctx->match_id_value)};

    LOG_DEBUG("creating communicator with id ", id.value(), " for match_id ", match_id);

    /* original comm is required to create new communicator with the same size */
    ccl_comm* original_comm = nullptr;

    {
        std::lock_guard<ccl_spinlock> lock{postponed_scheds_guard};
        auto sched = postponed_scheds.find(match_id);
        if (sched != postponed_scheds.end())
        {
            original_comm = sched->second->coll_param.comm;
            CCL_ASSERT(original_comm);
        }
    }

    if (!original_comm)
    {
        CCL_THROW_IF_NOT(coordination_comm->rank() != CCL_UNORDERED_COLL_COORDINATOR);
        /*
            coordinator broadcasted match_id
            but other ranks don't have collectives with the same match_id yet
        */
        LOG_DEBUG("can't find postponed sched for match_id ", match_id,
                  ", postpone comm creation, reserved comm_id ", id.value());
        std::lock_guard<ccl_spinlock> lock{unresolved_comms_guard};
        unresolved_comms.emplace(match_id, std::move(id));
    }
    else
    {
        auto new_comm = original_comm->clone_with_new_id(std::move(id));
        add_comm(match_id, new_comm);
        run_postponed_scheds(match_id, new_comm.get());
    }

    CCL_ASSERT(ctx->service_sched, "service_sched is null");
    std::lock_guard<ccl_spinlock> lock{service_scheds_guard};
    CCL_THROW_IF_NOT(service_scheds.find(match_id) == service_scheds.end());
    auto emplace_result = service_scheds.emplace(std::move(match_id), ctx->service_sched);
    CCL_ASSERT(emplace_result.second);
}

void ccl_unordered_coll_manager::run_postponed_scheds(const std::string& match_id, ccl_comm* comm)
{
    CCL_THROW_IF_NOT(comm, "communicator is null");

    std::vector<ccl_master_sched*> scheds_to_run;
    std::unique_lock<ccl_spinlock> lock{postponed_scheds_guard};
    auto scheds = postponed_scheds.equal_range(match_id);
    size_t sched_count = std::distance(scheds.first, scheds.second);
    LOG_DEBUG("found ", sched_count,
              " scheds for match_id ", match_id);

    scheds_to_run.reserve(sched_count);
    transform(scheds.first, scheds.second,
              back_inserter(scheds_to_run),
              [](const std::pair<std::string, ccl_master_sched*> &element) { return element.second; } );
    postponed_scheds.erase(scheds.first, scheds.second);
    lock.unlock();

    for (auto& sched: scheds_to_run)
    {
        run_sched(sched, comm);
    }
}

void ccl_unordered_coll_manager::run_sched(ccl_master_sched* sched, ccl_comm* comm) const
{
    ccl_sched_key old_key, new_key;
    old_key.set(sched->coll_param, sched->coll_attr);
    sched->coll_param.comm = comm;
    new_key.set(sched->coll_param, sched->coll_attr);

    if (sched->coll_attr.to_cache)
    {
        ccl::global_data::get().sched_cache->recache(old_key, std::move(new_key));
    }

    for (size_t part_idx = 0; part_idx < sched->partial_scheds.size(); ++part_idx)
    {
        sched->partial_scheds[part_idx]->coll_param.comm = comm;
        if (ccl::global_data::env().priority_mode == ccl_priority_lifo)
        {
            /*
                can't use real lifo priority here because it can be unsynchronized between nodes
                so use comm id as synchronized increasing value for priority
            */
            sched->partial_scheds[part_idx]->coll_attr.priority = comm->id();
        }
        sched->partial_scheds[part_idx]->coll_attr.match_id = sched->coll_attr.match_id;
    }

    LOG_DEBUG("running sched ", sched,
              ", type ", ccl_coll_type_to_str(sched->coll_param.ctype),
              ",  for match_id ", sched->coll_attr.match_id);

    sched->start(ccl::global_data::get().executor.get(), false);
}

void ccl_unordered_coll_manager::add_comm(const std::string& match_id, std::shared_ptr<ccl_comm> comm)
{
    std::lock_guard<ccl_spinlock> lock(match_id_to_comm_map_guard);
    auto emplace_result = match_id_to_comm_map.emplace(match_id, comm);
    CCL_ASSERT(emplace_result.second);
}

void ccl_unordered_coll_manager::postpone_sched(ccl_master_sched* sched)
{
    std::lock_guard<ccl_spinlock> lock{postponed_scheds_guard};
    size_t sched_count = postponed_scheds.count(sched->coll_attr.match_id);
    LOG_DEBUG("postponed_scheds contains ", sched_count,
              " entries for match_id ", sched->coll_attr.match_id);
    postponed_scheds.emplace(sched->coll_attr.match_id, sched);
}

size_t ccl_unordered_coll_manager::get_postponed_sched_count(const std::string& match_id)
{
    std::lock_guard<ccl_spinlock> lock{postponed_scheds_guard};
    return postponed_scheds.count(match_id);
}

void ccl_unordered_coll_manager::remove_service_scheds()
{
    std::lock_guard<ccl_spinlock> lock{service_scheds_guard};
    for (auto it = service_scheds.begin(); it != service_scheds.end();)
    {
        ccl_extra_sched* sched = it->second;
        if (sched->req->is_completed())
        {
            LOG_DEBUG("sched ", sched, ", match_id ", it->first);
            delete sched;
            it = service_scheds.erase(it);
        }
        else
        {
            ++it;
        }
    }
}
