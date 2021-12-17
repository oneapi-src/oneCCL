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
#include "common/utils/sync_object.hpp"
#include "common/utils/sycl_utils.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/cache/cache.hpp"
#include "sched/cache/key.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/extra_sched.hpp"
#include "sched/master_sched.hpp"
#include "sched/queue/queue.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>

#ifdef CCL_ENABLE_ZE
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_SYCL
constexpr ze_event_pool_desc_t get_event_pool_desc() {
    auto desc = ccl::ze::default_event_pool_desc;

    desc.count = 1;
    desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

    return desc;
}
#endif

ccl_master_sched::ccl_master_sched(const ccl_sched_create_param& param)
        : ccl_sched_base(param),
          ccl_request(),
          partial_scheds() {
#ifdef ENABLE_DEBUG
    set_dump_callback([this](std::ostream& out) {
        dump(out);
    });
#endif

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::utils::should_use_sycl_output_event(coll_param.stream)) {
        auto ze_context = coll_param.stream->get_ze_context();
        auto pool_desc = get_event_pool_desc();
        ccl::global_data::get().ze_cache->get(0, ze_context, pool_desc, &get_memory().sync_pool);

        ze_event_desc_t event_desc = ccl::ze::default_event_desc;
        event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        event_desc.index = 0;

        ZE_CALL(zeEventCreate, (get_memory().sync_pool, &event_desc, &get_memory().sync_event));
        LOG_DEBUG("created sync event: ", get_memory().sync_event);
    }
    else {
        LOG_DEBUG("skip sync event creation");
    }
#endif
}

ccl_master_sched::~ccl_master_sched() {
    for (auto& part_sched : partial_scheds) {
        part_sched.reset();
    }
    if (!memory.mr_list.empty())
        LOG_WARN("memory region list should be empty for master sched");

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::utils::should_use_sycl_output_event(coll_param.stream)) {
        // Sycl event might call wait on destruction meaning that it should be valid at that time
        // The problem is that the sync event is stored in request, which descrutor is called
        // after ccl_master_sched, which means its underlying l0 event will be already destroyed
        // by that time. As a workaround, reset the event, essentially calling its destructor before
        // destroying the corresponding l0 event
        set_sync_event(sycl::event());

        LOG_DEBUG("destroying sync event: ", get_memory().sync_event);
        ZE_CALL(zeEventDestroy, (get_memory().sync_event));

        auto ze_context = coll_param.stream->get_ze_context();
        auto pool_desc = get_event_pool_desc();
        ccl::global_data::get().ze_cache->push(0, ze_context, pool_desc, get_memory().sync_pool);
    }
    else {
        LOG_DEBUG("skip sync event destruction");
    }
#endif
}

void ccl_master_sched::commit(ccl_parallelizer* parallelizer) {
    if (ccl::global_data::env().priority_mode == ccl_priority_lifo) {
        coll_attr.priority = ccl_sched_base::get_lifo_priority();
    }

    if (partial_scheds.empty()) {
        /* single time operations */
        update_id();
        if (parallelizer) {
            parallelizer->process(this);
            CCL_THROW_IF_NOT(
                !partial_scheds.empty(),
                "ccl_master_sched must have at least 1 partial sched after parallelized");
        }
    }
    else {
        /* repeated operations, should happen each time to reuse schedule */
        for (size_t idx = 0; idx < partial_scheds.size(); idx++) {
            partial_scheds[idx]->coll_attr.priority = coll_attr.priority;
        }
    }

    LOG_DEBUG("sched ",
              this,
              ", sched_id ",
              sched_id,
              ", req ",
              static_cast<const ccl_request*>(this),
              ", partial_sched_count ",
              partial_scheds.size());
}

void ccl_master_sched::reset_state() {
    reset_request();

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::utils::should_use_sycl_output_event(coll_param.stream)) {
        // Reset sycl event while it's in complete state, similar case to destruction in ~ccl_master_sched
        set_sync_event(sycl::event());
        LOG_DEBUG("reset sync event: ", get_memory().sync_event);
        ZE_CALL(zeEventHostReset, (get_memory().sync_event));
    }
#endif
}

ccl_request* ccl_master_sched::start(ccl_executor* exec, bool reset_sched) {
    /* sanity check the schedule */
    CCL_THROW_IF_NOT(coll_param.comm);

    LOG_DEBUG("starting schedule ", this, ", type ", ccl_coll_type_to_str(coll_param.ctype));

    prepare_partial_scheds();

    if (reset_sched) {
        reset_state();
    }

    if (ccl::global_data::env().sched_dump) {
        std::stringstream ostream;
        dump(ostream);
        logger.info(ostream.str());
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::utils::should_use_sycl_output_event(coll_param.stream)) {
        LOG_DEBUG("convert L0 event: ",
                  get_memory().sync_event,
                  "into a SYCL event and submit a barrier");
        auto q = coll_param.stream->get_native_stream();
        auto context = q.get_context();
#ifdef CCL_ENABLE_SYCL_INTEROP_EVENT
        auto e = ccl::utils::make_event(context, get_memory().sync_event);
        set_sync_event(e);
        set_native_event(ccl::utils::submit_barrier(q, e));
#else // CCL_ENABLE_SYCL_INTEROP_EVENT
        CCL_THROW("interop event functionality is not available with current configuration, "
                  "please rebuild oneCCL using ENABLE_SYCL_INTEROP_EVENT option "
                  "and a DPCPP compiler that supports that feature");
#endif // CCL_ENABLE_SYCL_INTEROP_EVENT
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    exec->start(this);

    return this;
}

ccl_request* ccl_master_sched::reset_request() {
    set_counter(std::max(1, static_cast<int>(partial_scheds.size())));
    return this;
}

void ccl_master_sched::add_partial_sched(const ccl_coll_param& coll_param) {
    partial_scheds.emplace_back(std::make_shared<ccl_sched>(
        ccl_sched_create_param(
            sched_type, coll_param.comm->get_sched_id(sched_type != ccl_sched_regular), coll_param),
        this,
        this));
}

void ccl_master_sched::prepare_partial_scheds() {
    for (auto& sched : partial_scheds) {
        sched->renew(true);
    }
}

void ccl_master_sched::sync_partial_scheds() {
    CCL_THROW_IF_NOT(!partial_scheds.empty(), "no partial schedules");

    bool add_sync_entry = false;

    /* ensure all partial schedules have the same add_mode */
    ccl_sched_add_mode add_mode = partial_scheds[0]->get_add_mode();
    for (auto& sched : partial_scheds) {
        CCL_THROW_IF_NOT(sched->get_add_mode() == add_mode,
                         "unexpected add_mode ",
                         sched->get_add_mode(),
                         ", expected ",
                         add_mode);
    }

    /* check whether all partial schedules already have sync_entry at the tail */
    for (auto& sched : partial_scheds) {
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
        auto sync_obj = std::make_shared<sync_object>(partial_scheds.size());
        for (auto& sched : partial_scheds) {
            entry_factory::create<sync_entry>(sched.get(), sync_obj);
        }
    }
}

void ccl_master_sched::dump(std::ostream& out) const {
    if (!ccl::global_data::env().sched_dump) {
        return;
    }

    ccl_sched_base::dump(out, class_name());
    ccl_logger::format(out,
                       ", req: ",
                       static_cast<const ccl_request*>(this),
                       ", partial_scheds size: ",
                       partial_scheds.size());

    for (const auto& sched : partial_scheds) {
        sched->dump(out);
    }

    ccl_logger::format(out, "--------------------------------\n");
}

ccl_master_sched::ccl_master_sched_ptr ccl_master_sched::create(const ccl_coll_param& param,
                                                                const ccl_coll_attr& attr) {
    ccl_sched_key key;
    ccl_master_sched_ptr sched;
    bool is_created = false;
    auto create_fn = [param]() -> ccl_master_sched_ptr {
        return new ccl_master_sched({ ccl_sched_regular, param.comm->get_sched_id(false), param });
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
                  ", type ",
                  ccl_coll_type_to_str(sched->coll_param.ctype));
    }
    else {
        /* update some parameters and attributes in existing schedule
           as they could be changed since previous call */
        sched->update_coll_param_and_attr(param, attr);
        LOG_DEBUG(
            "found sched, reuse ", sched, ", type ", ccl_coll_type_to_str(sched->coll_param.ctype));
    }

    return sched;
}

#ifdef CCL_ENABLE_SYCL
bool ccl_master_sched::print_kernel_timer() const {
    if (ccl::global_data::env().enable_kernel_profile) {
        return kernel_timer.print();
    }

    // if we don't have env variable set, just return false to say that we haven't printed
    // anything so no more work would be done
    return false;
}

void ccl_master_sched::reset_kernel_timer() {
    kernel_timer.reset();
}
#endif // CCL_ENABLE_SYCL
