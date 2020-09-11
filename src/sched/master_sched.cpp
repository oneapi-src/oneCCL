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
#include "sched/cache/cache.hpp"
#include "sched/cache/key.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/extra_sched.hpp"
#include "sched/master_sched.hpp"
#include "sched/queue/queue.hpp"

ccl_master_sched::~ccl_master_sched() {
    for (auto& part_sched : partial_scheds) {
        part_sched.reset();
    }

    CCL_ASSERT(memory.mr_list.empty(), "memory list is not empty");
    free_buffers();
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
            CCL_ASSERT(!partial_scheds.empty(),
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

ccl_request* ccl_master_sched::start(ccl_executor* exec, bool reset_sched) {
    /* sanity check the schedule */
    CCL_ASSERT(coll_param.comm);

    LOG_DEBUG("starting schedule ", this, ", type ", ccl_coll_type_to_str(coll_param.ctype));

    prepare_partial_scheds();

    if (reset_sched) {
        reset_request();
    }

    if (ccl::global_data::env().sched_dump) {
        std::stringstream ostream;
        dump(ostream);
        LOG_INFO(ostream.str());
    }

    exec->start(this);
    return this;
}

ccl_request* ccl_master_sched::reset_request() {
    set_counter(std::max(1, static_cast<int>(partial_scheds.size())));
    return this;
}

void ccl_master_sched::add_partial_sched(ccl_coll_param& coll_param) {
    partial_scheds.emplace_back(std::make_shared<ccl_sched>(coll_param, this));
    partial_scheds.back()->internal_type = internal_type;
}

void ccl_master_sched::prepare_partial_scheds() {
    for (auto& sched : partial_scheds) {
        sched->renew(true);
    }
}

void ccl_master_sched::sync_partial_scheds() {
    CCL_THROW_IF_NOT(!partial_scheds.empty(), "no partial schedules");

    auto sync_obj = std::make_shared<sync_object>(partial_scheds.size());
    for (auto& sched : partial_scheds) {
        entry_factory::make_entry<sync_entry>(sched.get(), sync_obj);
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
                       ", worker_sched count: ",
                       partial_scheds.size());

    for (const auto& sched : partial_scheds) {
        sched->dump(out);
    }

#ifdef ENABLE_TIMERS
    ccl_logger::format(
        out,
        "\nlife time [us] ",
        std::setw(5),
        std::setbase(10),
        std::chrono::duration_cast<std::chrono::microseconds>(exec_complete_time - exec_start_time)
            .count(),
        "\n");
#endif

    ccl_logger::format(out, "--------------------------------\n");
}

ccl_master_sched::ccl_master_sched_ptr ccl_master_sched::create(const ccl_coll_param& param,
                                                                const ccl_coll_attr& attr) {
    /* check contracts at first */

    CCL_THROW_IF_NOT(ccl::global_data::env().atl_transport == ccl_atl_ofi || !(attr.reduction_fn),
                     "custom reduction is supported for OFI transport only");

    CCL_THROW_IF_NOT(ccl_datatype_storage::is_predefined_datatype(param.dtype.idx()) ||
                         ccl::global_data::env().atl_transport == ccl_atl_ofi,
                     "custom datatype is supported for OFI transport only");

    CCL_THROW_IF_NOT((param.ctype != ccl_coll_allreduce && param.ctype != ccl_coll_reduce &&
                      param.ctype != ccl_coll_sparse_allreduce) ||
                         ccl_datatype_storage::is_predefined_datatype(param.dtype.idx()) ||
                         attr.reduction_fn,
                     "custom datatype requires custom reduction");

    CCL_THROW_IF_NOT(param.ctype == ccl_coll_allreduce ||
                         !(attr.prologue_fn || attr.epilogue_fn || attr.reduction_fn),
                     "prologue/epilogue/custom reduction is supported for allreduce only");

    CCL_THROW_IF_NOT(param.ctype == ccl_coll_allgatherv || !(attr.vector_buf),
                     "vector buffer is supported for allgatherv only");

    if (param.ctype == ccl_coll_sparse_allreduce) {
        CCL_THROW_IF_NOT(
            ccl::global_data::env().sparse_allreduce_algo_raw != "mask" || !(attr.reduction_fn),
            "mask algorithm for sparse_allreduce does not support custom reduction");

        CCL_THROW_IF_NOT(
            (attr.sparse_allreduce_completion_fn || attr.sparse_allreduce_alloc_fn) &&
                !(reinterpret_cast<uintptr_t>(attr.sparse_allreduce_completion_fn) &
                  reinterpret_cast<uintptr_t>(attr.sparse_allreduce_alloc_fn)),
            "sparse_allreduce requires completion callback only or allocation callback only");
    }

    CCL_THROW_IF_NOT((param.dtype.idx() != ccl::datatype::bfloat16) ||
                         (ccl::global_data::get().bfp16_impl_type != ccl_bfp16_none),
                     "BFP16 datatype is requested but not supported");

    ccl_sched_key key;
    ccl_master_sched_ptr sched;
    bool is_created = false;
    auto create_fn = [param]() -> ccl_master_sched_ptr {
        return new ccl_master_sched(param);
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
        sched->alloc_buffers_for_sycl_copy();
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
