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
#pragma once

#include "sched/entry/entry.hpp"
#include "sched/extra_sched.hpp"
#include "sched/queue/queue.hpp"

class subsched_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "SUBSCHED";
    }

    subsched_entry() = delete;
    subsched_entry(ccl_sched* sched,
                   ccl_op_id_t op_id,
                   std::function<void(ccl_sched*)> fill_fn,
                   const char* subsched_name = nullptr)
            : sched_entry(sched),
              fill_fn(fill_fn),
              op_id(op_id),
              subsched_name(subsched_name) {
        if (subsched_name) {
            LOG_DEBUG("subsched name: ", subsched_name);
        }

        subsched.reset(new ccl_extra_sched({ sched->sched_id, sched->coll_param }));

        inherit_params(subsched.get(), sched, sched->coll_param.ctype);

        subsched->coll_param.ctype = ccl_coll_undefined;
        subsched->set_op_id(this->op_id);

        fill_fn(subsched.get());
    }

    static void inherit_params(ccl_sched* sched,
                               const ccl_sched* parent_sched,
                               ccl_coll_type ctype) {
        if (sched == parent_sched) {
            return;
        }

        if (ctype == ccl_coll_allreduce || ctype == ccl_coll_reduce ||
            ctype == ccl_coll_reduce_scatter) {
            sched->coll_attr.reduction_fn = parent_sched->coll_attr.reduction_fn;
            /* required to create ccl_fn_context in reduce/recv_reduce entries */
            sched->coll_attr.match_id = parent_sched->coll_attr.match_id;
        }
        sched->coll_attr.to_cache = parent_sched->coll_attr.to_cache;

#ifdef CCL_ENABLE_SYCL
        sched->coll_attr.is_sycl_buf = parent_sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL

        sched->flow_control.set_max_credits(parent_sched->flow_control.get_max_credits());
    }

    ~subsched_entry() {
        subsched.reset();
    }

    void start() override {
        subsched->renew();
        subsched->bin = sched->bin;
        subsched->queue = sched->queue;
        subsched->sched_id = sched->sched_id;
        status = ccl_sched_entry_status_started;
    }

    void update() override {
        subsched->do_progress();
        if (subsched->start_idx == subsched->entries.size()) {
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return !subsched_name.empty() ? subsched_name.c_str() : class_name();
    }

    ccl_sched* get_subsched() {
        return subsched.get();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "content:\n");

        for (size_t idx = 0; idx < subsched->entries.size(); ++idx) {
            ccl_logger::format(str, "\t");
            subsched->entries[idx]->dump(str, idx);
        }
    }

private:
    std::unique_ptr<ccl_extra_sched> subsched;
    std::function<void(ccl_sched*)> fill_fn;
    ccl_op_id_t op_id;
    std::string subsched_name;
};
