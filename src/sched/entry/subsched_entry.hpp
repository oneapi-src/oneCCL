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
#include "sched/sched.hpp"
#include "sched/queue/queue.hpp"

class subsched_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "SUBSCHED";
    }

    subsched_entry() = delete;
    subsched_entry(ccl_sched* sched,
                   ccl_op_id_t op_id,
                   std::function<void(ccl_sched*)> build_fn,
                   const char* subsched_name)
            : sched_entry(sched),
              build_fn(build_fn),
              op_id(op_id),
              subsched_name(subsched_name),
              build_sched_id(sched->sched_id),
              is_master_sched(false) {
        LOG_DEBUG("subsched name: ", subsched_name ? subsched_name : "<empty>");
    }

    subsched_entry(ccl_sched* sched,
                   ccl_op_id_t op_id,
                   ccl_sched_create_param sched_param,
                   const char* subsched_name)
            : sched_entry(sched),
              op_id(op_id),
              subsched_name(subsched_name),
              build_sched_id(sched_param.id),
              is_master_sched(true) {
        LOG_DEBUG("subsched name: ", subsched_name ? subsched_name : "<empty>");

        make_barrier();
        ccl::global_data& data = ccl::global_data::get();
        subsched.reset(new ccl_sched(sched_param));
        bool update_sched_id = false;
        subsched->commit(data.parallelizer.get(), update_sched_id);
        CCL_THROW_IF_NOT(subsched->sched_id == build_sched_id);
        auto& subscheds = subsched->get_subscheds();
        for (size_t i = 0; i < subscheds.size(); i++) {
            subscheds[i]->set_op_id(i);
        }
    }

    void set_params() {
        if (!is_master_sched) {
            subsched->set_op_id(op_id);

            if (subsched.get() == sched) {
                return;
            }
        }

        subsched->coll_attr.reduction_fn = sched->coll_attr.reduction_fn;
        subsched->coll_attr.priority = sched->coll_attr.priority;
        subsched->coll_attr.to_cache = sched->coll_attr.to_cache;
        subsched->coll_attr.match_id = sched->coll_attr.match_id;
#ifdef CCL_ENABLE_SYCL
        subsched->coll_attr.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL

        subsched->flow_control.set_max_credits(sched->flow_control.get_max_credits());
    }

    void build_subsched(const ccl_sched_create_param& create_param,
                        ccl_sched* master_sched = nullptr) {
        if (subsched || is_master_sched) {
            return;
        }

        subsched.reset(new ccl_sched(create_param, master_sched));
        set_params();
        build_fn(subsched.get());
    }

    void start() override {
        if (is_master_sched) {
            ccl::global_data& data = ccl::global_data::get();
            bool update_sched_id = false;
            subsched->start(data.executor.get(), true, update_sched_id);
        }
        else {
            build_subsched({ build_sched_id, sched->coll_param });
            subsched->renew();
            subsched->get_request()->set_counter(1);
            subsched->bin = sched->bin;
            subsched->queue = sched->queue;
            /* this makes effects on creation of tags in atl entries */
            subsched->sched_id = sched->sched_id;
            subsched->coll_param.comm = sched->coll_param.comm;
        }

        if (ccl::global_data::env().sched_dump) {
            std::stringstream ostream;
            subsched->dump(ostream);
            logger.info(ostream.str());
        }

        status = ccl_sched_entry_status_started;
        update();
    }

    void update() override {
        if (is_master_sched) {
            if (subsched->is_completed()) {
                status = ccl_sched_entry_status_complete;
            }
        }
        else {
            subsched->do_progress();
            if (subsched->start_idx == subsched->entries.size()) {
                status = ccl_sched_entry_status_complete;
                ((ccl_sched*)subsched.get())->complete();
            }
        }
    }

    const char* name() const override {
        return !subsched_name.empty() ? subsched_name.c_str() : class_name();
    }

    ccl_sched* get_subsched() {
        build_subsched({ build_sched_id, sched->coll_param });
        return subsched.get();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        if (!subsched) {
            return;
        }

        if (is_master_sched) {
            subsched->dump(std::cout);
        }
        else {
            ccl_logger::format(str, "content:\n");
            for (size_t idx = 0; idx < subsched->entries.size(); ++idx) {
                ccl_logger::format(str, "\t");
                subsched->entries[idx]->dump(str, idx);
            }
        }
    }

    std::unique_ptr<ccl_sched> subsched;

private:
    std::function<void(ccl_sched*)> build_fn;
    ccl_op_id_t op_id;
    std::string subsched_name;
    ccl_sched_id_t build_sched_id;
    bool is_master_sched;
};
