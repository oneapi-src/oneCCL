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
#include "sched/sched_base.hpp"

//todo: sequence diagram
//workflow:
//1. new ccl_sched()
//2. set_coll_attr [opt]
//3. sched->commit(parallelizer)
//4. sched->start(executor)
//  4.1 prepare_partial_scheds()
//      4.1.1 update_id()
//      4.1.2 renew()
//  4.2 reset_request()

enum ccl_sched_in_bin_status {
    ccl_sched_in_bin_none,
    ccl_sched_in_bin_added,
    ccl_sched_in_bin_erased
};

typedef ccl_status_t (*ccl_sched_finalize_fn_t)(ccl_sched*, const void*);

class ccl_extra_sched;

class alignas(CACHELINE_SIZE) ccl_sched : public ccl_sched_base {
public:
    static constexpr const char* class_name() {
        return "worker_sched";
    }

    ccl_sched(const ccl_coll_param& coll_param, ccl_request* master_request)
            : ccl_sched_base(coll_param) {
        req = master_request;
    }

    ccl_sched() = delete;
    ccl_sched(const ccl_sched& other) = delete;
    ccl_sched& operator=(const ccl_sched& other) = delete;

    virtual ~ccl_sched();

    bool is_strict_order_satisfied();

    void do_progress();

    virtual void complete();

    void clear() {
        entries.clear();
    }

    size_t get_start_idx() const {
        return start_idx;
    }

    /* communicators on build and execution stages can differ */
    ccl_comm_id_t get_comm_id();

    void set_op_id(ccl_op_id_t id) {
        op_id = id;
    }

    ccl_op_id_t get_op_id() {
        return op_id;
    }

    void set_in_bin_status(ccl_sched_in_bin_status status) {
        in_bin_status = status;
    }

    ccl_sched_in_bin_status get_in_bin_status() const {
        return in_bin_status;
    }

    /**
     * Reset runtime parameters and all entries
     */
    void renew(bool need_update_id = false);

    using ccl_sched_base::add_entry_front_t;
    using ccl_sched_base::add_entry_back_t;
    using add_entry_default_t = add_entry_mode_t<ccl_sched_add_mode_last_value>;

    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry) {
        entry->set_exec_mode(exec_mode);

        sched_entry* raw_ptr = entry.get();
        if (add_mode == ccl_sched_add_back)
            entries.push_back(std::move(entry));
        else if (add_mode == ccl_sched_add_front)
            entries.push_front(std::move(entry));
        else
            CCL_FATAL("unexpected mode ", add_mode);

        return raw_ptr;
    }

    /**
     * Policy-based add_entry
     */
    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry,
                           add_entry_mode_t<ccl_sched_add_mode_last_value>) {
        return add_entry(std::move(entry));
    }

    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry, add_entry_front_t) {
        entry->set_exec_mode(exec_mode);

        sched_entry* raw_ptr = entry.get();
        entries.push_front(std::move(entry));
        return raw_ptr;
    }

    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry, add_entry_back_t) {
        entry->set_exec_mode(exec_mode);

        sched_entry* raw_ptr = entry.get();
        entries.push_back(std::move(entry));
        return raw_ptr;
    }

    /**
     * Require that all previously added entries are completed before subsequent ops
     * may begin execution
     */
    void add_barrier();

    ccl_request* start_subsched(ccl_extra_sched* subsched);

    ccl_sched_bin* bin = nullptr; /* valid only during execution */
    ccl_sched_queue* queue = nullptr; /* cached pointer to queue, valid even after execution */
    size_t start_idx = 0; /* index to start */

    /* 
      used for unique ATL tag creation in algorithms with multiple parallel sub-schedules
      set once and then used for all entries
    */
    ccl_op_id_t op_id = 0;

    /* to track status of schedule wrt execution bin, not atomic as updated by single thread in time */
    ccl_sched_in_bin_status in_bin_status = ccl_sched_in_bin_none;

    using sched_entry_ptr = std::unique_ptr<sched_entry>;
    std::deque<sched_entry_ptr> entries{};

    /* whether sched should be started in the same order as in user code */
    bool strict_start_order = false;

    void set_finalize_fn(ccl_sched_finalize_fn_t fn, void* ctx) {
        finalize_fn = fn;
        finalize_fn_ctx = ctx;
    }
    ccl_request* req = nullptr;
    void dump(std::ostream& out) const;
    size_t entries_count() const;

private:
    ccl_sched_finalize_fn_t finalize_fn = nullptr;
    void* finalize_fn_ctx = nullptr;

#ifdef ENABLE_TIMERS
    using timer_type = std::chrono::system_clock;
    timer_type::time_point exec_start_time{};
    timer_type::time_point exec_complete_time{};
#endif
};
