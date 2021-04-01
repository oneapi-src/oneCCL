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

#include "atl/atl.h"
#include "coll/coll.hpp"
#include "common/global/global.hpp"
#include "common/request/request.hpp"
#include "exec/thread/listener.hpp"
#include "sched/extra_sched.hpp"
#include "internal_types.hpp"

#include <memory>
#include <vector>

class ccl_worker;
class ccl_service_worker;
class ccl_master_sched;
class ccl_extra_sched;

class alignas(CACHELINE_SIZE) ccl_executor {
    friend class ccl_listener;

public:
    ccl_executor(const ccl_executor& other) = delete;
    ccl_executor& operator=(const ccl_executor& other) = delete;
    ccl_executor(ccl_executor&& other) = delete;
    ccl_executor& operator=(ccl_executor&& other) = delete;

    ccl_executor(const char* main_addr = NULL);
    ~ccl_executor();

    struct worker_guard {
        ccl_executor* parent;
        worker_guard(ccl_executor* e) {
            parent = e;
            parent->lock_workers();
        }
        worker_guard(worker_guard&) = delete;
        worker_guard(worker_guard&& src) {
            parent = src.parent;
            src.parent = nullptr;
        }
        ~worker_guard() {
            if (parent)
                parent->unlock_workers();
        }
    };

    worker_guard get_worker_lock() {
        return worker_guard(this);
    }

    void start(ccl_extra_sched* extra_sched);
    void start(ccl_master_sched* sched);
    void wait(const ccl_request* req);
    bool test(const ccl_request* req);

    void start_workers();
    void start_workers(size_t local_proc_idx, size_t local_proc_count);
    bool are_workers_started() {
        return workers_started;
    };
    size_t get_worker_count() const;
    void update_wait_condition(size_t idx,
                               ccl_base_thread::wait_data::update_type type,
                               size_t delta);

    // TODO: Rework to support listener
    //    ccl::status create_listener(ccl_resize_fn_t resize_func);
    void update_workers();
    void lock_workers();
    void unlock_workers();
    bool is_locked = false;

    size_t get_local_proc_idx() const {
        return local_proc_idx;
    }
    size_t get_local_proc_count() const {
        return local_proc_count;
    }

    static size_t calculate_atl_ep_count(size_t worker_count);
    static atl_attr_t generate_atl_attr(const ccl::env_data& env);

private:
    size_t get_worker_idx_round_robin(ccl_sched* sched);
    size_t get_worker_idx_by_sched_id(ccl_sched* sched);

    std::unique_ptr<ccl_sched_queue> create_sched_queue(size_t idx, size_t ep_per_worker);
    void do_work();
    void set_local_coord(size_t proc_idx, size_t proc_count);

    std::vector<std::unique_ptr<ccl_worker>> workers;
    // TODO: Rework to support listener
    //  std::unique_ptr<ccl_listener> listener;

    typedef size_t (ccl_executor::*get_worker_idx_fn_t)(ccl_sched* sched);
    get_worker_idx_fn_t get_worker_idx_fn;
    size_t rr_worker_idx = 0; /* to distribute work in round-robin */
    size_t local_proc_idx;
    size_t local_proc_count;
    bool workers_started = false;
};

inline void ccl_release_sched(ccl_master_sched* sched) {
    if (sched->coll_attr.to_cache) {
        ccl::global_data::get().sched_cache->release(sched);
    }
    else {
        delete sched;
    }
}

inline void ccl_release_sched(ccl_extra_sched* sched) {
    delete sched;
}

template <class sched_type = ccl_master_sched>
inline void ccl_wait_impl(ccl_executor* exec, ccl_request* request) {
    exec->wait(request);
    if (!exec->is_locked) {
        LOG_DEBUG("req ",
                  request,
                  " completed, sched ",
                  ccl_coll_type_to_str(static_cast<sched_type*>(request)->coll_param.ctype));
        ccl_release_sched(static_cast<sched_type*>(request));
    }
}

template <class sched_type = ccl_master_sched>
inline bool ccl_test_impl(ccl_executor* exec, ccl_request* request) {
    bool completed = exec->test(request);

    if (completed) {
        sched_type* sched = static_cast<sched_type*>(request);
        LOG_DEBUG(
            "req ", request, " completed, sched ", ccl_coll_type_to_str(sched->coll_param.ctype));

        if (!exec->is_locked) {
            ccl_release_sched(sched);
        }
    }

    return completed;
}
