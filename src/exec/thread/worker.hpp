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

#include "exec/thread/base_thread.hpp"
#include "sched/queue/strict_queue.hpp"
#include "sched/queue/queue.hpp"

#include <memory>
#include <list>
#include <pthread.h>

class ccl_executor;

class ccl_worker : public ccl_base_thread
{
public:
    ccl_worker() = delete;
    ccl_worker(const ccl_worker& other) = delete;
    ccl_worker& operator= (const ccl_worker& other) = delete;
    ccl_worker(size_t idx, std::unique_ptr<ccl_sched_queue> queue);
    virtual ~ccl_worker() = default;
    virtual void* get_this() override { return static_cast<void*>(this); };

    virtual const std::string& name() const override
    {
        static const std::string name("worker");
        return name;
    };

    void add(ccl_sched* sched);

    virtual ccl_status_t do_work(size_t& processed_count);

    void clear_queue();

    void reset_queue(std::unique_ptr<ccl_sched_queue>&& queue)
    {
        clear_queue();
        sched_queue = std::move(queue);
    }

    std::atomic<bool> should_lock;
    std::atomic<bool> is_locked;

private:

    ccl_status_t process_strict_sched_queue();
    ccl_status_t process_sched_queue(size_t& processed_count, bool process_all);
    ccl_status_t process_sched_bin(ccl_sched_bin* bin, size_t& processed_count);

    size_t do_work_counter = 0;

    std::unique_ptr<ccl_strict_sched_queue> strict_sched_queue;
    std::unique_ptr<ccl_sched_queue> sched_queue;
};
