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

#include "common/utils/spinlock.hpp"
#include "sched/cache/key.hpp"
#include "sched/master_sched.hpp"

#include <atomic>
#include <functional>
#include <unordered_map>
#include <utility>

#define CCL_SCHED_CACHE_INITIAL_BUCKET_COUNT (4096)

class ccl_sched_cache
{
public:
    ccl_sched_cache() : reference_counter(0) {};
    ~ccl_sched_cache()
    {
        size_t iter = 0;
        static const size_t check_period = 1000;
        while (!try_flush())
        {
            if (iter % check_period)
            {
                LOG_DEBUG("can't destruct cache because reference_counter = ",
                    reference_counter, ", expected 0");
            }
            iter++;
        }
    }
    ccl_sched_cache(const ccl_sched_cache& other) = delete;
    ccl_sched_cache& operator= (const ccl_sched_cache& other) = delete;
    template<class Lambda>
    std::pair<ccl_master_sched*, bool> find_or_create(ccl_sched_key&& key,
                                                      Lambda create_fn);
    void recache(const ccl_sched_key& old_key, ccl_sched_key&& new_key);
    void release(ccl_master_sched* sched);
    bool try_flush();

private:
    ccl_master_sched* find_unsafe(const ccl_sched_key& key) const;
    using sched_cache_lock_t = ccl_spinlock;
    mutable sched_cache_lock_t guard{}; //could be changed in constant method
    //TODO use smart ptr for ccl_master_sched in table
    using sched_table_t = std::unordered_map<ccl_sched_key, ccl_master_sched* , ccl_sched_key_hasher>;
    sched_table_t table { CCL_SCHED_CACHE_INITIAL_BUCKET_COUNT };
    std::atomic<size_t> reference_counter;
};

template<class Lambda>
std::pair<ccl_master_sched*, bool> ccl_sched_cache::find_or_create(ccl_sched_key&& key,
                                                                   Lambda create_fn)
{
    ccl_master_sched* sched = nullptr;
    bool is_created = false;
    {
        std::lock_guard<sched_cache_lock_t> lock{guard};
        sched = find_unsafe(key);
        if (sched)
        {
            reference_counter++;
        }
        else
        {
            LOG_DEBUG("didn't find sched in cache, the new one will be created");
            sched = create_fn();
            {
                reference_counter++;
                auto emplace_result = table.emplace(std::move(key), sched);
                CCL_ASSERT(emplace_result.second);
            }
            is_created = true;

            LOG_DEBUG("size ", table.size(),
                ", bucket_count ", table.bucket_count(),
                ", load_factor ", table.load_factor(),
                ", max_load_factor ", table.max_load_factor());
        }
    }
    LOG_TRACE("reference_counter=", reference_counter);
    return std::make_pair(sched, is_created);
}
