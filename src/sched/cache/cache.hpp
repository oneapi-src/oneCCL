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

#pragma once

#include "common/utils/spinlock.hpp"
#include "sched/cache/key.hpp"
#include "sched/master_sched.hpp"

#include <unordered_map>

#define CCL_SCHED_CACHE_INITIAL_BUCKET_COUNT (4096)

class ccl_sched_cache
{
public:
    ccl_sched_cache() = default;
    ~ccl_sched_cache()
    {
        remove_all();
    }

    ccl_sched_cache(const ccl_sched_cache& other) = delete;
    ccl_sched_cache& operator= (const ccl_sched_cache& other) = delete;

    ccl_master_sched* find(ccl_sched_key& key);
    void add(ccl_sched_key&& key, ccl_master_sched* sched);

private:
    void remove_all();
    using sched_cache_lock_t = ccl_spinlock;
    sched_cache_lock_t guard{};
    //TODO use smart ptr for ccl_master_sched in table
    using sched_table_t = std::unordered_map<ccl_sched_key, ccl_master_sched* , ccl_sched_key_hasher>;
    sched_table_t table { CCL_SCHED_CACHE_INITIAL_BUCKET_COUNT };
};
