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

#include "comp/comp.hpp"

#include <unordered_map>

enum ccl_cache_key_type
{
    ccl_cache_key_full,
    ccl_cache_key_match_id,

    ccl_cache_key_last_value
};

const char* ccl_cache_key_type_to_str(ccl_cache_key_type type);

class ccl_sched_key
{
private:
    size_t hasher_result = 0;

public:
    ccl_sched_key() = default;
    ~ccl_sched_key() = default;
    ccl_sched_key(ccl_sched_key&& other) = default;
    ccl_sched_key& operator= (ccl_sched_key&& other) = default;

    ccl_sched_key(const ccl_coll_param& param, const ccl_coll_attr& attr);
    void set(const ccl_coll_param& param, const ccl_coll_attr& attr);
    bool check(const ccl_coll_param& param, const ccl_coll_attr& attr);

    size_t get_hasher_result() const
    {
        return hasher_result;
    }

    void set_hasher_result(size_t value)
    {
        has_hasher_result = true;
        hasher_result = value;
    }

    bool has_hasher_result = false;

    void* buf = nullptr; /* non-data buffer which can be used for caching */
    ccl_coll_type ctype = ccl_coll_internal;
    ccl_datatype_t dtype = ccl_dtype_char;
    ccl_datatype_t itype = ccl_dtype_char; /* used in sparse collective to store index type */
    ccl_reduction_t reduction = ccl_reduction_sum;
    size_t count1 = 0;
    size_t count2 = 0;
    size_t* count3 = nullptr; /* used in sparse collective to store recv index count */
    size_t* count4 = nullptr; /* used in sparse collective to store recv value count */
    size_t root = 0;
    const ccl_comm* comm = nullptr;
    ccl_prologue_fn_t prologue_fn = nullptr;
    ccl_epilogue_fn_t epilogue_fn = nullptr;
    ccl_reduction_fn_t reduction_fn = nullptr;

    std::string match_id{}; /* should always be last field */

    bool operator== (const ccl_sched_key& k) const;

    void print() const
    {
        LOG_DEBUG( "ctype ", ccl_coll_type_to_str(ctype),
                   ", dtype ", ccl_datatype_get(dtype)->name,
                   ", itype ", ccl_datatype_get(itype)->name,
                   ", reduction ", ccl_reduction_to_str(reduction),
                   ", buf ", buf,
                   ", count1 ", count1,
                   ", count2 ", count2,
                   ", count3 ", count3,
                   ", count4 ", count4,
                   ", root ", root,
                   ", comm ", comm,
                   ", prologue_fn ", (void*)prologue_fn,
                   ", epilogue_fn ", (void*)epilogue_fn,
                   ", reduction_fn ", (void*)reduction_fn,
                   ", match_id ", match_id);
    }
};

class ccl_sched_key_hasher
{
public:
    size_t operator()(const ccl_sched_key& k) const;

private:
    std::hash<std::string> string_hasher{};
};
