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
#include "common/log/log.hpp"
#include "common/utils/yield.hpp"

void ccl_yield(ccl_yield_type yield_type)
{
    struct timespec sleep_time;

    switch (yield_type)
    {
        case ccl_yield_none:
            break;
        case ccl_yield_pause:
            _mm_pause();
            break;
        case ccl_yield_sleep:
            sleep_time.tv_sec = 0;
            sleep_time.tv_nsec = 0;
            nanosleep(&sleep_time, nullptr);
            break;
        case ccl_yield_sched_yield:
            sched_yield();
            break;
        default:
            break;
    }
}

const char* ccl_yield_type_to_str(ccl_yield_type type)
{
    switch (type)
    {
        case ccl_yield_none:
            return "none";
        case ccl_yield_pause:
            return "pause";
        case ccl_yield_sleep:
            return "sleep";
        case ccl_yield_sched_yield:
            return "sched_yield";
        default:
            CCL_FATAL("unknown yield_type ", type);
    }
    return "unknown";
}
