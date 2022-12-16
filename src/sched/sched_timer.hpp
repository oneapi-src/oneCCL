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

#include <chrono>
#include <string>

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/api_wrapper/ze_api_wrapper.hpp"
#endif

namespace ccl {

class sched_timer {
public:
    sched_timer() = default;
    void start();
    void update();
    void reset();
    bool is_started() const;

    long double get_elapsed_usec() const;

private:
    bool started{};
    long double time_usec{};
    std::chrono::high_resolution_clock::time_point start_time{};
};

std::string to_string(const sched_timer& timer);

#ifdef CCL_ENABLE_ITT

namespace profile {
namespace itt {

void set_thread_name(const std::string& name);

enum class task_type : int {
    operation = 0,
    api_call = 1,
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    preparation = 2,
    device_work = 3,
    deps_handling = 4,
    completion = 5
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
};

void task_start(task_type type);
void task_end(task_type type);

} // namespace itt
} // namespace profile

#endif // CCL_ENABLE_ITT

} //namespace ccl
