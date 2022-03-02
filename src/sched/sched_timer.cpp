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
#include <iomanip>
#include <numeric>
#include <sstream>

#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "sched_timer.hpp"

#ifdef CCL_ENABLE_ITT
#include "ittnotify.h"
#endif // CCL_ENABLE_ITT

namespace ccl {

void sched_timer::start() noexcept {
    start_time = std::chrono::high_resolution_clock::now();
}

void sched_timer::stop() {
    time_usec = get_elapsed_usec();
}

std::string sched_timer::str() const {
    std::stringstream ss;
    ss.precision(2);
    ss << std::fixed << time_usec;
    return ss.str();
}

void sched_timer::print(std::string title) const {
    logger.info(title, ": ", this->str());
}

void sched_timer::reset() noexcept {
    time_usec = 0;
}

long double sched_timer::get_elapsed_usec() const noexcept {
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> time_span = current_time - start_time;
    return time_span.count();
}

#if defined(CCL_ENABLE_ITT)

namespace profile {
namespace itt {

static __itt_domain* get_domain() {
    static __itt_domain* domain = __itt_domain_create("oneCCL");
    return domain;
}

static __itt_string_handle* get_operation_handle() {
    static __itt_string_handle* handle = __itt_string_handle_create("ccl_operation");
    return handle;
}

static __itt_string_handle* get_api_call_handle() {
    static __itt_string_handle* handle = __itt_string_handle_create("ccl_api_call");
    return handle;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

static __itt_string_handle* get_preparation_handle() {
    static __itt_string_handle* handle = __itt_string_handle_create("ccl_init");
    return handle;
}

static __itt_string_handle* get_device_work_handle() {
    static __itt_string_handle* handle = __itt_string_handle_create("ccl_submit_and_execute");
    return handle;
}

static __itt_string_handle* get_deps_handling_handle() {
    static __itt_string_handle* handle = __itt_string_handle_create("ccl_deps_handling");
    return handle;
}

static __itt_string_handle* get_completion_handle() {
    static __itt_string_handle* handle = __itt_string_handle_create("ccl_finalize");
    return handle;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

void set_thread_name(const std::string& name) {
    __itt_thread_set_name(name.c_str());
}

static __itt_string_handle* get_handle_for_type(task_type type) {
    switch (type) {
        case task_type::operation: return get_operation_handle();
        case task_type::api_call: return get_api_call_handle();
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        case task_type::preparation: return get_preparation_handle();
        case task_type::device_work: return get_device_work_handle();
        case task_type::deps_handling: return get_deps_handling_handle();
        case task_type::completion: return get_completion_handle();
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
        default: CCL_THROW("invalid task type: ", (int)type);
    }
}

void task_start(task_type type) {
    if (ccl::global_data::env().itt_level == 0)
        return;

    auto* handle = get_handle_for_type(type);
    __itt_task_begin(get_domain(), __itt_null, __itt_null, handle);
}

void task_end(task_type type) {
    if (ccl::global_data::env().itt_level == 0)
        return;

    // ignore for now, use only to identify the task that's being ended
    (void)type;
    __itt_task_end(get_domain());
}

} // namespace itt
} // namespace profile

#endif // CCL_ENABLE_ITT

} // namespace ccl
