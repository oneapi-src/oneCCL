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
#include "common/request/request.hpp"
#include "common/event/impls/gpu_event.hpp"
#include "sched/gpu_sched.hpp"

namespace ccl {
gpu_event_impl::gpu_event_impl(std::unique_ptr<ccl_gpu_sched>&& sched)
        : gpu_sched(std::move(sched)) {
    if (!gpu_sched) {
        completed = true;
    }
}

gpu_event_impl::~gpu_event_impl() {
    if (!completed) {
        LOG_ERROR("not completed gpu event is destroyed");
    }
}

void gpu_event_impl::wait() {
    if (!completed && gpu_sched) {
        do {
            gpu_sched->do_progress();
            completed = gpu_sched->wait(0);
        } while (!completed);
    }
}

bool gpu_event_impl::test() {
    if (!completed && gpu_sched) {
        completed = gpu_sched->wait(0);
        gpu_sched->do_progress();
    }
    return completed;
}

bool gpu_event_impl::cancel() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

event::native_t& gpu_event_impl::get_native() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

gpu_shared_event_impl::gpu_shared_event_impl(std::shared_ptr<ccl_gpu_sched>&& sched)
        : gpu_sched(std::move(sched)) {
    if (!gpu_sched) {
        completed = true;
    }
}

gpu_shared_event_impl::~gpu_shared_event_impl() {
    if (!completed) {
        LOG_ERROR("not completed shared gpu event is destroyed");
    }
}

void gpu_shared_event_impl::wait() {
    if (!completed && gpu_sched) {
        do {
            gpu_sched->do_progress();
            completed = gpu_sched->wait(0);
        } while (!completed);
    }
}

bool gpu_shared_event_impl::test() {
    if (!completed && gpu_sched) {
        completed = gpu_sched->wait(0);
        gpu_sched->do_progress();
    }
    return completed;
}

bool gpu_shared_event_impl::cancel() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

event::native_t& gpu_shared_event_impl::get_native() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

gpu_shared_process_event_impl::gpu_shared_process_event_impl(
    std::shared_ptr<ccl_gpu_sched>&& sched) {}

gpu_shared_process_event_impl::~gpu_shared_process_event_impl() {}

void gpu_shared_process_event_impl::wait() {}

bool gpu_shared_process_event_impl::test() {
    return false;
}

bool gpu_shared_process_event_impl::cancel() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

event::native_t& gpu_shared_process_event_impl::get_native() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

} // namespace ccl
