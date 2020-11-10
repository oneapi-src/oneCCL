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
#include <memory>

#include "oneapi/ccl.hpp"
#include "common/event/impls/event_impl.hpp"

class ccl_gpu_sched;

namespace ccl {

class gpu_event_impl final : public event_impl {
public:
    explicit gpu_event_impl(std::unique_ptr<ccl_gpu_sched>&& sched);
    ~gpu_event_impl();

    void wait() override;
    bool test() override;
    bool cancel() override;
    event::native_t& get_native() override;

private:
    std::unique_ptr<ccl_gpu_sched> gpu_sched;
    bool completed = false;
};

class gpu_shared_event_impl final : public event_impl {
public:
    explicit gpu_shared_event_impl(std::shared_ptr<ccl_gpu_sched>&& sched);
    ~gpu_shared_event_impl();

    void wait() override;
    bool test() override;
    bool cancel() override;
    event::native_t& get_native() override;

private:
    std::shared_ptr<ccl_gpu_sched> gpu_sched;
    bool completed = false;
};

class gpu_shared_process_event_impl final : public event_impl {
public:
    explicit gpu_shared_process_event_impl(std::shared_ptr<ccl_gpu_sched>&& sched);
    ~gpu_shared_process_event_impl();

    void wait() override;
    bool test() override;
    bool cancel() override;
    event::native_t& get_native() override;

private:
    std::shared_ptr<ccl_gpu_sched> gpu_sched;
};

} // namespace ccl
