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

#include "sched/entry/entry.hpp"
#include "sched/sched.hpp"

class ze_event_wait_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_EVENT_WAIT";
    }

    const char* name() const override {
        return class_name();
    }

    bool is_strict_order_satisfied() override {
        return (status >= ccl_sched_entry_status_complete);
    }

    explicit ze_event_wait_entry(ccl_sched* sched, ze_event_handle_t event);

    void start() override;
    void update() override;

private:
    const ze_event_handle_t event;

    void check_event_status();
};
