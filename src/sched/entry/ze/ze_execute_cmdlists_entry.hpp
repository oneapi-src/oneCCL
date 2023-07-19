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

#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

class execute_cmdlists_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "EXEC_CMDLIST";
    }
    const char* name() const override {
        return class_name();
    }

    execute_cmdlists_entry(ccl_sched* sched) : ze_base_entry(sched, {}) {}

    void init() override {
        LOG_DEBUG("execute cmdlists entry");
        if (sched->use_single_list && !ze_command::bypass_command_flag()) {
            // submit commands to command lists
            sched->ze_commands_submit();

            // once command lists have commands, execute their associated cmdqueues
            sched->get_memory().list_manager->execute(this);

            sched->set_submitted_to_gpu(true);
        }
        is_initialized = true;
    }

    void update() override {
        ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
        ze_base_entry::update();
    }
};
