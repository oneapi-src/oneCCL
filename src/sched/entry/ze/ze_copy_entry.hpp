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

#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

struct copy_attr;

class ze_copy_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_COPY";
    }

    const char* name() const override {
        return class_name();
    }

    virtual std::string name_ext() const override {
        std::stringstream out;
        out << name() << " ";
        out << "size: " << count;
        return out.str();
    }

    explicit ze_copy_entry(ccl_sched* sched,
                           ccl_buffer in_buf,
                           ccl_buffer out_buf,
                           size_t count,
                           const ccl_datatype& dtype,
                           const copy_attr& attr = {},
                           std::vector<ze_event_handle_t> wait_events = {});

    void init_ze_hook() override;

private:
    ccl_sched* const sched;
    ccl_buffer in_buf{};
    ccl_buffer out_buf{};
    const ccl_datatype& dtype;
    const copy_attr attr;
    const size_t count;
};
