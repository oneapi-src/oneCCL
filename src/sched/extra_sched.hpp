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

#include "sched/sched.hpp"

class ccl_master_sched;

class ccl_extra_sched : public ccl_request, public ccl_sched {
public:
    static constexpr const char* class_name() {
        return "extra_sched";
    }

    ccl_extra_sched(const ccl_sched_create_param& param, ccl_master_sched* master_sched = nullptr)
            : ccl_request(),
              ccl_sched(param, this, master_sched) {
#ifdef ENABLE_DEBUG
        set_dump_callback([this](std::ostream& out) {
            dump(out);
        });
#endif
    }

    ~ccl_extra_sched() override = default;

    void dump(std::ostream& out) const;
};
