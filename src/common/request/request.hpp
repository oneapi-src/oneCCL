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

#include <atomic>
#include <functional>

#include "common/utils/utils.hpp"

class alignas(CACHELINE_SIZE) ccl_request {
public:
    using dump_func = std::function<void(std::ostream &)>;
#ifdef ENABLE_DEBUG
    void set_dump_callback(dump_func &&callback);
#endif

    virtual ~ccl_request();

    bool complete();

    bool is_completed() const;

    void set_counter(int counter);

    void increase_counter(int increment);

    mutable bool urgent = false;

private:
    std::atomic_int completion_counter{ 0 };

#ifdef ENABLE_DEBUG
    dump_func dump_callback;
    mutable size_t complete_checks_count = 0;
    static constexpr const size_t CHECK_COUNT_BEFORE_DUMP = 40000000;
#endif
};
