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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "common/event/impls/event_impl.hpp"

class ccl_request;

namespace ccl {

// event returned by ccl_comm(i.e. native backend)
class host_event_impl final : public event_impl {
public:
    explicit host_event_impl(ccl_request* r);
    ~host_event_impl() override;

    void wait() override;
    bool test() override;
    bool cancel() override;
    event::native_t& get_native() override;

private:
    ccl_request* req = nullptr;
    bool completed = false;
    bool synchronous = false;

#ifdef CCL_ENABLE_SYCL
    // the actual sycl::event returned to the user via ccl::event.get_native()
    std::shared_ptr<sycl::event> native_event;
#endif // CCL_ENABLE_SYCL
};

} // namespace ccl
