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
#include "common/event/impls/host_event.hpp"
#include "exec/exec.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

namespace ccl {

host_event_impl::host_event_impl(ccl_request* r) : req(r) {
    if (!req) {
        completed = true;
        return;
    }
#ifdef CCL_ENABLE_SYCL
    native_event = req->share_native_event();
#endif // CCL_ENABLE_SYCL
    if (req->synchronous) {
        if (!ccl::global_data::get().executor.get()->is_locked) {
            ccl_release_request(req);
        }
        // if the user calls collective with coll_attr->synchronous=1 then it will be progressed
        // in place and in this case we mark request as completed,
        // all calls to wait() or test() will do nothing
        completed = true;
        synchronous = true;
    }
}

host_event_impl::~host_event_impl() {
    // TODO: need to find a way to synchronize these 2 statuses, right now there are
    // some issues, e.g. in case of pure host event get_native() is an empty sycl
    // event which always complete, this way LOG_ERROR is never called
    if (!completed
#ifdef CCL_ENABLE_SYCL
        && (ccl::global_data::env().enable_sycl_output_event &&
            !utils::is_sycl_event_completed(get_native()))) {
        LOG_WARN("not completed event is destroyed");
    }
#else // CCL_ENABLE_SYCL
    ) {
        LOG_ERROR("not completed event is destroyed");
    }
#endif // NOT CCL_ENABLE_SYCL

    // when using native event user might not call wait/test on ccl event(complete = false)
    // but we need to ensure that the bound schedule is actually destroyed. For this
    // to happen, call wait() to do a proper finalization and cleanup.
    wait();
}

void host_event_impl::wait() {
    if (!completed) {
        auto* exec = ccl::global_data::get().executor.get();
        ccl_wait_impl(exec, req);
        if (synchronous && !exec->is_locked) {
            ccl_release_request(req);
        }
        completed = true;
    }
}

bool host_event_impl::test() {
    if (!completed) {
        completed = ccl_test_impl(ccl::global_data::get().executor.get(), req);
    }
    return completed;
}

bool host_event_impl::cancel() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

event::native_t& host_event_impl::get_native() {
#ifdef CCL_ENABLE_SYCL
    if (ccl::global_data::env().enable_sycl_output_event) {
        return *native_event;
    }
    else {
        CCL_THROW("get_native() is not available without CCL_SYCL_OUTPUT_EVENT=1 env variable");
    }
#else // CCL_ENABLE_SYCL
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
#endif // CCL_ENABLE_SYCL
}

} // namespace ccl
