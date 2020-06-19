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
#include "common/request/host_request.hpp"
#include "exec/exec.hpp"
namespace ccl
{
host_request_impl::host_request_impl(ccl_request* r) : req(r)
{
    if (!req)
    {
        // If the user calls collective with coll_attr->synchronous=1 then it will be progressed
        // in place and API will return null request. In this case mark cpp wrapper as completed,
        // all calls to wait() or test() will do nothing
        completed = true;
    }
}

host_request_impl::~host_request_impl()
{
    if (!completed)
    {
        LOG_ERROR("not completed request is destroyed");
    }
}

void host_request_impl::wait()
{
    if (!completed)
    {
        ccl_wait_impl(ccl::global_data::get().executor.get(), req);
        completed = true;
    }
}

bool host_request_impl::test()
{
    if (!completed)
    {
        completed = ccl_test_impl(ccl::global_data::get().executor.get(), req);
    }
    return completed;
}
}
