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
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif /* CCL_ENABLE_SYCL */

ccl_stream::ccl_stream(ccl_stream_type_t type,
                       void* native_stream) :
    type(type),
    native_stream(native_stream)
{
#ifdef CCL_ENABLE_SYCL
    if (native_stream && (type == ccl_stream_sycl))
        LOG_INFO("SYCL queue's device is ",
                 static_cast<cl::sycl::queue*>(native_stream)->get_device().get_info<cl::sycl::info::device::name>());
#else
    CCL_THROW_IF_NOT(type != ccl_stream_sycl, "unsupported stream type");
#endif /* CCL_ENABLE_SYCL */
}

