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
#include "common/stream/stream_provider_dispatcher_impl.hpp"

#ifdef CCL_ENABLE_SYCL
    template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(cl::sycl::queue& native_stream);
#endif

#ifdef MULTI_GPU_SUPPORT
    template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(ze_command_queue_handle_t& native_stream);
#else
    template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(void*& native_stream);
#endif
