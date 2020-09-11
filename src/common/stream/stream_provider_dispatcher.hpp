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
#ifdef MULTI_GPU_SUPPORT
#include <ze_api.h>
#endif

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif

#include "oneapi/ccl/ccl_type_traits.hpp"

class ccl_stream;
/*
#ifdef MULTI_GPU_SUPPORT
namespace native
{
    class ccl_device;
}
#endif
*/

class stream_provider_dispatcher {
public:
#ifdef MULTI_GPU_SUPPORT
    using stream_native_device_t = typename ccl::unified_device_type::ccl_native_t;
    using stream_native_context_t = typename ccl::unified_device_context_type::ccl_native_t;
    using stream_native_t = typename ccl::unified_stream_type::ccl_native_t;
    using stream_native_handle_t = typename ccl::unified_stream_type::handle_t;
#else
#ifdef CCL_ENABLE_SYCL
    using stream_native_t = cl::sycl::queue;
    using stream_native_device_t = cl::sycl::device;
    using stream_native_context_t = typename ccl::unified_device_context_type::ccl_native_t;
    using stream_native_handle_t = typename ccl::unified_stream_type::handle_t;
#else
    using stream_native_t = void*;
    using stream_native_device_t = void*;
    using stream_native_context_t = void*;
#endif
#endif
    stream_native_t get_native_stream() const;

    const stream_native_device_t& get_native_device() const;
    stream_native_device_t& get_native_device();

    std::string to_string() const;

    template <
        class NativeStream,
        typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                int>::type = 0>
    static std::unique_ptr<ccl_stream> create(NativeStream& native_stream,
                                              const ccl::library_version& version);

    template <class NativeStreamHandle,
              typename std::enable_if<
                  not std::is_class<typename std::remove_cv<NativeStreamHandle>::type>::value,
                  int>::type = 0>
    static std::unique_ptr<ccl_stream> create(NativeStreamHandle& native_stream,
                                              const ccl::library_version& version);

    static std::unique_ptr<ccl_stream> create(stream_native_device_t device,
                                              const ccl::library_version& version);
    static std::unique_ptr<ccl_stream> create(stream_native_device_t device,
                                              stream_native_context_t context,
                                              const ccl::library_version& version);

protected:
    template <
        class NativeStream,
        typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                int>::type = 0>
    stream_provider_dispatcher(NativeStream& stream);

    template <class NativeStreamHandle,
              typename std::enable_if<
                  not std::is_class<typename std::remove_cv<NativeStreamHandle>::type>::value,
                  int>::type = 0>
    stream_provider_dispatcher(NativeStreamHandle stream);
    stream_provider_dispatcher();

    stream_native_device_t native_device;
    stream_native_context_t native_context;
    bool creation_is_postponed{ false };
    stream_native_t native_stream;
};
