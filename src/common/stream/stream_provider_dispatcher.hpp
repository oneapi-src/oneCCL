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

class ccl_stream;

class stream_provider_dispatcher {
public:
#ifdef MULTI_GPU_SUPPORT
#ifdef CCL_ENABLE_SYCL
    using stream_native_t = cl::sycl::queue;
    using stream_native_handle_t = ze_command_queue_handle_t; //cl::sycl::queue::cl_command_queue;
#else
    using stream_native_t = ze_command_queue_handle_t;
    using stream_native_handle_t = ze_command_queue_handle_t;
#endif
#else
#ifdef CCL_ENABLE_SYCL
    using stream_native_t = cl::sycl::queue;
    using stream_native_handle_t = void*;
#else
    using stream_native_t = void*;
    using stream_native_handle_t = stream_native_t;
#endif
#endif
    static stream_native_handle_t get_native_stream_handle_impl(stream_native_t& handle);

    stream_native_t get_native_stream() const;
    stream_native_handle_t get_native_stream_handle() const;
    std::string to_string() const;

    template <
        class NativeStream,
        typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                int>::type = 0>
    static std::unique_ptr<ccl_stream> create(NativeStream& native_stream);

    template <class NativeStream,
              typename std::enable_if<
                  not std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                  int>::type = 0>
    static std::unique_ptr<ccl_stream> create(NativeStream& native_stream);

    static std::unique_ptr<ccl_stream> create();

protected:
    template <
        class NativeStream,
        typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                int>::type = 0>
    stream_provider_dispatcher(NativeStream& stream);
    template <class NativeStream,
              typename std::enable_if<
                  not std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                  int>::type = 0>
    stream_provider_dispatcher(NativeStream stream);

private:
    stream_native_t native_stream;
    bool native_stream_set; //TODO use std::variant in c++17
#ifdef CCL_ENABLE_SYCL
    stream_native_handle_t native_stream_handle;
#endif
};
