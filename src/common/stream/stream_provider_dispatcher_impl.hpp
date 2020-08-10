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

#include "common/stream/stream_provider_dispatcher.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif /* CCL_ENABLE_SYCL */

template <class NativeStream,
          typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                  int>::type>
stream_provider_dispatcher::stream_provider_dispatcher(NativeStream& stream)
        : native_stream(stream),
          native_stream_set(true) {
#ifdef CCL_ENABLE_SYCL
    native_stream_handle = get_native_stream_handle_impl(native_stream);
#endif
}

template <
    class NativeStream,
    typename std::enable_if<not std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                            int>::type>
stream_provider_dispatcher::stream_provider_dispatcher(NativeStream stream) {
#ifdef CCL_ENABLE_SYCL
    (void)native_stream;
    native_stream_set = false; //not set cl::sycl::queue
    native_stream_handle = static_cast<stream_native_handle_t>(stream);
#else
    native_stream = static_cast<stream_native_t>(stream);
    if (native_stream)
        native_stream_set = true;
    else
        native_stream_set = false;
#endif
}

stream_provider_dispatcher::stream_native_t stream_provider_dispatcher::get_native_stream() const {
    if (!native_stream_set) {
        throw ccl::ccl_error("native stream is not set");
    }

    return native_stream;
}

stream_provider_dispatcher::stream_native_handle_t
stream_provider_dispatcher::get_native_stream_handle() const {
#ifdef CCL_ENABLE_SYCL
    return native_stream_handle;
#else
    if (!native_stream_set) {
        throw ccl::ccl_error("native stream is not set");
    }
    return native_stream;
#endif
}

std::string stream_provider_dispatcher::to_string() const {
    std::stringstream ss;
#ifdef CCL_ENABLE_SYCL
    if (native_stream_set) {
        ss << "sycl: "
           << native_stream.get_info<cl::sycl::info::queue::device>()
                  .get_info<cl::sycl::info::device::name>();
    }
    else {
        ss << "native: " << native_stream_handle;
    }
#else
    ss << native_stream;
#endif
    return ss.str();
}

template <class NativeStream,
          typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                  int>::type>
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(NativeStream& native_stream) {
    ccl_stream_type_t type = ccl_stream_cpu;
#ifdef CCL_ENABLE_SYCL
    type = native_stream.get_device().is_host() ? ccl_stream_cpu : ccl_stream_gpu;
    LOG_INFO("SYCL queue's device is ",
             native_stream.get_device().template get_info<cl::sycl::info::device::name>());
#endif /* CCL_ENABLE_SYCL */

    return std::unique_ptr<ccl_stream>(new ccl_stream(type, native_stream));
}

template <
    class NativeStream,
    typename std::enable_if<not std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                            int>::type>
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(NativeStream& native_stream) {
    static_assert(std::is_same<NativeStream, stream_native_handle_t>::value,
                  "Unsupported 'NativeStream'");
    return std::unique_ptr<ccl_stream>(new ccl_stream(ccl_stream_gpu, native_stream));
}

std::unique_ptr<ccl_stream> stream_provider_dispatcher::create() {
    void* ptr = nullptr;
    return std::unique_ptr<ccl_stream>(new ccl_stream(ccl_stream_host, ptr));
}

stream_provider_dispatcher::stream_native_handle_t
stream_provider_dispatcher::get_native_stream_handle_impl(stream_native_t& handle) {
#ifdef CCL_ENABLE_SYCL
    if (!handle.get_device().is_host()) {
        return *reinterpret_cast<stream_native_handle_t*>(handle.get());
    }
    else {
        return *reinterpret_cast<stream_native_handle_t*>(&handle);
    }
#else
    return handle;
#endif
}
