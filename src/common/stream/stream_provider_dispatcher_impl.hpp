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

// Creation from class-type: cl::sycl::queue or native::ccl_device::devie_queue
template <class NativeStream,
          typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                  int>::type>
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    NativeStream& native_stream,
    const ccl::library_version& version) {
    static_assert(std::is_same<NativeStream, stream_native_t>::value, "Unsupported 'NativeStream'");

    ccl_stream_type_t type = ccl_stream_cpu;
#ifdef CCL_ENABLE_SYCL
    type = native_stream.get_device().is_host() ? ccl_stream_cpu : ccl_stream_gpu;
    LOG_INFO("SYCL queue's device is ",
             native_stream.get_device().template get_info<cl::sycl::info::device::name>());
#else
    //L0 at now for GPU only
    type = ccl_stream_gpu;
#endif /* CCL_ENABLE_SYCL */

    return std::unique_ptr<ccl_stream>(new ccl_stream(type, native_stream, version));
}

// Creation from handles: cl_queue or ze_device_queue_handle_t
template <class NativeStreamHandle,
          typename std::enable_if<
              not std::is_class<typename std::remove_cv<NativeStreamHandle>::type>::value,
              int>::type>
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    NativeStreamHandle& native_stream,
    const ccl::library_version& version) {
    static_assert(std::is_same<NativeStreamHandle, stream_native_handle_t>::value,
                  "Unsupported 'NativeStream'");
    return std::unique_ptr<ccl_stream>(new ccl_stream(ccl_stream_gpu, native_stream, version));
}

// Postponed creation from device
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    stream_native_device_t device,
    const ccl::library_version& version) {
    auto ret = std::unique_ptr<ccl_stream>(new ccl_stream(ccl_stream_gpu, version));
    ret->native_device = device;
    return ret;
}

// Postponed creation from device & context
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    stream_native_device_t device,
    stream_native_context_t context,
    const ccl::library_version& version) {
    auto ret = stream_provider_dispatcher::create(device, version);
    ret->native_context = context;
    return ret;
}

template <class NativeStream,
          typename std::enable_if<std::is_class<typename std::remove_cv<NativeStream>::type>::value,
                                  int>::type>
stream_provider_dispatcher::stream_provider_dispatcher(NativeStream& stream)
        : native_stream(stream) {}

template <class NativeStreamHandle,
          typename std::enable_if<
              not std::is_class<typename std::remove_cv<NativeStreamHandle>::type>::value,
              int>::type>
stream_provider_dispatcher::stream_provider_dispatcher(NativeStreamHandle stream) {
    creation_is_postponed = true;
    /*
#ifdef CCL_ENABLE_SYCL
    native_stream = stream_native_t{stream};
#else
    native_stream = ccl::unified_stream_type{stream}.get();
#endif*/
}

stream_provider_dispatcher::stream_provider_dispatcher() {
    creation_is_postponed = true;
}

stream_provider_dispatcher::stream_native_t stream_provider_dispatcher::get_native_stream() const {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("native stream is not set");
    }

    return native_stream;
}

const stream_provider_dispatcher::stream_native_device_t&
stream_provider_dispatcher::get_native_device() const {
    if (creation_is_postponed) {
        throw ccl::ccl_error("native device is not set");
    }
    return native_device;
}

stream_provider_dispatcher::stream_native_device_t&
stream_provider_dispatcher::get_native_device() {
    return const_cast<stream_provider_dispatcher::stream_native_device_t&>(
        static_cast<const stream_provider_dispatcher*>(this)->get_native_device());
}

std::string stream_provider_dispatcher::to_string() const {
    if (creation_is_postponed) {
        throw ccl::ccl_error("native device is not set");
    }
    std::stringstream ss;
#ifdef CCL_ENABLE_SYCL
    ss << "sycl: "
       << native_stream.get_info<cl::sycl::info::queue::device>()
              .get_info<cl::sycl::info::device::name>();
#else
    ss << reinterpret_cast<void*>(native_stream.get()); //TODO
#endif
    return ss.str();
}

/*
stream_provider_dispatcher::stream_native_handle_t
stream_provider_dispatcher::get_native_stream_handle_impl(stream_native_t &handle)
{
#ifdef CCL_ENABLE_SYCL
    if (!handle.get_device().is_host())
    {
        return *reinterpret_cast<stream_native_handle_t*>(handle.get());
    }
    else
    {
        return *reinterpret_cast<stream_native_handle_t*>(&handle);
    }
#else
        return handle;
#endif
}
*/
