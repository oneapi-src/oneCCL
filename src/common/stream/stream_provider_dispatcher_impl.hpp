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
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    stream_native_t& native_stream,
    const ccl::library_version& version) {
    stream_type type = stream_type::host;
#ifdef CCL_ENABLE_SYCL
    if (native_stream.get_device().is_host()) {
        type = stream_type::host;
    }
    else if (native_stream.get_device().is_cpu()) {
        type = stream_type::cpu;
    }
    else if (native_stream.get_device().is_gpu()) {
        type = stream_type::gpu;
    }
    else {
        throw ccl::invalid_argument(
            "CORE",
            "create_stream",
            std::string("Unsupported SYCL queue's device type:\n") +
                native_stream.get_device().template get_info<cl::sycl::info::device::name>() +
                std::string("Supported types: host, cpu, gpu"));
    }

    std::unique_ptr<ccl_stream> ret(new ccl_stream(type, native_stream, version));
    ret->native_device.second = native_stream.get_device();
    ret->native_device.first = true;
    ret->native_context.second = native_stream.get_context();
    ret->native_context.first = true;
    LOG_INFO("SYCL queue type: ",
             ::to_string(type),
             ", device: ",
             native_stream.get_device().template get_info<cl::sycl::info::device::name>());

#else
#ifdef MULTI_GPU_SUPPORT
    LOG_INFO("L0 queue type: gpu - supported only");
    type = stream_type::gpu;
    std::unique_ptr<ccl_stream> ret(new ccl_stream(type, native_stream, version));
    ret->native_device.second = native_stream->get_owner().lock();
    ret->native_device.first = true;
    ret->native_context.second = native_stream->get_ctx().lock();
    ret->native_context.first = true;
#else
    std::unique_ptr<ccl_stream> ret(new ccl_stream(type, native_stream, version));
#endif
#endif /* CCL_ENABLE_SYCL */

    return ret;
}

// Creation from handles: cl_queue or ze_device_queue_handle_t
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    stream_native_handle_t native_stream,
    const ccl::library_version& version) {
    return std::unique_ptr<ccl_stream>(new ccl_stream(stream_type::gpu, native_stream, version));
}

// Postponed creation from device
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    stream_native_device_t device,
    const ccl::library_version& version) {
    auto ret = std::unique_ptr<ccl_stream>(new ccl_stream(stream_type::gpu, version));
    ret->native_device.second = device;
    ret->native_device.first = true;
    return ret;
}

// Postponed creation from device & context
std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    stream_native_device_t device,
    stream_native_context_t context,
    const ccl::library_version& version) {
    auto ret = stream_provider_dispatcher::create(device, version);
    ret->native_context.second = context;
    ret->native_context.first = true;
    return ret;
}

stream_provider_dispatcher::stream_native_t stream_provider_dispatcher::get_native_stream() const {
    if (creation_is_postponed) {
        throw ccl::exception("native stream is not set");
    }

    return native_stream;
}

#ifdef CCL_ENABLE_SYCL
stream_provider_dispatcher::stream_native_t* stream_provider_dispatcher::get_native_stream(
    size_t idx) {
    if (creation_is_postponed) {
        throw ccl::exception("native stream is not set");
    }

    if (idx >= native_streams.size()) {
        throw ccl::exception("unexpected stream idx");
    }

    return &(native_streams[idx]);
}
#endif /* CCL_ENABLE_SYCL */

const stream_provider_dispatcher::stream_native_device_t&
stream_provider_dispatcher::get_native_device() const {
    if (!native_device.first) {
        throw ccl::exception(std::string(__FUNCTION__) + " - stream has no native device");
    }
    return native_device.second;
}

stream_provider_dispatcher::stream_native_device_t&
stream_provider_dispatcher::get_native_device() {
    return const_cast<stream_provider_dispatcher::stream_native_device_t&>(
        static_cast<const stream_provider_dispatcher*>(this)->get_native_device());
}

std::string stream_provider_dispatcher::to_string() const {
    if (creation_is_postponed) {
        throw ccl::exception("stream is not properly created yet");
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
