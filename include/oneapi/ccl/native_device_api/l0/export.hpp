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

#define CCL_BE_API CCL_HELPER_DLL_EXPORT

#define CL_BACKEND_TYPE ccl::cl_backend_type::l0

#include "oneapi/ccl/native_device_api/l0/declarations.hpp"

namespace ccl {

template <>
struct backend_info<CL_BACKEND_TYPE> {
    static constexpr ccl::cl_backend_type type() {
        return CL_BACKEND_TYPE;
    }
    static constexpr const char* name() {
        return "LEVEL_ZERO_BACKEND";
    }
};

template <>
struct generic_device_type<CL_BACKEND_TYPE> {
    using handle_t = device_index_type;
    using impl_t = native::ccl_device;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_device_type(device_index_type id);
    generic_device_type(ccl_native_t dev);
    device_index_type get_id() const noexcept;
    ccl_native_t get() noexcept;
    const ccl_native_t get() const noexcept;

    handle_t device;
};

template <>
struct generic_context_type<CL_BACKEND_TYPE> {
    using handle_t = ze_context_handle_t;
    using impl_t = native::ccl_context;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_context_type();
    generic_context_type(ccl_native_t ctx);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t context;
};

template <>
struct generic_platform_type<CL_BACKEND_TYPE> {
    using handle_t = native::ccl_device_platform;
    using impl_t = handle_t;
    using ccl_native_t = impl_t;

    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;
};

template <>
struct generic_stream_type<CL_BACKEND_TYPE> {
    using handle_t = ze_command_queue_handle_t;
    using impl_t = native::ccl_device::device_queue;
    using ccl_native_t = std::shared_ptr<impl_t>;

    generic_stream_type(handle_t q);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t queue;
};

template <>
struct generic_event_type<CL_BACKEND_TYPE> {
    using handle_t = ze_event_handle_t;
    using impl_t = handle_t;
    using ccl_native_t = std::shared_ptr<native::ccl_device::device_event>;

    generic_event_type(handle_t e);
    ccl_native_t& get() noexcept;
    const ccl_native_t& get() const noexcept;

    ccl_native_t event;
};

/**
 * Export CL native API supported types
 */
API_CLASS_TYPE_INFO(native::ccl_device::device_queue);
//API_CLASS_TYPE_INFO(ze_command_queue_handle_t);
API_CLASS_TYPE_INFO(ze_event_handle_t);
} // namespace ccl
