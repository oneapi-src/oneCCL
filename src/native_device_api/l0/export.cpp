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
#include "oneapi/ccl/config.h"
#if defined(CCL_ENABLE_ZE) and !defined(CCL_ENABLE_SYCL)

#include "oneapi/ccl/native_device_api/l0/export.hpp"
#include "common/log/log.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

namespace ccl {

/**
 * Context
 */
CCL_BE_API generic_context_type<cl_backend_type::l0>::generic_context_type() {}
CCL_BE_API generic_context_type<cl_backend_type::l0>::generic_context_type(ccl_native_t ctx)
        : context(ctx) {}

CCL_BE_API generic_context_type<cl_backend_type::l0>::ccl_native_t&
generic_context_type<cl_backend_type::l0>::get() noexcept {
    return const_cast<generic_context_type<cl_backend_type::l0>::ccl_native_t&>(
        static_cast<const generic_context_type<cl_backend_type::l0>*>(this)->get());
}

CCL_BE_API const generic_context_type<cl_backend_type::l0>::ccl_native_t&
generic_context_type<cl_backend_type::l0>::get() const noexcept {
    //TODO
    return context; //native::get_platform();
}

/**
 * Device
 */
CCL_BE_API generic_device_type<cl_backend_type::l0>::generic_device_type(device_index_type id)
        : device(id) {}

CCL_BE_API generic_device_type<cl_backend_type::l0>::generic_device_type(ccl_native_t dev)
        : device(dev->get_device_path()) {}

CCL_BE_API device_index_type generic_device_type<cl_backend_type::l0>::get_id() const noexcept {
    return device;
}

CCL_BE_API typename generic_device_type<cl_backend_type::l0>::ccl_native_t
generic_device_type<cl_backend_type::l0>::get() noexcept {
    return native::get_runtime_device(device);
}

CCL_BE_API const typename generic_device_type<cl_backend_type::l0>::ccl_native_t
generic_device_type<cl_backend_type::l0>::get() const noexcept {
    return native::get_runtime_device(device);
}

/**
 * Event
 */
CCL_BE_API generic_event_type<cl_backend_type::l0>::generic_event_type(handle_t e)
        : event(/*TODO use ccl_context to create event*/) {}

CCL_BE_API generic_event_type<cl_backend_type::l0>::ccl_native_t&
generic_event_type<cl_backend_type::l0>::get() noexcept {
    return const_cast<generic_event_type<cl_backend_type::l0>::ccl_native_t&>(
        static_cast<const generic_event_type<cl_backend_type::l0>*>(this)->get());
}

CCL_BE_API const generic_event_type<cl_backend_type::l0>::ccl_native_t&
generic_event_type<cl_backend_type::l0>::get() const noexcept {
    return event;
}

/**
 * Stream
 */
CCL_BE_API generic_stream_type<cl_backend_type::l0>::generic_stream_type(handle_t q)
        : queue(/*TODO use ccl_device to create event*/) {}

CCL_BE_API generic_stream_type<cl_backend_type::l0>::ccl_native_t&
generic_stream_type<cl_backend_type::l0>::get() noexcept {
    return const_cast<generic_stream_type<cl_backend_type::l0>::ccl_native_t&>(
        static_cast<const generic_stream_type<cl_backend_type::l0>*>(this)->get());
}

CCL_BE_API const generic_stream_type<cl_backend_type::l0>::ccl_native_t&
generic_stream_type<cl_backend_type::l0>::get() const noexcept {
    return queue;
}

/**
 * Platform
 */
CCL_BE_API generic_platform_type<cl_backend_type::l0>::ccl_native_t&
generic_platform_type<cl_backend_type::l0>::get() noexcept {
    return const_cast<generic_platform_type<cl_backend_type::l0>::ccl_native_t&>(
        static_cast<const generic_platform_type<cl_backend_type::l0>*>(this)->get());
}

CCL_BE_API const generic_platform_type<cl_backend_type::l0>::ccl_native_t&
generic_platform_type<cl_backend_type::l0>::get() const noexcept {
    return native::get_platform();
}
} // namespace ccl
#endif //CCL_ENABLE_ZE
