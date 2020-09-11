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

#include "oneapi/ccl/ccl_type_traits.hpp"
#include "common/log/log.hpp"

namespace ccl {
#ifdef CCL_ENABLE_SYCL

generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::generic_device_context_type(ccl_native_t ctx)
        : context(ctx) {}

generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::ccl_native_t&
generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::get() noexcept {
    return const_cast<generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::ccl_native_t&>(
        static_cast<const generic_device_context_type<CCL_ENABLE_SYCL_TRUE>*>(this)->get());
}

const generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::ccl_native_t&
generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::get() const noexcept {
    return context;
}

#else
#ifdef MULTI_GPU_SUPPORT

generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::generic_device_context_type(handle_t ctx)
        : context() {
    //TODO context
    (void)ctx;

    throw;
}

generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::ccl_native_t
generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::get() noexcept {
    return /*const_cast<generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::ccl_native_t>*/ (
        static_cast<const generic_device_context_type<CCL_ENABLE_SYCL_FALSE>*>(this)->get());
}

const generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::ccl_native_t&
generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::get() const noexcept {
    //TODO
    return context; //native::get_platform();
}
#endif //MULTI_GPU_SUPPORT
#endif
} // namespace ccl
