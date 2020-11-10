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
#if !defined(MULTI_GPU_SUPPORT) and !defined(CCL_ENABLE_SYCL)

#include "oneapi/ccl/native_device_api/empty/export.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "common/log/log.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

namespace ccl {

generic_context_type<cl_backend_type::empty_backend>::ccl_native_t
generic_context_type<cl_backend_type::empty_backend>::get() noexcept {
    return /*const_cast<generic_context_type<cl_backend_type::l0>::ccl_native_t>*/ (
        static_cast<const generic_context_type<cl_backend_type::empty_backend>*>(this)->get());
}

const generic_context_type<cl_backend_type::empty_backend>::ccl_native_t&
generic_context_type<cl_backend_type::empty_backend>::get() const noexcept {
    //TODO
    return context; //native::get_platform();
}
} // namespace ccl

#endif //#if !defined(MULTI_GPU_SUPPORT) and !defined(CCL_ENABLE_SYCL)
