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
#include "oneapi/ccl/type_traits.hpp"
#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif

namespace native {
namespace detail {

#ifdef CCL_ENABLE_SYCL
size_t get_sycl_device_id(const cl::sycl::device& dev);
size_t get_sycl_subdevice_id(const cl::sycl::device& dev);
#endif

} // namespace detail
} // namespace native
