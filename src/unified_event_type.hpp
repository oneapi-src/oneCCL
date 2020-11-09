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

#include "oneapi/ccl/type_traits.hpp"
#include "common/log/log.hpp"

namespace ccl {
#ifdef CCL_ENABLE_SYCL
#else

#ifdef MULTI_GPU_SUPPORT
#endif //MULTI_GPU_SUPPORT
#endif
} // namespace ccl
