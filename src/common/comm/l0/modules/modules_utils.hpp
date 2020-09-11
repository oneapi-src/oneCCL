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
#include "common/comm/l0/modules/base_entry_module.hpp"
#include "common/utils/tuple.hpp"

namespace native {
namespace detail {

struct kernel_entry_initializer {
    using loader_t =
        std::function<gpu_module_base::kernel_handle(const std::string& function_name)>;

    kernel_entry_initializer(loader_t&& f) : functor(std::move(f)) {}

    template <class typed_kernel>
    void operator()(typed_kernel& kernel) {
        kernel.handle =
            functor(std::string(typed_kernel::name()) + "_" +
                    ccl::native_type_info<typename typed_kernel::processing_type>::name());
    }

private:
    loader_t functor;
};
} // namespace detail
} // namespace native
