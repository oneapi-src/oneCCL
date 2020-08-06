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
#include "common/comm/l0/base_connector.hpp"
#include "common/utils/tuple.hpp"

template <class managed_kernel, class entry, class... binded_kernels>
struct kernel_connector : public base_connector_interface<managed_kernel> {
    using base = base_connector_interface<managed_kernel>;
    using binding_type = std::tuple<std::reference_wrapper<binded_kernels>...>;
    kernel_connector(entry& e, binded_kernels&... args)
            : base(),
              executor(e),
              deferred_kernels(args...) {}

    bool operator()(managed_kernel& kernel_to_connect) override {
        return connect_impl(
            kernel_to_connect,
            typename sequence_generator<std::tuple_size<binding_type>::value>::type());
    }

private:
    template <int... connected_arguments_indices>
    bool connect_impl(managed_kernel& kernel_to_connect,
                      numeric_sequence<connected_arguments_indices...>) {
        return executor.execute(kernel_to_connect,
                                std::get<connected_arguments_indices>(deferred_kernels).get()...);
    }

    entry& executor;
    binding_type deferred_kernels;
};
