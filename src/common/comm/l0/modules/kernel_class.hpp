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
#include <tuple>
#include "common/comm/l0/modules/kernel_params.hpp"
#include "common/utils/tuple.hpp"

namespace native {

#define SUPPORTED_KERNEL_NATIVE_DATA_TYPES \
    int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, ccl::float16, float, \
        double, ccl::bfloat16

template <ccl_coll_type type, template <typename> class kernel_function_impl>
struct kernel_class {
    template <class native_data_type>
    using kernel_param_t = kernel_params_default<native_data_type>;

    template <class kernel_param>
    using kernel_t = kernel_function_impl<kernel_param>;

    template <class... native_data_types>
    using kernels_t = std::tuple<kernel_t<kernel_param_t<native_data_types>>...>;

    using kernel_class_container_t = kernels_t<SUPPORTED_KERNEL_NATIVE_DATA_TYPES>;

    // getter
    template <class kernel_param>
    const kernel_t<kernel_param> &get() const {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

    template <class kernel_param>
    kernel_t<kernel_param> &get() {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

protected:
    kernel_class_container_t value;
};

template <template <typename> class kernel_function_impl>
struct kernel_class<ccl_coll_allreduce, kernel_function_impl> {
    template <class native_data_type, ccl_coll_reduction reduction>
    using kernel_param_t = kernel_reduction_params_traits<native_data_type, reduction>;

    template <class kernel_param>
    using kernel_t = kernel_function_impl<kernel_param>;

    template <class first_param, ccl_coll_reduction... second_params>
    using kernel_second_params_expanded_t =
        std::tuple<kernel_t<kernel_param_t<first_param, second_params>>...>;

    template <class... first_params>
    using kernel_first_param_expanded_t = decltype(std::tuple_cat(
        std::declval<kernel_second_params_expanded_t<first_params, REDUCE_TYPES> &&>()...));

    using kernel_class_container_t =
        kernel_first_param_expanded_t<SUPPORTED_KERNEL_NATIVE_DATA_TYPES>;

    // getter
    template <class kernel_param>
    const kernel_t<kernel_param> &get() const {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

    template <class kernel_param>
    kernel_t<kernel_param> &get() {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

protected:
    kernel_class_container_t value;
};

template <template <typename> class kernel_function_impl>
struct kernel_class<ccl_coll_reduce, kernel_function_impl> {
    template <class native_data_type, ccl_coll_reduction reduction>
    using kernel_param_t = kernel_reduction_params_traits<native_data_type, reduction>;

    template <class kernel_param>
    using kernel_t = kernel_function_impl<kernel_param>;

    template <class first_param, ccl_coll_reduction... second_params>
    using kernel_second_params_expanded_t =
        std::tuple<kernel_t<kernel_param_t<first_param, second_params>>...>;

    template <class... first_params>
    using kernel_first_param_expanded_t = decltype(std::tuple_cat(
        std::declval<kernel_second_params_expanded_t<first_params, REDUCE_TYPES> &&>()...));

    using kernel_class_container_t =
        kernel_first_param_expanded_t<SUPPORTED_KERNEL_NATIVE_DATA_TYPES>;

    // getter
    template <class kernel_param>
    const kernel_t<kernel_param> &get() const {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

    template <class kernel_param>
    kernel_t<kernel_param> &get() {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

protected:
    kernel_class_container_t value;
};

template <template <typename> class kernel_function_impl>
struct kernel_class<ccl_coll_reduce_scatter, kernel_function_impl> {
    template <class native_data_type, ccl_coll_reduction reduction>
    using kernel_param_t = kernel_reduction_params_traits<native_data_type, reduction>;

    template <class kernel_param>
    using kernel_t = kernel_function_impl<kernel_param>;

    template <class first_param, ccl_coll_reduction... second_params>
    using kernel_second_params_expanded_t =
        std::tuple<kernel_t<kernel_param_t<first_param, second_params>>...>;

    template <class... first_params>
    using kernel_first_param_expanded_t = decltype(std::tuple_cat(
        std::declval<kernel_second_params_expanded_t<first_params, REDUCE_TYPES> &&>()...));

    using kernel_class_container_t =
        kernel_first_param_expanded_t<SUPPORTED_KERNEL_NATIVE_DATA_TYPES>;

    // getter
    template <class kernel_param>
    const kernel_t<kernel_param> &get() const {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

    template <class kernel_param>
    kernel_t<kernel_param> &get() {
        return ccl_tuple_get<kernel_t<kernel_param>>(value);
    }

protected:
    kernel_class_container_t value;
};
} //namespace native
