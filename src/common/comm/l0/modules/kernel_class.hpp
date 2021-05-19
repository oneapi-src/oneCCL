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
#include "common/utils/tuple.hpp"
#include <unordered_map>

namespace native {

template <ccl_coll_type type, class kernel_function_impl, class Enable = void>
struct kernel_class {
    using kernel_t = kernel_function_impl;

    using key_type = ccl::datatype;

    struct hasher {
        size_t operator()(const ccl::datatype& dtype) const {
            return std::hash<size_t>{}((size_t)dtype);
        }
    };

    using kernel_class_container_t = std::unordered_map<key_type, kernel_t, hasher>;

    kernel_class() {
        for (ccl::datatype idx = ccl::datatype::int8; idx <= ccl::datatype::bfloat16; idx++) {
            key_type key{ idx };
            // Have to use this ugly inplace construction because kernel_t have deleted copy and move
            // constructor and there is no other way to do that.
            value.emplace(std::piecewise_construct,
                          std::make_tuple(key),
                          std::make_tuple(coll_param_gpu(type, idx)));
        }
    }
    // getter
    kernel_t& get(const coll_param_gpu& params) {
        assert(!params.is_reduction());
        key_type key{ params.get_datatype() };

        auto it = value.find(key);
        if (it == value.end()) {
            // TODO: sycl error
            throw std::runtime_error("Kernel not found");
        }

        return it->second;
    }

protected:
    kernel_class_container_t value;
};

template <ccl_coll_type type, class kernel_function_impl>
struct kernel_class<type,
                    kernel_function_impl,
                    typename std::enable_if<is_reduction_coll_type<type>::value>::type> {
    using kernel_t = kernel_function_impl;

    using key_type = std::pair<ccl::datatype, ccl::reduction>;

    struct hasher {
        size_t operator()(const std::pair<ccl::datatype, ccl::reduction>& key) const {
            return std::hash<size_t>{}((size_t)key.first) ^ std::hash<size_t>{}((size_t)key.second);
        }
    };

    using kernel_class_container_t = std::unordered_map<key_type, kernel_t, hasher>;

    kernel_class() {
        for (ccl::datatype idx = ccl::datatype::int8; idx <= ccl::datatype::bfloat16; idx++) {
            // TODO: allow to iterate over reduction values(need to implement operator++)
            auto insert_kernel = [this, idx](ccl::reduction red) {
                key_type key{ idx, red };
                value.emplace(std::piecewise_construct,
                              std::make_tuple(key),
                              std::make_tuple(coll_param_gpu(type, idx, red)));
            };

            insert_kernel(ccl::reduction::sum);
            insert_kernel(ccl::reduction::prod);
            insert_kernel(ccl::reduction::min);
            insert_kernel(ccl::reduction::max);
        }
    }

    // getter
    kernel_t& get(const coll_param_gpu& params) {
        assert(params.is_reduction());

        key_type key{ params.get_datatype(), params.get_reduction() };

        auto it = value.find(key);
        if (it == value.end()) {
            // TODO: sycl error
            throw std::runtime_error("Kernel not found");
        }

        return it->second;
    }

protected:
    // TODO: threadsafety? Looks like this should be fine as different threads access different devices.
    // Need to double check IPC/NUMA case.
    kernel_class_container_t value;
};

} //namespace native
