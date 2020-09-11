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

#include "ccl_types.h"

#include <bitset>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace ccl {

/**
 * Supported reduction operations
 */
enum class reduction {
    sum = ccl_reduction_sum,
    prod = ccl_reduction_prod,
    min = ccl_reduction_min,
    max = ccl_reduction_max,
    custom = ccl_reduction_custom,

    last_value = ccl_reduction_last_value
};

/**
 * Supported datatypes
 */
enum datatype : int {
    dt_char = ccl_dtype_char,
    dt_int = ccl_dtype_int,
    dt_bfp16 = ccl_dtype_bfp16,
    dt_float = ccl_dtype_float,
    dt_double = ccl_dtype_double,
    dt_int64 = ccl_dtype_int64,
    dt_uint64 = ccl_dtype_uint64,

    dt_last_value = ccl_dtype_last_value
};

/**
 * Supported stream types
 */
enum class stream_type {
    host = ccl_stream_host,
    cpu = ccl_stream_cpu,
    gpu = ccl_stream_gpu,

    last_value = ccl_stream_last_value
};

typedef ccl_coll_attr_t coll_attr;

typedef ccl_comm_attr_t comm_attr;

typedef ccl_datatype_attr_t datatype_attr;

template <ccl_host_attributes attrId>
struct ccl_host_attributes_traits {};

/**
 * Exception type that may be thrown by ccl API
 */
class ccl_error : public std::runtime_error {
public:
    explicit ccl_error(const std::string& message) : std::runtime_error(message) {}

    explicit ccl_error(const char* message) : std::runtime_error(message) {}
};

/**
 * Type traits, which describes how-to types would be interpretered by ccl API
 */
template <class ntype_t,
          size_t size_of_type,
          ccl_datatype_t ccl_type_v,
          bool iclass = false,
          bool supported = false>
struct ccl_type_info_export {
    using native_type = ntype_t;
    using ccl_type = std::integral_constant<ccl_datatype_t, ccl_type_v>;
    static constexpr size_t size = size_of_type;
    static constexpr ccl_datatype_t ccl_type_value = ccl_type::value;
    static constexpr datatype ccl_datatype_value = static_cast<datatype>(ccl_type_value);
    static constexpr bool is_class = iclass;
    static constexpr bool is_supported = supported;
};

} // namespace ccl
#ifdef MULTI_GPU_SUPPORT
#include "ccl_device_types.hpp"
#endif
