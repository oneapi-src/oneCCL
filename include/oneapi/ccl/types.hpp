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

#include <stdint.h>
#include <stdlib.h>
#include "oneapi/ccl/config.h"

#include <bitset>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "oneapi/ccl/aliases.hpp"
#include "oneapi/ccl/exception.hpp"

namespace ccl {

namespace v1 {

/**
 * Supported reduction operations
 */
enum class reduction : int {
    sum = 0,
    prod,
    min,
    max,
    custom,

    last_value
};

/**
 * Supported datatypes
 */
enum class datatype : int {
    int8 = 0,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,

    float16,
    float32,
    float64,

    bfloat16,

    last_predefined = bfloat16
};

/**
 * Supported CL backend types
 */
enum class cl_backend_type : int {
    empty_backend = 0x0,
    dpcpp_sycl = 0x1,
    l0 = 0x2,
    dpcpp_sycl_l0 = 0x3,

    last_value
};

} // namespace v1

using v1::reduction;
using v1::datatype;
using v1::cl_backend_type;

/**
 * Type traits, which describes how-to types would be interpretered by ccl API
 */
template <class ntype_t,
          size_t size_of_type,
          ccl::datatype ccl_type_v,
          bool iclass = false,
          bool supported = false>
struct ccl_type_info_export {
    using native_type = ntype_t;
    using ccl_type = std::integral_constant<ccl::datatype, ccl_type_v>;
    static constexpr size_t size = size_of_type;
    static constexpr datatype dtype = static_cast<enum datatype>(ccl_type::value);
    static constexpr bool is_class = iclass;
    static constexpr bool is_supported = supported;
};

namespace v1 {

/**
 * Library version description
 */
typedef struct {
    unsigned int major;
    unsigned int minor;
    unsigned int update;
    const char* product_status;
    const char* build_date;
    const char* full;
    string_class cl_backend_name;
} library_version;

typedef struct {
    const char* match_id;
    const size_t offset;
} fn_context;

/* in_buf, in_count, inout_buf, out_count, dtype, context */
typedef void (
    *reduction_fn)(const void*, size_t, void*, size_t*, ccl::datatype, const ccl::v1::fn_context*);

struct ccl_empty_attr {
    static ccl::v1::library_version version;

    template <class attr>
    static attr create_empty();
};

/**
 * Sparse coalesce modes
 * 
 * Use this variable to set sparse_allreduce coalescing mode:
 * regular        - run regular coalesce funtion;
 * disable        - disables coalesce function in sparse_allreduce,
 *                  allgathered data is returned;
 * keep_precision - on every local reduce bf16 data is converted to fp32,
 *                  reduced and then converted back to bf16.
 */
enum class sparse_coalesce_mode : int {
    regular = 0,
    disable,
    keep_precision,

    last_value
};

/* idx_buf, idx_count, idx_dtype, val_buf, val_count, val_dtype, user_context */
typedef void (*sparse_allreduce_completion_fn)(const void*,
                                               size_t,
                                               ccl::datatype,
                                               const void*,
                                               size_t,
                                               ccl::datatype,
                                               const void*);

/* idx_count, idx_dtype, val_count, val_dtype, user_context, out_idx_buf, out_val_buf */
typedef void (*sparse_allreduce_alloc_fn)(size_t,
                                          ccl::datatype,
                                          size_t,
                                          ccl::datatype,
                                          const void*,
                                          void**,
                                          void**);
} // namespace v1

using v1::library_version;
using v1::fn_context;
using v1::reduction_fn;
using v1::ccl_empty_attr;

using v1::sparse_coalesce_mode;
using v1::sparse_allreduce_completion_fn;
using v1::sparse_allreduce_alloc_fn;

/**
 * API object attributes traits
 */
namespace info {
template <class param_type, param_type value>
struct param_traits {};

} //namespace info
} // namespace ccl

#include "oneapi/ccl/device_types.hpp"
