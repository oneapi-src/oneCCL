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
#include "oneapi/ccl/ccl_config.h"

#include <bitset>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "oneapi/ccl/ccl_aliases.hpp"

// TODO: tmp enums, refactor core code and remove them
/************************************************/
typedef int ccl_datatype_t;

typedef int ccl_reduction_t;

typedef int ccl_status_t;
#define ccl_status_success               (0)
#define ccl_status_out_of_resource       (1)
#define ccl_status_invalid_arguments     (2)
#define ccl_status_unimplemented         (3)
#define ccl_status_runtime_error         (4)
#define ccl_status_blocked_due_to_resize (5)
#define ccl_status_last_value            (6)

/** Resize action types. */
typedef enum ccl_resize_action {
    /* Wait additional changes for number of ranks */
    ccl_ra_wait = 0,
    /* Run with current number of ranks */
    ccl_ra_run = 1,
    /* Finalize work */
    ccl_ra_finalize = 2,
} ccl_resize_action_t;

/* comm_size */
typedef ccl_resize_action_t (*ccl_resize_fn_t)(size_t comm_size);

/** Stream types. */
typedef enum {
    ccl_stream_host = 0,
    ccl_stream_cpu = 1,
    ccl_stream_gpu = 2,

    ccl_stream_last_value
} ccl_stream_type_t;
/************************************************/

namespace ccl {

/** Library version description. */
typedef struct {
    unsigned int major;
    unsigned int minor;
    unsigned int update;
    const char* product_status;
    const char* build_date;
    const char* full;
} library_version;

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

inline std::ostream& operator<<(std::ostream& os, const ccl::datatype& dt) {
    os << static_cast<std::underlying_type<ccl::datatype>::type>(dt);
    return os;
}

typedef struct {
    const char* match_id;
    const size_t offset;
} fn_context;

/* Sparse coalesce modes */
/* Use this variable to set sparse_allreduce coalescing mode:
   ccl_sparse_coalesce_regular run regular coalesce funtion;
   ccl_sparse_coalesce_disable disables coalesce function in sparse_allreduce,
                               allgathered data is returned;
   ccl_sparse_coalesce_keep_precision on every local reduce bfp16 data is
                               converted to fp32, reduced and then converted
                               back to bfp16.
*/

enum class sparse_coalesce_mode : int { regular = 0, disable = 1, keep_precision = 2 };

/* comm_size */
typedef ccl_resize_action_t (*ccl_resize_fn_t)(size_t comm_size);

/* in_buf, in_count, in_dtype, out_buf, out_count, out_dtype, context */
typedef void (*prologue_fn)(const void*,
                            size_t,
                            ccl::datatype,
                            void**,
                            size_t*,
                            ccl::datatype*,
                            const ccl::fn_context*);

/* in_buf, in_count, in_dtype, out_buf, out_count, out_dtype, context */
typedef void (*epilogue_fn)(const void*,
                            size_t,
                            ccl::datatype,
                            void*,
                            size_t*,
                            ccl::datatype,
                            const ccl::fn_context*);

/* in_buf, in_count, inout_buf, out_count, dtype, context */
typedef void (
    *reduction_fn)(const void*, size_t, void*, size_t*, ccl::datatype, const ccl::fn_context*);

/* idx_buf, idx_count, idx_dtype, val_buf, val_count, val_dtype, fn_context */
typedef void (*sparse_allreduce_completion_fn)(const void*,
                                               size_t,
                                               ccl::datatype,
                                               const void*,
                                               size_t,
                                               ccl::datatype,
                                               const void*);

/* idx_count, idx_dtype, val_count, val_dtype, fn_context, out_idx_buf, out_val_buf */
typedef void (*sparse_allreduce_alloc_fn)(size_t,
                                          ccl::datatype,
                                          size_t,
                                          ccl::datatype,
                                          const void*,
                                          void**,
                                          void**);

// using datatype_attr_t = ccl_datatype_attr_t;
/**
 * Supported stream types
 */
enum class stream_type : int {
    host = 0,
    cpu,
    gpu,

    last_value
};

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

struct ccl_empty_attr {
    static ccl::library_version version;

    template <class attr>
    static attr create_empty();
};

/**
 * API object attributes traits
 */
namespace info {
template <class param_type, param_type value>
struct param_traits {};

} //namespace info
} // namespace ccl

/* TODO: tmp mappings */

/*********************************************************/

#define ccl_dtype_char       (int)(ccl::datatype::int8)
#define ccl_dtype_int        (int)(ccl::datatype::int32)
#define ccl_dtype_int64      (int)(ccl::datatype::int64)
#define ccl_dtype_uint64     (int)(ccl::datatype::uint64)
#define ccl_dtype_bfp16      (int)(ccl::datatype::bfloat16)
#define ccl_dtype_float      (int)(ccl::datatype::float32)
#define ccl_dtype_double     (int)(ccl::datatype::float64)
#define ccl_dtype_last_value (int)(ccl::datatype::last_predefined)

#define ccl_reduction_sum        (int)(ccl::reduction::sum)
#define ccl_reduction_min        (int)(ccl::reduction::min)
#define ccl_reduction_max        (int)(ccl::reduction::max)
#define ccl_reduction_prod       (int)(ccl::reduction::prod)
#define ccl_reduction_custom     (int)(ccl::reduction::custom)
#define ccl_reduction_last_value (int)(ccl::reduction::last_value)

// TODO: tmp struct, refactor core code and remove it
/*********************************************************/

/** Extendable list of collective attributes. */
typedef struct {
    /**
     * Callbacks into application code
     * for pre-/post-processing data
     * and custom reduction operation
     */
    ccl::prologue_fn prologue_fn;
    ccl::epilogue_fn epilogue_fn;
    ccl::reduction_fn reduction_fn;

    /* Priority for collective operation */
    size_t priority;

    /* Blocking/non-blocking */
    int synchronous;

    /* Persistent/non-persistent */
    int to_cache;

    /* Treat buffer as vector/regular - applicable for allgatherv only */
    int vector_buf;

    /**
     * Id of the operation. If specified, new communicator will be created and collective
     * operations with the same @b match_id will be executed in the same order.
     */
    const char* match_id;

    /* Sparse allreduce specific */
    ccl::sparse_allreduce_completion_fn sparse_allreduce_completion_fn;
    ccl::sparse_allreduce_alloc_fn sparse_allreduce_alloc_fn;
    const void* sparse_allreduce_fn_ctx;
    ccl::sparse_coalesce_mode sparse_coalesce_mode;

} ccl_coll_attr_t;

#include "oneapi/ccl/ccl_device_types.hpp"
