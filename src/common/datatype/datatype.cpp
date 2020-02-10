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
#include "common/datatype/datatype.hpp"
#include "common/log/log.hpp"

ccl_datatype_internal_t ccl_dtype_internal_none;
ccl_datatype_internal_t ccl_dtype_internal_char;
ccl_datatype_internal_t ccl_dtype_internal_int;
ccl_datatype_internal_t ccl_dtype_internal_bfp16;
ccl_datatype_internal_t ccl_dtype_internal_float;
ccl_datatype_internal_t ccl_dtype_internal_double;
ccl_datatype_internal_t ccl_dtype_internal_int64;
ccl_datatype_internal_t ccl_dtype_internal_uint64;

const ccl_datatype_internal ccl_dtype_internal_none_value = { .type = ccl_dtype_char, .size = 0, .name = "NONE" };
const ccl_datatype_internal ccl_dtype_internal_char_value = { .type = ccl_dtype_char, .size = 1, .name = "CHAR" };
const ccl_datatype_internal ccl_dtype_internal_int_value = { .type = ccl_dtype_int, .size = 4, .name = "INT" };
const ccl_datatype_internal ccl_dtype_internal_bfp16_value = { .type = ccl_dtype_bfp16, .size = 2, .name = "BFP16" };
const ccl_datatype_internal ccl_dtype_internal_float_value = { .type = ccl_dtype_float, .size = 4, .name = "FLOAT" };
const ccl_datatype_internal ccl_dtype_internal_double_value = { .type = ccl_dtype_double, .size = 8, .name = "DOUBLE" };
const ccl_datatype_internal ccl_dtype_internal_int64_value = { .type = ccl_dtype_int64, .size = 8, .name = "INT64" };
const ccl_datatype_internal ccl_dtype_internal_uint64_value = { .type = ccl_dtype_uint64, .size = 8, .name = "UINT64" };

ccl_status_t ccl_datatype_init()
{
    ccl_dtype_internal_none = &ccl_dtype_internal_none_value;
    ccl_dtype_internal_char = &ccl_dtype_internal_char_value;
    ccl_dtype_internal_int = &ccl_dtype_internal_int_value;
    ccl_dtype_internal_bfp16 = &ccl_dtype_internal_bfp16_value;
    ccl_dtype_internal_float = &ccl_dtype_internal_float_value;
    ccl_dtype_internal_double = &ccl_dtype_internal_double_value;
    ccl_dtype_internal_int64 = &ccl_dtype_internal_int64_value;
    ccl_dtype_internal_uint64 = &ccl_dtype_internal_uint64_value;
    return ccl_status_success;
}

size_t ccl_datatype_get_size(ccl_datatype_internal_t dtype)
{
    CCL_THROW_IF_NOT(dtype, "empty dtype");
    CCL_ASSERT(dtype->size > 0);
    return dtype->size;
}

const char* ccl_datatype_get_name(ccl_datatype_internal_t dtype)
{
    CCL_ASSERT(dtype);
    return dtype->name;
}

ccl_datatype_internal_t ccl_datatype_get(ccl_datatype_t type)
{
    ccl_datatype_internal_t dtype = NULL;
    switch (type)
    {
        case ccl_dtype_char: { dtype = ccl_dtype_internal_char; break; }
        case ccl_dtype_int: { dtype = ccl_dtype_internal_int; break; }
        case ccl_dtype_bfp16: { dtype = ccl_dtype_internal_bfp16; break; }
        case ccl_dtype_float: { dtype = ccl_dtype_internal_float; break; }
        case ccl_dtype_double: { dtype = ccl_dtype_internal_double; break; }
        case ccl_dtype_int64: { dtype = ccl_dtype_internal_int64; break; }
        case ccl_dtype_uint64: { dtype = ccl_dtype_internal_uint64; break; }
        default: CCL_FATAL("unexpected dtype ", type);
    }
    return dtype;
}
