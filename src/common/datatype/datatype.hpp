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

typedef struct
{
    ccl_datatype_t type;
    size_t size;
    const char* name;
}
ccl_datatype_internal;
typedef const ccl_datatype_internal* ccl_datatype_internal_t;

ccl_status_t ccl_datatype_init();
size_t ccl_datatype_get_size(ccl_datatype_internal_t dtype);
const char* ccl_datatype_get_name(ccl_datatype_internal_t dtype);
ccl_datatype_internal_t ccl_datatype_get(ccl_datatype_t type);

extern ccl_datatype_internal_t ccl_dtype_internal_none;
extern ccl_datatype_internal_t ccl_dtype_internal_char;
extern ccl_datatype_internal_t ccl_dtype_internal_int;
extern ccl_datatype_internal_t ccl_dtype_internal_bfp16;
extern ccl_datatype_internal_t ccl_dtype_internal_float;
extern ccl_datatype_internal_t ccl_dtype_internal_double;
extern ccl_datatype_internal_t ccl_dtype_internal_int64;
extern ccl_datatype_internal_t ccl_dtype_internal_uint64;
