/*
 Copyright 2016-2019 Intel Corporation
 
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

#include "common/datatype/datatype.hpp"
#include "ccl_types.h"

ccl_status_t ccl_comp_copy(const void *in_buf, void *out_buf, size_t count, ccl_datatype_internal_t dtype);
ccl_status_t ccl_comp_reduce(const void *in_buf, size_t in_count, void *inout_buf, size_t *out_count,
                             ccl_datatype_internal_t dtype, ccl_reduction_t reduction,
                             ccl_reduction_fn_t reduction_fn, const ccl_fn_context_t* context = nullptr);
const char *ccl_reduction_to_str(ccl_reduction_t type);
