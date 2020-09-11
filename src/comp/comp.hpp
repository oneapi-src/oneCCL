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

#include "common/datatype/datatype.hpp"
#include "oneapi/ccl/ccl_types.hpp"

ccl_status_t ccl_comp_copy(const void* in_buf,
                           void* out_buf,
                           size_t count,
                           const ccl_datatype& dtype);
ccl_status_t ccl_comp_reduce(const void* in_buf,
                             size_t in_count,
                             void* inout_buf,
                             size_t* out_count,
                             const ccl_datatype& dtype,
                             ccl::reduction reduction,
                             ccl::reduction_fn reduction_fn,
                             const ccl::fn_context* context = nullptr);
ccl_status_t ccl_comp_batch_reduce(const void* in_buf,
                                   const std::vector<size_t>& offsets,
                                   size_t in_count,
                                   void* inout_buf,
                                   size_t* out_count,
                                   const ccl_datatype& dtype,
                                   ccl::reduction reduction,
                                   ccl::reduction_fn reduction_fn,
                                   const ccl::fn_context* context,
                                   int bfp16_keep_precision_mode,
                                   float* tmp,
                                   float* acc);
const char* ccl_reduction_to_str(ccl::reduction type);
