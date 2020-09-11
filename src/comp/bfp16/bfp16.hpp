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

#ifdef CCL_BFP16_TARGET_ATTRIBUTES
#ifdef CCL_BFP16_AVX512BF_COMPILER
__attribute__((target("avx512bw,avx512vl,avx512bf16")))
#else
__attribute__((target("avx512bw,avx512vl")))
#endif
void ccl_bfp16_reduce(const void* in_buf, size_t in_cnt,
                     void* inout_buf, size_t* out_cnt,
                     ccl_reduction_t reduction_op);
#else
void ccl_bfp16_reduce(const void* in_buf,
                      size_t in_cnt,
                      void* inout_buf,
                      size_t* out_cnt,
                      ccl_reduction_t reduction_op);
#endif

void ccl_convert_fp32_to_bfp16_arrays(void*, void*, size_t);
void ccl_convert_bfp16_to_fp32_arrays(void*, float*, size_t);

#ifdef CCL_BFP16_COMPILER

#ifdef CCL_BFP16_TARGET_ATTRIBUTES
#ifdef CCL_BFP16_AVX512BF_COMPILER
void ccl_convert_fp32_to_bfp16(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void ccl_convert_fp32_to_bfp16(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif

#ifdef CCL_BFP16_TARGET_ATTRIBUTES
#ifdef CCL_BFP16_AVX512BF_COMPILER
void ccl_convert_bfp16_to_fp32(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void ccl_convert_bfp16_to_fp32(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif

#endif /* CCL_BFP16_COMPILER */
