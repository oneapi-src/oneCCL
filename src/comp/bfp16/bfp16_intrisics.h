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

#ifdef CCL_BFP16_COMPILER

#include <immintrin.h>
#include <inttypes.h>

#include "comp/bfp16/bfp16_utils.h"

#ifdef CCL_BFP16_TARGET_ATTRIBUTES

#ifdef CCL_BFP16_AVX512BF_COMPILER
#define ALL_BFP16_ATTRS "avx512bw,avx512vl,avx512f,avx512bf16"
#else
#define ALL_BFP16_ATTRS "avx512bw,avx512vl,avx512f"
#endif

#define TARGET_ATTRIBUTE_BWF         __attribute__((target("avx512bw,avx512f")))
#define TARGET_ATTRIBUTE_ALL         __attribute__((target(ALL_BFP16_ATTRS)))
#define INLINE_TARGET_ATTRIBUTE_BW   __attribute__((__always_inline__, target("avx512bw"))) inline
#define INLINE_TARGET_ATTRIBUTE_BF16 __attribute__((__always_inline__, target("avx512bf16"))) inline
#define INLINE_TARGET_ATTRIBUTE_ALL  __attribute__((__always_inline__, target(ALL_BFP16_ATTRS))) inline

#else /* CCL_BFP16_TARGET_ATTRIBUTES */

#define TARGET_ATTRIBUTE_BWF
#define TARGET_ATTRIBUTE_ALL
#define INLINE_TARGET_ATTRIBUTE_BW   __attribute__((__always_inline__)) inline
#define INLINE_TARGET_ATTRIBUTE_BF16 __attribute__((__always_inline__)) inline
#define INLINE_TARGET_ATTRIBUTE_ALL  __attribute__((__always_inline__)) inline

#endif /* CCL_BFP16_TARGET_ATTRIBUTES */

typedef __m512 (*ccl_bfp16_reduction_func_ptr)(__m512 a, __m512 b);

TARGET_ATTRIBUTE_BWF __m512
sum_wrap(__m512 a, __m512 b)
{
    return _mm512_add_ps(a, b);
}

TARGET_ATTRIBUTE_BWF __m512
prod_wrap(__m512 a, __m512 b)
{
    return _mm512_mul_ps(a, b);
}

TARGET_ATTRIBUTE_BWF __m512
min_wrap(__m512 a, __m512 b)
{
    return _mm512_min_ps(a, b);
}

TARGET_ATTRIBUTE_BWF __m512
max_wrap(__m512 a, __m512 b)
{
    return _mm512_max_ps(a, b);
}

TARGET_ATTRIBUTE_BWF __m512
ccl_m512_reduce(__m512 a, __m512 b, ccl_bfp16_reduction_func_ptr op)
{
    return (*op)(a, b);
}

INLINE_TARGET_ATTRIBUTE_BW void
ccl_bfp16_load_as_fp32(const void* src, void* dst)
{
    __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
    _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}

INLINE_TARGET_ATTRIBUTE_BW void
ccl_fp32_store_as_bfp16_avx512f(const void* src, void* dst)
{
    _mm256_storeu_si256((__m256i*)(dst), _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_loadu_si512(src), 2)));
}

#ifdef CCL_BFP16_AVX512BF_COMPILER
INLINE_TARGET_ATTRIBUTE_BF16 void
ccl_fp32_store_as_bfp16_avx512bf(const void* src, void* dst)
{
    _mm256_storeu_si256((__m256i*)(dst), _mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
}
#endif

#define CCL_BFP16_REDUCE_FUNC_DEFINITIONS(impl_type)                              \
                                                                                  \
    INLINE_TARGET_ATTRIBUTE_ALL void                                              \
    ccl_bfp16_reduce_inputs_##impl_type(const void* a, const void* b, void* res,  \
                                        ccl_bfp16_reduction_func_ptr op)          \
    {                                                                             \
        __m512 vfp32_in, vfp32_inout;                                             \
        ccl_bfp16_load_as_fp32(a, (void*)&vfp32_in);                              \
        ccl_bfp16_load_as_fp32(b, (void*)&vfp32_inout);                           \
        __m512 vfp32_out = ccl_m512_reduce(vfp32_in, vfp32_inout, op);            \
        ccl_fp32_store_as_bfp16_##impl_type((const void*)&vfp32_out, res);        \
    }                                                                             \
                                                                                  \
    INLINE_TARGET_ATTRIBUTE_ALL void                                              \
    ccl_bfp16_reduce_256_##impl_type(const void* in, const void* inout,           \
                                     ccl_bfp16_reduction_func_ptr op)             \
    {                                                                             \
        __m256i vbfp16_out;                                                       \
        ccl_bfp16_reduce_inputs_##impl_type(in, inout, (void*)&vbfp16_out, op);   \
        _mm256_storeu_si256((__m256i*)(inout), vbfp16_out);                       \
    }                                                                             \
                                                                                  \
    INLINE_TARGET_ATTRIBUTE_ALL void                                              \
    ccl_bfp16_reduce_masked_##impl_type(const void* in, void* inout, uint8_t len, \
                                        ccl_bfp16_reduction_func_ptr op)          \
    {                                                                             \
        if (len == 0) return;                                                     \
        uint16_t mask = ((uint16_t)0xFFFF) >> (16 - len);                         \
        __m256i vbfp16_out;                                                       \
        ccl_bfp16_reduce_inputs_##impl_type(in, inout, (void*)&vbfp16_out, op);   \
        _mm256_mask_storeu_epi16(inout, (__mmask16)mask, vbfp16_out);             \
    }                                                                             \
                                                                                  \
    INLINE_TARGET_ATTRIBUTE_ALL void                                              \
    ccl_bfp16_reduce_impl_##impl_type(const void* in_buf,                         \
                                      void* inout_buf,                            \
                                      size_t in_cnt,                              \
                                      ccl_bfp16_reduction_func_ptr op)            \
    {                                                                             \
        int i = 0;                                                                \
        for (i = 0; i <= (int)in_cnt - 16; i += 16)                               \
        {                                                                         \
            ccl_bfp16_reduce_256_##impl_type((uint16_t*)in_buf + i,               \
                                             (uint16_t*)inout_buf + i, op);       \
        }                                                                         \
        ccl_bfp16_reduce_masked_##impl_type((uint16_t*)in_buf + i,                \
                                            (uint16_t*)inout_buf + i,             \
                                            (uint8_t)(in_cnt - i), op);           \
    }

CCL_BFP16_REDUCE_FUNC_DEFINITIONS(avx512f);
#ifdef CCL_BFP16_AVX512BF_COMPILER
CCL_BFP16_REDUCE_FUNC_DEFINITIONS(avx512bf);
#endif

INLINE_TARGET_ATTRIBUTE_ALL void
ccl_bfp16_reduce_impl(const void* in_buf, void* inout_buf, size_t in_cnt,
                      ccl_bfp16_reduction_func_ptr op,
                      ccl_bfp16_impl_type impl_type)
{
    if (impl_type == ccl_bfp16_avx512f)
        ccl_bfp16_reduce_impl_avx512f(in_buf, inout_buf, in_cnt, op);
#ifdef CCL_BFP16_AVX512BF_COMPILER
    else if (impl_type == ccl_bfp16_avx512bf)
        ccl_bfp16_reduce_impl_avx512bf(in_buf, inout_buf, in_cnt, op);
#endif

}

#endif /* CCL_BFP16_COMPILER */
