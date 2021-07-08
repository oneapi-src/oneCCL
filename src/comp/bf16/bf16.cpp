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
#include "oneapi/ccl/types.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "comp/bf16/bf16.hpp"
#include "comp/bf16/bf16_intrisics.hpp"
#include "common/utils/enums.hpp"

#define CCL_FLOATS_IN_M512 16
#define CCL_BF16_SHIFT     16

std::map<ccl_bf16_impl_type, std::string> bf16_impl_names = {
    std::make_pair(ccl_bf16_no_compiler_support, "no_compiler_support"),
    std::make_pair(ccl_bf16_no_hardware_support, "no_hardware_support"),
    std::make_pair(ccl_bf16_avx512f, "avx512f"),
    std::make_pair(ccl_bf16_avx512bf, "avx512bf")
};

std::map<ccl_bf16_impl_type, std::string> bf16_env_impl_names = {
    std::make_pair(ccl_bf16_avx512f, "avx512f"),
    std::make_pair(ccl_bf16_avx512bf, "avx512bf")
};

#ifdef CCL_BF16_COMPILER

void ccl_bf16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction op) {
    LOG_DEBUG("BF16 reduction for %zu elements\n", in_cnt);

    if (out_cnt != nullptr) {
        *out_cnt = in_cnt;
    }

    ccl_bf16_reduce_impl(in_buf, inout_buf, in_cnt, op);
}

void ccl_convert_fp32_to_bf16(const void* src, void* dst) {
#ifdef CCL_BF16_AVX512BF_COMPILER
    if (ccl::global_data::env().bf16_impl_type == ccl_bf16_avx512bf) {
        _mm256_storeu_si256((__m256i*)(dst), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
    }
    else
#endif
    {
        _mm256_storeu_si256((__m256i*)(dst),
                            _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_loadu_si512(src), 2)));
    }
}

void ccl_convert_bf16_to_fp32(const void* src, void* dst) {
    _mm512_storeu_si512(
        dst,
        _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src)), 2));
}

void ccl_convert_fp32_to_bf16_arrays(void* fp32_buf, void* bf16_buf, size_t count) {
    int int_val = 0, int_val_shifted = 0;
    float* fp32_buf_float = (float*)fp32_buf;
    size_t limit = (count / CCL_FLOATS_IN_M512) * CCL_FLOATS_IN_M512;

    for (size_t i = 0; i < limit; i += CCL_FLOATS_IN_M512) {
        ccl_convert_fp32_to_bf16(fp32_buf_float + i, ((unsigned char*)bf16_buf) + (2 * i));
    }

    /* proceed remaining float's in buffer */
    for (size_t i = limit; i < count; i++) {
        /* iterate over bf16_buf */
        int* send_bfp_tail = (int*)(((char*)bf16_buf) + (2 * i));
        /* copy float (4 bytes) data as is to int variable, */
        memcpy(&int_val, &fp32_buf_float[i], 4);
        /* then perform shift and */
        int_val_shifted = int_val >> CCL_BF16_SHIFT;
        /* save pointer to result */
        *send_bfp_tail = int_val_shifted;
    }
}

void ccl_convert_bf16_to_fp32_arrays(void* bf16_buf, float* fp32_buf, size_t count) {
    int int_val = 0, int_val_shifted = 0;
    size_t limit = (count / CCL_FLOATS_IN_M512) * CCL_FLOATS_IN_M512;

    for (size_t i = 0; i < limit; i += CCL_FLOATS_IN_M512) {
        ccl_convert_bf16_to_fp32((char*)bf16_buf + (2 * i), fp32_buf + i);
    }

    /* proceed remaining bf16's in buffer */
    for (size_t i = limit; i < count; i++) {
        /* iterate over bf16_buf */
        int* recv_bfp_tail = (int*)((char*)bf16_buf + (2 * i));
        /* copy bf16 data as is to int variable, */
        memcpy(&int_val, recv_bfp_tail, 4);
        /* then perform shift and */
        int_val_shifted = int_val << CCL_BF16_SHIFT;
        /* copy result to output */
        memcpy((fp32_buf + i), &int_val_shifted, 4);
    }
}

#else /* CCL_BF16_COMPILER */

void ccl_bf16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction reduction_op) {
    CCL_FATAL("BF16 reduction was requested but CCL was compiled w/o BF16 support");
}

void ccl_convert_fp32_to_bf16_arrays(void* fp32_buf, void* bf16_buf, size_t count) {
    CCL_FATAL("FP32->BF16 conversion was requested but CCL was compiled w/o BF16 support");
}

void ccl_convert_bf16_to_fp32_arrays(void* bf16_buf, float* fp32_buf, size_t count) {
    CCL_FATAL("BF16->FP32 conversion was requested but CCL was compiled w/o BF16 support");
}

#endif /* CCL_BF16_COMPILER */
