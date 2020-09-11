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
#ifndef BASE_BFP16_HPP
#define BASE_BFP16_HPP

#ifdef CCL_BFP16_COMPILER

#include <immintrin.h>
#include <math.h>

#include "base.hpp"

#define FLOATS_IN_M512  16
#define BFP16_SHIFT     16
#define BFP16_PRECISION 0.0390625 // 2^-8

int is_avx512bf_enabled() {
#ifdef CCL_BFP16_AVX512BF_COMPILER
    static int is_enabled = -1;

    if (is_enabled == -1) {
        uint32_t reg[4];

        __asm__ __volatile__("cpuid"
                             : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                             : "a"(7), "c"(1));
        is_enabled = (reg[0] & (1 << 5)) >> 5;
    }

    return is_enabled;
#else
    return 0;
#endif
}

// float32 -> bfloat16
#ifdef CCL_BFP16_TARGET_ATTRIBUTES
#ifdef CCL_BFP16_AVX512BF_COMPILER
void convert_fp32_to_bfp16(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void convert_fp32_to_bfp16(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif
void convert_fp32_to_bfp16(const void* src, void* dst) {
#ifdef CCL_BFP16_AVX512BF_COMPILER
    if (is_avx512bf_enabled())
        _mm256_storeu_si256((__m256i*)(dst), _mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
    else
#endif
        _mm256_storeu_si256((__m256i*)(dst),
                            _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_loadu_si512(src), 2)));
}

// bfloat16 -> float32
#ifdef CCL_BFP16_TARGET_ATTRIBUTES
#ifdef CCL_BFP16_AVX512BF_COMPILER
void convert_bfp16_to_fp32(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void convert_bfp16_to_fp32(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif
void convert_bfp16_to_fp32(const void* src, void* dst) {
    __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
    _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}

template <typename T>
void convert_fp32_to_bfp16_arrays(T* send_buf, void* send_buf_bfp16, size_t count) {
    int int_val = 0, int_val_shifted = 0;

    for (size_t i = 0; i < (count / FLOATS_IN_M512) * FLOATS_IN_M512; i += FLOATS_IN_M512) {
        convert_fp32_to_bfp16(send_buf + i, ((char*)send_buf_bfp16) + (2 * i));
    }

    for (size_t i = (count / FLOATS_IN_M512) * FLOATS_IN_M512; i < count; i++) {
        int* send_bfp_tail = (int*)(((char*)send_buf_bfp16) + (2 * i));
        memcpy(&int_val, &send_buf[i], 4);
        int_val_shifted = int_val >> BFP16_SHIFT;
        *send_bfp_tail = int_val_shifted;
    }
}

template <typename T>
void convert_bfp16_to_fp32_arrays(void* recv_buf_bfp16, T* recv_buf, size_t count) {
    int int_val = 0, int_val_shifted = 0;

    for (size_t i = 0; i < (count / FLOATS_IN_M512) * FLOATS_IN_M512; i += FLOATS_IN_M512) {
        convert_bfp16_to_fp32((char*)recv_buf_bfp16 + (2 * i), recv_buf + i);
    }

    for (size_t i = (count / FLOATS_IN_M512) * FLOATS_IN_M512; i < count; i++) {
        float recv_bfp_tail = *(float*)((char*)recv_buf_bfp16 + (2 * i));
        memcpy(&int_val, &recv_bfp_tail, 4);
        int_val_shifted = int_val << BFP16_SHIFT;
        memcpy((recv_buf + i), &int_val_shifted, 4);
    }
}

template <typename T>
void make_bfp16_prologue(typed_test_param<T>& param, size_t size) {
    for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
        size_t new_idx = param.buf_indexes[buf_idx];

        if (param.test_conf.place_type == PT_IN) {
            T* recv_buf_orig = param.recv_buf[new_idx].data();
            void* recv_buf_bfp16 = param.recv_buf_bfp16[new_idx].data();
            convert_fp32_to_bfp16_arrays(recv_buf_orig, recv_buf_bfp16, size);
        }
        else {
            T* send_buf_orig = param.send_buf[new_idx].data();
            void* send_buf_bfp16 = param.send_buf_bfp16[new_idx].data();
            convert_fp32_to_bfp16_arrays(send_buf_orig, send_buf_bfp16, size);
        }
    }
}

template <typename T>
void make_bfp16_epilogue(typed_test_param<T>& param, size_t size) {
    for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
        size_t new_idx = param.buf_indexes[buf_idx];
        T* recv_buf_orig = param.recv_buf[new_idx].data();
        void* recv_buf_bfp16 = static_cast<void*>(param.recv_buf_bfp16[new_idx].data());
        convert_bfp16_to_fp32_arrays(recv_buf_bfp16, recv_buf_orig, size);
    }
}

#endif /* CCL_BFP16_COMPILER */

#endif /* BASE_BFP16_HPP */
