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
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ccl.h"

#define COUNT           (1048576 / 256)
#define FLOATS_IN_M512  16
#define BFP16_SHIFT     16
 
/*

 https://www.johndcook.com/blog/2018/11/15/bfloat16/ 

 In this example we use the accuracy 0.00781250
 of calculations performed in the bfloat16, but don't take
 into account the error that may occur during conversion
 from float32 datatype to bfloat16. 
 
 */

#define BFP16_PRECISION 0.00781250 /* 2^-7 = 0.00781250 */

int is_bfp16_enabled()
{
#ifdef CCL_BFP16_COMPILER
    int is_avx512f_enabled = 0;
    uint32_t reg[4];

    __asm__ __volatile__ ("cpuid" :
                          "=a" (reg[0]), "=b" (reg[1]), "=c" (reg[2]), "=d" (reg[3]) :
                          "a" (7), "c" (0));
    is_avx512f_enabled = (( reg[1] & (1 << 16) ) >> 16) &
                         (( reg[1] & (1 << 30) ) >> 30) &
                         (( reg[1] & (1 << 31) ) >> 31);

    return (is_avx512f_enabled) ? 1 : 0;
#else
    return 0;
#endif
}

#define CHECK_ERROR(send_buf, recv_buf)                                                            \
  {                                                                                                \
      /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */                                        \
                                                                                                   \
      double max_error = 0;                                                                        \
      double log_base2 = log(size) / log(2);                                                       \
      double g = (log_base2 * BFP16_PRECISION) / (1 - (log_base2 * BFP16_PRECISION));              \
      for (size_t i = 0; i < COUNT; i++)                                                           \
      {                                                                                            \
          double expected = ((size * (size - 1) / 2) + ((float)(i) * size));                       \
          double max_error = g * expected;                                                         \
          if (fabs(max_error) < fabs(expected - recv_buf[i]))                                      \
          {                                                                                        \
              printf("[%zu] got recvBuf[%zu] = %0.7f, but expected = %0.7f, max_error = %0.16f\n", \
                      rank, i, recv_buf[i], (float)expected, (double) max_error);                  \
              exit(1);                                                                             \
          }                                                                                        \
      }                                                                                            \
  }

#ifdef CCL_BFP16_COMPILER

/* float32 -> bfloat16 */
#ifdef CCL_BFP16_TARGET_ATTRIBUTES
void convert_fp32_to_bfp16(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
void convert_fp32_to_bfp16(const void* src, void* dst)
{
    __m512i y = _mm512_bsrli_epi128(_mm512_loadu_si512(src), 2);
    _mm256_storeu_si256((__m256i*)(dst), _mm512_cvtepi32_epi16(y));
}

/* bfloat16 -> float32 */
#ifdef CCL_BFP16_TARGET_ATTRIBUTES
void convert_bfp16_to_fp32(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
void convert_bfp16_to_fp32(const void* src, void* dst)
{
    __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
    _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}

void convert_fp32_to_bfp16_arrays(float* send_buf, void* send_buf_bfp16)
{
    int int_val = 0, int_val_shifted = 0;

    for (int i = 0; i < (COUNT / FLOATS_IN_M512) * FLOATS_IN_M512; i += FLOATS_IN_M512)
    {
        convert_fp32_to_bfp16(send_buf + i, ((unsigned char*)send_buf_bfp16)+(2 * i));
    }

    /* proceed remaining float's in buffer */
    for (int i = (COUNT / FLOATS_IN_M512) * FLOATS_IN_M512; i < COUNT; i ++)
    {
        /* iterate over send_buf_bfp16 */
        int* send_bfp_tail = (int*)(((char*)send_buf_bfp16) + (2 * i));
        /* copy float (4 bytes) data as is to int variable, */
        memcpy(&int_val,&send_buf[i], 4);
        /* then perform shift and */
        int_val_shifted = int_val >> BFP16_SHIFT;
        /* save pointer to result */
        *send_bfp_tail = int_val_shifted;
    }
}

void convert_bfp16_to_fp32_arrays(void* recv_buf_bfp16, float* recv_buf)
{
    int int_val = 0, int_val_shifted = 0;

    for (int i = 0; i < (COUNT / FLOATS_IN_M512) * FLOATS_IN_M512; i += FLOATS_IN_M512)
    {
        convert_bfp16_to_fp32((char*)recv_buf_bfp16 + (2 * i), recv_buf+i);
    }

    /* proceed remaining bfp16's in buffer */
    for (int i = (COUNT / FLOATS_IN_M512) * FLOATS_IN_M512; i < COUNT; i ++)
    {
        /* iterate over recv_buf_bfp16 */
        int* recv_bfp_tail = (int*)((char*)recv_buf_bfp16 + (2 * i));
        /* copy bfp16 data as is to int variable, */
        memcpy(&int_val,recv_bfp_tail,4);
        /* then perform shift and */
        int_val_shifted = int_val << BFP16_SHIFT;
        /* copy result to output */
        memcpy((recv_buf+i), &int_val_shifted, 4);
    }
}
#endif /* CCL_BFP16_COMPILER */

int main()
{
    size_t idx = 0;
    size_t size = 0;
    size_t rank = 0;

    float* send_buf = (float*)malloc(sizeof(float) * COUNT);
    float* recv_buf = (float*)malloc(sizeof(float) * COUNT);
    void* recv_buf_bfp16 = (short*)malloc(sizeof(short) * COUNT);
    void* send_buf_bfp16 = (short*)malloc(sizeof(short) * COUNT);

    ccl_request_t request;
    ccl_stream_t stream;

    ccl_init();

    ccl_get_comm_rank(NULL, &rank);
    ccl_get_comm_size(NULL, &size);

    for (idx = 0; idx < COUNT; idx++)
    {
        send_buf[idx] = rank + idx;
        recv_buf[idx] = 0.0;
    }

    if (is_bfp16_enabled() == 0)
    {
        printf("WARNING: BFP16 is not enabled, skipped.\n");
        return 0;
    }
    else
    {
        printf("BFP16 is enabled\n");
#ifdef CCL_BFP16_COMPILER
        convert_fp32_to_bfp16_arrays(send_buf, send_buf_bfp16);
#endif /* CCL_BFP16_COMPILER */
        ccl_allreduce(send_buf_bfp16, 
                      recv_buf_bfp16, 
                      COUNT, 
                      ccl_dtype_bfp16,
                      ccl_reduction_sum,
                      NULL, /* attr */
                      NULL, /* comm */
                      stream,
                      &request);
        ccl_wait(request);
#ifdef CCL_BFP16_COMPILER
        convert_bfp16_to_fp32_arrays(recv_buf_bfp16, recv_buf);
#endif /* CCL_BFP16_COMPILER */
    }

    CHECK_ERROR(send_buf, recv_buf);

    free(send_buf);
    free(recv_buf);
    free(send_buf_bfp16);
    free(recv_buf_bfp16);
    
    ccl_stream_free(stream);
 
    ccl_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
