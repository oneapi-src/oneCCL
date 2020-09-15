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
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bfp16.h"
#include "ccl.h"

#define COUNT (1048576 / 256)

#define CHECK_ERROR(send_buf, recv_buf) \
    { \
        /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */ \
\
        double log_base2 = log(size) / log(2); \
        double g = (log_base2 * BFP16_PRECISION) / (1 - (log_base2 * BFP16_PRECISION)); \
        for (size_t i = 0; i < COUNT; i++) { \
            double expected = ((size * (size - 1) / 2) + ((float)(i)*size)); \
            double max_error = g * expected; \
            if (fabs(max_error) < fabs(expected - recv_buf[i])) { \
                printf( \
                    "[%zu] got recvBuf[%zu] = %0.7f, but expected = %0.7f, max_error = %0.16f\n", \
                    rank, \
                    i, \
                    recv_buf[i], \
                    (float)expected, \
                    (double)max_error); \
                exit(1); \
            } \
        } \
    }

int main() {
    size_t idx = 0;
    size_t size = 0;
    size_t rank = 0;

    float* send_buf = (float*)malloc(sizeof(float) * COUNT);
    float* recv_buf = (float*)malloc(sizeof(float) * COUNT);
    void* recv_buf_bfp16 = (short*)malloc(sizeof(short) * COUNT);
    void* send_buf_bfp16 = (short*)malloc(sizeof(short) * COUNT);

    ccl_request_t request;

    ccl_init();

    ccl_get_comm_rank(NULL, &rank);
    ccl_get_comm_size(NULL, &size);

    for (idx = 0; idx < COUNT; idx++) {
        send_buf[idx] = rank + idx;
        recv_buf[idx] = 0.0;
    }

    if (is_bfp16_enabled() == 0) {
        printf("WARNING: BFP16 is not enabled, skipped.\n");
        return 0;
    }
    else {
        printf("BFP16 is enabled\n");
#ifdef CCL_BFP16_COMPILER
        convert_fp32_to_bfp16_arrays(send_buf, send_buf_bfp16, COUNT);
#endif /* CCL_BFP16_COMPILER */
        ccl_allreduce(send_buf_bfp16,
                      recv_buf_bfp16,
                      COUNT,
                      ccl_dtype_bfp16,
                      ccl_reduction_sum,
                      NULL, /* attr */
                      NULL, /* comm */
                      NULL, /* stream */
                      &request);
        ccl_wait(request);
#ifdef CCL_BFP16_COMPILER
        convert_bfp16_to_fp32_arrays(recv_buf_bfp16, recv_buf, COUNT);
#endif /* CCL_BFP16_COMPILER */
    }

    CHECK_ERROR(send_buf, recv_buf);

    free(send_buf);
    free(recv_buf);
    free(send_buf_bfp16);
    free(recv_buf_bfp16);

    ccl_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
