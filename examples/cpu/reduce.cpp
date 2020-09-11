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
#include "base.h"

#define RUN_COLLECTIVE(start_cmd, name) \
    do { \
        t = 0; \
        for (iter_idx = 0; iter_idx < ITERS; iter_idx++) { \
            for (idx = 0; idx < COUNT; idx++) { \
                send_buf[idx] = 1.0; \
                recv_buf[idx] = 0.0; \
            } \
            t1 = when(); \
            CCL_CALL(start_cmd); \
            CCL_CALL(ccl_wait(request)); \
            t2 = when(); \
            t += (t2 - t1); \
        } \
        ccl_barrier(NULL, NULL); \
        float expected = size; \
        for (idx = 0; idx < COUNT; idx++) { \
            if ((rank == COLL_ROOT) && (recv_buf[idx] != expected)) { \
                printf("iter %zu, idx %zu, expected %f, got %f\n", \
                       iter_idx, \
                       idx, \
                       expected, \
                       recv_buf[idx]); \
                ASSERT(0, "unexpected value"); \
            } \
        } \
        printf("[%zu] avg %s time: %8.2lf us\n", rank, name, t / ITERS); \
        fflush(stdout); \
    } while (0)

int main() {
    float send_buf[COUNT];
    float recv_buf[COUNT];

    test_init();

    coll_attr.to_cache = 1;
    RUN_COLLECTIVE(ccl_reduce(send_buf,
                              recv_buf,
                              COUNT,
                              ccl_dtype_float,
                              ccl_reduction_sum,
                              COLL_ROOT,
                              &coll_attr,
                              NULL,
                              NULL,
                              &request),
                   "persistent_reduce");

    coll_attr.to_cache = 0;
    RUN_COLLECTIVE(ccl_reduce(send_buf,
                              recv_buf,
                              COUNT,
                              ccl_dtype_float,
                              ccl_reduction_sum,
                              COLL_ROOT,
                              &coll_attr,
                              NULL,
                              NULL,
                              &request),
                   "regular_reduce");

    test_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
