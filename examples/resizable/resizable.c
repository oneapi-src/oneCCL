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
                send_buf[idx] = (float)rank; \
                recv_buf[idx] = 0.0; \
            } \
            t1 = when(); \
            CCL_CALL(start_cmd); \
            CCL_CALL(ccl_wait(request)); \
            t2 = when(); \
            t += (t2 - t1); \
        } \
        ccl_barrier(NULL, NULL); \
        float expected = (size - 1) * ((float)size / 2); \
        for (idx = 0; idx < COUNT; idx++) { \
            if (recv_buf[idx] != expected) { \
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

typedef enum resizable_test_case_t {
    resizable_test_simple = 0,
    resizable_test_reinit,
    resizable_test_reconnect,
    resizable_test_last,
};

const char* test_case_strs[resizable_test_last] = { "simple", "reinit", "reconnect" };

resizable_test_case_t string_to_option(const char* str) {
    resizable_test_case_t opt = resizable_test_simple;
    for (; opt < resizable_test_last; opt++) {
        if (strstr(string_opt[opt], str)) {
            break;
        }
    }
    return opt;
}

void print_help() {
    printf("You must use this example like:\n"
           "resizable test_case\n"
           "where option:\n"
           "\t%s - for one init/finalize test\n"
           "\t%s - for multi init/finalize test\n"
           "\t%s - in this case part of ranks will be finalized and must be recreated by user\n",
           string_opt[resizable_test_simple],
           string_opt[resizable_test_reinit],
           string_opt[resizable_test_reconnect]);
}

int main(int argc, char** argv) {
    int ret = 0;
    float* send_buf = malloc(sizeof(float) * COUNT);
    float* recv_buf = malloc(sizeof(float) * COUNT);
    int reinit_count = 1;
    int repeat_count = 1;
    resizable_test_case_t opt = resizable_test_simple;

    if (argc > 1) {
        opt = string_to_option(argv[1]);
        switch (opt) {
            case resizable_test_simple:
                reinit_count = 1;
                repeat_count = 1;
                break;
            case resizable_test_reinit:
                reinit_count = 4;
                repeat_count = 1;
                break;
            case resizable_test_reconnect:
                printf("Unsupported yet\n");
                return -1;
                reinit_count = 1;
                repeat_count = 2;
                break;
            case resizable_test_last:
            default:
                printf("Unknown option!\n");
                print_help();
                return -1;
        }
    }
    else {
        print_help();
        return -1;
    }

    for (int i = 0; i < reinit_count; ++i) {
        test_init();
        for (int j = 0; j < repeat_count; ++j) {
            coll_attr.to_cache = 0;
            RUN_COLLECTIVE(ccl_allreduce(send_buf,
                                         recv_buf,
                                         COUNT,
                                         ccl_dtype_float,
                                         ccl_reduction_sum,
                                         &coll_attr,
                                         NULL,
                                         NULL,
                                         &request),
                           "regular_allreduce");
            if (opt == resizable_test_reconnect && rank % 2) {
                ret = 1;
                break;
            }
        }
        sleep(size - rank);
        test_finalize();
        sleep(rank);
    }
    free(send_buf);
    free(recv_buf);

    if (rank == 0)
        printf("PASSED\n");

    return ret;
}
