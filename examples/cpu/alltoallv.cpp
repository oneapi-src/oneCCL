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

#define EVEN_RANK_SEND_COUNT 100
#define ODD_RANK_SEND_COUNT  200

#define RUN_COLLECTIVE(start_cmd, name) \
    do { \
        t = 0; \
        size_t rank_idx, elem_idx; \
        for (iter_idx = 0; iter_idx < ITERS; iter_idx++) { \
            memset(recv_buf, 0, total_recv_count * sizeof(*recv_buf)); \
            elem_idx = 0; \
            for (rank_idx = 0; rank_idx < size; rank_idx++) { \
                for (idx = 0; idx < send_counts[rank_idx]; idx++) { \
                    send_buf[elem_idx] = rank; \
                    elem_idx++; \
                } \
            } \
            t1 = when(); \
            CCL_CALL(start_cmd); \
            CCL_CALL(ccl_wait(request)); \
            t2 = when(); \
            t += (t2 - t1); \
        } \
        ccl_barrier(NULL, NULL); \
        int expected; \
        elem_idx = 0; \
        for (rank_idx = 0; rank_idx < size; rank_idx++) { \
            expected = rank_idx; \
            for (idx = 0; idx < recv_counts[rank_idx]; idx++) { \
                if (recv_buf[elem_idx] != expected) { \
                    printf("iter %zu, idx %zu, expected %d, got %d\n", \
                           iter_idx, \
                           elem_idx, \
                           expected, \
                           recv_buf[elem_idx]); \
                    ASSERT(0, "unexpected value"); \
                } \
                elem_idx++; \
            } \
        } \
        printf("[%zu] avg %s time: %8.2lf us\n", rank, name, t / ITERS); \
        fflush(stdout); \
    } while (0)

int main() {
    int* send_buf;
    int* recv_buf;
    size_t* send_counts;
    size_t* recv_counts;
    int is_even;
    size_t send_count;
    size_t total_send_count, total_recv_count;

    test_init();

    ASSERT((EVEN_RANK_SEND_COUNT + ODD_RANK_SEND_COUNT) % 2 == 0,
           "unexpected send counts (%d, %d)",
           EVEN_RANK_SEND_COUNT,
           ODD_RANK_SEND_COUNT);

    ASSERT(size % 2 == 0, "unexpected rank count (%zu)", size);

    is_even = (rank % 2 == 0) ? 1 : 0;
    send_count = (is_even) ? EVEN_RANK_SEND_COUNT : ODD_RANK_SEND_COUNT;

    total_send_count = send_count * size;
    total_recv_count = (EVEN_RANK_SEND_COUNT + ODD_RANK_SEND_COUNT) * (size / 2);

    send_buf = static_cast<int*>(malloc(total_send_count * sizeof(int)));
    recv_buf = static_cast<int*>(malloc(total_recv_count * sizeof(int)));

    send_counts = static_cast<size_t*>(malloc(size * sizeof(size_t)));
    recv_counts = static_cast<size_t*>(malloc(size * sizeof(size_t)));

    for (idx = 0; idx < size; idx++) {
        int is_even_peer = (idx % 2 == 0) ? 1 : 0;
        send_counts[idx] = send_count;
        recv_counts[idx] = (is_even_peer) ? EVEN_RANK_SEND_COUNT : ODD_RANK_SEND_COUNT;
    }

    coll_attr.to_cache = 0;
    RUN_COLLECTIVE(ccl_alltoallv(send_buf,
                                 send_counts,
                                 recv_buf,
                                 recv_counts,
                                 ccl_dtype_int,
                                 &coll_attr,
                                 NULL,
                                 NULL,
                                 &request),
                   "warmup_alltoallv");

    coll_attr.to_cache = 1;
    RUN_COLLECTIVE(ccl_alltoallv(send_buf,
                                 send_counts,
                                 recv_buf,
                                 recv_counts,
                                 ccl_dtype_int,
                                 &coll_attr,
                                 NULL,
                                 NULL,
                                 &request),
                   "persistent_alltoallv");

    coll_attr.to_cache = 0;
    RUN_COLLECTIVE(ccl_alltoallv(send_buf,
                                 send_counts,
                                 recv_buf,
                                 recv_counts,
                                 ccl_dtype_int,
                                 &coll_attr,
                                 NULL,
                                 NULL,
                                 &request),
                   "regular_alltoallv");

    test_finalize();

    free(send_buf);
    free(recv_buf);
    free(send_counts);
    free(recv_counts);

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
