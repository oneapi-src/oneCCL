/*
 Copyright 2016-2019 Intel Corporation
 
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

#define RUN_COLLECTIVE(start_cmd, name)                                    \
  do {                                                                     \
      t = 0;                                                               \
      float expected = 1.0;                                                \
      for (iter_idx = 0; iter_idx < ITERS; iter_idx++)                     \
      {                                                                    \
          for (idx = 0; idx < COUNT; idx++)                                \
              send_buf[idx] = expected;                                    \
          for (idx = 0; idx < size; idx++)                                 \
              memset(recv_bufs[idx], 0, COUNT * sizeof(float));            \
          t1 = when();                                                     \
          CCL_CALL(start_cmd);                                             \
          CCL_CALL(ccl_wait(request));                                     \
          t2 = when();                                                     \
          t += (t2 - t1);                                                  \
      }                                                                    \
      ccl_barrier(NULL, NULL);                                             \
      for (idx = 0; idx < size; idx++)                                     \
      {                                                                    \
          for (size_t elem_idx = 0; elem_idx < COUNT; elem_idx++)          \
          {                                                                \
              if (recv_bufs[idx][elem_idx] != expected)                    \
              {                                                            \
                  printf("iter %zu, buf_idx %zu, elem_idx %zu, "           \
                         "expected %f, got %f\n",                          \
                          iter_idx, idx, elem_idx, expected,               \
                          recv_bufs[idx][elem_idx]);                       \
                  ASSERT(0, "unexpected value");                           \
              }                                                            \
          }                                                                \
      }                                                                    \
      printf("[%zu] avg %s time: %8.2lf us\n", rank, name, t / ITERS);     \
      fflush(stdout);                                                      \
  } while (0)

int main()
{
    float send_buf[COUNT];
    float **recv_bufs;
    size_t *recv_counts;

    setenv("CCL_ALLGATHERV_IOV", "1", 1);

    test_init();

    recv_bufs = static_cast<float**>(malloc(size * sizeof(float*)));
    for (idx = 0; idx < size; idx++)
        recv_bufs[idx] = static_cast<float*>(malloc(COUNT * sizeof(float)));

    recv_counts = static_cast<size_t*>(malloc(size * sizeof(size_t)));
    for (idx = 0; idx < size; idx++)
        recv_counts[idx] = COUNT;

    coll_attr.to_cache = 0;
    RUN_COLLECTIVE(ccl_allgatherv(send_buf, COUNT, recv_bufs, recv_counts, ccl_dtype_float, &coll_attr, NULL, NULL, &request),
                   "warmup_allgatherv");

    coll_attr.to_cache = 1;
    RUN_COLLECTIVE(ccl_allgatherv(send_buf, COUNT, recv_bufs, recv_counts, ccl_dtype_float, &coll_attr, NULL, NULL, &request),
                   "persistent_allgatherv");

    coll_attr.to_cache = 0;
    RUN_COLLECTIVE(ccl_allgatherv(send_buf, COUNT, recv_bufs, recv_counts, ccl_dtype_float, &coll_attr, NULL, NULL, &request),
                   "regular_allgatherv");

    for (idx = 0; idx < size; idx++)
        free(recv_bufs[idx]);
    free(recv_counts);

    test_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}

