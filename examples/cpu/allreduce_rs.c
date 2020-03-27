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

#include <signal.h>

#define RUN_COLLECTIVE(start_cmd, name)                                \
  do {                                                                 \
      ccl_status_t status = ccl_status_success;                        \
      int is_completed = 0;                                            \
      t = 0;                                                           \
      for (iter_idx = 0; iter_idx < ITERS; iter_idx++)                 \
      {                                                                \
          for (idx = 0; idx < COUNT; idx++)                            \
          {                                                            \
              send_buf[idx] = (float)rank;                             \
              recv_buf[idx] = 0.0;                                     \
          }                                                            \
          t1 = when();                                                 \
          status = start_cmd;                                          \
          if (should_exit)                                             \
              goto exit;                                               \
          if (status == ccl_status_blocked_due_to_resize)              \
              goto update;                                             \
          do                                                           \
          {                                                            \
              status = ccl_test(request, &is_completed);               \
              if (should_exit)                                         \
                  goto exit;                                           \
              if (status == ccl_status_blocked_due_to_resize)          \
                  goto update;                                         \
          }                                                            \
          while (!is_completed);                                       \
          t2 = when();                                                 \
          t += (t2 - t1);                                              \
      }                                                                \
      ccl_barrier(NULL, NULL);                                         \
      if (should_exit)                                                 \
          goto exit;                                                   \
      float expected = (size - 1) * ((float)size / 2);                 \
      for (idx = 0; idx < COUNT; idx++)                                \
      {                                                                \
          if (recv_buf[idx] != expected)                               \
          {                                                            \
              printf("iter %zu, idx %zu, expected %f, got %f\n",       \
                      iter_idx, idx, expected, recv_buf[idx]);         \
              ASSERT(0, "unexpected value");                           \
          }                                                            \
      }                                                                \
      printf("[%zu] avg %s time: %8.2lf us\n", rank, name, t / ITERS); \
      fflush(stdout);                                                  \
  } while (0)

struct sigaction old_sigact;
size_t should_exit = 0;

void finalize(int sig)
{
    should_exit = 1;
    old_sigact.sa_handler(sig);
}

ccl_resize_action_t simple_resize_fn(size_t comm_size)
{
    if (comm_size > 1)
    {
        // 2 or more ranks, can start communication
        return ccl_ra_run;
    }

    if (comm_size == 1)
    {
        // only 1 rank, wait additional rank(s)
        return ccl_ra_wait;
    }
    else
    {
        // no ranks, finalize
        return ccl_ra_finalize;
    }
}

int main()
{
    float* send_buf = malloc(sizeof(float) * COUNT);
    float* recv_buf = malloc(sizeof(float) * COUNT);

    char* pm_type_env = getenv("CCL_PM_TYPE");
    int set_resize_fn = 0;
    if (pm_type_env && !strcmp(pm_type_env, "resizable"))
        set_resize_fn = 1;

    test_init();

    if (set_resize_fn)
        ccl_set_resize_fn(&simple_resize_fn);
    else
        should_exit = 1;

    struct sigaction sigact;
    memset(&sigact, 0, sizeof(sigact));
    sigact.sa_handler = &finalize;
    sigact.sa_flags = 0;
    sigaction(SIGTERM, &sigact, &old_sigact);
    sigaction(SIGINT, &sigact, 0);

    do
    {
        coll_attr.to_cache = 0;

        RUN_COLLECTIVE(ccl_allreduce(send_buf, recv_buf, COUNT, ccl_dtype_float,
                                     ccl_reduction_sum, &coll_attr, NULL, NULL, &request),
                       "warmup_allreduce");
        coll_attr.to_cache = 1;

        RUN_COLLECTIVE(ccl_allreduce(send_buf, recv_buf, COUNT, ccl_dtype_float,
                                     ccl_reduction_sum, &coll_attr, NULL, NULL, &request),
                       "persistent_allreduce");

        coll_attr.to_cache = 0;

        RUN_COLLECTIVE(ccl_allreduce(send_buf, recv_buf, COUNT, ccl_dtype_float,
                                     ccl_reduction_sum, &coll_attr, NULL, NULL, &request),
                       "regular_allreduce");
        usleep(10000);

        if (0)
        {
update:
            while (ccl_get_comm_rank(NULL, &rank) == ccl_status_blocked_due_to_resize)
            {
                usleep(1000);
            }

            ccl_get_comm_size(NULL, &size);
        }
    }
    while (!should_exit);

exit:

    test_finalize();

    free(send_buf);
    free(recv_buf);

    return 0;
}
