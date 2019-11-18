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

#include <signal.h>

#define RUN_COLLECTIVE(start_cmd, name)                                    \
  do {                                                                     \
      ccl_status_t status = ccl_status_success;                            \
      int is_compl;                                                        \
      t = 0;                                                               \
      for (iter_idx = 0; iter_idx < ITERS; iter_idx++)                     \
      {                                                                    \
          for (idx = 0; idx < COUNT; idx++)                                \
          {                                                                \
              send_buf[idx] = (float)rank;                                 \
              recv_buf[idx] = 0.0;                                         \
          }                                                                \
          t1 = when();                                                     \
          status = start_cmd;                                              \
          if (is_finish) goto exit;                                        \
          if (status == ccl_status_blocked_due_to_resize)                  \
              goto update_;                                                \
          do{                                                              \
              status = ccl_test(request,&is_compl);                        \
              if (is_finish) goto exit;                                    \
              if (status == ccl_status_blocked_due_to_resize)              \
                  goto update_;                                            \
          } while(!is_compl);                                              \
          t2 = when();                                                     \
          t += (t2 - t1);                                                  \
      }                                                                    \
      ccl_barrier(NULL, NULL);                                             \
      if (is_finish) goto exit;                                            \
      float expected = (size - 1) * ((float)size / 2);                     \
      for (idx = 0; idx < COUNT; idx++)                                    \
      {                                                                    \
          if (recv_buf[idx] != expected)                                   \
          {                                                                \
              printf("iter %zu, idx %zu, expected %f, got %f\n",           \
                      iter_idx, idx, expected, recv_buf[idx]);             \
              ASSERT(0, "unexpected value");                               \
          }                                                                \
      }                                                                    \
      printf("[%zu] avg %s time: %8.2lf us\n", rank, name, t / ITERS);     \
      fflush(stdout);                                                      \
  } while (0)

struct sigaction act_old;
size_t is_finish = 0;

void finalize(int sig)
{
    is_finish = 1;
    act_old.sa_handler(sig);
}

ccl_resize_action_t simple_framework_func(size_t comm_size)
{

    // We have 2 or more ranks, so we can to start communication.
    if (comm_size > 1)
    {
        return  ccl_ra_run;
    }

    // We have only 1 rank, so we should to wait additional rank\ranks.
    if (comm_size == 1)
    {
        return  ccl_ra_wait;
    }
    // We have less that 1 rank, so we should to finalize.
    else
    {
        return ccl_ra_finalize;
    }
}

int main()
{
    float* send_buf = malloc(sizeof(float) * COUNT);
    float* recv_buf = malloc(sizeof(float) * COUNT);

    struct sigaction act1;
    char* type_str = getenv("CCL_PM_TYPE");
    int type = 0;

    if (type_str && strstr(type_str, "resizable"))
    {
        type = 1;
    }

    test_init();
    if (type)
    {
        ccl_set_resize_fn(&simple_framework_func);
    }
    else
    {
        is_finish = 1;
    }

    memset(&act1, 0, sizeof(act1));
    act1.sa_handler = &finalize;
    act1.sa_flags = 0;
    sigaction(SIGTERM, &act1, &act_old);
    sigaction(SIGINT, &act1, 0);

    do
    {
        coll_attr.to_cache = 0;

        RUN_COLLECTIVE(ccl_allreduce(send_buf, recv_buf, COUNT, ccl_dtype_float, ccl_reduction_sum, &coll_attr, NULL, NULL, &request),
                       "warmup_allreduce");
        coll_attr.to_cache = 1;

        RUN_COLLECTIVE(ccl_allreduce(send_buf, recv_buf, COUNT, ccl_dtype_float, ccl_reduction_sum, &coll_attr, NULL, NULL, &request),
                       "persistent_allreduce");

        coll_attr.to_cache = 0;

        RUN_COLLECTIVE(ccl_allreduce(send_buf, recv_buf, COUNT, ccl_dtype_float, ccl_reduction_sum, &coll_attr, NULL, NULL, &request),
                       "regular_allreduce");
        usleep(10000);
        if (0)
        {
update_:
            while (ccl_get_comm_rank(NULL, &rank) == ccl_status_blocked_due_to_resize)
            {
                usleep(1000);
            }

            ccl_get_comm_size(NULL, &size);

            coll_attr.prologue_fn = NULL;
            coll_attr.epilogue_fn = NULL;
            coll_attr.reduction_fn = NULL;
            coll_attr.priority = 0;
            coll_attr.synchronous = 0;
            coll_attr.match_id = "tensor_name";
            coll_attr.to_cache = 0;
        }
    } while ( !is_finish );
exit:
    test_finalize();

    free(send_buf);
    free(recv_buf);

    return 0;
}
