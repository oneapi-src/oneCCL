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
#ifndef BASE_H
#define BASE_H

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "ccl.h"

#define COUNT     (1048576 / 256)
#define ITERS     (100)
#define COLL_ROOT (0)

void test_finalize(void);

#define ASSERT(cond, fmt, ...)                            \
  do                                                      \
  {                                                       \
      if (!(cond))                                        \
      {                                                   \
          printf("FAILED\n");                             \
          fprintf(stderr, "ASSERT '%s' FAILED " fmt "\n", \
              #cond, ##__VA_ARGS__);                      \
          test_finalize();                                \
          exit(1);                                        \
      }                                                   \
  } while (0)

#define CCL_CALL(expr)                                     \
  do {                                                     \
        ccl_status_t status = ccl_status_success;          \
        status = expr;                                     \
        ASSERT(status == ccl_status_success, "CCL error"); \
        (void)status;                                      \
  } while (0)

ccl_coll_attr_t coll_attr;
ccl_request_t request;
size_t rank, size;
double t1, t2, t;
size_t idx, iter_idx;

double when(void)
{
    struct timeval tv;
    static struct timeval tv_base;
    static int is_first = 1;

    if (gettimeofday(&tv, NULL))
    {
        perror("gettimeofday");
        return 0;
    }

    if (is_first)
    {
        tv_base = tv;
        is_first = 0;
    }

    return (double)(tv.tv_sec - tv_base.tv_sec) * 1.0e6 +
           (double)(tv.tv_usec - tv_base.tv_usec);
}

void test_init()
{
    CCL_CALL(ccl_init());

    CCL_CALL(ccl_get_comm_rank(NULL, &rank));
    CCL_CALL(ccl_get_comm_size(NULL, &size));

    coll_attr.prologue_fn = NULL;
    coll_attr.epilogue_fn = NULL;
    coll_attr.reduction_fn = NULL;
    coll_attr.priority = 0;
    coll_attr.synchronous = 0;
    coll_attr.match_id = "tensor_name";
    coll_attr.to_cache = 0;
    coll_attr.vector_buf = 0;
}

void test_finalize()
{
    CCL_CALL(ccl_finalize());
}

#endif /* BASE_H */
