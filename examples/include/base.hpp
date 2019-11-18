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

#ifndef BASE_HPP
#define BASE_HPP

#include "ccl.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include <sys/time.h>
#include <vector>

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
using namespace cl::sycl;
using namespace cl::sycl::access;
#endif

#define DEFAULT_BACKEND "cpu"

#define ITERS                (16)
#define COLL_ROOT            (0)
#define MSG_SIZE_COUNT       (6)
#define START_MSG_SIZE_POWER (10)

#define PRINT(fmt, ...)             \
    printf(fmt"\n", ##__VA_ARGS__); \

#define PRINT_BY_ROOT(fmt, ...)         \
    if (comm->rank() == 0)               \
    {                                   \
        printf(fmt"\n", ##__VA_ARGS__); \
    }

#define ASSERT(cond, fmt, ...)                            \
  do                                                      \
  {                                                       \
      if (!(cond))                                        \
      {                                                   \
          printf("FAILED\n");                             \
          fprintf(stderr, "ASSERT '%s' FAILED " fmt "\n", \
                  #cond, ##__VA_ARGS__);                  \
          throw std::runtime_error("ASSERT FAILED");      \
      }                                                   \
  } while (0)

#define MSG_LOOP(per_msg_code)                                  \
  do                                                            \
  {                                                             \
      PRINT_BY_ROOT("iters=%d, msg_size_count=%d, "             \
                    "start_msg_size_power=%d, coll_root=%d",    \
                    ITERS, MSG_SIZE_COUNT,                      \
                    START_MSG_SIZE_POWER, COLL_ROOT);           \
      std::vector<size_t> msg_counts(MSG_SIZE_COUNT);           \
      std::vector<std::string> msg_match_ids(MSG_SIZE_COUNT);   \
      for (size_t idx = 0; idx < MSG_SIZE_COUNT; ++idx)         \
      {                                                         \
          msg_counts[idx] = 1u << (START_MSG_SIZE_POWER + idx); \
          msg_match_ids[idx] = std::to_string(msg_counts[idx]); \
      }                                                         \
      try                                                       \
      {                                                         \
          for (size_t idx = 0; idx < MSG_SIZE_COUNT; ++idx)     \
          {                                                     \
              size_t msg_count = msg_counts[idx];               \
              coll_attr.match_id = msg_match_ids[idx].c_str();  \
              PRINT_BY_ROOT("msg_count=%zu, match_id=%s",       \
                            msg_count, coll_attr.match_id);     \
              per_msg_code;                                     \
          }                                                     \
      }                                                         \
      catch (ccl::ccl_error& e)                                 \
      {                                                         \
          printf("FAILED\n");                                   \
          fprintf(stderr, "ccl exception:\n%s\n", e.what());    \
      }                                                         \
      catch (...)                                               \
      {                                                         \
          printf("FAILED\n");                                   \
          fprintf(stderr, "other exception\n");                 \
      }                                                         \
      PRINT_BY_ROOT("PASSED");                                  \
  } while (0)

double when(void)
{
    struct timeval tv;
    static struct timeval tv_base;
    static int is_first = 1;

    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        return 0;
    }

    if (is_first) {
        tv_base = tv;
        is_first = 0;
    }
    return (double)(tv.tv_sec - tv_base.tv_sec) * 1.0e6 +
           (double)(tv.tv_usec - tv_base.tv_usec);
}

void print_timings(ccl::communicator& comm,
                  double* timer, size_t elem_count,
                  size_t elem_size, size_t buf_count,
                  size_t rank, size_t size)
{
    double* timers = (double*)malloc(size * sizeof(double));
    size_t* recv_counts = (size_t*)malloc(size * sizeof(size_t));

    size_t idx;
    for (idx = 0; idx < size; idx++)
        recv_counts[idx] = 1;

    ccl::coll_attr attr;
    memset(&attr, 0, sizeof(ccl_coll_attr_t));

    comm.allgatherv(timer,
                    1,
                    timers,
                    recv_counts,
                    &attr,
                    nullptr)->wait();

    if (rank == 0)
    {
        double avg_timer = 0;
        double avg_timer_per_buf = 0;
        for (idx = 0; idx < size; idx++)
        {
            avg_timer += timers[idx];
        }
        avg_timer /= (ITERS * size);
        avg_timer_per_buf = avg_timer / buf_count;

        double stddev_timer = 0;
        double sum = 0;
        for (idx = 0; idx < size; idx++)
        {
            double val = timers[idx] / ITERS;
            sum += (val - avg_timer) * (val - avg_timer);
        }
        stddev_timer = sqrt(sum / size) / avg_timer * 100;
        printf("size %10zu x %5zu bytes, avg %10.2lf us, avg_per_buf %10.2f, stddev %5.1lf %%\n",
                elem_count * elem_size, buf_count, avg_timer, avg_timer_per_buf, stddev_timer);
    }
    comm.barrier();

    free(timers);
    free(recv_counts);
}

#endif /* BASE_HPP */
