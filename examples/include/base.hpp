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
#pragma once

#include "oneapi/ccl.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdexcept>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <vector>
#include <unistd.h>

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
using namespace cl::sycl;
using namespace cl::sycl::access;
#endif // CCL_ENABLE_SYCL

#define GETTID() syscall(SYS_gettid)

#define ITERS                (16)
#define COLL_ROOT            (0)
#define MSG_SIZE_COUNT       (6)
#define START_MSG_SIZE_POWER (10)

#define PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__);

#define PRINT_BY_ROOT(comm, fmt, ...) \
    if (comm.rank() == 0) { \
        printf(fmt "\n", ##__VA_ARGS__); \
    }

#define ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            printf("FAILED\n"); \
            fprintf(stderr, \
                    "(%ld): %s:%s:%d: ASSERT '%s' FAILED: " fmt "\n", \
                    GETTID(), \
                    __FILE__, \
                    __FUNCTION__, \
                    __LINE__, \
                    #cond, \
                    ##__VA_ARGS__); \
            fflush(stderr); \
            throw std::runtime_error("ASSERT FAILED"); \
        } \
    } while (0)

#define MSG_LOOP(comm, per_msg_code) \
    do { \
        PRINT_BY_ROOT(comm, \
                      "iters=%d, msg_size_count=%d, " \
                      "start_msg_size_power=%d, coll_root=%d", \
                      ITERS, \
                      MSG_SIZE_COUNT, \
                      START_MSG_SIZE_POWER, \
                      COLL_ROOT); \
        std::vector<size_t> msg_counts(MSG_SIZE_COUNT); \
        std::vector<ccl::string_class> msg_match_ids(MSG_SIZE_COUNT); \
        for (size_t idx = 0; idx < MSG_SIZE_COUNT; ++idx) { \
            msg_counts[idx] = 1u << (START_MSG_SIZE_POWER + idx); \
            msg_match_ids[idx] = std::to_string(msg_counts[idx]); \
        } \
        try { \
            for (size_t idx = 0; idx < MSG_SIZE_COUNT; ++idx) { \
                size_t msg_count = msg_counts[idx]; \
                attr.set<ccl::operation_attr_id::match_id>(msg_match_ids[idx]); \
                PRINT_BY_ROOT(comm, \
                              "msg_count=%zu, match_id=%s", \
                              msg_count, \
                              attr.get<ccl::operation_attr_id::match_id>().c_str()); \
                per_msg_code; \
            } \
        } \
        catch (ccl::exception & e) { \
            printf("FAILED\n"); \
            fprintf(stderr, "ccl exception:\n%s\n", e.what()); \
        } \
        catch (...) { \
            printf("FAILED\n"); \
            fprintf(stderr, "other exception\n"); \
        } \
        PRINT_BY_ROOT(comm, "PASSED"); \
    } while (0)

inline double when(void) {
    auto time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(time.time_since_epoch());
    return duration.count();
}

inline void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized)
        MPI_Finalize();
}
