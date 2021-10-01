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

#ifdef HOST_CTX

#define __global
using namespace ccl;
#include <cstdint>

#ifdef ENABLE_KERNEL_ATOMICS
// type for sync flags for atomics support
typedef atomic_int sync_flag_type;
#else
// default type for sync flags
typedef volatile int sync_flag_type;
#endif // ENABLE_KERNEL_ATOMICS

#else // HOST_CTX

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#include "lp.h"

#define FORMAT_int8_t  "%hhd"
#define FORMAT_int16_t "%d"
#define FORMAT_int32_t "%d"
#define FORMAT_int64_t "%ld"

#define FORMAT_uint8_t  "%hhu"
#define FORMAT_uint16_t "%u"
#define FORMAT_uint32_t "%u"
#define FORMAT_uint64_t "%lu"

#define FORMAT_float  "%f"
#define FORMAT_double "%f"

#define FORMAT_ushort "%u"
#define FORMAT_half   "%f"

#define FORMAT_4(format) #format ", " #format ", " #format ", " #format
#define FORMAT_char4     FORMAT_4(% hhd)
#define FORMAT_uchar4    FORMAT_4(% hhu)
#define FORMAT_short4    FORMAT_4(% d)
#define FORMAT_ushort4   FORMAT_4(% u)
#define FORMAT_int4      FORMAT_4(% d)
#define FORMAT_uint4     FORMAT_4(% u)
#define FORMAT_long4     FORMAT_4(% ld)
#define FORMAT_ulong4    FORMAT_4(% lu)
#define FORMAT_float4    FORMAT_4(% f)
#define FORMAT_double4   FORMAT_4(% f)

#define ELEMENTS_1(X) X
#define ELEMENTS_4(X) (X)[0], (X)[1], (X)[2], (X)[3]

// define aliases for OpenCL types
typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;
typedef half float16_t;
typedef float float32_t;
typedef double float64_t;
typedef ushort bfloat16;

#define DEFINE_SUM_OP(T) \
    T __sum_##T(T lhs, T rhs) { \
        return lhs + rhs; \
    }

#define DEFINE_PROD_OP(T) \
    T __prod_##T(T lhs, T rhs) { \
        return lhs * rhs; \
    }

#define DEFINE_MIN_OP(T) \
    T __min_##T(T lhs, T rhs) { \
        return min(lhs, rhs); \
    }

#define DEFINE_MAX_OP(T) \
    T __max_##T(T lhs, T rhs) { \
        return max(lhs, rhs); \
    }

#ifdef ENABLE_KERNEL_DEBUG
#define DEBUG_BLOCK(block) block
#else
#define DEBUG_BLOCK(block)
#endif

#ifdef ENABLE_KERNEL_DEBUG
#define LOG_INPUT_DATA_START(rank)    printf("kernel %d, wait income data\n", rank)
#define LOG_INPUT_DATA_END(rank)      printf("kernel %d, received data\n", rank)
#define LOG_OUTGOING_DATA_START(rank) printf("kernel %d, wait signal to send\n", rank)
#define LOG_OUTGOING_DATA_END(rank)   printf("kernel %d, received signal to send\n", rank)
#define LOG_SEND_PROGRESS(rank, thread_id, flag, desired) \
    printf("kernel %d.%d, send %d/%d\n", rank, thread_id, flag, desired)
#define LOG_BARRIER_PASSED(rank, thread_id) \
    printf("kernel %d.%d barrier passed\n", rank, thread_id);
#define LOG_IN_BARRIER(rank, thread_id, flag, desired) \
    printf("kernel %d.%d barrier %d/%d\n", rank, thread_id, flag, desired);
#else // ENABLE_KERNEL_DEBUG
#define LOG_INPUT_DATA_START(rank)
#define LOG_INPUT_DATA_END(rank)
#define LOG_OUTGOING_DATA_START(rank)
#define LOG_OUTGOING_DATA_END(rank)
#define LOG_BARRIER_PASSED(rank, thread_id)
#define LOG_IN_BARRIER(rank, thread_id, flag, desired)
#endif // ENABLE_KERNEL_DEBUG

#define SWAP_VARIABLES(var1, var2, type) \
    do { \
        type tmp; \
        tmp = var1; \
        var1 = var2; \
        var2 = tmp; \
    } while (0);

int get_left_rank(int rank, int comm_size) {
    return rank == 0 ? comm_size - 1 : rank - 1;
}

int get_right_rank(int rank, int comm_size) {
    return rank == (comm_size - 1) ? 0 : rank + 1;
}

#ifdef ENABLE_KERNEL_ATOMICS

// type for sync flags for atomics support
typedef atomic_int sync_flag_type;

#define PUT_READY_TO_RECEIVE(_sync_flag) \
    if (thread_id == 0) { \
        atomic_fetch_add_explicit( \
            _sync_flag, 1, memory_order_seq_cst, memory_scope_all_svm_devices); \
    }

#define I_SENT(_sync_flag) \
    if (thread_id == 0) { \
        atomic_fetch_add_explicit( \
            _sync_flag, 1, memory_order_seq_cst, memory_scope_all_svm_devices); \
    }

#define WAIT_INPUT_DATA(_sync_flag, _desired) \
    if (thread_id == 0) { \
        LOG_INPUT_DATA_START(my_rank); \
        while (1) { \
            int _old_value = atomic_load_explicit( \
                _sync_flag, memory_order_seq_cst, memory_scope_all_svm_devices); \
            if (_old_value == _desired) { \
                LOG_INPUT_DATA_END(my_rank); \
                ++_desired; \
                break; \
            } \
        } \
    }

#define WAIT_SIGNAL_TO_SEND(_sync_flag, _desired) \
    if (thread_id == 0) { \
        LOG_OUTGOING_DATA_START(my_rank); \
        while (_desired != atomic_load_explicit( \
                               _sync_flag, memory_order_seq_cst, memory_scope_all_svm_devices)) { \
        } \
        LOG_OUTGOING_DATA_END(my_rank); \
        ++_desired; \
    }

#define SET_PROXY_SIZE(_sync_flag, size) \
    if (thread_id == 0) { \
        atomic_store_explicit( \
            _sync_flag, size, memory_order_seq_cst, memory_scope_all_svm_devices); \
    }

#define GET_PROXY_SIZE(_sync_flag, size) \
    size = atomic_load_explicit(_sync_flag, memory_order_seq_cst, memory_scope_all_svm_devices);

#else // ENABLE_KERNEL_ATOMICS

// default type for sync flags
typedef volatile int sync_flag_type;

#define PUT_READY_TO_RECEIVE(_sync_flag) \
    if (thread_id == 0) { \
        (*_sync_flag)++; \
    }

#define I_SENT(_sync_flag) \
    if (thread_id == 0) { \
        (*_sync_flag)++; \
    }

#define WAIT_INPUT_DATA(_sync_flag, _desired) \
    if (thread_id == 0) { \
        LOG_INPUT_DATA_START(my_rank); \
        while (1) { \
            if (*_sync_flag == _desired) { \
                LOG_INPUT_DATA_END(my_rank); \
                ++_desired; \
                break; \
            } \
        } \
    }

#define WAIT_SIGNAL_TO_SEND(_sync_flag, _desired) \
    if (thread_id == 0) { \
        LOG_OUTGOING_DATA_START(my_rank); \
        while (_desired != *_sync_flag) { \
        }; \
        LOG_OUTGOING_DATA_END(my_rank); \
        ++_desired; \
    }

#define SET_PROXY_SIZE(_sync_flag, size) \
    if (thread_id == 0) { \
        *_sync_flag = size; \
    }

#define GET_PROXY_SIZE(_sync_flag, size) size = *_sync_flag;

#endif // ENABLE_KERNEL_ATOMICS

/*
#define KERNEL_BARRIER(_barrier_flag, _desired, _increment)                         \
    do                                                                              \
    {                                                                               \
        int _barrier_value = atomic_add(_barrier_flag, 0);                          \
        atomic_inc(_barrier_flag);                                                  \
        int _old_value = _barrier_value;                                            \
        while(1)                                                                    \
        {                                                                           \
            / *thread that last reached the barrier will reset it                    \
              other threads may expect to receive _desired value while it can be 0  \
              check if received value is less than initially received* /             \
            if(_old_value == _desired || _old_value < _barrier_value)               \
            {                                                                       \
                BARRIER_PASSED(my_rank, thread_id);                                 \
                break;                                                              \
            }                                                                       \
            IN_BARRIER(my_rank, thread_id, _old_value, _desired);                   \
            _old_value = atomic_add(_barrier_flag, 0);                \
        }                                                                           \
    } while (0);
*/

/* for A2A */
/*#define WAIT_INPUT_DATA(_sync_flag, _desired) \
    if (local_thread_id == 0) { \
        LOG_INPUT_DATA_START(rank_id); \
        while (1) { \
            int _old_value = atomic_cmpxchg(_sync_flag, _desired, _desired); \
            if (_old_value == _desired) { \
                LOG_INPUT_DATA_END(rank_id); \
                _desired += 1 + comm_size; \
                break; \
            } \
        } \
    }

#define WAIT_SIGNAL_TO_SEND(_sync_flag, _desired) \
    if (local_thread_id == 0) { \
        LOG_OUTGOING_DATA_START(rank_id); \
        while (_desired != atomic_cmpxchg(_sync_flag, _desired, _desired)) { \
        }; \
        LOG_OUTGOING_DATA_END(rank_id); \
        _desired += comm_size; \
    }*/

#endif // HOST_CTX
