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

#pragma once

#if defined(__INTEL_COMPILER) || defined(__ICC)
#include <malloc.h>
#else

#include <mm_malloc.h>

#endif

#include <immintrin.h>
#include <stdlib.h>
#include <stddef.h>
#include <algorithm>
#include <chrono>
#include <time.h>

/* common */

#define CCL_CALL(expr)                                   \
  do {                                                   \
        status = (expr);                                 \
        CCL_ASSERT(status == ccl_status_success,         \
            "bad status ", status);                      \
  } while (0)

#define unlikely(x_) __builtin_expect(!!(x_), 0)
#define likely(x_)   __builtin_expect(!!(x_), 1)

#ifndef container_of
#define container_of(ptr, type, field)                  \
    ((type *) ((char *)ptr - offsetof(type, field)))
#endif

#define CACHELINE_SIZE 64
#define ONE_MB         1048576
#define TWO_MB         2097152

#define CCL_MEMCPY(dest, src, n)                                \
    std::copy((char*)(src), (char*)(src) + (n), (char*)(dest))

/* Single-linked list */

struct ccl_slist_entry_t
{
    ccl_slist_entry_t* next;
};

struct ccl_slist_t
{
    ccl_slist_entry_t* head;
    ccl_slist_entry_t* tail;
};

/* malloc/realloc/free */

#if defined(__INTEL_COMPILER) || defined(__ICC)
#define CCL_MEMALIGN_IMPL(size, align) _mm_malloc(size, align)
#define CCL_REALLOC_IMPL(old_ptr, old_size, new_size, align)            \
      ({                                                                \
          void* new_ptr = NULL;                                         \
          if (!old_ptr) new_ptr = _mm_malloc(new_size, align);          \
          else if (!old_size) _mm_free(old_ptr);                        \
          else                                                          \
          {                                                             \
              new_ptr = _mm_malloc(new_size, align);                    \
              memcpy(new_ptr, old_ptr, std::min(old_size, new_size));   \
              _mm_free(old_ptr);                                        \
          }                                                             \
          new_ptr;                                                      \
      })
#define CCL_CALLOC_IMPL(size, align)           \
      ({                                       \
          void* ptr = _mm_malloc(size, align); \
          memset(ptr, 0, size);                \
          ptr;                                 \
      })
#define CCL_FREE_IMPL(ptr) _mm_free(ptr)
#elif defined(__GNUC__)
#define CCL_MEMALIGN_IMPL(size, align)                                                  \
    ({                                                                                  \
      void* ptr = NULL;                                                                 \
      int pm_ret __attribute__((unused)) = posix_memalign((void**)(&ptr), align, size); \
      ptr;                                                                              \
    })
#define CCL_REALLOC_IMPL(old_ptr, old_size, new_size, align) realloc(old_ptr, new_size)
#define CCL_CALLOC_IMPL(size, align) calloc(size, 1)
#define CCL_FREE_IMPL(ptr) free(ptr)
#else
# error "this compiler is not supported" 
#endif

#define CCL_MEMALIGN_WRAPPER(size, align, name)                 \
    ({                                                          \
        void *ptr = CCL_MEMALIGN_IMPL(size, align);             \
        CCL_THROW_IF_NOT(ptr, "CCL Out of memory, ", name);     \
        ptr;                                                    \
    })

#define CCL_REALLOC_WRAPPER(old_ptr, old_size, new_size, align, name)       \
    ({                                                                      \
        void *ptr = CCL_REALLOC_IMPL(old_ptr, old_size, new_size, align);   \
        CCL_THROW_IF_NOT(ptr, "CCL Out of memory, ", name);                 \
        ptr;                                                                \
    })

#define CCL_CALLOC_WRAPPER(size, align, name)                   \
    ({                                                          \
        void *ptr = CCL_CALLOC_IMPL(size, align);               \
        CCL_THROW_IF_NOT(ptr, "CCL Out of memory, ", name);     \
        ptr;                                                    \
    })

#define CCL_MALLOC(size, name)                                CCL_MEMALIGN_WRAPPER(size, CACHELINE_SIZE, name)
#define CCL_MEMALIGN(size, align, name)                       CCL_MEMALIGN_WRAPPER(size, align, name)
#define CCL_CALLOC(size, name)                                CCL_CALLOC_WRAPPER(size, CACHELINE_SIZE, name)
#define CCL_REALLOC(old_ptr, old_size, new_size, align, name) CCL_REALLOC_WRAPPER(old_ptr, old_size, new_size, align, name)
#define CCL_FREE(ptr)                                         CCL_FREE_IMPL(ptr)

/* other */

static inline size_t ccl_pof2(size_t number)
{
    size_t pof2 = 1;

    while (pof2 <= number)
    {
        pof2 <<= 1;
    }
    pof2 >>= 1;

    return pof2;
}

static inline size_t ccl_aligned_sz(size_t size,
                                     size_t alignment)
{
    return ((size % alignment) == 0) ?
           size : ((size / alignment) + 1) * alignment;
}

static inline timespec from_time_point(const std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> point)
{
    auto sec = std::chrono::time_point_cast<std::chrono::seconds>(point);
    auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(point) - std::chrono::time_point_cast<std::chrono::nanoseconds>(sec);

    return timespec { .tv_sec = sec.time_since_epoch().count(), .tv_nsec = ns.count() };
}
