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

#if defined(__INTEL_COMPILER) || defined(__ICC) || defined(__INTEL_LLVM_COMPILER)
#include <immintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <functional>
#include <malloc.h>
#include <map>
#include <mutex>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <vector>

#include "common/utils/spinlock.hpp"
#include "internal_types.hpp"

/* common */

#ifndef gettid
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)
#endif

#define CCL_CALL(expr) \
    do { \
        status = (expr); \
        CCL_ASSERT(status == ccl::status::success, "bad status ", status); \
    } while (0)

#define unlikely(x_) __builtin_expect(!!(x_), 0)
#define likely(x_)   __builtin_expect(!!(x_), 1)

#ifndef container_of
#define container_of(ptr, type, field) ((type*)((char*)ptr - offsetof(type, field)))
#endif

#define CCL_UNDEFINED_CPU_ID    (-1)
#define CCL_UNDEFINED_NUMA_NODE (-1)

#define CACHELINE_SIZE 64

#define CCL_REG_MSG_ALIGNMENT   (4096)
#define CCL_LARGE_MSG_ALIGNMENT (2 * 1024 * 1024)
#define CCL_LARGE_MSG_THRESHOLD (1 * 1024 * 1024)

/* malloc/realloc/free */

#if 0 // defined(__INTEL_COMPILER) || defined(__ICC)
#define CCL_MEMALIGN_IMPL(size, align) _mm_malloc(size, align)
#define CCL_REALLOC_IMPL(old_ptr, old_size, new_size, align) \
    ({ \
        void* new_ptr = NULL; \
        if (!old_ptr) \
            new_ptr = _mm_malloc(new_size, align); \
        else if (!old_size) \
            _mm_free(old_ptr); \
        else { \
            new_ptr = _mm_malloc(new_size, align); \
            memcpy(new_ptr, old_ptr, std::min(old_size, new_size)); \
            _mm_free(old_ptr); \
        } \
        new_ptr; \
    })
#define CCL_CALLOC_IMPL(size, align) \
    ({ \
        void* ptr = _mm_malloc(size, align); \
        memset(ptr, 0, size); \
        ptr; \
    })
#define CCL_FREE_IMPL(ptr) _mm_free(ptr)
#elif defined(__GNUC__)
#define CCL_MEMALIGN_IMPL(size, align) \
    ({ \
        void* ptr = NULL; \
        int pm_ret __attribute__((unused)) = posix_memalign((void**)(&ptr), align, size); \
        ptr; \
    })
#define CCL_REALLOC_IMPL(old_ptr, old_size, new_size, align) realloc(old_ptr, new_size)
#define CCL_CALLOC_IMPL(size, align)                         calloc(size, 1)
#define CCL_FREE_IMPL(ptr)                                   free(ptr)
#else
#error "this compiler is not supported"
#endif

#define CCL_MALLOC_WRAPPER(size, name) \
    ({ \
        size_t alignment = CCL_REG_MSG_ALIGNMENT; \
        if (size >= CCL_LARGE_MSG_THRESHOLD) \
            alignment = CCL_LARGE_MSG_ALIGNMENT; \
        void* ptr = CCL_MEMALIGN_IMPL(size, alignment); \
        CCL_THROW_IF_NOT(ptr, "CCL cannot allocate bytes: ", size, ", out of memory, ", name); \
        ptr; \
    })

#define CCL_MEMALIGN_WRAPPER(size, align, name) \
    ({ \
        void* ptr = CCL_MEMALIGN_IMPL(size, align); \
        CCL_THROW_IF_NOT(ptr, "CCL cannot allocate bytes: ", size, ", out of memory, ", name); \
        ptr; \
    })

#define CCL_REALLOC_WRAPPER(old_ptr, old_size, new_size, align, name) \
    ({ \
        void* ptr = CCL_REALLOC_IMPL(old_ptr, old_size, new_size, align); \
        CCL_THROW_IF_NOT(ptr, "CCL cannot allocate bytes: ", new_size, ", out of memory, ", name); \
        ptr; \
    })

#define CCL_CALLOC_WRAPPER(size, align, name) \
    ({ \
        void* ptr = CCL_CALLOC_IMPL(size, align); \
        CCL_THROW_IF_NOT(ptr, "CCL cannot allocate bytes: ", size, ", out of memory, ", name); \
        ptr; \
    })

#define CCL_MALLOC(size, name)          CCL_MALLOC_WRAPPER(size, name)
#define CCL_MEMALIGN(size, align, name) CCL_MEMALIGN_WRAPPER(size, align, name)
#define CCL_CALLOC(size, name)          CCL_CALLOC_WRAPPER(size, CACHELINE_SIZE, name)
#define CCL_REALLOC(old_ptr, old_size, new_size, align, name) \
    CCL_REALLOC_WRAPPER(old_ptr, old_size, new_size, align, name)
#define CCL_FREE(ptr) CCL_FREE_IMPL(ptr)

/* other */
namespace ccl {
namespace utils {

size_t get_ptr_diff(const void* ptr1, const void* ptr2);
size_t pof2(size_t number);
size_t aligned_sz(size_t size, size_t alignment);

template <class container>
container tokenize(const std::string& input, char delimeter) {
    std::istringstream ss(input);
    container ret;
    std::string str;
    while (std::getline(ss, str, delimeter)) {
        //use c++14 regex
        std::stringstream converter;
        converter << str;
        typename container::value_type value;
        converter >> value;
        ret.push_back(value);
    }
    return ret;
}

template <class Container>
std::string vec_to_string(Container& elems) {
    if (elems.empty()) {
        return "<empty>";
    }

    size_t idx = 0;
    std::ostringstream ss;
    for (auto elem : elems) {
        ss << elem;
        idx++;
        if (idx < elems.size()) {
            ss << " ";
        }
    }
    return ss.str();
}

template <typename T>
inline T from_string(const std::string& str) {
    T val;
    std::stringstream ss(str);
    ss >> val;
    return val;
}

template <class T>
void str_to_array(const std::string& input_str, std::string delims, std::vector<T>& result) {
    size_t beg, pos = 0;
    while ((beg = input_str.find_first_not_of(delims, pos)) != std::string::npos) {
        pos = input_str.find_first_of(delims, beg + 1);
        auto str = input_str.substr(beg, pos - beg);
        if (str.size() == 0) {
            throw ccl::exception("unexpected result string size: 0");
        }
        result.push_back(from_string<T>(str));
    }
}

void str_to_array(const std::string& input_str,
                  std::string delimiter,
                  std::vector<std::string>& result);

std::string get_substring_between_delims(std::string& full_str,
                                         const std::string& start_delim,
                                         const std::string& stop_delim);
} // namespace utils
} // namespace ccl
