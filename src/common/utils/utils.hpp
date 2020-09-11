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

#if defined(__INTEL_COMPILER) || defined(__ICC)
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
        CCL_ASSERT(status == ccl_status_success, "bad status ", status); \
    } while (0)

#define unlikely(x_) __builtin_expect(!!(x_), 0)
#define likely(x_)   __builtin_expect(!!(x_), 1)

#ifndef container_of
#define container_of(ptr, type, field) ((type*)((char*)ptr - offsetof(type, field)))
#endif

#define CACHELINE_SIZE 64
#define ONE_MB         1048576
#define TWO_MB         2097152

#define CCL_MEMCPY(dest, src, n) std::copy((char*)(src), (char*)(src) + (n), (char*)(dest))

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

#define CCL_MALLOC(size, name)          CCL_MEMALIGN_WRAPPER(size, CACHELINE_SIZE, name)
#define CCL_MEMALIGN(size, align, name) CCL_MEMALIGN_WRAPPER(size, align, name)
#define CCL_CALLOC(size, name)          CCL_CALLOC_WRAPPER(size, CACHELINE_SIZE, name)
#define CCL_REALLOC(old_ptr, old_size, new_size, align, name) \
    CCL_REALLOC_WRAPPER(old_ptr, old_size, new_size, align, name)
#define CCL_FREE(ptr) CCL_FREE_IMPL(ptr)

/* other */

static inline size_t ccl_pof2(size_t number) {
    size_t last_bit_mask = ((size_t)1 << (8 * sizeof(size_t) - 1));
    if (number & last_bit_mask) {
        return last_bit_mask;
    }

    size_t pof2 = 1;
    while (pof2 <= number) {
        pof2 <<= 1;
    }
    pof2 >>= 1;
    return pof2;
}

static inline size_t ccl_aligned_sz(size_t size, size_t alignment) {
    return ((size % alignment) == 0) ? size : ((size / alignment) + 1) * alignment;
}

static inline timespec ccl_from_time_point(
    const std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> point) {
    auto sec = std::chrono::time_point_cast<std::chrono::seconds>(point);
    auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(point) -
              std::chrono::time_point_cast<std::chrono::nanoseconds>(sec);

    return timespec{ .tv_sec = sec.time_since_epoch().count(), .tv_nsec = ns.count() };
}

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

template <typename T>
void ccl_str_to_array(const char* input, std::vector<T>& output, char delimiter) {
    std::stringstream ss(input);
    T temp{};
    while (ss >> temp) {
        output.push_back(temp);
        if (ss.peek() == delimiter) {
            ss.ignore();
        }
    }
}

//TODO naite implementation, use TBB
template <class Key,
          class Value,
          class = typename std::enable_if<std::is_pointer<Value>::value>::type>
class concurrent_map {
public:
    using implementation = std::map<Key, Value>;
    using value_type = typename implementation::value_type;
    using lock_t = std::unique_lock<ccl_spinlock>;

    template <class Impl>
    using accessor = std::tuple<Impl, lock_t>;

    using read_accessor =
        std::tuple<std::reference_wrapper<typename std::add_const<implementation>::type>, lock_t>;
    using write_accessor = std::tuple<std::reference_wrapper<implementation>, lock_t>;

    concurrent_map() = default;
    concurrent_map(concurrent_map<Key, Value>&& src) {
        src.swap(get_write());
    }

    concurrent_map<Key, Value>& operator=(const concurrent_map<Key, Value>&& src) {
        src.swap(get_write());
        return *this;
    }

    concurrent_map(const concurrent_map<Key, Value>&) = delete;
    concurrent_map<Key, Value>& operator=(const concurrent_map<Key, Value>&) = delete;

    std::pair<Value, bool> insert(value_type&& value) {
        Value ret = nullptr;
        bool find = false;
        {
            std::unique_lock<ccl_spinlock> lock(guard);
            auto pair = map.insert(std::move(value));
            find = pair.second;
            ret = pair.first->second;
        }
        return { ret, find };
    }

    Value find(const Key& key) {
        Value ret = nullptr;
        {
            std::unique_lock<ccl_spinlock> lock(guard);
            auto it = map.find(key);
            if (it != map.end()) {
                ret = it->second;
            }
        }
        return ret;
    }

    read_accessor get_read() const {
        return { std::cref(map), locker() };
    }

    write_accessor get_write() {
        return { std::ref(map), locker() };
    }

    void swap(write_accessor&& rhs) {
        {
            std::unique_lock<ccl_spinlock> lock(guard);
            std::swap(map, std::get<0>(rhs).get());
        }
    }

    void swap(write_accessor& rhs) {
        swap(rhs);
    }

private:
    std::unique_lock<ccl_spinlock> locker() const {
        return std::unique_lock<ccl_spinlock>(guard);
    }

    mutable ccl_spinlock guard;
    implementation map;
};
