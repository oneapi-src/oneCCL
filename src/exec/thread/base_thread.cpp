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
#include "common/global/global.hpp"
#include "common/utils/yield.hpp"
#include "exec/thread/base_thread.hpp"

ccl_status_t ccl_base_thread::start(int affinity) {
    LOG_DEBUG(name(), " ", idx);

    start_affinity = affinity;

    /* start thread with initial affinity */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    cpu_set_t cpuset;
    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
    __CPU_SET_S(affinity, sizeof(cpu_set_t), &cpuset);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);

    int err = pthread_create(&thread, &attr, progress_function, get_this());
    if (err) {
        LOG_ERROR(
            "error while creating ", name(), " thread #", idx, " pthread_create returns ", err);
        return ccl_status_runtime_error;
    }

    while (!started.load(std::memory_order_relaxed)) {
        ccl_yield(ccl::global_data::env().yield_type);
    }

    return ccl_status_success;
}

ccl_status_t ccl_base_thread::stop() {
    LOG_DEBUG(name(), " # ", idx);

    void* exit_code;
    int err;

    should_stop = true;

    while (started.load(std::memory_order_relaxed)) {
        ccl_yield(ccl::global_data::env().yield_type);
    }

    err = pthread_join(thread, &exit_code);
    if (err) {
        LOG_INFO("error while joining progress thread # ", idx, " , pthread_join returns ", err);
    }
    else {
        LOG_DEBUG("progress thread # ",
                  idx,
                  ", exited with code (",
                  (uintptr_t)exit_code,
                  (exit_code == PTHREAD_CANCELED) ? "PTHREAD_CANCELED" : "?",
                  ")");
    }

    return ccl_status_success;
}

ccl_status_t ccl_base_thread::set_affinity(int affinity) {
    LOG_DEBUG(name(), " # ", idx, ", affinity ", affinity);

    int pthread_err;
    cpu_set_t cpuset;

    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
    __CPU_SET_S(affinity, sizeof(cpu_set_t), &cpuset);

    if ((pthread_err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0) {
        LOG_ERROR("pthread_setaffinity_np failed, err ", pthread_err);
        return ccl_status_runtime_error;
    }

    if (get_affinity() != affinity) {
        LOG_ERROR(name(), " ", idx, " is not pinned ", affinity);
        return ccl_status_runtime_error;
    }

    return ccl_status_success;
}

int ccl_base_thread::get_affinity() {
    int pthread_err;
    int result = CCL_UNDEFINED_CPU_ID;
    cpu_set_t cpuset;

    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);

    if ((pthread_err = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0) {
        LOG_ERROR("pthread_getaffinity_np failed, err ", pthread_err);
    }

    for (int idx = 0; idx < CPU_SETSIZE; idx++) {
        if (__CPU_ISSET_S(idx, sizeof(cpu_set_t), &cpuset)) {
            if (result == CCL_UNDEFINED_CPU_ID) {
                result = idx;
            }
            else {
                CCL_THROW("multiple affinity cores, previous ", result, ", new ", idx);
            }
        }
    }

    CCL_THROW_IF_NOT(result != CCL_UNDEFINED_CPU_ID, "can't retrieve affinity");

    return result;
}
