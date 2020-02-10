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
#include "exec/thread/base_thread.hpp"

ccl_status_t ccl_base_thread::start()
{
    LOG_DEBUG(name(), " ", idx);
    int err = pthread_create(&thread, nullptr, progress_function, get_this());
    if (err)
    {
        LOG_ERROR("error while creating ", name(), " thread #", idx, " pthread_create returns ", err);
        return ccl_status_runtime_error;
    }
    return ccl_status_success;
}

ccl_status_t ccl_base_thread::stop()
{
    LOG_DEBUG(name(), " # ", idx);

    void* exit_code;
    int err = pthread_cancel(thread);
    if (err)
        LOG_INFO("error while canceling progress thread # ", idx, ", pthread_cancel returns ", err);

    err = pthread_join(thread, &exit_code);
    if (err)
    {
        LOG_INFO("error while joining progress thread # ", idx, " , pthread_join returns ", err);
    }
    else
    {
        LOG_DEBUG("progress thread # ", idx, ", exited with code ",
                  idx, " (", (uintptr_t) exit_code, (exit_code == PTHREAD_CANCELED) ? "PTHREAD_CANCELED" : "?", ")");
    }
    return ccl_status_success;
}

ccl_status_t ccl_base_thread::pin(int proc_id)
{
    LOG_DEBUG(name(), " # ", idx, ", proc_id ", proc_id);
    int pthread_err;
    cpu_set_t cpuset;

    __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
    __CPU_SET_S(proc_id, sizeof(cpu_set_t), &cpuset);

    if ((pthread_err = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0)
    {
        LOG_ERROR("pthread_setaffinity_np failed, err ", pthread_err);
        return ccl_status_runtime_error;
    }

    if ((pthread_err = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0)
    {
        LOG_ERROR("pthread_getaffinity_np failed, err ", pthread_err);
        return ccl_status_runtime_error;
    }

    if (!__CPU_ISSET_S(proc_id, sizeof(cpu_set_t), &cpuset))
    {
        LOG_ERROR(name(), " ", idx, " is not pinned on proc_id ", proc_id);
        return ccl_status_runtime_error;
    }

    return ccl_status_success;
}
