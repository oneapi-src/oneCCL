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
#define Collective_Name "CCL_ALLTOALL"

#include <chrono>
#include <functional>
#include <vector>

#include "base_impl.hpp"

template <typename T> class alltoall_test : public base_test <T> {
public:
     int check(typed_test_param<T>& param)
     {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t proc_idx = 0; proc_idx < param.process_count; proc_idx++)
            {
                for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++)
                {
                    T expected = static_cast<T>(proc_idx);
                    if (param.recv_buf[buf_idx][(param.elem_count * proc_idx) + elem_idx] != expected)
                    {
                        sprintf(alltoall_test::get_err_message(), "[%zu] got recv_buf[%zu][%zu]  = %f, but expected = %f\n",
                                param.process_idx, buf_idx, (param.elem_count * proc_idx) + elem_idx, (double) param.recv_buf[buf_idx][(param.elem_count * proc_idx) + elem_idx], (double) expected);
                        return TEST_FAILURE;
                    }
                }
            }
        }
        return TEST_SUCCESS;
    }
    void fill_buffers(typed_test_param<T>& param)
    {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            param.recv_buf[buf_idx].resize(param.elem_count * param.process_count * sizeof(T));
            param.send_buf[buf_idx].resize(param.elem_count * param.process_count * sizeof(T));
        }
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t proc_idx = 0; proc_idx < param.process_count * param.elem_count; proc_idx++)
            {
                param.send_buf[buf_idx][proc_idx] = param.process_idx;
                if (param.test_conf.place_type == PT_OOP)
                {
                    param.recv_buf[buf_idx][proc_idx] = static_cast<T>SOME_VALUE;
                }
            }
        }
        if (param.test_conf.place_type != PT_OOP)
            param.recv_buf = param.send_buf;
    }
    void run_derived(typed_test_param<T>& param)
    {
        const ccl_test_conf& test_conf = param.get_conf();
        size_t count = param.elem_count;
        ccl::coll_attr* attr = &param.coll_attr;
        ccl::stream_t& stream = param.get_stream();
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            param.prepare_coll_attr(param.buf_indexes[buf_idx]);
            T* send_buf = param.send_buf[param.buf_indexes[buf_idx]].data();
            T* recv_buf = param.recv_buf[param.buf_indexes[buf_idx]].data();
            param.reqs[buf_idx] =
                param.global_comm->alltoall((test_conf.place_type == PT_IN) ? recv_buf : send_buf,
                 recv_buf, count, attr, stream);
        }
    }
};

RUN_METHOD_DEFINITION(alltoall_test);
TEST_CASES_DEFINITION(alltoall_test);
MAIN_FUNCTION();
