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
#define Collective_Name "CCL_ALLGATHERV"
#define TEST_CCL_ALLGATHERV

#include <chrono>
#include <functional>
#include <vector>

#include "base_impl.hpp"

template <typename T> class allgatherv_test : public base_test <T>
{
public:
    std::vector<size_t> recv_counts;
    std::vector<size_t> offsets;
    int check(typed_test_param<T>& param)
    {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t elem_idx = 0; elem_idx < param.process_count; elem_idx++)
            {
                for (size_t recv_count_idx = 0; recv_count_idx < recv_counts[elem_idx]; recv_count_idx++)
                {
                    size_t idx = offsets[elem_idx] + recv_count_idx;
                    T expected = static_cast<T>(elem_idx + recv_count_idx);
                    if ((param.test_conf.place_type != PT_OOP) && (param.recv_buf[buf_idx][idx] != expected))
                    {
                        snprintf(allgatherv_test::get_err_message(), ERR_MESSAGE_MAX_LEN, "[%zu] got recv_buf[%zu][%zu]  = %f, but expected = %f\n",
                                 param.process_idx, buf_idx, idx, (double) param.recv_buf[buf_idx][idx], (double) expected);
                        return TEST_FAILURE;
                    }
                    else if (param.recv_buf[buf_idx][idx] != expected)
                    {
                        snprintf(allgatherv_test::get_err_message(), ERR_MESSAGE_MAX_LEN, "[%zu] got recv_buf[%zu][%zu]  = %f, but expected = %f\n",
                                 param.process_idx, buf_idx, idx, (double) param.recv_buf[buf_idx][idx], (double) expected);
                        return TEST_FAILURE;
                    }
                }
            }
        }
        return TEST_SUCCESS;
    }
    void alloc_buffers(typed_test_param<T>& param)
    {
        base_test<T>::alloc_buffers(param);
        recv_counts.resize(param.process_count);
        offsets.resize(param.process_count);
        offsets[0] = 0;
        recv_counts[0] = param.elem_count;
        if (param.test_conf.place_type == PT_OOP)
        {
            for (size_t elem_idx = 0; elem_idx < param.buffer_count; elem_idx++)
            {
                /* each buffer is different size */
                param.recv_buf[elem_idx].resize(param.elem_count * param.process_count * sizeof(T));
            }
        }
    }
    void fill_buffers(typed_test_param<T>& param)
    {
        for (size_t proc_idx = 1; proc_idx < param.process_count; proc_idx++)
        {
            recv_counts[proc_idx] = (param.elem_count > proc_idx) ? param.elem_count - proc_idx : param.elem_count;
            offsets[proc_idx] = recv_counts[proc_idx - 1] + offsets[proc_idx - 1];
        }
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t elem_idx = 0; elem_idx < param.process_count; elem_idx++)
            {
                for (size_t recv_count_idx = 0; recv_count_idx < recv_counts[elem_idx]; recv_count_idx++)
                {
                    param.send_buf[buf_idx][offsets[elem_idx] + recv_count_idx] = param.process_idx + elem_idx + recv_count_idx;
                    if (param.test_conf.place_type == PT_OOP)
                    {
                        param.recv_buf[buf_idx][offsets[elem_idx] + recv_count_idx] = static_cast<T>SOME_VALUE;
                    }
                }
            }
            /* in case of in-place i-th process already has result in i-th block of send buffer */
            if (param.test_conf.place_type != PT_OOP)
            {
                param.recv_buf[buf_idx] = param.send_buf[buf_idx];
                for (size_t recv_count_idx = 0; recv_count_idx < recv_counts[param.process_idx]; recv_count_idx++)
                {
                    param.send_buf[buf_idx][offsets[param.process_idx] + recv_count_idx] = param.process_idx + recv_count_idx;
                }
            }
        }
        if (param.test_conf.place_type != PT_OOP)
        {
            param.recv_buf = param.send_buf;
        }
    }

    void run_derived(typed_test_param<T>& param)
    {
        const ccl_test_conf& test_conf = param.get_conf();
        size_t count = recv_counts[param.process_idx];
        size_t* recv_count = recv_counts.data();
        ccl::coll_attr* attr = &param.coll_attr;
        ccl::stream_t& stream = param.get_stream();
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            param.prepare_coll_attr(param.buf_indexes[buf_idx]);
            T* send_buf = param.send_buf[param.buf_indexes[buf_idx]].data();
            T* recv_buf = param.recv_buf[param.buf_indexes[buf_idx]].data();
            param.reqs[buf_idx] =
                param.global_comm->allgatherv((test_conf.place_type == PT_IN) ? recv_buf : send_buf,
                count, recv_buf, recv_count, attr, stream);
        }
    }
};

RUN_METHOD_DEFINITION(allgatherv_test);
TEST_CASES_DEFINITION(allgatherv_test);
MAIN_FUNCTION();
