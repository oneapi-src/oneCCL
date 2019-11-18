
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

#define Collective_Name "CCL_BCAST"

#include <chrono>
#include <functional>
#include <vector>

#include "base_impl.hpp"

template <typename T> class bcast_test : public base_test <T>
{
public:
    int check(typed_test_param<T>& param)
    {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++)
            {
                if (param.send_buf[buf_idx][elem_idx] != static_cast<T>(elem_idx))
                {
                    snprintf(bcast_test::get_err_message(), ERR_MESSAGE_MAX_LEN, "[%zu] got send_buf[%zu][%zu] = %f, but expected = %f\n",
                             param.process_idx, buf_idx, elem_idx, (double)param.send_buf[buf_idx][elem_idx], (double)elem_idx);
                    return TEST_FAILURE;
                }
            }
        }
        return TEST_SUCCESS;
    }
    void fill_buffers(typed_test_param<T>& param)
    {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++)
            {
                if (param.process_idx == ROOT_PROCESS_IDX)
                    param.send_buf[buf_idx][elem_idx] = elem_idx;
                else
                    param.send_buf[buf_idx][elem_idx] = static_cast<T>(SOME_VALUE);
            }
        }
    }

    void run_derived(typed_test_param<T>& param)
    {
        size_t count = param.elem_count;
        ccl::coll_attr* attr = &param.coll_attr;
        ccl::stream_t &stream = param.get_stream();
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) 
        {
            param.prepare_coll_attr(param.buf_indexes[buf_idx]);
            T* send_buf = param.send_buf[param.buf_indexes[buf_idx]].data();
            param.reqs[buf_idx] = param.global_comm->bcast(send_buf, count, ROOT_PROCESS_IDX, attr, stream);
        }
    }
};

RUN_METHOD_DEFINITION(bcast_test);
TEST_CASES_DEFINITION(bcast_test);
MAIN_FUNCTION();
