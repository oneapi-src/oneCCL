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
#define TEST_CCL_ALLTOALLV

#define COLL_NAME "CCL_ALLTOALLV"

#include "base_impl.hpp"

template <typename T> class alltoallv_test : public base_test <T>
{
public:

    std::vector<size_t> send_counts;
    std::vector<size_t> recv_counts;

     int check(typed_test_param<T>& param)
     {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            size_t elem_idx = 0;
            for (size_t proc_idx = 0; proc_idx < param.process_count; proc_idx++)
            {
                T expected = static_cast<T>(proc_idx);
                for (size_t idx = 0; idx < recv_counts[proc_idx]; idx++)
                {
                    if (base_test<T>::check_error(param, expected, buf_idx, elem_idx))
                        return TEST_FAILURE;
                    elem_idx++;
                }
            }
        }
        return TEST_SUCCESS;
    }

    void alloc_buffers(typed_test_param<T>& param)
    {
        base_test<T>::alloc_buffers(param);

        send_counts.resize(param.process_count);
        recv_counts.resize(param.process_count);

        if (param.test_conf.place_type == PT_OOP)
        {
            for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
            {
                param.recv_buf[buf_idx].resize(param.elem_count * param.process_count);
            }
        }
    }

    void fill_buffers(typed_test_param<T>& param)
    {
        /* TODO: this already happens in alloc_buffers, remove duplicated logic, review all func test */
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            param.recv_buf[buf_idx].resize(param.elem_count * param.process_count);
            param.send_buf[buf_idx].resize(param.elem_count * param.process_count);
        }

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            for (size_t elem_idx = 0; elem_idx < param.process_count * param.elem_count; elem_idx++)
            {
                param.send_buf[buf_idx][elem_idx] = param.process_idx;
                if (param.test_conf.place_type == PT_OOP)
                {
                    param.recv_buf[buf_idx][elem_idx] = static_cast<T>SOME_VALUE;
                }
            }
        }

        if (param.test_conf.place_type == PT_IN)
        {
            param.recv_buf = param.send_buf;

            /* 
               Specifying the in-place option indicates that 
               the same amount and type of data is sent and received
               between any two processes in the group of the communicator.
               Different pairs of processes can exchange different amounts of data.
               https://docs.microsoft.com/en-us/message-passing-interface/mpi-alltoallv-function#remarks
             */
            for (size_t idx = 0; idx < param.process_count; idx++)
            {
                size_t common_size = (param.process_idx + idx) * (param.elem_count / 4);
                recv_counts[idx] = ((common_size > param.elem_count) || (common_size == 0)) ?
                    param.elem_count : common_size;
                send_counts[idx] = recv_counts[idx];
            }
        }
        else
        {
            bool is_even_rank = (param.process_idx % 2 == 0) ? true : false;
            size_t send_count = (is_even_rank) ? (param.elem_count / 2) : param.elem_count;
            for (size_t idx = 0; idx < param.process_count; idx++)
            {
                int is_even_peer = (idx % 2 == 0) ? true : false;
                send_counts[idx] = send_count;
                recv_counts[idx] = (is_even_peer) ? (param.elem_count / 2) : param.elem_count;
            }
        }
    }

    size_t get_recv_buf_size(typed_test_param<T>& param)
    {
        return param.elem_count * param.process_count;
    }

    void run_derived(typed_test_param<T>& param)
    {
        void* send_buf;
        void* recv_buf;
        const ccl_test_conf& test_conf = param.get_conf();
        ccl::coll_attr* attr = &param.coll_attr;
        ccl::stream_t& stream = param.get_stream();
        ccl::data_type data_type = static_cast<ccl::data_type>(test_conf.data_type);

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
            size_t new_idx = param.buf_indexes[buf_idx];
            param.prepare_coll_attr(param.buf_indexes[buf_idx]);

            send_buf = param.get_send_buf(new_idx);
            recv_buf = param.get_recv_buf(new_idx);

            param.reqs[buf_idx] =
                param.global_comm->alltoallv((test_conf.place_type == PT_IN) ? recv_buf : send_buf, 
                                              send_counts.data(), recv_buf, recv_counts.data(), data_type, attr, stream);    
        }
    }
};

RUN_METHOD_DEFINITION(alltoallv_test);
TEST_CASES_DEFINITION(alltoallv_test);
MAIN_FUNCTION();
