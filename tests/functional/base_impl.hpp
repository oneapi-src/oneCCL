
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

#include "base.hpp"

template <typename T>
void typed_test_param<T>::prepare_coll_attr(size_t idx)
{
    coll_attr.priority = generate_priority_value(idx);
    coll_attr.to_cache = test_conf.cache_type;
    char* test_unordered_coll = getenv("CCL_UNORDERED_COLL");
    if (test_unordered_coll && atoi(test_unordered_coll) == 1)
    {
        coll_attr.synchronous = 0;
    }
    else
    {
        coll_attr.synchronous = test_conf.sync_type;
    }
    match_id = create_match_id(idx);
    coll_attr.match_id = match_id.c_str();
}
template <typename T>
std::string typed_test_param<T>::create_match_id(size_t buf_idx)
{
    return (std::to_string(buf_idx) +
            std::to_string(process_count) +
            std::to_string(elem_count) +
            std::to_string(buffer_count) +
            std::to_string(test_conf.reduction_type) +
            std::to_string(test_conf.sync_type) +
            std::to_string(test_conf.cache_type) +
            std::to_string(test_conf.size_type) +
            std::to_string(test_conf.data_type) +
            std::to_string(test_conf.completion_type) +
            std::to_string(test_conf.place_type) +
            std::to_string(test_conf.start_order_type) +
            std::to_string(test_conf.complete_order_type) +
            std::to_string(test_conf.prolog_type) +
            std::to_string(test_conf.epilog_type));
}
template <typename T>
bool typed_test_param<T>::complete_request(std::shared_ptr < ccl::request > reqs)
{
    if (test_conf.completion_type == CMPT_TEST)
    {
        return reqs->test();
    }
    else if (test_conf.completion_type == CMPT_WAIT)
    {
        reqs->wait();
        return true;
    }
    else
    {
        ASSERT(0, "unexpected completion type %d", test_conf.completion_type);
        return false;
    }
}
template <typename T>
void typed_test_param<T>::define_start_order()
{
    if (test_conf.start_order_type == ORDER_DIRECT || test_conf.start_order_type == ORDER_DISABLE)
    {
        
        std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
    }
    else if (test_conf.start_order_type == ORDER_INDIRECT)
    {
        std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
        std::reverse(buf_indexes.begin(),buf_indexes.end());
    }
    else if (test_conf.start_order_type == ORDER_RANDOM)
    {
        char* test_unordered_coll = getenv("CCL_UNORDERED_COLL");
        if (test_unordered_coll && atoi(test_unordered_coll) == 1)
        {
            size_t buf_idx;
            srand(process_idx * SEED_STEP);
            for (buf_idx = 0; buf_idx < buffer_count; buf_idx++)
            {
                buf_indexes[buf_idx] = buf_idx;
            }
            for(int idx=buffer_count; idx > 1; idx--)
            {
                buf_idx = rand() % idx;
                int tmp_idx = buf_indexes[idx-1];
                buf_indexes[idx-1] = buf_indexes[buf_idx];
                buf_indexes[buf_idx] = tmp_idx;
            }
                    
        }
        else {
            std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
        }
    }
    else
    {
        std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
    }
}
template <typename T>
bool typed_test_param<T>::complete()
{
    size_t idx, msg_idx;
    size_t completions = 0;
    int msg_completions[buffer_count];
    memset(msg_completions, 0, buffer_count * sizeof(int));
    while (completions < buffer_count)
    {
        for (idx = 0; idx < buffer_count; idx++)
        {
            if (test_conf.complete_order_type == ORDER_DIRECT || test_conf.complete_order_type == ORDER_DISABLE)
            {
                msg_idx = idx;
            }
            else if (test_conf.complete_order_type == ORDER_INDIRECT)
            {
                msg_idx = (buffer_count - idx - 1);
            }
            else if (test_conf.complete_order_type == ORDER_RANDOM)
            {
                msg_idx = rand() % buffer_count;
            }
            else
            {
                msg_idx = idx;
            }
            if (msg_completions[msg_idx]) continue;
            if (complete_request(reqs[msg_idx]))
            {
                completions++;
                msg_completions[msg_idx] = 1;
            }
        }
    }
    return TEST_SUCCESS;
}
template <typename T>
void typed_test_param<T>::swap_buffers(size_t iter)
{
    char* test_dynamic_pointer = getenv("TEST_DYNAMIC_POINTER");
    if (test_dynamic_pointer && atoi(test_dynamic_pointer) == 1)
    {
        if (iter == 1)
        {
            if (process_idx % 2 )
            {
                std::vector<std::vector<T>> (send_buf.begin(), send_buf.end()).swap(send_buf);
            }
        }
    }
}
template <typename T>
size_t typed_test_param<T>::generate_priority_value(size_t buf_idx)
{
    return buf_idx++;
}
template <typename T>
void typed_test_param<T>::print(std::ostream &output)
{
    output << 
            "test conf:\n" << test_conf <<
            "\nprocess_count: " << process_count <<
            "\nprocess_idx: " << process_idx <<
            "\nelem_count: " << elem_count <<
            "\nbuffer_count: " << buffer_count <<
            "\nmatch_id: " << match_id <<
            "\n-------------\n" << std::endl;
}
template <typename T>
base_test<T>::base_test()
{
    comm = ccl::environment::instance().create_communicator();
    global_process_idx = comm->rank();
    global_process_count = comm->size();
    memset(err_message, '\0', ERR_MESSAGE_MAX_LEN);
}
template <typename T>
void base_test<T>::alloc_buffers(typed_test_param<T>& param)
{
    param.send_buf.resize(param.buffer_count);
    param.recv_buf.resize(param.buffer_count);
    param.reqs.resize(param.buffer_count);
    for (size_t elem_idx = 0; elem_idx < param.buffer_count; elem_idx++)
    {
        param.send_buf[elem_idx].resize(param.elem_count * param.process_count);
    }
}
template <typename T>
void base_test<T>::fill_buffers(typed_test_param<T>& param)
{
    for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
    {
        std::iota(param.send_buf[buf_idx].begin(), param.send_buf[buf_idx].end(), param.process_idx + buf_idx);
    }
    if (param.test_conf.place_type == PT_OOP)
    {
        for (size_t elem_idx = 0; elem_idx < param.buffer_count; elem_idx++)
        {
            // TODO: add parameter resize to SOME_VALUE
            param.recv_buf[elem_idx].resize(param.elem_count * param.process_count);
        }
    }
    else
    {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++)
        {
                param.recv_buf[buf_idx] = param.send_buf[buf_idx];
        }
    }
}
template <typename T>
int base_test<T>::run(typed_test_param<T>& param)
{
    size_t result = 0;
    SHOW_ALGO(Collective_Name);
    for (size_t iter = 0; iter < ITER_COUNT; iter++)
    {
        try
        {
            this->alloc_buffers(param);
            this->fill_buffers(param);
            param.swap_buffers(iter);
            param.define_start_order();
            this->run_derived(param);
            param.complete();
            result += check(param);
        }
        catch (const std::exception& ex)
        {
            result += TEST_FAILURE;
            printf("WARNING! %s iter number: %zu", ex.what(), iter);
        }
    }
    return result;
}