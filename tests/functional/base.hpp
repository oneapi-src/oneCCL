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

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "oneapi/ccl.hpp"
#include "conf.hpp"

class global_data {
public:
    std::vector<ccl::communicator> comms;
    ccl::shared_ptr_class<ccl::kvs> kvs;

    global_data(global_data& gd) = delete;
    void operator=(const global_data&) = delete;
    static global_data& instance() {
        static global_data gd;
        return gd;
    }

protected:
    global_data(){};
    ~global_data(){};
};

#include "utils.hpp"

bool is_lp_datatype(ccl_data_type dtype);

template <typename T>
struct test_operation {
    test_param param;

    size_t elem_count;
    size_t buffer_count;
    ccl::datatype datatype;
    ccl::reduction reduction;

    int comm_size;
    int comm_rank;

    std::vector<size_t> buf_indexes;

    std::vector<std::vector<T>> send_bufs;
    std::vector<std::vector<T>> recv_bufs;

    // buffers for 16-bits low precision datatype
    std::vector<std::vector<short>> send_bufs_lp;
    std::vector<std::vector<short>> recv_bufs_lp;

    std::vector<ccl::event> events;
    ccl::string_class match_id;

    test_operation(test_param param)
            : param(param),
              elem_count(get_elem_count(param)),
              buffer_count(get_buffer_count(param)),
              datatype(get_ccl_datatype(param)),
              reduction(get_ccl_reduction(param)) {
        comm_size = global_data::instance().comms[0].size();
        comm_rank = global_data::instance().comms[0].rank();
        buf_indexes.resize(buffer_count);
    }

    template <class coll_attr_type>
    void prepare_attr(coll_attr_type& coll_attr, size_t idx);

    std::string create_match_id(size_t buf_idx);
    void change_buffer_pointers();
    size_t generate_priority_value(size_t buf_idx);

    void define_start_order(std::default_random_engine& rand_engine);

    bool complete_events();
    bool complete_event(ccl::event& e);

    const test_param& get_param() {
        return param;
    }

    void print(std::ostream& output);

    void* get_send_buf(size_t buf_idx) {
        if (is_lp_datatype(param.datatype))
            return static_cast<void*>(send_bufs_lp[buf_idx].data());
        else
            return static_cast<void*>(send_bufs[buf_idx].data());
    }

    void* get_recv_buf(size_t buf_idx) {
        if (is_lp_datatype(param.datatype))
            return static_cast<void*>(recv_bufs_lp[buf_idx].data());
        else
            return static_cast<void*>(recv_bufs[buf_idx].data());
    }

    size_t get_check_step(size_t elem_idx) {
        size_t step;
        if (param.size_type == SIZE_SMALL)
            step = 1;
        else if (param.size_type == SIZE_MEDIUM)
            step = 4;
        else
            step = 32;

        if ((step > 1) && (elem_idx + step >= elem_count)) {
            /* 
                to process tail elements
                when elem_count is not dividable by step
            */
            step = 1;
        }

        return step;
    }
};

template <typename T>
class base_test {
public:
    int global_comm_rank;
    int global_comm_size;
    char err_message[ERR_MESSAGE_MAX_LEN]{};

    std::random_device rand_device;
    std::default_random_engine rand_engine;

    char* get_err_message() {
        return err_message;
    }

    base_test();

    void alloc_buffers_base(test_operation<T>& op);
    virtual void alloc_buffers(test_operation<T>& op);

    void fill_send_buffers_base(test_operation<T>& op);
    virtual void fill_send_buffers(test_operation<T>& op);

    void fill_recv_buffers_base(test_operation<T>& op);
    virtual void fill_recv_buffers(test_operation<T>& op);

    virtual T calculate_reduce_value(test_operation<T>& op, size_t buf_idx, size_t elem_idx);

    int run(test_operation<T>& op);
    virtual void run_derived(test_operation<T>& op) = 0;

    virtual int check(test_operation<T>& op) = 0;
    virtual int check_error(test_operation<T>& op, T expected, size_t buf_idx, size_t elem_idx);
};

class MainTest : public ::testing ::TestWithParam<test_param> {
    template <typename T>
    int run(test_param param);

public:
    int test(test_param& param) {
        switch (param.datatype) {
            case DATATYPE_INT8: return run<int8_t>(param);
            case DATATYPE_UINT8: return run<uint8_t>(param);
            case DATATYPE_INT16: return run<int16_t>(param);
            case DATATYPE_UINT16: return run<uint16_t>(param);
            case DATATYPE_INT32: return run<int32_t>(param);
            case DATATYPE_UINT32: return run<uint32_t>(param);
            case DATATYPE_INT64: return run<int64_t>(param);
            case DATATYPE_UINT64: return run<uint64_t>(param);
            case DATATYPE_FLOAT16: return run<float>(param);
            case DATATYPE_FLOAT32: return run<float>(param);
            case DATATYPE_FLOAT64: return run<double>(param);
            case DATATYPE_BFLOAT16: return run<float>(param);
            default:
                EXPECT_TRUE(false) << "Unexpected data type: " << param.datatype;
                return TEST_FAILURE;
        }
    }
};
