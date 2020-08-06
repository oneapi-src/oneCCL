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
#ifndef BASE_HPP
#define BASE_HPP

#include <chrono>
#include <functional>
#include <vector>

#include "ccl.hpp"
#include "ccl_test_conf.hpp"
#include "utils.hpp"

#define SEED_STEP 10

template <typename T>
struct typed_test_param {
    ccl_test_conf test_conf;
    size_t elem_count;
    size_t buffer_count;
    size_t process_count;
    size_t process_idx;
    static size_t priority;
    std::vector<size_t> buf_indexes;
    std::vector<std::vector<T>> send_buf;
    std::vector<std::vector<T>> recv_buf;

    // buffers for bfp16
    std::vector<std::vector<short>> send_buf_bfp16;
    std::vector<std::vector<short>> recv_buf_bfp16;

    std::vector<std::shared_ptr<ccl::request>> reqs;
    std::string match_id;
    ccl::communicator_t comm;
    ccl::communicator_t global_comm;
    ccl::coll_attr coll_attr{};
    ccl::stream_t stream;

    typed_test_param(ccl_test_conf tconf) : test_conf(tconf) {
        init_coll_attr(&coll_attr);
        elem_count = get_ccl_elem_count(test_conf);
        buffer_count = get_ccl_buffer_count(test_conf);
        comm = ccl::environment::instance().create_communicator();
        global_comm = ccl::environment::instance().create_communicator();
        process_count = comm->size();
        process_idx = comm->rank();
        buf_indexes.resize(buffer_count);
    }

    void prepare_coll_attr(size_t idx);
    std::string create_match_id(size_t buf_idx);
    bool complete_request(std::shared_ptr<ccl::request> reqs);
    void define_start_order();
    bool complete();
    void swap_buffers(size_t iter);
    size_t generate_priority_value(size_t buf_idx);

    const ccl_test_conf& get_conf() {
        return test_conf;
    }

    void print(std::ostream& output);

    ccl::stream_t& get_stream() {
        return stream;
    }

    void* get_send_buf(size_t buf_idx) {
        if (test_conf.data_type == DT_BFP16)
            return static_cast<void*>(send_buf_bfp16[buf_idx].data());
        else
            return static_cast<void*>(send_buf[buf_idx].data());
    }

    void* get_recv_buf(size_t buf_idx) {
        if (test_conf.data_type == DT_BFP16)
            return static_cast<void*>(recv_buf_bfp16[buf_idx].data());
        else
            return static_cast<void*>(recv_buf[buf_idx].data());
    }
};

template <typename T>
class base_test {
public:
    size_t global_process_idx;
    size_t global_process_count;
    ccl::communicator_t comm;
    char err_message[ERR_MESSAGE_MAX_LEN]{};

    char* get_err_message() {
        return err_message;
    }

    base_test();

    virtual void alloc_buffers(typed_test_param<T>& param);
    virtual void fill_buffers(typed_test_param<T>& param);

    int run(typed_test_param<T>& param);
    virtual void run_derived(typed_test_param<T>& param) = 0;

    virtual size_t get_recv_buf_size(typed_test_param<T>& param) = 0;

    virtual int check(typed_test_param<T>& param) = 0;
    virtual int check_error(typed_test_param<T>& param,
                            T expected,
                            size_t buf_idx,
                            size_t elem_idx);
};

class MainTest : public ::testing ::TestWithParam<ccl_test_conf> {
    template <typename T>
    int run(ccl_test_conf param);

public:
    int test(ccl_test_conf& param) {
        switch (param.data_type) {
            case DT_CHAR: return run<char>(param);
            case DT_INT: return run<int>(param);
            //TODO: add additional type to testing
            case DT_BFP16: return run<float>(param);
            case DT_FLOAT: return run<float>(param);
            case DT_DOUBLE: return run<double>(param);
            // case DT_INT64:
            // return TEST_SUCCESS;
            // case DT_UINT64:
            // return TEST_SUCCESS;
            default:
                EXPECT_TRUE(false) << "Unknown data type: " << param.data_type;
                return TEST_FAILURE;
        }
    }
};
#endif /* BASE_HPP */
