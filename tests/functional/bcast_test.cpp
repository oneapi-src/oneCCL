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
#define TEST_CCL_BCAST

#define COLL_NAME "CCL_BCAST"

#include "base_impl.hpp"

template <typename T>
class bcast_test : public base_test<T> {
public:
    int check(typed_test_param<T>& param) {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++) {
                T expected = static_cast<T>(elem_idx);
                if (base_test<T>::check_error(param, expected, buf_idx, elem_idx))
                    return TEST_FAILURE;
            }
        }
        return TEST_SUCCESS;
    }

    void fill_buffers(typed_test_param<T>& param) {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++) {
                if (param.process_idx == ROOT_PROCESS_IDX) {
                    param.recv_buf[buf_idx][elem_idx] = elem_idx;
                }
                else {
                    param.recv_buf[buf_idx][elem_idx] = static_cast<T>(SOME_VALUE);
                    if (param.test_conf.data_type == DT_BFP16) {
                        param.recv_buf_bfp16[buf_idx][elem_idx] = static_cast<short>(SOME_VALUE);
                    }
                }
            }
            param.send_buf[buf_idx] = param.recv_buf[buf_idx];
        }
    }

    size_t get_recv_buf_size(typed_test_param<T>& param) {
        return param.elem_count;
    }

    void run_derived(typed_test_param<T>& param) {
        void* recv_buf;
        size_t count = param.elem_count;
        const ccl_test_conf& test_conf = param.get_conf();
        ccl::coll_attr* attr = &param.coll_attr;
        ccl::stream_t& stream = param.get_stream();
        ccl::datatype data_type = static_cast<ccl::datatype>(test_conf.data_type);

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            size_t new_idx = param.buf_indexes[buf_idx];
            param.prepare_coll_attr(param.buf_indexes[buf_idx]);

            recv_buf = param.get_recv_buf(new_idx);

            param.reqs[buf_idx] = param.global_comm->bcast(
                recv_buf, count, data_type, ROOT_PROCESS_IDX, attr, stream);
        }
    }
};

RUN_METHOD_DEFINITION(bcast_test);
TEST_CASES_DEFINITION(bcast_test);
MAIN_FUNCTION();
