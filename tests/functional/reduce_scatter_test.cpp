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
#define TEST_CCL_REDUCE

#define COLL_NAME "CCL_REDUCE_SCATTER"

#include "base_impl.hpp"

template <typename T>
class reduce_scatter_test : public base_test<T> {
public:
    int check(typed_test_param<T>& param) {

        size_t my_rank = GlobalData::instance().comms[0].rank();

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++) {

                size_t real_elem_idx = my_rank * param.elem_count + elem_idx;

                if (param.test_conf.reduction == RT_SUM) {
                    T expected = ((param.process_count * (param.process_count - 1) / 2) +
                                  ((real_elem_idx + buf_idx) * param.process_count));
                    if (base_test<T>::check_error(param, expected, buf_idx, elem_idx))
                        return TEST_FAILURE;
                }

                if (param.test_conf.reduction == RT_MAX) {
                    T expected = get_expected_max<T>(real_elem_idx, buf_idx, param.process_count);
                    if (base_test<T>::check_error(param, expected, buf_idx, elem_idx))
                        return TEST_FAILURE;
                }

                if (param.test_conf.reduction == RT_MIN) {
                    T expected = get_expected_min<T>(real_elem_idx, buf_idx, param.process_count);
                    if (base_test<T>::check_error(param, expected, buf_idx, elem_idx))
                        return TEST_FAILURE;
                }

                if (param.test_conf.reduction == RT_PROD) {
                    T expected = 1;
                    for (size_t proc_idx = 0; proc_idx < param.process_count; proc_idx++) {
                        expected *= real_elem_idx + buf_idx + proc_idx;
                    }
                    if (base_test<T>::check_error(param, expected, buf_idx, elem_idx))
                        return TEST_FAILURE;
                }
            }
        }
        return TEST_SUCCESS;
    }

    size_t get_recv_buf_size(typed_test_param<T>& param) {
        return param.elem_count;
    }

    void run_derived(typed_test_param<T>& param) {
        void* send_buf;
        void* recv_buf;
        size_t count = param.elem_count;

        const ccl_test_conf& test_conf = param.get_conf();

        auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();

        ccl::reduction reduction = get_ccl_lib_reduction(test_conf);
        ccl::datatype datatype = get_ccl_lib_datatype(test_conf);

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            size_t new_idx = param.buf_indexes[buf_idx];
            param.prepare_coll_attr(attr, param.buf_indexes[buf_idx]);

            send_buf = param.get_send_buf(new_idx);
            recv_buf = param.get_recv_buf(new_idx);

            param.reqs[buf_idx] = ccl::reduce_scatter(
                (test_conf.place_type == PT_IN) ? recv_buf : send_buf,
                recv_buf,
                count,
                datatype,
                reduction,
                GlobalData::instance().comms[0],
                attr);
        }
    }
};

RUN_METHOD_DEFINITION(reduce_scatter_test);
TEST_CASES_DEFINITION(reduce_scatter_test);
MAIN_FUNCTION();
