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
#include <math.h>

#include "base.hpp"
#include "base_bfp16.hpp"

template <typename T>
void typed_test_param<T>::prepare_coll_attr(size_t idx) {
    coll_attr.priority = generate_priority_value(idx);
    coll_attr.to_cache = test_conf.cache_type;
    coll_attr.vector_buf = 0;

    char* test_unordered_coll = getenv("CCL_UNORDERED_COLL");
    if (test_unordered_coll && atoi(test_unordered_coll) == 1) {
        coll_attr.synchronous = 0;
    }
    else {
        coll_attr.synchronous = test_conf.sync_type;
    }

    match_id = create_match_id(idx);
    coll_attr.match_id = match_id.c_str();
}

template <typename T>
std::string typed_test_param<T>::create_match_id(size_t buf_idx) {
    return (std::to_string(buf_idx) + std::to_string(process_count) + std::to_string(elem_count) +
            std::to_string(buffer_count) + std::to_string(test_conf.reduction_type) +
            std::to_string(test_conf.sync_type) + std::to_string(test_conf.cache_type) +
            std::to_string(test_conf.size_type) + std::to_string(test_conf.data_type) +
            std::to_string(test_conf.completion_type) + std::to_string(test_conf.place_type) +
            std::to_string(test_conf.start_order_type) +
            std::to_string(test_conf.complete_order_type) + std::to_string(test_conf.prolog_type) +
            std::to_string(test_conf.epilog_type));
}

template <typename T>
bool typed_test_param<T>::complete_request(std::shared_ptr<ccl::request> reqs) {
    if (test_conf.completion_type == CMPT_TEST) {
        return reqs->test();
    }
    else if (test_conf.completion_type == CMPT_WAIT) {
        reqs->wait();
        return true;
    }
    else {
        ASSERT(0, "unexpected completion type %d", test_conf.completion_type);
        return false;
    }
}

template <typename T>
void typed_test_param<T>::define_start_order() {
    if (test_conf.start_order_type == ORDER_DIRECT || test_conf.start_order_type == ORDER_DISABLE) {
        std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
    }
    else if (test_conf.start_order_type == ORDER_INDIRECT) {
        std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
        std::reverse(buf_indexes.begin(), buf_indexes.end());
    }
    else if (test_conf.start_order_type == ORDER_RANDOM) {
        char* test_unordered_coll = getenv("CCL_UNORDERED_COLL");
        if (test_unordered_coll && atoi(test_unordered_coll) == 1) {
            size_t buf_idx;
            srand(process_idx * SEED_STEP);
            for (buf_idx = 0; buf_idx < buffer_count; buf_idx++) {
                buf_indexes[buf_idx] = buf_idx;
            }
            for (int idx = buffer_count; idx > 1; idx--) {
                buf_idx = rand() % idx;
                int tmp_idx = buf_indexes[idx - 1];
                buf_indexes[idx - 1] = buf_indexes[buf_idx];
                buf_indexes[buf_idx] = tmp_idx;
            }
        }
        else {
            std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
        }
    }
    else {
        std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
    }
}

template <typename T>
bool typed_test_param<T>::complete() {
    size_t idx, msg_idx;
    size_t completions = 0;
    int msg_completions[buffer_count];
    memset(msg_completions, 0, buffer_count * sizeof(int));

    while (completions < buffer_count) {
        for (idx = 0; idx < buffer_count; idx++) {
            if (test_conf.complete_order_type == ORDER_DIRECT ||
                test_conf.complete_order_type == ORDER_DISABLE) {
                msg_idx = idx;
            }
            else if (test_conf.complete_order_type == ORDER_INDIRECT) {
                msg_idx = (buffer_count - idx - 1);
            }
            else if (test_conf.complete_order_type == ORDER_RANDOM) {
                msg_idx = rand() % buffer_count;
            }
            else {
                msg_idx = idx;
            }

            if (msg_completions[msg_idx])
                continue;

            if (complete_request(reqs[msg_idx])) {
                completions++;
                msg_completions[msg_idx] = 1;
            }
        }
    }
    return TEST_SUCCESS;
}

template <typename T>
void typed_test_param<T>::swap_buffers(size_t iter) {
    char* test_dynamic_pointer = getenv("TEST_DYNAMIC_POINTER");
    if (test_dynamic_pointer && atoi(test_dynamic_pointer) == 1) {
        if (iter == 1) {
            if (process_idx % 2) {
                std::vector<std::vector<T>>(send_buf.begin(), send_buf.end()).swap(send_buf);
            }
        }
    }
}

template <typename T>
size_t typed_test_param<T>::generate_priority_value(size_t buf_idx) {
    return buf_idx++;
}

template <typename T>
void typed_test_param<T>::print(std::ostream& output) {
    output << "test conf:\n"
           << test_conf << "\nprocess_count: " << process_count << "\nprocess_idx: " << process_idx
           << "\nelem_count: " << elem_count << "\nbuffer_count: " << buffer_count
           << "\nmatch_id: " << match_id << "\n-------------\n"
           << std::endl;
}

template <typename T>
base_test<T>::base_test() {
    comm = ccl::environment::instance().create_communicator();
    global_process_idx = comm->rank();
    global_process_count = comm->size();
    memset(err_message, '\0', ERR_MESSAGE_MAX_LEN);
}

template <typename T>
int base_test<T>::check_error(typed_test_param<T>& param,
                              T expected,
                              size_t buf_idx,
                              size_t elem_idx) {
    double max_error = 0;

    if (param.test_conf.data_type == DT_BFP16) {
        /* TODO: handle float and double */

        // sources https://www.mcs.anl.gov/papers/P4093-0713_1.pdf

#ifdef CCL_BFP16_COMPILER
        double log_base2 = log(param.process_count) / log(2);
        double precision = BFP16_PRECISION;
        double g = (log_base2 * precision) / (1 - (log_base2 * precision));
        max_error = g * expected;
#else
        ASSERT(0, "unexpected data_type %d", param.test_conf.data_type);
#endif
    }

    if (fabs(max_error) < fabs((double)expected - (double)param.recv_buf[buf_idx][elem_idx])) {
        printf(
            "[%zu] got param.recvBuf[%zu][%zu] = %0.7f, but expected = %0.7f, max_error = %0.16f\n",
            param.process_idx,
            buf_idx,
            elem_idx,
            (double)param.recv_buf[buf_idx][elem_idx],
            (double)expected,
            (double)max_error);
        return TEST_FAILURE;
    }

    return TEST_SUCCESS;
}

template <typename T>
void base_test<T>::alloc_buffers(typed_test_param<T>& param) {
    param.send_buf.resize(param.buffer_count);
    param.recv_buf.resize(param.buffer_count);
    param.reqs.resize(param.buffer_count);

    for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
        param.send_buf[buf_idx].resize(param.elem_count * param.process_count);
        param.recv_buf[buf_idx].resize(param.elem_count * param.process_count);
    }

    if (param.test_conf.data_type == DT_BFP16) {
        param.send_buf_bfp16.resize(param.buffer_count);
        param.recv_buf_bfp16.resize(param.buffer_count);

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            param.send_buf_bfp16[buf_idx].resize(param.elem_count * param.process_count);
            param.recv_buf_bfp16[buf_idx].resize(param.elem_count * param.process_count);
        }
    }
}

template <typename T>
void base_test<T>::fill_buffers(typed_test_param<T>& param) {
    for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
        std::iota(param.send_buf[buf_idx].begin(),
                  param.send_buf[buf_idx].end(),
                  param.process_idx + buf_idx);
    }

    if (param.test_conf.place_type == PT_IN) {
        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            param.recv_buf[buf_idx] = param.send_buf[buf_idx];
        }
    }
}

template <typename T>
int base_test<T>::run(typed_test_param<T>& param) {
    size_t result = 0;

    SHOW_ALGO(COLL_NAME);

    for (size_t iter = 0; iter < ITER_COUNT; iter++) {
        try {
            alloc_buffers(param);
            fill_buffers(param);
            param.swap_buffers(iter);
            param.define_start_order();

            if (param.test_conf.data_type == DT_BFP16) {
#ifdef CCL_BFP16_COMPILER
                make_bfp16_prologue<T>(param, get_recv_buf_size(param));
#else
                ASSERT(0, "unexpected data_type %d", param.test_conf.data_type);
#endif
            }

            run_derived(param);
            param.complete();

            if (param.test_conf.data_type == DT_BFP16) {
#ifdef CCL_BFP16_COMPILER
                make_bfp16_epilogue<T>(param, get_recv_buf_size(param));
#else
                ASSERT(0, "unexpected data_type %d", param.test_conf.data_type);
#endif
            }

            result += check(param);
        }
        catch (const std::exception& ex) {
            result += TEST_FAILURE;
            printf("WARNING! %s iter number: %zu", ex.what(), iter);
        }
    }

    return result;
}
