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

#include <math.h>

#include "base.hpp"
#include "lp.hpp"

#define FIRST_FP_COEFF  (0.1)
#define SECOND_FP_COEFF (0.01)

template <typename T>
template <class coll_attr_type>
void test_operation<T>::prepare_attr(coll_attr_type& attr, size_t idx) {
    attr.template set<ccl::operation_attr_id::priority>(generate_priority_value(idx));
    attr.template set<ccl::operation_attr_id::to_cache>(param.cache_type == CACHE_TRUE ? true
                                                                                       : false);

    char* unordered_coll_env = getenv("CCL_UNORDERED_COLL");
    if (unordered_coll_env && atoi(unordered_coll_env) == 1) {
        attr.template set<ccl::operation_attr_id::synchronous>(false);
    }
    else {
        attr.template set<ccl::operation_attr_id::synchronous>(
            param.sync_type == SYNC_TRUE ? true : false);
    }

    match_id = create_match_id(idx);
    attr.template set<ccl::operation_attr_id::match_id>(match_id);
}

template <typename T>
std::string test_operation<T>::create_match_id(size_t buf_idx) {
    return (std::to_string(param.datatype) + std::to_string(param.size_type) +
            std::to_string(param.buf_count_type) + std::to_string(param.place_type) +
            std::to_string(param.start_order) + std::to_string(param.complete_order) +
            std::to_string(param.complete_type) + std::to_string(param.cache_type) +
            std::to_string(param.sync_type) + std::to_string(param.reduction) +
            std::to_string(buf_idx) + std::to_string(comm_size));
}

template <typename T>
void test_operation<T>::define_start_order(std::default_random_engine& rand_engine) {
    std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
    if (param.start_order == ORDER_INDIRECT) {
        std::reverse(buf_indexes.begin(), buf_indexes.end());
    }
    else if (param.start_order == ORDER_RANDOM) {
        char* unordered_coll_env = getenv("CCL_UNORDERED_COLL");
        if (unordered_coll_env && atoi(unordered_coll_env) == 1) {
            std::shuffle(buf_indexes.begin(), buf_indexes.end(), rand_engine);
        }
    }
}

template <typename T>
bool test_operation<T>::complete_events() {
    size_t idx, msg_idx;
    size_t completions = 0;
    std::vector<bool> msg_completions(buffer_count, false);

    while (completions < buffer_count) {
        for (idx = 0; idx < buffer_count; idx++) {
            if (param.complete_order == ORDER_DIRECT) {
                msg_idx = idx;
            }
            else if (param.complete_order == ORDER_INDIRECT) {
                msg_idx = (buffer_count - idx - 1);
            }
            else if (param.complete_order == ORDER_RANDOM) {
                msg_idx = rand() % buffer_count;
            }
            else {
                msg_idx = idx;
            }

            if (msg_completions[msg_idx])
                continue;

            if (complete_event(events[msg_idx])) {
                completions++;
                msg_completions[msg_idx] = true;
            }
        }
    }

    events.clear();

    return TEST_SUCCESS;
}

template <typename T>
bool test_operation<T>::complete_event(ccl::event& e) {
    if (param.complete_type == COMPLETE_TEST) {
        return e.test();
    }
    else if (param.complete_type == COMPLETE_WAIT) {
        e.wait();
        return true;
    }
    else {
        ASSERT(0, "unexpected completion type %d", param.complete_type);
        return false;
    }
}

template <typename T>
void test_operation<T>::change_buffer_pointers() {
    char* dynamic_pointer_env = getenv("CCL_TEST_DYNAMIC_POINTER");
    if (dynamic_pointer_env && atoi(dynamic_pointer_env) == 1) {
        /*
            create deep copy of vector with buffers and swap it with original one
            as result buffers in updated vector will have original content
            but in new memory locations
        */
        if (comm_rank % 2) {
            std::vector<std::vector<T>>(send_bufs.begin(), send_bufs.end()).swap(send_bufs);
        }
        else {
            std::vector<std::vector<T>>(recv_bufs.begin(), recv_bufs.end()).swap(recv_bufs);
        }
    }
}

template <typename T>
size_t test_operation<T>::generate_priority_value(size_t buf_idx) {
    return buf_idx++;
}

template <typename T>
void test_operation<T>::print(std::ostream& output) {
    output << "test op:\n"
           << param << "\ncomm_size: " << comm_size << "\ncomm_rank: " << comm_rank
           << "\nelem_count: " << elem_count << "\nbuffer_count: " << buffer_count
           << "\nmatch_id: " << match_id << "\n-------------\n"
           << std::endl;
}

template <typename T>
base_test<T>::base_test() {
    global_comm_rank = global_data::instance().comms[0].rank();
    global_comm_size = global_data::instance().comms[0].size();
    memset(err_message, '\0', ERR_MESSAGE_MAX_LEN);
    rand_engine = std::default_random_engine{ rand_device() };
}

template <typename T>
int base_test<T>::check_error(test_operation<T>& op, T expected, size_t buf_idx, size_t elem_idx) {
    double max_error = 0;
    double precision = 0;

    if (op.param.datatype == DATATYPE_FLOAT16) {
        precision = 2 * FP16_PRECISION;
    }
    else if (op.param.datatype == DATATYPE_FLOAT32) {
        precision = FP32_PRECISION;
    }
    else if (op.param.datatype == DATATYPE_FLOAT64) {
        precision = FP64_PRECISION;
    }
    else if (op.param.datatype == DATATYPE_BFLOAT16) {
        precision = 2 * BF16_PRECISION;
    }

    if (precision) {
        if (op.comm_size == 1) {
            max_error = precision;
        }
        else {
            /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */
            double log_base2 = log(op.comm_size) / log(2);
            double g = (log_base2 * precision) / (1 - (log_base2 * precision));
            max_error = g * expected;
        }
    }

    if (fabs(max_error) < fabs((double)expected - (double)op.recv_bufs[buf_idx][elem_idx])) {
        printf("[%d] got op.recvBuf[%zu][%zu] = %0.7f, but expected = %0.7f, "
               "max_error = %0.10f, precision = %0.7f\n",
               op.comm_rank,
               buf_idx,
               elem_idx,
               (double)op.recv_bufs[buf_idx][elem_idx],
               (double)expected,
               (double)max_error,
               precision);
        return TEST_FAILURE;
    }

    return TEST_SUCCESS;
}

template <typename T>
void base_test<T>::alloc_buffers_base(test_operation<T>& op) {
    op.send_bufs.resize(op.buffer_count);
    op.recv_bufs.resize(op.buffer_count);
    if (is_lp_datatype(op.param.datatype)) {
        op.send_bufs_lp.resize(op.buffer_count);
        op.recv_bufs_lp.resize(op.buffer_count);
    }

    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        op.send_bufs[buf_idx].resize(op.elem_count * op.comm_size);
        op.recv_bufs[buf_idx].resize(op.elem_count * op.comm_size);

        if (is_lp_datatype(op.param.datatype)) {
            op.send_bufs_lp[buf_idx].resize(op.elem_count * op.comm_size);
            op.recv_bufs_lp[buf_idx].resize(op.elem_count * op.comm_size);
        }
    }
}

template <typename T>
void base_test<T>::alloc_buffers(test_operation<T>& op) {}

template <typename T>
void base_test<T>::fill_send_buffers_base(test_operation<T>& op) {
    if (!is_lp_datatype(op.param.datatype))
        return;

    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        std::fill(op.send_bufs_lp[buf_idx].begin(), op.send_bufs_lp[buf_idx].end(), (T)SOME_VALUE);
    }
}

template <typename T>
void base_test<T>::fill_send_buffers(test_operation<T>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        for (size_t elem_idx = 0; elem_idx < op.send_bufs[buf_idx].size(); elem_idx++) {
            op.send_bufs[buf_idx][elem_idx] = op.comm_rank + buf_idx;

            if (op.param.reduction == REDUCTION_PROD) {
                op.send_bufs[buf_idx][elem_idx] += 1;
            }
        }
    }
}

template <typename T>
void base_test<T>::fill_recv_buffers_base(test_operation<T>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        if (op.param.place_type == PLACE_IN) {
            std::copy(op.send_bufs[buf_idx].begin(),
                      op.send_bufs[buf_idx].end(),
                      op.recv_bufs[buf_idx].begin());
        }
        else {
            std::fill(op.recv_bufs[buf_idx].begin(), op.recv_bufs[buf_idx].end(), (T)SOME_VALUE);
        }
        if (is_lp_datatype(op.param.datatype)) {
            std::fill(op.recv_bufs_lp[buf_idx].begin(), op.recv_bufs_lp[buf_idx].end(), SOME_VALUE);
        }
    }
}

template <typename T>
void base_test<T>::fill_recv_buffers(test_operation<T>& op) {}

template <typename T>
T base_test<T>::calculate_reduce_value(test_operation<T>& op, size_t buf_idx, size_t elem_idx) {
    T expected = 0;
    switch (op.param.reduction) {
        case REDUCTION_SUM:
            expected = (op.comm_size * (op.comm_size - 1)) / 2 + op.comm_size * buf_idx;
            break;
        case REDUCTION_PROD:
            expected = 1;
            for (int rank = 0; rank < op.comm_size; rank++) {
                expected *= rank + buf_idx + 1;
            }
            break;
        case REDUCTION_MIN: expected = (T)(buf_idx); break;
        case REDUCTION_MAX: expected = (T)(op.comm_size - 1 + buf_idx); break;
        default: ASSERT(0, "unexpected reduction %d", op.param.reduction); break;
    }
    return expected;
}

template <>
void base_test<float>::fill_send_buffers(test_operation<float>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        for (size_t elem_idx = 0; elem_idx < op.send_bufs[buf_idx].size(); elem_idx++) {
            op.send_bufs[buf_idx][elem_idx] =
                FIRST_FP_COEFF * op.comm_rank + SECOND_FP_COEFF * buf_idx;

            if (op.param.reduction == REDUCTION_PROD) {
                op.send_bufs[buf_idx][elem_idx] += 1;
            }
        }
    }
}

template <>
float base_test<float>::calculate_reduce_value(test_operation<float>& op,
                                               size_t buf_idx,
                                               size_t elem_idx) {
    float expected = 0;
    switch (op.param.reduction) {
        case REDUCTION_SUM:
            expected = op.comm_size *
                       (FIRST_FP_COEFF * (op.comm_size - 1) / 2 + SECOND_FP_COEFF * buf_idx);
            break;
        case REDUCTION_PROD:
            expected = 1;
            for (int rank = 0; rank < op.comm_size; rank++) {
                expected *= FIRST_FP_COEFF * rank + SECOND_FP_COEFF * buf_idx + 1;
            }
            break;
        case REDUCTION_MIN: expected = SECOND_FP_COEFF * buf_idx; break;
        case REDUCTION_MAX:
            expected = FIRST_FP_COEFF * (op.comm_size - 1) + SECOND_FP_COEFF * buf_idx;
            break;
        default: ASSERT(0, "unexpected reduction %d", op.param.reduction); break;
    }
    return expected;
}

template <typename T>
int base_test<T>::run(test_operation<T>& op) {
    size_t result = 0;

    char* algo = getenv(ALGO_SELECTION_ENV);
    if (algo)
        std::cout << ALGO_SELECTION_ENV << " = " << algo << "\n";
    std::cout << op.param << "\n";

    for (size_t iter = 0; iter < ITER_COUNT; iter++) {
        try {
            alloc_buffers_base(op);
            alloc_buffers(op);

            fill_send_buffers_base(op);
            fill_send_buffers(op);

            fill_recv_buffers_base(op);
            fill_recv_buffers(op);

            if (iter > 0) {
                op.change_buffer_pointers();
            }

            op.define_start_order(rand_engine);

            if (is_lp_datatype(op.param.datatype)) {
                make_lp_prologue(op, op.comm_size * op.elem_count);
            }

            run_derived(op);

            op.complete_events();

            if (is_lp_datatype(op.param.datatype)) {
                make_lp_epilogue(op, op.comm_size * op.elem_count);
            }

            result += check(op);
        }
        catch (const std::exception& ex) {
            result += TEST_FAILURE;
            printf("WARNING! %s iter number: %zu", ex.what(), iter);
        }
    }

    return result;
}
