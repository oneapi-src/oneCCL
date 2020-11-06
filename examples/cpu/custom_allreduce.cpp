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
#include <string>

#include "base.hpp"

#define CUSTOM_BASE_DTYPE        int
#define CUSTOM_REPEAT_COUNT      8
#define CUSTOM_BASE_DTYPE_FORMAT "%d"
#define CUSTOM_DTYPE_SIZE        (CUSTOM_REPEAT_COUNT * sizeof(CUSTOM_BASE_DTYPE))

int size, rank;
ccl::datatype custom_dtype;
std::string global_match_id;

typedef void (*expected_fn_t)(void*, size_t);
typedef void (*fill_fn_t)(void*, size_t, size_t);
typedef int (*check_fn_t)(void*, size_t, expected_fn_t);

#define RUN_COLLECTIVE(start_cmd, fill_fn, check_fn, expected_fn, name) \
    t = 0; \
    do { \
        for (int iter_idx = 0; iter_idx < ITERS; iter_idx++) { \
            global_match_id = match_id; \
            fill_fn(send_buf, MSG_SIZE_COUNT, rank + 1); \
            fill_fn(recv_buf, MSG_SIZE_COUNT, 1); \
            t1 = when(); \
            auto req = start_cmd; \
            req.wait(); \
            t2 = when(); \
            t += (t2 - t1); \
        } \
        check_values(recv_buf, MSG_SIZE_COUNT, check_fn, expected_fn); \
        printf("[%d] avg %s time: %8.2lf us\n", rank, name, t / ITERS); \
        fflush(stdout); \
    } while (0)

/* primitive operations for custom datatype */
void custom_2x(void* in_elem, void* out_elem) {
    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        ((CUSTOM_BASE_DTYPE*)out_elem)[idx] = 2 * ((CUSTOM_BASE_DTYPE*)in_elem)[idx];
    }
}

void custom_sum(void* in_elem, void* inout_elem) {
    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        ((CUSTOM_BASE_DTYPE*)inout_elem)[idx] += ((CUSTOM_BASE_DTYPE*)in_elem)[idx];
    }
}

void custom_to_char(void* in_elem, char* out_elem) {
    *out_elem = ((CUSTOM_BASE_DTYPE*)in_elem)[0];
}

void custom_from_char(char* in_elem, void* out_elem) {
    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        ((CUSTOM_BASE_DTYPE*)out_elem)[idx] = (CUSTOM_BASE_DTYPE)(*in_elem);
    }
}

void custom_set(void* elem, size_t base_value) {
    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        ((CUSTOM_BASE_DTYPE*)elem)[idx] = (CUSTOM_BASE_DTYPE)(base_value);
    }
}

void custom_zeroize(void* elem) {
    memset(elem, 0, CUSTOM_DTYPE_SIZE);
}

int custom_compare(void* elem1, void* elem2) {
    return memcmp(elem1, elem2, CUSTOM_DTYPE_SIZE);
}

void custom_print(void* elem) {
    char print_buf[1024] = { 0 };
    size_t print_bytes = 0;

    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        print_bytes += sprintf(
            print_buf + print_bytes, CUSTOM_BASE_DTYPE_FORMAT " ", ((CUSTOM_BASE_DTYPE*)elem)[idx]);
    }
    printf("%s\n", print_buf);
}

void fill_value_float(void* buf, size_t count, size_t base_value) {
    for (size_t idx = 0; idx < count; idx++) {
        ((float*)buf)[idx] = (float)base_value;
    }
}

void fill_value_custom(void* buf, size_t count, size_t base_value) {
    for (size_t idx = 0; idx < count; idx++) {
        custom_set(((char*)buf) + idx * CUSTOM_DTYPE_SIZE, base_value);
    }
}

void check_values(void* buf, size_t count, check_fn_t check_fn, expected_fn_t expected_fn) {
    for (size_t idx = 0; idx < MSG_SIZE_COUNT; idx++) {
        if (check_fn(buf, idx, expected_fn)) {
            ASSERT(0, "unexpected value on idx %zu", idx);
        }
    }
}

int check_value_float(void* buf, size_t idx, expected_fn_t expected_fn) {
    int res = 0;

    float expected = 0;
    expected_fn(&expected, idx);

    float got = ((float*)buf)[idx];

    if (got != expected) {
        printf("expected %f, got %f\n", expected, got);
        res = 1;
    }

    return res;
}

int check_value_custom(void* buf, size_t idx, expected_fn_t expected_fn) {
    int res = 0;

    char expected[CUSTOM_DTYPE_SIZE] = { 0 };
    expected_fn(expected, idx);

    char* got = (char*)buf + CUSTOM_DTYPE_SIZE * idx;

    if (custom_compare(expected, got)) {
        printf("expected ");
        custom_print(expected);
        printf(", got ");
        custom_print(got);
        printf("\n");
        res = 1;
    }

    return res;
}

void expected_float_1(void* elem, size_t idx) {
    *((float*)elem) = (size + 1) * ((float)size / 2);
}

void expected_float_2(void* elem, size_t idx) {
    *((float*)elem) = (size + 1) * size;
}

void expected_float_3(void* elem, size_t idx) {
    *((float*)elem) = 0;
}

void expected_float_4(void* elem, size_t idx) {
    *((float*)elem) = 2 * (size + 1) * size;
}

void expected_float_5(void* elem, size_t idx) {
    *((float*)elem) = (float)idx;
}

void expected_float_6(void* elem, size_t idx) {
    *((float*)elem) = (float)(2 * idx);
}

void expected_custom_1(void* elem, size_t idx) {
    custom_set(elem, (size + 1) * ((float)size / 2));
}

void expected_custom_2(void* elem, size_t idx) {
    custom_set(elem, (size + 1) * size);
}

void expected_custom_3(void* elem, size_t idx) {
    custom_zeroize(elem);
}

void expected_custom_4(void* elem, size_t idx) {
    custom_set(elem, 2 * (size + 1) * size);
}

void expected_custom_5(void* elem, size_t idx) {
    custom_set(elem, idx);
}

void expected_custom_6(void* elem, size_t idx) {
    custom_set(elem, 2 * idx);
}

void do_prologue_2x(const void* in_buf,
                    size_t in_count,
                    ccl::datatype in_dtype,
                    void** out_buf,
                    size_t* out_count,
                    ccl::datatype* out_dtype,
                    const ccl::fn_context* context) {
    ASSERT((in_dtype == ccl::datatype::float32) || (in_dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(in_dtype));
    ASSERT(out_buf, "null ptr");
    ASSERT(context->offset == 0, "wrong offset for prologue func, should be 0");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_buf)
        *out_buf = (void*)in_buf;
    if (out_count)
        *out_count = in_count;
    if (out_dtype)
        *out_dtype = in_dtype;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (in_dtype == ccl::datatype::float32) {
            ((float*)(*out_buf))[idx] = ((float*)in_buf)[idx] * 2;
        }
        else if (in_dtype == custom_dtype) {
            custom_2x((char*)in_buf + idx * CUSTOM_DTYPE_SIZE,
                      (char*)(*out_buf) + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(in_dtype));
        }
    }
}

void do_epilogue_2x(const void* in_buf,
                    size_t in_count,
                    ccl::datatype in_dtype,
                    void* out_buf,
                    size_t* out_count,
                    ccl::datatype* out_dtype,
                    const ccl::fn_context* context) {
    ASSERT((in_dtype == ccl::datatype::float32) || (in_dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(in_dtype));
    ASSERT(context->offset == 0, "wrong offset for epilogue func, should be 0");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (in_dtype == ccl::datatype::float32) {
            ((float*)out_buf)[idx] = ((float*)in_buf)[idx] * 2;
        }
        else if (in_dtype == custom_dtype) {
            custom_2x((char*)in_buf + idx * CUSTOM_DTYPE_SIZE,
                      (char*)out_buf + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(in_dtype));
        }
    }
}

void do_prologue_dtype_to_char(const void* in_buf,
                               size_t in_count,
                               ccl::datatype in_dtype,
                               void** out_buf,
                               size_t* out_count,
                               ccl::datatype* out_dtype,
                               const ccl::fn_context* context) {
    ASSERT((in_dtype == ccl::datatype::float32) || (in_dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(in_dtype));
    ASSERT(out_buf, "null ptr");
    ASSERT(context->offset == 0, "wrong offset for prologue func, should be 0");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_buf)
        *out_buf = malloc(in_count); /* will be deallocated in do_epilogue_char_to_dtype */
    if (out_count)
        *out_count = in_count;
    if (out_dtype)
        *out_dtype = ccl::datatype::int8;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (in_dtype == ccl::datatype::float32) {
            float fval = ((float*)in_buf)[idx];
            int ival = (int)fval;
            ((char*)(*out_buf))[idx] = (char)(ival % 256);
        }
        else if (in_dtype == custom_dtype) {
            custom_to_char((char*)in_buf + idx * CUSTOM_DTYPE_SIZE, (char*)(*out_buf) + idx);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(in_dtype));
        }
    }
}

void do_epilogue_char_to_dtype(const void* in_buf,
                               size_t in_count,
                               ccl::datatype in_dtype,
                               void* out_buf,
                               size_t* out_count,
                               ccl::datatype out_dtype,
                               const ccl::fn_context* context) {
    ASSERT(in_dtype == ccl::datatype::int8, "unexpected in_dtype %d", static_cast<int>(in_dtype));
    ASSERT((out_dtype == ccl::datatype::float32) || (out_dtype == custom_dtype),
           "unexpected out_dtype %d",
           static_cast<int>(out_dtype));
    ASSERT(context->offset == 0, "wrong offset for epilogue func, should be 0");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (out_dtype == ccl::datatype::float32) {
            ((float*)out_buf)[idx] = (float)(((char*)in_buf)[idx]);
        }
        else if (out_dtype == custom_dtype) {
            custom_from_char((char*)in_buf + idx, (char*)out_buf + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(out_dtype));
        }
    }

    if (in_buf != out_buf)
        free((void*)in_buf);
}

void do_reduction_sum(const void* in_buf,
                      size_t in_count,
                      void* inout_buf,
                      size_t* out_count,
                      ccl::datatype dtype,
                      const ccl::fn_context* context) {
    size_t dtype_size;
    auto& env = ccl::environment::instance();
    dtype_size = env.get_datatype_size(dtype);

    ASSERT((dtype == ccl::datatype::int8) || (dtype == ccl::datatype::float32) ||
               (dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(dtype));
    ASSERT(context->offset < MSG_SIZE_COUNT * dtype_size,
           "wrong offset for reduction_sum func, should be less than COUNT");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (dtype == ccl::datatype::int8) {
            ((char*)inout_buf)[idx] += ((char*)in_buf)[idx];
        }
        else if (dtype == ccl::datatype::float32) {
            ((float*)inout_buf)[idx] += ((float*)in_buf)[idx];
        }
        else if (dtype == custom_dtype) {
            custom_sum((char*)in_buf + idx * CUSTOM_DTYPE_SIZE,
                       (char*)inout_buf + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(dtype));
        }
    }
}

void do_reduction_null(const void* in_buf,
                       size_t in_count,
                       void* inout_buf,
                       size_t* out_count,
                       ccl::datatype dtype,
                       const ccl::fn_context* context) {
    size_t dtype_size;
    auto& env = ccl::environment::instance();
    dtype_size = env.get_datatype_size(dtype);

    ASSERT((dtype == ccl::datatype::int8) || (dtype == ccl::datatype::float32) ||
               (dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(dtype));
    ASSERT(context->offset < MSG_SIZE_COUNT * dtype_size,
           "wrong offset for reduction_null func, should be less than COUNT");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (dtype == ccl::datatype::int8) {
            ((char*)inout_buf)[idx] = (char)0;
        }
        else if (dtype == ccl::datatype::float32) {
            ((float*)inout_buf)[idx] = (float)0;
        }
        else if (dtype == custom_dtype) {
            custom_zeroize((char*)inout_buf + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(dtype));
        }
    }
}

void do_reduction_custom(const void* in_buf,
                         size_t in_count,
                         void* inout_buf,
                         size_t* out_count,
                         ccl::datatype dtype,
                         const ccl::fn_context* context) {
    size_t dtype_size;
    auto& env = ccl::environment::instance();
    dtype_size = env.get_datatype_size(dtype);

    ASSERT((dtype == ccl::datatype::float32) || (dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(dtype));
    ASSERT(context->offset < MSG_SIZE_COUNT * dtype_size,
           "wrong offset for reduction_sum func, should be less than COUNT");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    size_t global_elem_offset = context->offset / dtype_size;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (dtype == ccl::datatype::float32) {
            ((float*)inout_buf)[idx] = (float)(global_elem_offset + idx);
        }
        else if (dtype == custom_dtype) {
            custom_set((char*)inout_buf + idx * CUSTOM_DTYPE_SIZE, global_elem_offset + idx);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(dtype));
        }
    }
}

int main() {

    setenv("CCL_ATL_TRANSPORT", "ofi", 1);

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto& env = ccl::environment::instance();

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = env.create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = env.create_kvs(main_addr);
    }

    auto comm = env.create_communicator(size, rank, kvs);
    auto attr = env.create_operation_attr<ccl::allreduce_attr>();

    float float_send_buf[MSG_SIZE_COUNT];
    float float_recv_buf[MSG_SIZE_COUNT];

    auto dt_attr =
        ccl::create_datatype_attr(ccl::attr_val<ccl::datatype_attr_id::size>(CUSTOM_DTYPE_SIZE));
    custom_dtype = ccl::register_datatype(dt_attr);

    char custom_send_buf[MSG_SIZE_COUNT * CUSTOM_DTYPE_SIZE];
    char custom_recv_buf[MSG_SIZE_COUNT * CUSTOM_DTYPE_SIZE];

    std::string base_match_id = attr.get<ccl::operation_attr_id::match_id>();
    attr.set<ccl::operation_attr_id::to_cache>(true);
    std::string match_id;

    for (size_t idx = 0; idx < 2; idx++) {
        if (rank == 0)
            printf("Running tests for %s datatype\n", (idx == 0) ? "FLOAT" : "CUSTOM");
        ccl::datatype dtype = (idx == 0) ? ccl::datatype::float32 : custom_dtype;
        void* send_buf = (idx == 0) ? (void*)float_send_buf : (void*)custom_send_buf;
        void* recv_buf = (idx == 0) ? (void*)float_recv_buf : (void*)custom_recv_buf;

        fill_fn_t fill_fn = (idx == 0) ? fill_value_float : fill_value_custom;
        check_fn_t check_fn = (idx == 0) ? check_value_float : check_value_custom;
        expected_fn_t expected_fn;

        if (idx == 0) {
            /* regular sum allreduce */
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
            match_id = base_match_id + "_regular_" + std::to_string(idx);
            attr.set<ccl::operation_attr_id::match_id>(match_id);
            RUN_COLLECTIVE(
                ccl::allreduce(
                    send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::sum, comm, attr),
                fill_fn,
                check_fn,
                expected_fn,
                "regular_allreduce");

            /* prologue */
            expected_fn = (idx == 0) ? expected_float_2 : expected_custom_2;
            match_id = base_match_id + "_prologue_" + std::to_string(idx);
            attr.set<ccl::operation_attr_id::match_id>(match_id);
            attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn)do_prologue_2x);
            RUN_COLLECTIVE(
                ccl::allreduce(
                    send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::sum, comm, attr),
                fill_fn,
                check_fn,
                expected_fn,
                "allreduce_with_prologue");

            /* epilogue */
            expected_fn = (idx == 0) ? expected_float_2 : expected_custom_2;
            match_id = base_match_id + "_epilogue_" + std::to_string(idx);
            attr.set<ccl::operation_attr_id::match_id>(match_id);
            attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn) nullptr);
            attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)do_epilogue_2x);
            RUN_COLLECTIVE(
                ccl::allreduce(
                    send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::sum, comm, attr),
                fill_fn,
                check_fn,
                expected_fn,
                "allreduce_with_epilogue");

            /* prologue and epilogue */
            expected_fn = (idx == 0) ? expected_float_4 : expected_custom_4;
            match_id = base_match_id + "_prologue_and_epilogue_" + std::to_string(idx);
            attr.set<ccl::operation_attr_id::match_id>(match_id);
            attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn)do_prologue_2x);
            attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)do_epilogue_2x);
            RUN_COLLECTIVE(
                ccl::allreduce(
                    send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::sum, comm, attr),
                fill_fn,
                check_fn,
                expected_fn,
                "allreduce_with_prologue_and_epilogue");
        }

        /* reduction_sum */
        expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        match_id = base_match_id + "_reduction_sum_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn) nullptr);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn) nullptr);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_sum);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_reduction_sum");

        /* reduction_null */
        if (size == 1)
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        else
            expected_fn = (idx == 0) ? expected_float_3 : expected_custom_3;
        match_id = base_match_id + "_reduction_null_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn) nullptr);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn) nullptr);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_null);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_reduction_null");

        /* reduction_custom */
        if (size == 1)
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        else
            expected_fn = (idx == 0) ? expected_float_5 : expected_custom_5;
        match_id = base_match_id + "_reduction_custom_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn) nullptr);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn) nullptr);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_custom);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_reduction_custom");

        /* prologue and reduction_sum */
        expected_fn = (idx == 0) ? expected_float_2 : expected_custom_2;
        match_id = base_match_id + "_prologue_and_reduction_sum_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn)do_prologue_2x);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn) nullptr);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_sum);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_prologue_and_reduction_sum");

        /* epilogue and reduction_sum */
        expected_fn = (idx == 0) ? expected_float_2 : expected_custom_2;
        match_id = base_match_id + "_epilogue_and_reduction_sum_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn) nullptr);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)do_epilogue_2x);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_sum);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_epilogue_and_reduction_sum");

        /* prologue and epilogue and reduction_sum */
        expected_fn = (idx == 0) ? expected_float_4 : expected_custom_4;
        match_id =
            base_match_id + "_prologue_and_epilogue_and_reduction_sum_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn)do_prologue_2x);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)do_epilogue_2x);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_sum);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_prologue_and_epilogue_and_reduction_sum");

        /* prologue and epilogue and reduction_null */
        if (size == 1)
            expected_fn = (idx == 0) ? expected_float_4 : expected_custom_4;
        else
            expected_fn = (idx == 0) ? expected_float_3 : expected_custom_3;
        match_id =
            base_match_id + "_prologue_and_epilogue_and_reduction_null_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn)do_prologue_2x);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)do_epilogue_2x);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_null);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_prologue_and_epilogue_and_reduction_null");

        /* prologue and epilogue and reduction_sum */
        expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        match_id =
            base_match_id + "_prologue_and_epilogue_and_reduction_sum2_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>(
            (ccl::prologue_fn)do_prologue_dtype_to_char);
        attr.set<ccl::operation_attr_id::epilogue_fn>(
            (ccl::epilogue_fn)do_epilogue_char_to_dtype);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_sum);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_prologue_and_epilogue_and_reduction_sum2");

        /* epilogue and reduction_custom */
        if (size == 1)
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        else
            expected_fn = (idx == 0) ? expected_float_6 : expected_custom_6;
        match_id =
            base_match_id + "_prologue_and_epilogue_and_reduction_custom_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::operation_attr_id::prologue_fn>((ccl::prologue_fn) nullptr);
        attr.set<ccl::operation_attr_id::epilogue_fn>((ccl::epilogue_fn)do_epilogue_2x);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_custom);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_epilogue_and_reduction_custom");
    }

    if (rank == 0)
        printf("PASSED\n");

    MPI_Finalize();

    return 0;
}
