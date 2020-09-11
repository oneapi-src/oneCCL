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
#define TEST_CCL_CUSTOM_PROLOG
#define TEST_CCL_CUSTOM_EPILOG
#define TEST_CCL_CUSTOM_REDUCE

#define COLL_NAME "CCL_ALLREDUCE"

#include <cmath>
#include <mutex>

#include "base_impl.hpp"

static size_t COUNT;
static std::vector<std::pair<std::string, int>> glob_match_id;
std::mutex inc_mutex;

size_t get_dtype_size(ccl_datatype_t dtype) {
    size_t dtype_size = 1;
    switch (dtype) {
        case ccl_dtype_char: {
            dtype_size = 1;
            break;
        }
        case ccl_dtype_int: {
            dtype_size = 4;
            break;
        }
        case ccl_dtype_bfp16: {
            dtype_size = 2;
            break;
        }
        case ccl_dtype_float: {
            dtype_size = 4;
            break;
        }
        case ccl_dtype_double: {
            dtype_size = 8;
            break;
        }
        case ccl_dtype_int64: {
            dtype_size = 8;
            break;
        }
        case ccl_dtype_uint64: {
            dtype_size = 8;
            break;
        }
        default: ASSERT(0, "unexpected dtype %d", dtype);
    }
    return dtype_size;
}

template <typename T>
ccl_status_t do_prologue_T_2x(const void* in_buf,
                              size_t in_count,
                              ccl_datatype_t in_dtype,
                              void** out_buf,
                              size_t* out_count,
                              ccl_datatype_t* out_dtype,
                              const ccl_fn_context_t* ctx) {
    size_t buf_idx;
    ASSERT(out_buf, "null ptr");
    //TODO: this assert works only for single prologue.
    ASSERT(ctx->offset == 0, "wrong offset for prologue func, should be 0");
    {
        std::lock_guard<std::mutex> guard(inc_mutex);
        auto match = std::find_if(glob_match_id.begin(),
                                  glob_match_id.end(),
                                  [&ctx](std::pair<std::string, int>& element) {
                                      return !strcmp(element.first.c_str(), ctx->match_id);
                                  });
        ASSERT(match != glob_match_id.end(), "wrong match_id");
        match->second++;
    }

    if (out_buf)
        *out_buf = (void*)in_buf;
    if (out_count)
        *out_count = in_count;
    if (out_dtype)
        *out_dtype = in_dtype;

    for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
        ((T*)(*out_buf))[buf_idx] = ((T*)in_buf)[buf_idx] * 2;
    }
    return ccl_status_success;
}

template <typename T>
ccl_status_t do_epilogue_T_2x(const void* in_buf,
                              size_t in_count,
                              ccl_datatype_t in_dtype,
                              void* out_buf,
                              size_t* out_count,
                              ccl_datatype_t out_dtype,
                              const ccl_fn_context_t* ctx) {
    size_t buf_idx;
    if (out_count)
        *out_count = in_count;
    //TODO: this assert works only for single epilogue.
    ASSERT(ctx->offset == 0, "wrong offset for epilogue func, should be 0");
    {
        std::lock_guard<std::mutex> guard(inc_mutex);
        auto match = std::find_if(glob_match_id.begin(),
                                  glob_match_id.end(),
                                  [&ctx](std::pair<std::string, int>& element) {
                                      return !strcmp(element.first.c_str(), ctx->match_id);
                                  });
        ASSERT(match != glob_match_id.end(), "wrong match_id");
        match->second++;
    }
    for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
        ((T*)out_buf)[buf_idx] = ((T*)in_buf)[buf_idx] * 2;
    }
    return ccl_status_success;
}

template <typename T>
ccl_status_t do_prologue_T_to_char(const void* in_buf,
                                   size_t in_count,
                                   ccl_datatype_t in_dtype,
                                   void** out_buf,
                                   size_t* out_count,
                                   ccl_datatype_t* out_dtype,
                                   const ccl_fn_context_t* ctx) {
    size_t buf_idx;
    ASSERT(out_buf, "null ptr");
    //TODO: this assert works only for single prologue.
    ASSERT(ctx->offset == 0, "wrong offset for prologue func, should be 0");
    {
        std::lock_guard<std::mutex> guard(inc_mutex);
        auto match = std::find_if(glob_match_id.begin(),
                                  glob_match_id.end(),
                                  [&ctx](std::pair<std::string, int>& element) {
                                      return !strcmp(element.first.c_str(), ctx->match_id);
                                  });
        ASSERT(match != glob_match_id.end(), "wrong match_id");
        match->second++;
    }

    if (out_buf)
        *out_buf = malloc(in_count);
    if (out_count)
        *out_count = in_count;
    if (out_dtype)
        *out_dtype = ccl_dtype_char;

    for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
        T fval = ((T*)in_buf)[buf_idx];
        int ival = (int)fval;
        ((char*)(*out_buf))[buf_idx] = (char)(ival % 256);
    }
    return ccl_status_success;
}

template <typename T>
ccl_status_t do_epilogue_char_to_T(const void* in_buf,
                                   size_t in_count,
                                   ccl_datatype_t in_dtype,
                                   void* out_buf,
                                   size_t* out_count,
                                   ccl_datatype_t out_dtype,
                                   const ccl_fn_context_t* ctx) {
    size_t buf_idx;
    if (out_count)
        *out_count = in_count;
    //TODO: this assert works only for single epilogue.
    ASSERT(in_dtype == ccl_dtype_char, "unexpected dtype %d", in_dtype);
    ASSERT(ctx->offset == 0, "wrong offset for epilogue func, should be 0");
    {
        std::lock_guard<std::mutex> guard(inc_mutex);
        auto match = std::find_if(glob_match_id.begin(),
                                  glob_match_id.end(),
                                  [&ctx](std::pair<std::string, int>& element) {
                                      return !strcmp(element.first.c_str(), ctx->match_id);
                                  });
        ASSERT(match != glob_match_id.end(), "wrong match_id");
        match->second++;
    }
    for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
        ((T*)out_buf)[buf_idx] = (T)(((char*)in_buf)[buf_idx]);
    }
    if (in_buf != out_buf) {
        free((void*)in_buf);
    }
    return ccl_status_success;
}

template <typename T>
ccl_status_t do_reduction_null(const void* in_buf,
                               size_t in_count,
                               void* inout_buf,
                               size_t* out_count,
                               ccl_datatype_t dtype,
                               const ccl_fn_context_t* ctx) {
    size_t buf_idx;
    ASSERT(ctx->offset < COUNT * get_dtype_size(dtype),
           "wrong offset for reduction_null func, should be less than COUNT");
    {
        std::lock_guard<std::mutex> guard(inc_mutex);
        auto match = std::find_if(glob_match_id.begin(),
                                  glob_match_id.end(),
                                  [&ctx](std::pair<std::string, int>& element) {
                                      return !strcmp(element.first.c_str(), ctx->match_id);
                                  });
        ASSERT(match != glob_match_id.end(), "wrong match_id");
        match->second++;
    }
    if (out_count)
        *out_count = in_count;
    switch (dtype) {
        case ccl_dtype_char:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((char*)inout_buf)[buf_idx] = (char)0;
            }
            break;
        case ccl_dtype_float:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((float*)inout_buf)[buf_idx] = (float)0;
            }
            break;
        case ccl_dtype_double:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((double*)inout_buf)[buf_idx] = (double)0;
            }
            break;
        case ccl_dtype_int:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((int*)inout_buf)[buf_idx] = (int)0;
            }
            break;
        default: ASSERT(0, "unexpected dtype %d", dtype); break;
    }
    return ccl_status_success;
}

template <typename T>
ccl_status_t do_reduction_custom(const void* in_buf,
                                 size_t in_count,
                                 void* inout_buf,
                                 size_t* out_count,
                                 ccl_datatype_t dtype,
                                 const ccl_fn_context_t* ctx) {
    size_t buf_idx;
    ASSERT(ctx->offset < COUNT * get_dtype_size(dtype),
           "wrong offset for reduction_custom func, should be less than COUNT");
    {
        std::lock_guard<std::mutex> guard(inc_mutex);
        auto match = std::find_if(glob_match_id.begin(),
                                  glob_match_id.end(),
                                  [&ctx](std::pair<std::string, int>& element) {
                                      return !strcmp(element.first.c_str(), ctx->match_id);
                                  });
        ASSERT(match != glob_match_id.end(), "wrong match_id");
        match->second++;
    }
    if (out_count)
        *out_count = in_count;
    switch (dtype) {
        case ccl_dtype_char:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((char*)inout_buf)[buf_idx] += ((char*)in_buf)[buf_idx];
            }
            break;
        case ccl_dtype_float:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((float*)inout_buf)[buf_idx] += ((float*)in_buf)[buf_idx];
            }
            break;
        case ccl_dtype_double:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((double*)inout_buf)[buf_idx] += ((double*)in_buf)[buf_idx];
            }
            break;
        case ccl_dtype_int:
            for (buf_idx = 0; buf_idx < in_count; buf_idx++) {
                ((int*)inout_buf)[buf_idx] += ((int*)in_buf)[buf_idx];
            }
            break;
        default: ASSERT(0, "unexpected dtype %d", dtype); break;
    }
    return ccl_status_success;
}

template <typename T>
int set_custom_reduction(typed_test_param<T>& param) {
    ccl_reduction_type customFuncName = param.test_conf.reduction_type;
    switch (customFuncName) {
        case RT_CUSTOM: param.coll_attr.reduction_fn = do_reduction_custom<T>; break;
        case RT_CUSTOM_NULL: param.coll_attr.reduction_fn = do_reduction_null<T>; break;
        default: return TEST_FAILURE;
    }
    return TEST_SUCCESS;
}

template <typename T>
class allreduce_custom_test : public base_test<T> {
public:
    int check(typed_test_param<T>& param) {
        size_t epilog_coeff = 1;
        size_t prolog_coeff = 1;
        size_t prolog_coeff_prod = 1;
        size_t epilog_coeff_prod = 1;
        T expected_fin = 0;

        if (param.test_conf.prolog_type == PTYPE_T_TO_2X) {
            prolog_coeff = 2;
            prolog_coeff_prod = pow(2, param.process_count);
        }

        if (param.test_conf.epilog_type == ETYPE_T_TO_2X) {
            epilog_coeff = 2;
            epilog_coeff_prod = epilog_coeff;
        }

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < param.elem_count; elem_idx++) {
                if (param.test_conf.reduction_type == RT_SUM) {
                    T expected = ((param.process_count * (param.process_count - 1) / 2) +
                                  ((elem_idx + buf_idx) * param.process_count));

                    if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                        param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                        expected_fin = ((char)(expected * prolog_coeff * epilog_coeff));
                    }
                    else {
                        expected_fin = expected * prolog_coeff * epilog_coeff;
                    }

                    if (param.recv_buf[buf_idx][elem_idx] != expected_fin) {
                        snprintf(
                            allreduce_custom_test::get_err_message(),
                            ERR_MESSAGE_MAX_LEN,
                            "[%zu] sent send_buf[%zu][%zu] = %f, got recv_buf[%zu][%zu] = %f, but expected = %f\n",
                            param.process_idx,
                            buf_idx,
                            elem_idx,
                            (double)param.send_buf[buf_idx][elem_idx],
                            buf_idx,
                            elem_idx,
                            (double)param.recv_buf[buf_idx][elem_idx],
                            (double)expected_fin);
                        return TEST_FAILURE;
                    }
                }
                else if (param.test_conf.reduction_type == RT_MAX) {
                    T expected = 0;

                    if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                        param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                        expected = get_expected_max<char>(
                            elem_idx, buf_idx, param.process_count, prolog_coeff);
                        expected_fin = ((char)(expected * epilog_coeff));
                    }
                    else {
                        expected = get_expected_max<T>(
                            elem_idx, buf_idx, param.process_count, prolog_coeff);
                        expected_fin = expected * epilog_coeff;
                    }

                    if (param.recv_buf[buf_idx][elem_idx] != expected_fin) {
                        snprintf(
                            allreduce_custom_test::get_err_message(),
                            ERR_MESSAGE_MAX_LEN,
                            "[%zu] sent send_buf[%zu][%zu] = %f, got recv_buf[%zu][%zu] = %f, but expected = %f\n",
                            param.process_idx,
                            buf_idx,
                            elem_idx,
                            (double)param.send_buf[buf_idx][elem_idx],
                            buf_idx,
                            elem_idx,
                            (double)param.recv_buf[buf_idx][elem_idx],
                            (double)expected_fin);
                        return TEST_FAILURE;
                    }
                }
                else if (param.test_conf.reduction_type == RT_MIN) {
                    T expected = 0;

                    if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                        param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                        expected = get_expected_min<char>(
                            elem_idx, buf_idx, param.process_count, prolog_coeff);
                        expected_fin = ((char)(expected * epilog_coeff));
                    }
                    else {
                        expected = get_expected_min<T>(
                            elem_idx, buf_idx, param.process_count, prolog_coeff);
                        expected_fin = expected * epilog_coeff;
                    }

                    if (param.recv_buf[buf_idx][elem_idx] != expected_fin) {
                        snprintf(
                            allreduce_custom_test::get_err_message(),
                            ERR_MESSAGE_MAX_LEN,
                            "[%zu] sent send_buf[%zu][%zu] = %f, got recv_buf[%zu][%zu] = %f, but expected = %f\n",
                            param.process_idx,
                            buf_idx,
                            elem_idx,
                            (double)param.send_buf[buf_idx][elem_idx],
                            buf_idx,
                            elem_idx,
                            (double)param.recv_buf[buf_idx][elem_idx],
                            (double)expected_fin);
                        return TEST_FAILURE;
                    }
                }
                else if (param.test_conf.reduction_type == RT_PROD) {
                    T expected = 1;

                    for (size_t proc_idx = 0; proc_idx < param.process_count; proc_idx++) {
                        if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                            param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                            expected *= ((char)elem_idx + (char)buf_idx + (char)proc_idx);
                        }
                        else
                            expected *= (elem_idx + buf_idx + proc_idx);
                    }

                    if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                        param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                        expected_fin = ((char)(expected * prolog_coeff_prod * epilog_coeff_prod));
                    }
                    else {
                        expected_fin = expected * prolog_coeff_prod * epilog_coeff_prod;
                    }

                    if (param.recv_buf[buf_idx][elem_idx] != expected_fin) {
                        snprintf(
                            allreduce_custom_test::get_err_message(),
                            ERR_MESSAGE_MAX_LEN,
                            "[%zu] sent send_buf[%zu][%zu] = %f, got recv_buf[%zu][%zu] = %f, but expected = %f\n",
                            param.process_idx,
                            buf_idx,
                            elem_idx,
                            (double)param.send_buf[buf_idx][elem_idx],
                            buf_idx,
                            elem_idx,
                            (double)param.recv_buf[buf_idx][elem_idx],
                            (double)expected_fin);
                        return TEST_FAILURE;
                    }
                }
                else if (param.test_conf.reduction_type == RT_CUSTOM) {
                    T expected = ((param.process_count * (param.process_count - 1) / 2) +
                                  ((elem_idx + buf_idx) * param.process_count));

                    if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                        param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                        expected_fin = ((char)(expected * prolog_coeff * epilog_coeff));
                    }
                    else {
                        expected_fin = expected * prolog_coeff * epilog_coeff;
                    }

                    if (param.recv_buf[buf_idx][elem_idx] != expected_fin) {
                        snprintf(
                            allreduce_custom_test::get_err_message(),
                            ERR_MESSAGE_MAX_LEN,
                            "[%zu] sent send_buf[%zu][%zu] = %f, got recv_buf[%zu][%zu] = %f, but expected = %f\n",
                            param.process_idx,
                            buf_idx,
                            elem_idx,
                            (double)param.send_buf[buf_idx][elem_idx],
                            buf_idx,
                            elem_idx,
                            (double)param.recv_buf[buf_idx][elem_idx],
                            (double)expected_fin);
                        return TEST_FAILURE;
                    }
                }
                else if (param.test_conf.reduction_type == RT_CUSTOM_NULL) {
                    T expected = 0;

                    if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                        param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                        expected_fin = ((char)(expected * prolog_coeff * epilog_coeff));
                    }
                    else {
                        expected_fin = expected * prolog_coeff * epilog_coeff;
                    }

                    if (param.recv_buf[buf_idx][elem_idx] != expected_fin) {
                        snprintf(
                            allreduce_custom_test::get_err_message(),
                            ERR_MESSAGE_MAX_LEN,
                            "[%zu] sent send_buf[%zu][%zu] = %f, got recv_buf[%zu][%zu] = %f, but expected = %f\n",
                            param.process_idx,
                            buf_idx,
                            elem_idx,
                            (double)param.send_buf[buf_idx][elem_idx],
                            buf_idx,
                            elem_idx,
                            (double)param.recv_buf[buf_idx][elem_idx],
                            (double)expected_fin);
                        return TEST_FAILURE;
                    }
                }
                else
                    return TEST_FAILURE;
            }
        }
        return TEST_SUCCESS;
    }

    void run_derived(typed_test_param<T>& param) {
        const ccl_test_conf& test_conf = param.get_conf();
        size_t count = param.elem_count;
        ccl::reduction reduction = (ccl::reduction)get_ccl_lib_reduction_type(test_conf);
        ccl::coll_attr* attr = &param.coll_attr;
        ccl::stream_t& stream = param.get_stream();

        for (size_t buf_idx = 0; buf_idx < param.buffer_count; buf_idx++) {
            param.prepare_coll_attr(param.buf_indexes[buf_idx]);
            T* send_buf = param.send_buf[param.buf_indexes[buf_idx]].data();
            T* recv_buf = param.recv_buf[param.buf_indexes[buf_idx]].data();
            glob_match_id[buf_idx].first = param.match_id;
            glob_match_id[buf_idx].second = 0;

            if (param.test_conf.prolog_type == PTYPE_T_TO_2X) {
                param.coll_attr.prologue_fn = do_prologue_T_2x<T>;
            }

            if (param.test_conf.epilog_type == ETYPE_T_TO_2X) {
                param.coll_attr.epilogue_fn = do_epilogue_T_2x<T>;
            }

            if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                param.coll_attr.prologue_fn = do_prologue_T_to_char<T>;
                param.coll_attr.epilogue_fn = do_epilogue_char_to_T<T>;
            }
            else if (param.test_conf.prolog_type != PTYPE_T_TO_CHAR &&
                     param.test_conf.epilog_type == ETYPE_CHAR_TO_T) {
                param.coll_attr.epilogue_fn = NULL;
            }
            else if (param.test_conf.prolog_type == PTYPE_T_TO_CHAR &&
                     param.test_conf.epilog_type != ETYPE_CHAR_TO_T) {
                param.coll_attr.prologue_fn = NULL;
            }

            if (reduction == ccl::reduction::custom) {
                set_custom_reduction<T>(param);
            }

            COUNT = param.elem_count;
            param.reqs[buf_idx] =
                param.global_comm->allreduce((test_conf.place_type == PT_IN) ? recv_buf : send_buf,
                                             recv_buf,
                                             count,
                                             reduction,
                                             attr,
                                             stream);
        }
    }

    size_t get_recv_buf_size(typed_test_param<T>& param) {
        ASSERT(0, "unsupported case for custom_allreduce");
        return 0;
    }

    int run(typed_test_param<T>& param) {
        size_t result = 0;

        SHOW_ALGO(COLL_NAME);

        const ccl_test_conf& test_conf = param.get_conf();
        glob_match_id.resize(param.buffer_count);

        if (test_conf.data_type == DT_BFP16) {
            printf("WARNING! BFP16 is not supported for custom reduction, test skipped");
            return result;
        }

        for (size_t iter = 0; iter < ITER_COUNT; iter++) {
            try {
                base_test<T>::alloc_buffers(param);
                base_test<T>::fill_buffers(param);
                param.swap_buffers(iter);
                param.define_start_order();
                run_derived(param);
                param.complete();
                result += check(param);
                // free_buffers(param);

                if (((get_ccl_lib_reduction_type(test_conf) == ccl_reduction_custom ||
                      param.test_conf.prolog_type != PTYPE_NULL ||
                      param.test_conf.epilog_type != ETYPE_NULL) &&
                     (param.coll_attr.prologue_fn != NULL ||
                      param.coll_attr.epilogue_fn != NULL))) {
                    for (auto it : glob_match_id) {
                        if (it.second == 0) {
                            snprintf(allreduce_custom_test::get_err_message(),
                                     ERR_MESSAGE_MAX_LEN,
                                     "[%zu] Wrong count match_id %s %d\n",
                                     param.process_idx,
                                     it.first.c_str(),
                                     it.second);
                            throw std::runtime_error(allreduce_custom_test::get_err_message());
                        }
                    }
                }
            }
            catch (const std::exception& ex) {
                result += TEST_FAILURE;
                printf("WARNING! %s iter number: %zu", ex.what(), iter);
            }
        }

        glob_match_id.clear();
        return result;
    }
};

RUN_METHOD_DEFINITION(allreduce_custom_test);
TEST_CASES_DEFINITION(allreduce_custom_test);
MAIN_FUNCTION();
