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

#include "oneapi/ccl.hpp"

#define PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__);

#ifndef PRINT_BY_ROOT
#define PRINT_BY_ROOT(comm, fmt, ...) \
    if (comm.rank() == 0) { \
        printf(fmt "\n", ##__VA_ARGS__); \
    }
#endif //PRINT_BY_ROOT

#define ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            printf("FAILED\n"); \
            fprintf(stderr, "ASSERT '%s' FAILED " fmt "\n", #cond, ##__VA_ARGS__); \
            throw std::runtime_error("ASSERT FAILED"); \
        } \
    } while (0)

// TODO: add ccl::bfloat16
constexpr std::initializer_list<ccl::datatype> all_dtypes = {
    ccl::datatype::int8,    ccl::datatype::int32, ccl::datatype::float32,
    ccl::datatype::float64, ccl::datatype::int64, ccl::datatype::uint64
};

typedef enum { BACKEND_HOST, BACKEND_SYCL } backend_type_t;
typedef enum { LOOP_REGULAR, LOOP_UNORDERED } loop_type_t;

typedef enum { SYCL_DEV_HOST, SYCL_DEV_CPU, SYCL_DEV_GPU } sycl_dev_type_t;
typedef enum { SYCL_MEM_USM, SYCL_MEM_BUF } sycl_mem_type_t;
typedef enum { SYCL_USM_SHARED, SYCL_USM_DEVICE } sycl_usm_type_t;

std::map<backend_type_t, std::string> backend_names = { std::make_pair(BACKEND_HOST, "host"),
                                                        std::make_pair(BACKEND_SYCL, "sycl") };

std::map<loop_type_t, std::string> loop_names = { std::make_pair(LOOP_REGULAR, "regular"),
                                                  std::make_pair(LOOP_UNORDERED, "unordered") };

std::map<sycl_dev_type_t, std::string> sycl_dev_names = { std::make_pair(SYCL_DEV_HOST, "host"),
                                                          std::make_pair(SYCL_DEV_CPU, "cpu"),
                                                          std::make_pair(SYCL_DEV_GPU, "gpu") };

std::map<sycl_mem_type_t, std::string> sycl_mem_names = { std::make_pair(SYCL_MEM_USM, "usm"),
                                                          std::make_pair(SYCL_MEM_BUF, "buf") };

std::map<sycl_usm_type_t, std::string> sycl_usm_names = { std::make_pair(SYCL_USM_SHARED, "shared"),
                                                          std::make_pair(SYCL_USM_DEVICE,
                                                                         "device") };

// TODO: add ccl::bfloat16
std::map<ccl::datatype, std::string> dtype_names = {
    std::make_pair(ccl::datatype::int8, "int8"),
    std::make_pair(ccl::datatype::int32, "int32"),
    std::make_pair(ccl::datatype::int64, "int64"),
    std::make_pair(ccl::datatype::uint64, "uint64"),
    std::make_pair(ccl::datatype::float32, "float32"),
    std::make_pair(ccl::datatype::float64, "float64")
};

std::map<ccl::reduction, std::string> reduction_names = {
    std::make_pair(ccl::reduction::sum, "sum"),
    std::make_pair(ccl::reduction::prod, "prod"),
    std::make_pair(ccl::reduction::min, "min"),
    std::make_pair(ccl::reduction::max, "max"),
};

std::list<std::string> tokenize(const std::string& input, char delimeter) {
    std::stringstream ss(input);
    std::list<std::string> ret;
    std::string value;
    while (std::getline(ss, value, delimeter)) {
        ret.push_back(value);
    }
    return ret;
}

typedef struct user_options_t {
    backend_type_t backend;
    loop_type_t loop;
    size_t iters;
    size_t warmup_iters;
    size_t buf_count;
    size_t min_elem_count;
    size_t max_elem_count;
    int check_values;
    size_t v2i_ratio;
    sycl_dev_type_t sycl_dev_type;
    sycl_mem_type_t sycl_mem_type;
    sycl_usm_type_t sycl_usm_type;
    size_t ranks_per_proc;
    std::list<std::string> coll_names;
    std::list<std::string> dtypes;
    std::list<std::string> reductions;
    std::string csv_filepath;

    user_options_t() {
        backend = DEFAULT_BACKEND;
        loop = DEFAULT_LOOP;
        iters = DEFAULT_ITERS;
        warmup_iters = DEFAULT_WARMUP_ITERS;
        buf_count = DEFAULT_BUF_COUNT;
        min_elem_count = DEFAULT_MIN_ELEM_COUNT;
        max_elem_count = DEFAULT_MAX_ELEM_COUNT;
        check_values = DEFAULT_CHECK_VALUES;
        v2i_ratio = DEFAULT_V2I_RATIO;
        sycl_dev_type = DEFAULT_SYCL_DEV_TYPE;
        sycl_mem_type = DEFAULT_SYCL_MEM_TYPE;
        sycl_usm_type = DEFAULT_SYCL_USM_TYPE;
        ranks_per_proc = DEFAULT_RANKS_PER_PROC;
        coll_names = tokenize(DEFAULT_COLL_LIST, ',');
        dtypes = tokenize(DEFAULT_DTYPES_LIST, ',');
        reductions = tokenize(DEFAULT_REDUCTIONS_LIST, ',');
        csv_filepath = std::string(DEFAULT_CSV_FILEPATH);
    }
} user_options_t;
