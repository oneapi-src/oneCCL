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
#ifndef SPARSE_TEST_ALGO_HPP
#define SPARSE_TEST_ALGO_HPP

#include <iterator>
#include <random>
#include <unordered_map>
#include <utility>

#include "base.h"
#include "base_utils.hpp"
#include "bfp16.h"
#include "ccl.hpp"

#define COUNT_I 1024
#define VDIM_SIZE 64
#define RANGE 255

typedef enum
{
    sparse_test_callback_completion,
    sparse_test_callback_alloc
} sparse_test_callback_mode_t;

#define PRINT_ERROR(itype_name, vtype_name)\
({                                        \
    std::string str = "\n\nexpected [";   \
    for (auto x : it->second)             \
        str += std::to_string(x) + ",";   \
    str[str.length()-1] = ']';            \
                                          \
    str += "\n\ngot [";                   \
    for (auto x : v)                      \
        str += std::to_string(x) + ",";   \
    str[str.length()-1] = ']';            \
                                          \
    printf("\nrank [%zu]: recv_idx %zu, " \
           "i_type %s, v_type %s, %s\n",  \
            rank, (size_t)rcv_idx[idx],   \
            itype_name, vtype_name,       \
            str.c_str());                 \
    ASSERT(0, "unexpected value");        \
})

#define PRINT_FLAGGED_ERROR(itype_name, vtype_name)                              \
({                                                                               \
    str = "expected " + std::to_string(evb[idx * VDIM_SIZE + j]) +               \
          ", but received " + std::to_string(rcv_val[idx * VDIM_SIZE + j]) +     \
          " on position " + std::to_string(j) +                                  \
          "(for i_type " + itype_name + ", v_type " + vtype_name + ")";          \
    printf("rank [%zu]: idx %zu, %s\n", rank, (size_t)rcv_idx[idx], str.c_str());\
    ASSERT(0, "unexpected value");                                               \
})

#define CHECK(itype_name, vtype_name, is_bfp16)                         \
({                                                                      \
    i_t* rcv_idx = (i_t*)recv_ibuf;                                     \
    std::vector<v_t> rcv_val((v_t*)recv_vbuf,                           \
                             (v_t*)recv_vbuf + recv_vcount);            \
                                                                        \
    for (idx = 0; idx < recv_icount; idx++)                             \
    {                                                                   \
        auto it = expected.find(rcv_idx[idx]);                          \
        if (it == expected.end())                                       \
        {                                                               \
            printf("rank [%zu]: idx %zu is not expected to be found\n", \
                    rank, (size_t)rcv_idx[idx]);                        \
            ASSERT(0, "unexpected value");                              \
        }                                                               \
        else                                                            \
        {                                                               \
            std::vector<v_t> v(rcv_val.begin() + VDIM_SIZE * idx,       \
                               rcv_val.begin() + VDIM_SIZE * (idx + 1));\
            if (is_bfp16)                                               \
            {                                                           \
                for(size_t i = 0; i < v.size(); i++)                    \
                {                                                       \
                    double max_error = g * it->second[i];               \
                    if (fabs(max_error) < fabs(it->second[i] - v[i]))   \
                    {                                                   \
                        PRINT_ERROR(itype_name, vtype_name);            \
                        break;                                          \
                    }                                                   \
                }                                                       \
            }                                                           \
            else                                                        \
            {                                                           \
                if (v != it->second)                                    \
                {                                                       \
                    PRINT_ERROR(itype_name, vtype_name);                \
                }                                                       \
            }                                                           \
        }                                                               \
    }                                                                   \
})

#define CHECK_WO_COALESCE(itype_name, vtype_name, is_bfp16)             \
({                                                                      \
    i_t* rcv_idx = (i_t*)recv_ibuf;                                     \
    std::vector<v_t> rcv_val((v_t*)recv_vbuf,                           \
                             (v_t*)recv_vbuf + recv_vcount);            \
                                                                        \
    if (recv_icount != expected_count)                                  \
    {                                                                   \
        printf("rank [%zu]: expected count (%zu) and received           \
                count (%zu) differ\n", rank, expected_count,            \
                recv_icount);                                           \
        ASSERT(0, "unexpected value");                                  \
    }                                                                   \
    else                                                                \
    {                                                                   \
        i_t* eib = (i_t*)expected_buf;                                  \
        v_t* evb = (v_t*)((i_t*)expected_buf + expected_count);         \
        std::string str;                                                \
        for (idx = 0; idx < recv_icount; idx++)                         \
        {                                                               \
            if (rcv_idx[idx] != eib[idx])                               \
            {                                                           \
                printf("rank [%zu]: idx %zu is not expected to be       \
                        found on position %zu\n", rank,                 \
                        (size_t)rcv_idx[idx], (size_t)idx);             \
                ASSERT(0, "unexpected value");                          \
            }                                                           \
            else                                                        \
            {                                                           \
                if (is_bfp16)                                           \
                {                                                       \
                    for(size_t j = 0; j < VDIM_SIZE; j++)               \
                    {                                                   \
                        double max_error = g * evb[idx * VDIM_SIZE + j];\
                        if (fabs(max_error) <                           \
                            fabs(evb[idx * VDIM_SIZE + j] -             \
                                 rcv_val[idx * VDIM_SIZE + j]))         \
                        {                                               \
                            PRINT_FLAGGED_ERROR(itype_name, vtype_name);\
                            break;                                      \
                        }                                               \
                    }                                                   \
                }                                                       \
                else                                                    \
                {                                                       \
                    for (size_t j = 0; j < VDIM_SIZE; j++)              \
                    {                                                   \
                        if (rcv_val[idx * VDIM_SIZE + j] !=             \
                            evb[idx * VDIM_SIZE + j])                   \
                        {                                               \
                            PRINT_FLAGGED_ERROR(itype_name, vtype_name);\
                        }                                               \
                    }                                                   \
                }                                                       \
            }                                                           \
        }                                                               \
    }                                                                   \
})

#define RUN_COLLECTIVE(start_cmd, itype_name, vtype_name)                 \
  do {                                                                    \
      t = 0;                                                              \
      for (iter_idx = 0; iter_idx < 1; iter_idx++)                        \
      {                                                                   \
          t1 = when();                                                    \
          CCL_CALL(start_cmd);                                            \
          CCL_CALL(ccl_wait(request));                                    \
          t2 = when();                                                    \
          t += (t2 - t1);                                                 \
          ccl_barrier(NULL, NULL);                                        \
      }                                                                   \
      printf("[%zu] idx_type: %s, val_type: %s, avg time: %8.2lf us\n",   \
             rank, itype_name, vtype_name, t / 1);                        \
      fflush(stdout);                                                     \
  } while (0)

template<ccl_datatype_t i_type, ccl_datatype_t v_type> void
gather_expected_data(const std::vector<typename ccl::type_info<i_type>::native_type>& ibuffer,
                     const std::vector<typename ccl::type_info<v_type>::native_type>& vbuffer,
                     void** result,
                     size_t* result_count)
{
    ASSERT(result, "void** result buffer mustn't be nullptr");
    ASSERT(result_count, "size_t* result_count mustn't be nullptr");

    if (!result || !result_count)
    {
        throw std::runtime_error("gather_expected_data: result and result_count parameters mustn't be nullptr");
    }
    else
    {
        using i_t = typename ccl::type_info<i_type>::native_type;
        using v_t = typename ccl::type_info<v_type>::native_type;

        void* recv_buf;
        size_t sum_nnz;
        size_t count = ibuffer.size();

        if (size > 1)
        {
            /* gather the number of non-zero (NNZ) values from all the ranks */
            std::vector<size_t> recv_counts(size, sizeof(size_t));
            size_t nnz = count;
            std::vector<size_t> recv_nnz(size);

            ccl_allgatherv(&nnz, sizeof(size_t),
                            recv_nnz.data(), recv_counts.data(),
                            ccl_dtype_char,
                            &coll_attr,
                            nullptr, nullptr,
                            &request);
            ccl_wait(request);

            /* gather indices and values */
            std::copy(recv_nnz.begin(), recv_nnz.end(), recv_counts.begin());
            sum_nnz = 0;
            for (unsigned int i = 0; i < size; i++)
            {
                sum_nnz += recv_nnz[i];
                recv_counts[i] *= VDIM_SIZE;
            }

            recv_buf = malloc(sum_nnz * (sizeof(i_t) + VDIM_SIZE * sizeof(v_t)));

            ccl_allgatherv(ibuffer.data(), ibuffer.size(),
                            recv_buf, recv_nnz.data(),
                            i_type,
                            &coll_attr,
                            nullptr, nullptr,
                            &request);
            ccl_wait(request);

            ccl_allgatherv(vbuffer.data(), vbuffer.size(),
                            ((i_t*)recv_buf + sum_nnz), recv_counts.data(),
                            v_type,
                            &coll_attr,
                            nullptr, nullptr,
                            &request);
            ccl_wait(request);
        }
        else
        {
            recv_buf = malloc(count * (sizeof(i_t) + VDIM_SIZE * sizeof(v_t)));
            std::copy(ibuffer.begin(), ibuffer.end(), (i_t*)recv_buf);
            std::copy(vbuffer.begin(), vbuffer.end(), (v_t*)((i_t*)recv_buf + count));
            sum_nnz = count;
        }

        *result = recv_buf;
        *result_count = sum_nnz;
    }
}

template<ccl_datatype_t i_type, ccl_datatype_t v_type>
std::map<typename ccl::type_info<i_type>::native_type,
         std::vector<typename ccl::type_info<v_type>::native_type> >
coalesce_expected_data(void* recv_buf, size_t nnz)
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;

    /* calculate expected values */
    std::map<i_t, std::vector<v_t> > exp_vals;
    i_t* idx_buf = (i_t*)recv_buf;
    v_t* val_buf = (v_t*)((i_t*)recv_buf + nnz);
    std::vector<v_t> tmp(VDIM_SIZE);
    for (unsigned int idx = 0; idx < nnz; idx++)
    {
        auto it = exp_vals.find(idx_buf[idx]);
        if (it == exp_vals.end())
        {
           std::copy(val_buf + idx * VDIM_SIZE, val_buf + (idx + 1) * VDIM_SIZE, tmp.begin());
           exp_vals.emplace(idx_buf[idx], tmp);
        }
        else
        {
            for (unsigned int jdx = 0; jdx < VDIM_SIZE; jdx++)
            {
                it->second[jdx] += val_buf[idx * VDIM_SIZE + jdx];
            }
        }
    }
    
    return exp_vals;
}

/* =================== */
/* these fields will be updated/allocated inside completion callback */
size_t recv_icount;
size_t recv_vcount;
void* recv_ibuf;
void* recv_vbuf;
#ifdef CCL_BFP16_COMPILER
void* recv_vbuf_bfp16;
#endif
/* =================== */

ccl_status_t
completion_fn(const void* i_buf, size_t i_cnt, ccl_datatype_t itype,
              const void* v_buf, size_t v_cnt, ccl_datatype_t vtype,
              const void* fn_ctx) 
{ 
    recv_icount = i_cnt;
    recv_vcount = v_cnt;

    size_t itype_size, vtype_size;
    ccl_get_datatype_size(itype, &itype_size);
    ccl_get_datatype_size(vtype, &vtype_size);

    recv_ibuf = malloc(itype_size * recv_icount);
    recv_vbuf = malloc(vtype_size * recv_vcount);

    memcpy(recv_ibuf, i_buf, itype_size * recv_icount);
    memcpy(recv_vbuf, v_buf, vtype_size * recv_vcount);
   
    return ccl_status_success;
}

ccl_status_t
alloc_fn(size_t i_cnt, ccl_datatype_t itype,
         size_t v_cnt, ccl_datatype_t vtype,
         const void* fn_ctx,
         void** out_i_buf, void** out_v_buf) 
{
    ASSERT(out_i_buf && out_v_buf, "out_i_buf or out_v_buf");

    recv_icount = i_cnt;
    recv_vcount = v_cnt;

    size_t itype_size, vtype_size;
    ccl_get_datatype_size(itype, &itype_size);
    ccl_get_datatype_size(vtype, &vtype_size);

    recv_ibuf = malloc(itype_size * recv_icount);
    recv_vbuf = malloc(vtype_size * recv_vcount);

    *out_i_buf = recv_ibuf;
    *out_v_buf = recv_vbuf;
   
    return ccl_status_success;
}

#ifdef CCL_BFP16_COMPILER
ccl_status_t
completion_bfp16_fn(const void* i_buf, size_t i_cnt, ccl_datatype_t itype,
                    const void* v_buf, size_t v_cnt, ccl_datatype_t vtype,
                    const void* fn_ctx) 
{ 
    recv_icount = i_cnt;
    recv_vcount = v_cnt;

    size_t itype_size, vtype_size;
    ccl_get_datatype_size(itype, &itype_size);
    ccl_get_datatype_size(vtype, &vtype_size);

    recv_ibuf = malloc(itype_size * recv_icount);
    recv_vbuf = malloc(sizeof(float) * recv_vcount);
    recv_vbuf_bfp16 = malloc(vtype_size * recv_vcount);

    memcpy(recv_ibuf, i_buf, itype_size * recv_icount);
    memcpy(recv_vbuf_bfp16, v_buf, vtype_size * recv_vcount);
      
    return ccl_status_success;
}

ccl_status_t
alloc_bfp16_fn(size_t i_cnt, ccl_datatype_t itype,
               size_t v_cnt, ccl_datatype_t vtype,
               const void* fn_ctx,
               void** out_i_buf, void** out_v_buf) 
{
    ASSERT(out_i_buf && out_v_buf, "out_i_buf or out_v_buf");

    recv_icount = i_cnt;
    recv_vcount = v_cnt;

    size_t itype_size, vtype_size;
    ccl_get_datatype_size(itype, &itype_size);
    ccl_get_datatype_size(vtype, &vtype_size);

    recv_ibuf = malloc(itype_size * recv_icount);
    recv_vbuf = malloc(sizeof(float) * recv_vcount);
    recv_vbuf_bfp16 = malloc(vtype_size * recv_vcount);

    *out_i_buf = recv_ibuf;
    *out_v_buf = recv_vbuf_bfp16;
      
    return ccl_status_success;
}

template<ccl_datatype_t i_type, ccl_datatype_t v_type, 
        typename std::enable_if< v_type == ccl_dtype_bfp16, int>::type = 0>
void sparse_test_run(ccl_sparse_coalesce_mode_t coalesce_mode,
                     sparse_test_callback_mode_t callback_mode)
{
    if (is_bfp16_enabled() == 0)
    {
        printf("WARNING: BFP16 is not enabled, skipped.\n");
        return;
    }
    else
    {
        using i_t = typename ccl::type_info<i_type>::native_type;
        using v_t = float;

        size_t count = COUNT_I - rank;
        std::vector<i_t> send_ibuf(count);
        std::vector<v_t> send_vbuf(count * VDIM_SIZE); 

        /* generate pseudo-random indices and calculate values */
        std::random_device seed;
        std::mt19937 gen(seed());
        std::uniform_int_distribution<> dist(0, RANGE);
        for (size_t i = 0; i < count; i++)
        {
            send_ibuf[i] = dist(gen);
            for (unsigned int j = 0; j < VDIM_SIZE; j++)
            {
                send_vbuf[i * VDIM_SIZE + j] = (rank + j)/100;
            }
        }

        void* expected_buf = nullptr;
        size_t expected_count = 0;
        gather_expected_data<i_type, ccl_dtype_float>(send_ibuf, send_vbuf, &expected_buf, &expected_count);
        ASSERT(expected_buf, "expected_buf is null");
        ASSERT(expected_count, "expected_count is zero");

        std::map<i_t, std::vector<v_t> > expected{};
        if (coalesce_mode != ccl_sparse_coalesce_disable)
        {
            expected = coalesce_expected_data<i_type, ccl_dtype_float>(expected_buf, expected_count);
        }

        /* run sparse collective */
        memset(&coll_attr, 0, sizeof(ccl_coll_attr_t));
        coll_attr.to_cache = 0;
        if (callback_mode == sparse_test_callback_completion)
            coll_attr.sparse_allreduce_completion_fn = completion_bfp16_fn;
        else
            coll_attr.sparse_allreduce_alloc_fn = alloc_bfp16_fn;
        coll_attr.sparse_allreduce_fn_ctx = nullptr;
        coll_attr.sparse_coalesce_mode = coalesce_mode;
        void* send_vbuf_bfp16 = malloc(sizeof(ccl::bfp16) * send_vbuf.size());

        convert_fp32_to_bfp16_arrays(send_vbuf.data(), send_vbuf_bfp16, send_vbuf.size());

        recv_icount = 0;
        recv_vcount = 0;
        recv_ibuf = nullptr;
        recv_vbuf = nullptr;
        recv_vbuf_bfp16 = nullptr;

        RUN_COLLECTIVE(ccl_sparse_allreduce(send_ibuf.data(), send_ibuf.size(),
                                            send_vbuf_bfp16, send_vbuf.size(),
                                            nullptr, 0,
                                            nullptr, 0,
                                            i_type, v_type,
                                            ccl_reduction_sum,
                                            &coll_attr,
                                            nullptr, nullptr, &request),
                       ccl::type_info<i_type>::name(),
                       ccl::type_info<v_type>::name());

        ASSERT(recv_icount, "recv_icount is zero");
        ASSERT(recv_vcount, "recv_vcount is zero");
        ASSERT(recv_ibuf, "recv_ibuf is null");
        ASSERT(recv_vbuf, "recv_vbuf is null");
        ASSERT(recv_vbuf_bfp16, "recv_vbuf_bfp16 is null");

        convert_bfp16_to_fp32_arrays(recv_vbuf_bfp16, (float*)recv_vbuf, (int)recv_vcount);

        /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */
        double log_base2 = log(size) / log(2);
        double g = (log_base2 * BFP16_PRECISION) / (1 - (log_base2 * BFP16_PRECISION));

        if (coalesce_mode == ccl_sparse_coalesce_disable)
        {
            CHECK_WO_COALESCE(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name(), true);
        }
        else
        {
            CHECK(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name(), true);    
        }

        free(expected_buf);

        free(send_vbuf_bfp16);
        
        free(recv_ibuf);
        free(recv_vbuf);
        free(recv_vbuf_bfp16);
    }
}
#endif  /* CCL_BFP16_COMPILER */

template<ccl_datatype_t i_type, ccl_datatype_t v_type, 
        typename std::enable_if< v_type != ccl_dtype_bfp16, int>::type = 0>
void sparse_test_run(ccl_sparse_coalesce_mode_t coalesce_mode,
                     sparse_test_callback_mode_t callback_mode)
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;
    size_t count = COUNT_I - rank;
    std::vector<i_t> send_ibuf(count);
    std::vector<v_t> send_vbuf(count * VDIM_SIZE);

    /* generate pseudo-random indices and calculate values */
    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_int_distribution<> dist(0, RANGE);
    for (size_t i = 0; i < count; i++)
    {
        send_ibuf[i] = dist(gen);
        for (unsigned int j = 0; j < VDIM_SIZE; j++)
        {
            send_vbuf[i * VDIM_SIZE + j] = (rank + j);
        }
    }

    void* expected_buf = nullptr;
    size_t expected_count = 0;
    gather_expected_data<i_type, v_type>(send_ibuf, send_vbuf, &expected_buf, &expected_count);
    ASSERT(expected_buf, "expected_buf is nullptr");
    ASSERT(expected_count, "expected_count is zero");

    std::map<i_t, std::vector<v_t> > expected{};
    if (coalesce_mode != ccl_sparse_coalesce_disable)
    {
        expected = coalesce_expected_data<i_type, v_type>(expected_buf, expected_count);
    }
    
    /* run sparse collective */
    memset(&coll_attr, 0, sizeof(ccl_coll_attr_t));
    coll_attr.to_cache = 0;
    if (callback_mode == sparse_test_callback_completion)
        coll_attr.sparse_allreduce_completion_fn = completion_fn;
    else
        coll_attr.sparse_allreduce_alloc_fn = alloc_fn;
    coll_attr.sparse_allreduce_fn_ctx = nullptr;
    coll_attr.sparse_coalesce_mode = coalesce_mode;

    recv_icount = 0;
    recv_vcount = 0;
    recv_ibuf = nullptr;
    recv_vbuf = nullptr;

    RUN_COLLECTIVE(ccl_sparse_allreduce(send_ibuf.data(), send_ibuf.size(),
                                        send_vbuf.data(), send_vbuf.size(),
                                        nullptr, 0,
                                        nullptr, 0,
                                        i_type, v_type,
                                        ccl_reduction_sum,
                                        &coll_attr,
                                        nullptr, nullptr, &request),
                   ccl::type_info<i_type>::name(),
                   ccl::type_info<v_type>::name());

    ASSERT(recv_icount, "recv_icount is zero");
    ASSERT(recv_vcount, "recv_vcount is zero");
    ASSERT(recv_ibuf, "recv_ibuf is null");
    ASSERT(recv_vbuf, "recv_vbuf is null");

    double g;

    if (coalesce_mode == ccl_sparse_coalesce_disable)
    {
        CHECK_WO_COALESCE(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name(), false);
    }
    else
    {      
        CHECK(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name(), false);
    }

    free(expected_buf);

    free(recv_ibuf);
    free(recv_vbuf);
}
#endif /* SPARSE_TEST_ALGO_HPP */
