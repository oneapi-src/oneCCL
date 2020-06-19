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

#define ERROR(itype_name, vtype_name)     \
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

#define CHECK_BFP16(itype_name, vtype_name)                          \
{                                                                    \
    /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */            \
    double log_base2 = log(size) / log(2);                           \
    double g =                                                       \
        (log_base2 * BFP16_PRECISION) /                              \
        (1 - (log_base2 * BFP16_PRECISION));                         \
                                                                     \
    i_t* rcv_idx = (i_t*)recv_ibuf;                                  \
    v_t* rcv_val = (v_t*)recv_vbuf;                                  \
                                                                     \
    std::vector<v_t> vb(rcv_val, rcv_val + recv_vcount);             \
                                                                     \
    for (idx = 0; idx < recv_icount; idx++)                          \
    {                                                                \
        auto it = expected.find(rcv_idx[idx]);                       \
        if (it == expected.end())                                    \
        {                                                            \
            printf("rank [%zu]: idx %zu is not expected to be found\n",\
                    rank, (size_t)rcv_idx[idx]);                     \
            ASSERT(0, "unexpected value");                           \
        }                                                            \
        else                                                         \
        {                                                            \
            std::vector<v_t> v(vb.begin() + VDIM_SIZE * idx,         \
            vb.begin() + VDIM_SIZE * (idx + 1));                     \
                                                                     \
            for(size_t i = 0; i < v.size(); i++)                     \
            {                                                        \
                double max_error = g * it->second[i];                \
                if (fabs(max_error) < fabs(it->second[i] - v[i]))    \
                {                                                    \
                    ERROR(itype_name, vtype_name);                   \
                    break;                                           \
                }                                                    \
            }                                                        \
        }                                                            \
    }                                                                \
}

#define CHECK(itype_name, vtype_name)                                \
{                                                                    \
    i_t* rcv_idx = (i_t*)recv_ibuf;                                  \
    v_t* rcv_val = (v_t*)recv_vbuf;                                  \
                                                                     \
    std::vector<v_t> vb(rcv_val, rcv_val + recv_vcount);             \
                                                                     \
    for (idx = 0; idx < recv_icount; idx++)                          \
    {                                                                \
        auto it = expected.find(rcv_idx[idx]);                       \
        if (it == expected.end())                                    \
        {                                                            \
            printf("iter %zu, idx %zu is not expected to be found\n",\
                    iter_idx, (size_t)rcv_idx[idx]);                 \
            ASSERT(0, "unexpected value");                           \
        }                                                            \
        else                                                         \
        {                                                            \
            std::vector<v_t> v(vb.begin() + VDIM_SIZE * idx,         \
                               vb.begin() + VDIM_SIZE * (idx + 1));  \
            if (v != it->second)                                     \
            {                                                        \
                ERROR(itype_name, vtype_name);                       \
            }                                                        \
        }                                                            \
    }                                                                \
}

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

template<ccl_datatype_t i_type, ccl_datatype_t v_type>
std::map<typename ccl::type_info<i_type>::native_type,
         std::vector<typename ccl::type_info<v_type>::native_type> >
get_expected_result(void* ibuffer, void* vbuffer)
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;

    i_t* ibuf = static_cast<i_t*>(ibuffer);
    v_t* vbuf = static_cast<v_t*>(vbuffer);

    void* recv_buf;
    size_t sum_nnz;
    size_t count = COUNT_I - rank;

    if (size > 1)
    {
        /* gather the number of non-zero (NNZ) values from all the ranks */
        std::vector<size_t> recv_counts(size, sizeof(size_t));

        size_t nnz = count;
        size_t recv_nnz[size];

        ccl_allgatherv(&nnz, sizeof(size_t),
                       recv_nnz, recv_counts.data(),
                       ccl_dtype_char,
                       &coll_attr,
                       nullptr, nullptr,
                       &request);
        ccl_wait(request);

        /* gather indices and values */
        memcpy(recv_counts.data(), recv_nnz, sizeof(size_t) * size);
        sum_nnz = 0;
        for (unsigned int i = 0; i < size; i++)
        {
            sum_nnz += recv_nnz[i];
            recv_counts[i] *= VDIM_SIZE;
        }

        recv_buf = malloc(sum_nnz * (sizeof(i_t) + VDIM_SIZE * sizeof(v_t)));

        ccl_allgatherv(ibuf, count,
                       recv_buf, recv_nnz,
                       i_type,
                       &coll_attr,
                       nullptr, nullptr,
                       &request);
        ccl_wait(request);

        ccl_allgatherv(vbuf, count * VDIM_SIZE,
                       ((char*)recv_buf + sum_nnz * sizeof(i_t)), recv_counts.data(),
                       v_type,
                       &coll_attr,
                       nullptr, nullptr,
                       &request);
        ccl_wait(request);
    }
    else
    {
        recv_buf = malloc(count * (sizeof(i_t) + VDIM_SIZE * sizeof(v_t)));
        memcpy(recv_buf, ibuffer, sizeof(i_t) * count);
        memcpy((char*)recv_buf + sizeof(i_t) * count, vbuffer, count * VDIM_SIZE * sizeof(v_t));
        sum_nnz = count;
    }
    

    /* calculate expected values */
    std::map<i_t, std::vector<v_t> > exp_vals;
    i_t* idx_buf = (i_t*)recv_buf;
    v_t* val_buf = (v_t*)((char*)recv_buf + sizeof(i_t) * sum_nnz);
    std::vector<v_t> tmp(VDIM_SIZE);
    for (unsigned int idx = 0; idx < sum_nnz; idx++)
    {
        auto it = exp_vals.find(idx_buf[idx]);
        if (it == exp_vals.end())
        {
           memcpy(tmp.data(), val_buf + idx * VDIM_SIZE, VDIM_SIZE * sizeof(v_t));
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
    free(recv_buf);

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
callback_fn(const void* i_buf, size_t i_cnt, ccl_datatype_t itype,
            const void* v_buf, size_t v_cnt, ccl_datatype_t vtype,
            const ccl_fn_context_t* fn_ctx, const void* user_ctx) 
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

#ifdef CCL_BFP16_COMPILER
ccl_status_t
callback_bfp16_fn(const void* i_buf, size_t i_cnt, ccl_datatype_t itype,
                  const void* v_buf, size_t v_cnt, ccl_datatype_t vtype,
                  const ccl_fn_context_t* context, const void* user_ctx) 
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

template<ccl_datatype_t i_type, ccl_datatype_t v_type, 
        typename std::enable_if< v_type == ccl_dtype_bfp16, int>::type = 0>
void sparse_test_run()
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
        i_t* send_ibuf = (i_t*)malloc(sizeof(i_t) * count);
        v_t* send_vbuf = (v_t*)malloc(sizeof(v_t) * count * VDIM_SIZE); 

        /* generate pseudo-random indices and calculate values */
        std::random_device seed;
        std::mt19937 gen(seed());
        std::uniform_int_distribution<> dist(0, RANGE);
        for (size_t i = 0; i < count; i++)
        {
            send_ibuf[i] = dist(gen);
            for (unsigned int j = 0; j < VDIM_SIZE; j++)
            {
                send_vbuf[i * VDIM_SIZE + j] = rank/(rank + 1 + j);
            }
        }

        auto expected = get_expected_result<i_type, ccl_dtype_float>(send_ibuf, send_vbuf);

        /* run sparse collective */
        coll_attr.to_cache = 0;
        coll_attr.sparse_allreduce_completion_fn = callback_bfp16_fn;
        coll_attr.sparse_allreduce_completion_ctx = nullptr;
        void* send_vbuf_bfp16 = malloc(sizeof(ccl::bfp16) * count * VDIM_SIZE);

        convert_fp32_to_bfp16_arrays(send_vbuf, send_vbuf_bfp16, count * VDIM_SIZE);

        recv_icount = 0;
        recv_vcount = 0;
        recv_ibuf = nullptr;
        recv_vbuf = nullptr;
        recv_vbuf_bfp16 = nullptr;

        RUN_COLLECTIVE(ccl_sparse_allreduce(send_ibuf, count,
                                            send_vbuf_bfp16, count * VDIM_SIZE,
                                            recv_ibuf, 0,
                                            recv_vbuf_bfp16, 0,
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
        CHECK_BFP16(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name());   

        free(send_ibuf);
        free(send_vbuf);
        free(send_vbuf_bfp16);
        
        free(recv_ibuf);
        free(recv_vbuf);
        free(recv_vbuf_bfp16);
    }
}
#endif  /* CCL_BFP16_COMPILER */

template<ccl_datatype_t i_type, ccl_datatype_t v_type, 
        typename std::enable_if< v_type != ccl_dtype_bfp16, int>::type = 0>
void sparse_test_run()
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;
    size_t count = COUNT_I - rank;
    i_t* send_ibuf = (i_t*)malloc(sizeof(i_t) * count);
    v_t* send_vbuf = (v_t*)malloc(sizeof(v_t) * count * VDIM_SIZE);

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

    auto expected = get_expected_result<i_type, v_type>(send_ibuf, send_vbuf);

    /* run sparse collective */
    coll_attr.to_cache = 0;
    coll_attr.sparse_allreduce_completion_fn = callback_fn;
    coll_attr.sparse_allreduce_completion_ctx = nullptr;

    recv_icount = 0;
    recv_vcount = 0;
    recv_ibuf = nullptr;
    recv_vbuf = nullptr;

    RUN_COLLECTIVE(ccl_sparse_allreduce(send_ibuf, count,
                                        send_vbuf, count * VDIM_SIZE,
                                        recv_ibuf, 0,
                                        recv_vbuf, 0,
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

    CHECK(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name());

    free(send_ibuf);
    free(send_vbuf);
        
    free(recv_ibuf);
    free(recv_vbuf);
}
#endif /* SPARSE_TEST_ALGO_HPP */
