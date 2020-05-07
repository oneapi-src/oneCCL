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
#include <iterator>
#include <random>
#include <unordered_map>
#include <utility>

#include "base.h"
#include "bfp16.h"
#include "ccl.hpp"

#define COUNT_I 1024
#define VDIM_SIZE 64
#define RANGE 255

#define ERROR(itype_name, vtype_name)                      \
({                                                         \
    std::string str = "[";                                 \
    for (auto x : it->second)                              \
        str += std::to_string(x) + ",";                    \
    str[str.length()-1] = ']';                             \
                                                           \
    str += ", got [";                                      \
    for (auto x : v)                                       \
        str += std::to_string(x) + ",";                    \
    str[str.length()-1] = ']';                             \
    printf("rank [%zu]: idx %zu, expected %s (for i_type: %s, v_type: %s)\n",\
            rank, (size_t)rcv_idx[idx], str.c_str(),       \
            itype_name, vtype_name);                       \
    ASSERT(0, "unexpected value");                         \
})

#define CHECK_BFP16(itype_name, vtype_name)                          \
{                                                                    \
    /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */            \
    double max_error = 0;                                            \
    double log_base2 = log(size) / log(2);                           \
    double g = (log_base2 * BFP16_PRECISION)/(1 - (log_base2 * BFP16_PRECISION));\
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
            vb.begin() + VDIM_SIZE * (idx + 1));                     \
            if (v != it->second)                                     \
            {                                                        \
                ERROR(itype_name, vtype_name);                       \
            }                                                        \
        }                                                            \
    }                                                                \
}

#define RUN_COLLECTIVE(start_cmd, name, itype_name, vtype_name)           \
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
      printf("[%zu] idx_type: %s, val_type: %s, avg %s time: %8.2lf us\n",\
             rank, itype_name, vtype_name,                                \
             name, t / 1);                                                \
      fflush(stdout);                                                     \
  } while (0)

template<ccl_datatype_t i_type, ccl_datatype_t v_type>
std::unordered_map<typename ccl::type_info<i_type>::native_type, std::vector<typename ccl::type_info<v_type>::native_type> > prep_data(void* ibuffer, void* vbuffer)
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;

    i_t* ibuf = static_cast<i_t*>(ibuffer);
    v_t* vbuf = static_cast<v_t*>(vbuffer);

    /*gather the number of non-zero (NNZ) values from all the ranks*/
    size_t recv_counts[size];
    for (unsigned int i = 0; i < size; i++)
    {
        recv_counts[i] = sizeof(size_t);
    }

    size_t nnz = COUNT_I;
    size_t recv_nnz[size];

    ccl_allgatherv(&nnz, sizeof(size_t), recv_nnz, recv_counts, ccl_dtype_char, &coll_attr, NULL, NULL, &request);
    ccl_wait(request);

    /*gather indices and values*/
    memcpy(recv_counts, recv_nnz, sizeof(size_t) * size);
    size_t sum_nnz = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        sum_nnz += recv_nnz[i];
        recv_counts[i] *= VDIM_SIZE;
    }

    void* recv_buf = malloc(sum_nnz * (sizeof(i_t) + VDIM_SIZE * sizeof(v_t)));

    ccl_allgatherv(ibuf, COUNT_I, recv_buf, recv_nnz, i_type, &coll_attr, NULL, NULL, &request);
    ccl_wait(request);
    ccl_allgatherv(vbuf, COUNT_I * VDIM_SIZE, ((char*)recv_buf + sum_nnz * sizeof(i_t)), recv_counts, v_type, &coll_attr, NULL, NULL, &request);
    ccl_wait(request);

    /*calculate expected values*/
    std::unordered_map<i_t, std::vector<v_t> > exp_vals;
    i_t* idx_buf = (i_t*)recv_buf;
    v_t* val_buf = (v_t*)((char*)recv_buf + sizeof(i_t) * sum_nnz);
    std::vector<v_t> tmp(VDIM_SIZE);
    size_t out_cnt = 0;
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

#ifdef CCL_BFP16_COMPILER
template<ccl_datatype_t i_type, ccl_datatype_t v_type, 
        typename std::enable_if< v_type == ccl_dtype_bfp16, int>::type = 0>
void sparse_test_run(const std::string& algo)
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
        i_t* send_ibuf = (i_t*)malloc(sizeof(i_t) * COUNT_I);
        v_t* send_vbuf = (v_t*)malloc(sizeof(v_t) * COUNT_I * VDIM_SIZE);
        void* recv_ibuf = calloc(COUNT_I, sizeof(i_t));
        void* recv_vbuf = calloc(COUNT_I * VDIM_SIZE, sizeof(v_t));    
        size_t recv_icount = COUNT_I;
        size_t recv_vcount = COUNT_I * VDIM_SIZE;

        /*generate pseudo-random indices and calculate values*/
        std::random_device seed;
        std::mt19937 gen(seed());
        std::uniform_int_distribution<> dist(0, RANGE);
        for (int i = 0; i < COUNT_I; i++)
        {
            send_ibuf[i] = dist(gen);
            for (unsigned int j = 0; j < VDIM_SIZE; j++)
            {
                send_vbuf[i * VDIM_SIZE + j] = rank/(rank + 1 + j);
            }
        }

        auto expected = prep_data<i_type, ccl_dtype_float>(send_ibuf, send_vbuf);

        /*run sparse collective*/
        coll_attr.to_cache = 0;

        void* recv_vbuf_bfp16 = malloc(sizeof(ccl::bfp16) * COUNT_I * VDIM_SIZE);
        void* send_vbuf_bfp16 = calloc(COUNT_I * VDIM_SIZE, sizeof(ccl::bfp16));

        convert_fp32_to_bfp16_arrays(send_vbuf, send_vbuf_bfp16, COUNT_I * VDIM_SIZE);
        RUN_COLLECTIVE(ccl_sparse_allreduce(send_ibuf, COUNT_I, send_vbuf_bfp16, COUNT_I * VDIM_SIZE,
                                            &recv_ibuf, &recv_icount, &recv_vbuf_bfp16, &recv_vcount,
                                            i_type, v_type, ccl_reduction_sum,
                                            &coll_attr, NULL, NULL, &request),
                       (algo + "_sparse_allreduce").c_str(), ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name());
        if (COUNT_I * VDIM_SIZE < recv_vcount)
        {
            recv_vbuf = realloc(recv_vbuf, sizeof(v_t) * recv_vcount); 
        }

        convert_bfp16_to_fp32_arrays(recv_vbuf_bfp16, (float*)recv_vbuf, (int)recv_vcount);
        CHECK_BFP16(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name());   

        free(send_vbuf_bfp16);
        free(recv_vbuf_bfp16);
        free(send_ibuf);
        free(send_vbuf);
        free(recv_ibuf);
        free(recv_vbuf);
    }
}
#endif  /* CCL_BFP16_COMPILER */

template<ccl_datatype_t i_type, ccl_datatype_t v_type, 
        typename std::enable_if< v_type != ccl_dtype_bfp16, int>::type = 0>
void sparse_test_run(const std::string& algo)
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;
    i_t* send_ibuf = (i_t*)malloc(sizeof(i_t) * COUNT_I);
    v_t* send_vbuf = (v_t*)malloc(sizeof(v_t) * COUNT_I * VDIM_SIZE);
    void* recv_ibuf = calloc(COUNT_I, sizeof(i_t));
    void* recv_vbuf = calloc(COUNT_I * VDIM_SIZE, sizeof(v_t));    
    size_t recv_icount = COUNT_I;
    size_t recv_vcount = COUNT_I * VDIM_SIZE;

    /*generate pseudo-random indices and calculate values*/
    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_int_distribution<> dist(0, RANGE);
    for (int i = 0; i < COUNT_I; i++)
    {
        send_ibuf[i] = dist(gen);
        for (unsigned int j = 0; j < VDIM_SIZE; j++)
        {
            send_vbuf[i * VDIM_SIZE + j] = (rank + 1 + j);
        }
    }

    auto expected = prep_data<i_type, v_type>(send_ibuf, send_vbuf);

    /*run sparse collective*/
    coll_attr.to_cache = 0;
    RUN_COLLECTIVE(ccl_sparse_allreduce(send_ibuf, COUNT_I, send_vbuf, COUNT_I * VDIM_SIZE,
                                        &recv_ibuf, &recv_icount, &recv_vbuf, &recv_vcount,
                                        i_type, v_type, ccl_reduction_sum,
                                        &coll_attr, NULL, NULL, &request),
                   (algo + "_sparse_allreduce").c_str(), ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name());
    CHECK(ccl::type_info<i_type>::name(), ccl::type_info<v_type>::name());

    free(send_ibuf);
    free(send_vbuf);
    free(recv_ibuf);
    free(recv_vbuf);
}


template<class specific_tuple, class functor, size_t cur_index>
void ccl_tuple_for_each_impl(specific_tuple&& t, functor f, 
                             std::true_type tuple_finished)
{
    // nothing to do
}

template<class specific_tuple, class functor, size_t cur_index>
void ccl_tuple_for_each_impl(specific_tuple&& t, functor f, 
                             std::false_type tuple_not_finished)
{
    f(std::get<cur_index>(std::forward<specific_tuple>(t)));

    constexpr std::size_t tuple_size = 
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = 
        std::integral_constant<bool, cur_index + 1>= tuple_size>;

    ccl_tuple_for_each_impl<specific_tuple,
                            functor,
                            cur_index + 1>(std::forward<specific_tuple>(t), f, 
                                           is_tuple_finished_t{});
}

template<class specific_tuple, class functor, size_t cur_index = 0>
void ccl_tuple_for_each(specific_tuple&& t, functor f)
{
    constexpr std::size_t tuple_size = 
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::integral_constant<bool,
                                                       cur_index >= tuple_size>;
    ccl_tuple_for_each_impl<specific_tuple, 
                            functor,
                            cur_index>(std::forward<specific_tuple>(t), f,
                                       is_tuple_finished_t{});
}


template<typename specific_tuple, size_t cur_index, 
         typename functor, class ...FunctionArgs>
void ccl_tuple_for_each_indexed_impl(functor, std::true_type tuple_finished, 
                                     const FunctionArgs&...args)
{}

template<typename specific_tuple, size_t cur_index,
         typename functor, class ...FunctionArgs>
void ccl_tuple_for_each_indexed_impl(functor f, std::false_type tuple_not_finished, 
                                     const FunctionArgs& ...args)
{
    using tuple_element_t = typename std::tuple_element<cur_index, specific_tuple>::type;

    f.template invoke<cur_index, tuple_element_t>(args...);

    constexpr std::size_t tuple_size = 
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;

    using is_tuple_finished_t = 
        std::integral_constant<bool, cur_index + 1 >= tuple_size>;

    ccl_tuple_for_each_indexed_impl<specific_tuple, 
                                    cur_index + 1,
                                    functor>(f, is_tuple_finished_t{}, args...);
}

template<typename specific_tuple, typename functor, class ...FunctionArgs>
void ccl_tuple_for_each_indexed(functor f, const FunctionArgs& ...args)
{
    constexpr std::size_t tuple_size = 
        std::tuple_size<typename std::remove_reference<specific_tuple>::type>::value;
    static_assert(tuple_size != 0, "Nothing to do, tuple is empty");

    using is_tuple_finished_t = std::false_type; //non-empty tuple started
    ccl_tuple_for_each_indexed_impl<specific_tuple,
                                    0, functor,
                                    FunctionArgs...>(f, is_tuple_finished_t{}, args...);
}
