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
#include <utility>
#include <unordered_map>
#include <random>

#include "base.h"
#include "ccl_type_traits.hpp"

#define COUNT_I 10
#define VDIM_SIZE 2
#define RANGE 10

#define RUN_COLLECTIVE(start_cmd, name, itype_name, vtype_name)            \
  do {                                                                     \
      t = 0;                                                               \
      for (iter_idx = 0; iter_idx < 1; iter_idx++)                         \
      {                                                                    \
          t1 = when();                                                     \
          CCL_CALL(start_cmd);                                             \
          CCL_CALL(ccl_wait(request));                                     \
          t2 = when();                                                     \
          t += (t2 - t1);                                                  \
          ccl_barrier(NULL, NULL);                                         \
                                                                           \
          i_t* rcv_idx = (i_t*)recv_ibuf;                                  \
          rcv_val = (v_t*)recv_vbuf;                                       \
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
                    std::string str = "[";                                 \
                    for (auto x : it->second)                              \
                        str += std::to_string(x) + ",";                    \
                    str[str.length()-1] = ']';                             \
                                                                           \
                    str += ", got [";                                      \
                    for (auto x : v)                                       \
                        str += std::to_string(x) + ",";                    \
                    str[str.length()-1] = ']';                             \
                    printf("iter %zu, idx %zu, expected %s (for i_type: %s, v_type: %s)\n", \
                            iter_idx, (size_t)rcv_idx[idx], str.c_str(),   \
                            itype_name, vtype_name);\
                    ASSERT(0, "unexpected value");                         \
                  }                                                        \
              }                                                            \
          }                                                                \
      }                                                                    \
      printf("[%zu] idx_type: %s, val_type: %s, avg %s time: %8.2lf us\n", \
             rank, itype_name, vtype_name,   \
             name, t / 1);                                                 \
      fflush(stdout);                                                      \
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
    ccl_allgatherv(vbuf, COUNT_I * VDIM_SIZE, ((char*)recv_buf + sum_nnz * sizeof(i_t)), recv_counts, v_type, &coll_attr, NULL, NULL, &request);
    ccl_wait(request);

    /*calculate expected values*/
    std::unordered_map<i_t, std::vector<v_t> > exp_vals;
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


template<ccl_datatype_t i_type, ccl_datatype_t v_type>
void sparse_test_run(const std::string& algo)
{
    using i_t = typename ccl::type_info<i_type>::native_type;
    using v_t = typename ccl::type_info<v_type>::native_type;
    i_t* send_ibuf = (i_t*)malloc(sizeof(i_t) * COUNT_I);
    v_t* send_vbuf = (v_t*)malloc(sizeof(v_t) * COUNT_I * VDIM_SIZE);
    void* recv_ibuf = malloc(COUNT_I * sizeof(i_t) + COUNT_I * VDIM_SIZE * sizeof(v_t));
    void* recv_vbuf = (char*)recv_ibuf + COUNT_I * sizeof(i_t);
    size_t recv_icount = 0;
    size_t recv_vcount = 0;

    /*generate pseudo-random indices and calculate values*/
    v_t* rcv_val = static_cast<v_t*>(recv_vbuf);

    std::random_device seed;
    std::default_random_engine gen(seed());
    std::uniform_int_distribution<i_t> dist(0, RANGE - 1);
    for (int i = 0; i < 10; i++)
    {
        send_ibuf[i] = dist(gen);
        for (unsigned int j = 0; j < VDIM_SIZE; j++)
        {
             send_vbuf[i * VDIM_SIZE + j] = (rank + 1 + j);
             rcv_val[i * VDIM_SIZE + j] = 0;
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

    free(send_ibuf);
    free(send_vbuf);
    free(recv_ibuf);
}

template<typename TupleType, typename FunctionType>
void ccl_tuple_for_each(TupleType&&, FunctionType,
                        std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>)
{}

template<std::size_t I, typename TupleType, typename FunctionType
       , typename = typename std::enable_if<I!=std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
void ccl_tuple_for_each(TupleType&& t, FunctionType f, std::integral_constant<size_t, I>)
{
    f(std::get<I>(std::forward<TupleType>(t)));
    ccl_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template<typename TupleType, typename FunctionType>
void ccl_tuple_for_each(TupleType&& t, FunctionType f)
{
    ccl_tuple_for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}


template<typename TupleType, typename FunctionType, class ...FunctionArgs>
void ccl_tuple_for_each_indexed(FunctionType,
                        std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>, const FunctionArgs&...args)
{}

template<typename TupleType, typename FunctionType, std::size_t I, class ...FunctionArgs,
       typename = typename std::enable_if<I!=std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
void ccl_tuple_for_each_indexed(FunctionType f, std::integral_constant<size_t, I>, const FunctionArgs& ...args)
{
    f.template invoke<I, typename std::tuple_element<I, TupleType>::type>(args...);
    ccl_tuple_for_each_indexed<TupleType, FunctionType>(f, std::integral_constant<size_t, I + 1>(), args...);
}

template<typename TupleType, typename FunctionType, class ...FunctionArgs>
void ccl_tuple_for_each_indexed(FunctionType f, const FunctionArgs& ...args)
{
    ccl_tuple_for_each_indexed<TupleType, FunctionType, 0, FunctionArgs...>(f, std::integral_constant<size_t, 0>(), args...);
}
