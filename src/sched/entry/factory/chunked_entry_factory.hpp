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

#include "common/env/env.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_CHUNKED_ENTRY_FUNCTION(entry_name, dtype, cnt, entry_expr) \
  do                                                                   \
  {                                                                    \
      LOG_DEBUG("creating chunked ", entry_name, " entry");            \
      size_t dtype_size = ccl_datatype_get_size(dtype);                \
      size_t bytes = cnt * dtype_size;                                 \
      size_t chunk_count =                                             \
          (bytes >= env_data.min_chunk_size &&                         \
           bytes >= env_data.chunk_count) ?                            \
              env_data.chunk_count : 1;                                \
      while ((chunk_count > 1) &&                                      \
             (bytes / chunk_count < env_data.min_chunk_size))          \
      {                                                                \
          chunk_count--;                                               \
      }                                                                \
      if (chunk_count == 0)                                            \
      {                                                                \
          LOG_ERROR("unexpected chunk_count");                         \
          chunk_count = 1;                                             \
      }                                                                \
      LOG_DEBUG("cnt ", cnt, ", chunk_count ", chunk_count);           \
      size_t main_chunk_size = cnt / chunk_count;                      \
      size_t last_chunk_size = main_chunk_size + cnt % chunk_count;    \
      size_t chunk_size, chunk_offset;                                 \
      for (size_t chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) \
      {                                                                \
          chunk_size = (chunk_idx == (chunk_count - 1)) ?              \
              last_chunk_size : main_chunk_size;                       \
          chunk_offset = chunk_idx * main_chunk_size * dtype_size;     \
          entry_expr;                                                  \
      }                                                                \
  } while (0)

namespace entry_factory
{
    void make_chunked_send_entry(ccl_sched* sched,
                                 const ccl_buffer buf,
                                 size_t cnt,
                                 ccl_datatype_internal_t dtype,
                                 size_t dst,
                                 ccl_comm* comm);

    void make_chunked_recv_entry(ccl_sched* sched,
                                 const ccl_buffer buf,
                                 size_t cnt,
                                 ccl_datatype_internal_t dtype,
                                 size_t src,
                                 ccl_comm* comm);

    void make_chunked_recv_reduce_entry(ccl_sched* sched,
                                        ccl_buffer inout_buf,
                                        size_t cnt,
                                        size_t* out_cnt,
                                        ccl_datatype_internal_t dtype,
                                        ccl_reduction_t reduction_op,
                                        size_t src,
                                        ccl_buffer comm_buf,
                                        ccl_comm* comm,
                                        ccl_recv_reduce_result_buf_type result_buf_type
                                            = ccl_recv_reduce_local_buf);
}
