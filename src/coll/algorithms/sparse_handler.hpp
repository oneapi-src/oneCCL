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

#include <unordered_map>

typedef std::unordered_map<size_t, size_t> idx_offset_map;

struct ccl_sparse_allreduce_handler
{
   size_t val_dim_cnt;
   size_t recv_buf_size;
   size_t send_buf_size;
   size_t itype_size;
   size_t vtype_size;
   size_t comm_size;
   size_t recv_count;
   size_t buf_size;

   size_t send_count[2];
   size_t dst_count[2];

   void* dst_buf;
   void* send_tmp_buf;
   void** recv_buf;
   void** recv_ibuf;
   void** recv_vbuf;
   size_t* recv_sizes;
   size_t* recv_icount;
   size_t* recv_vcount;
   size_t* size_per_rank;
   size_t* recv_counts;

   void* all_idx_buf;
   void* all_val_buf;

   ccl_datatype value_dtype;
   ccl_reduction_t op;

   std::unique_ptr<idx_offset_map> iv_map;
   ccl_sched* sched;
   ccl_comm* comm;
};

