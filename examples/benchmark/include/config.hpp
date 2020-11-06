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

#define ALIGNMENT      (4096)
#define DTYPE          float

#define ALL_DTYPES_LIST     "char,int,float,double,int64_t,uint64_t"
#define ALL_REDUCTIONS_LIST "sum,prod,min,max"

#define DEFAULT_BACKEND         BACKEND_HOST
#define DEFAULT_LOOP            LOOP_REGULAR
#define DEFAULT_COLL_LIST \
    "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce," \
    "reduce_scatter,sparse_allreduce,sparse_allreduce_bf16," \
    "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce," \
    "reduce_scatter,sparse_allreduce,sparse_allreduce_bf16"
#define DEFAULT_ITERS           (16)
#define DEFAULT_WARMUP_ITERS    (16)
#define DEFAULT_BUF_COUNT       (16)
#define DEFAULT_MIN_ELEM_COUNT  (1)
#define DEFAULT_MAX_ELEM_COUNT  (128)
#define DEFAULT_CHECK_VALUES    (1)
#define DEFAULT_BUF_TYPE        BUF_MULTI
#define DEFAULT_V2I_RATIO       (128)
#define DEFAULT_DTYPES_LIST     "float"
#define DEFAULT_REDUCTIONS_LIST "sum"
#define DEFAULT_CSV_FILEPATH    ""
