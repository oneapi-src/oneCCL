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

#include <vector>

#include "common/utils/enums.hpp"
#include "oneapi/ccl/types.hpp"

#define CCL_COLL_LIST \
    ccl_coll_allgatherv, ccl_coll_allreduce, ccl_coll_alltoall, ccl_coll_alltoallv, \
        ccl_coll_barrier, ccl_coll_bcast, ccl_coll_reduce, ccl_coll_reduce_scatter

enum ccl_coll_allgatherv_algo {
    ccl_coll_allgatherv_undefined = 0,

    ccl_coll_allgatherv_direct,
    ccl_coll_allgatherv_naive,
    ccl_coll_allgatherv_ring,
    ccl_coll_allgatherv_flat,
    ccl_coll_allgatherv_multi_bcast,
    ccl_coll_allgatherv_topo
};

enum ccl_coll_allreduce_algo {
    ccl_coll_allreduce_undefined = 0,

    ccl_coll_allreduce_direct,
    ccl_coll_allreduce_rabenseifner,
    ccl_coll_allreduce_nreduce,
    ccl_coll_allreduce_ring,
    ccl_coll_allreduce_ring_rma,
    ccl_coll_allreduce_double_tree,
    ccl_coll_allreduce_recursive_doubling,
    ccl_coll_allreduce_2d,
    ccl_coll_allreduce_topo
};

enum ccl_coll_alltoall_algo {
    ccl_coll_alltoall_undefined = 0,

    ccl_coll_alltoall_direct,
    ccl_coll_alltoall_naive,
    ccl_coll_alltoall_scatter,
    ccl_coll_alltoall_topo
};

enum ccl_coll_alltoallv_algo {
    ccl_coll_alltoallv_undefined = 0,

    ccl_coll_alltoallv_direct,
    ccl_coll_alltoallv_naive,
    ccl_coll_alltoallv_scatter,
    ccl_coll_alltoallv_topo
};

enum ccl_coll_barrier_algo {
    ccl_coll_barrier_undefined = 0,

    ccl_coll_barrier_direct,
    ccl_coll_barrier_ring
};

enum ccl_coll_bcast_algo {
    ccl_coll_bcast_undefined = 0,

    ccl_coll_bcast_direct,
    ccl_coll_bcast_ring,
    ccl_coll_bcast_double_tree,
    ccl_coll_bcast_naive,
    ccl_coll_bcast_topo
};

enum ccl_coll_reduce_algo {
    ccl_coll_reduce_undefined = 0,

    ccl_coll_reduce_direct,
    ccl_coll_reduce_rabenseifner,
    ccl_coll_reduce_ring,
    ccl_coll_reduce_tree,
    ccl_coll_reduce_double_tree,
    ccl_coll_reduce_topo
};

enum ccl_coll_reduce_scatter_algo {
    ccl_coll_reduce_scatter_undefined = 0,

    ccl_coll_reduce_scatter_direct,
    ccl_coll_reduce_scatter_ring,
    ccl_coll_reduce_scatter_topo
};

union ccl_coll_algo {
    ccl_coll_allgatherv_algo allgatherv;
    ccl_coll_allreduce_algo allreduce;
    ccl_coll_alltoall_algo alltoall;
    ccl_coll_alltoallv_algo alltoallv;
    ccl_coll_barrier_algo barrier;
    ccl_coll_bcast_algo bcast;
    ccl_coll_reduce_algo reduce;
    ccl_coll_reduce_scatter_algo reduce_scatter;
    int value;

    ccl_coll_algo() : value(0) {}
    bool has_value() const {
        return (value != 0);
    }
};

enum ccl_coll_type {
    ccl_coll_allgatherv,
    ccl_coll_allreduce,
    ccl_coll_alltoall,
    ccl_coll_alltoallv,
    ccl_coll_barrier,
    ccl_coll_bcast,
    ccl_coll_reduce,
    ccl_coll_reduce_scatter,
    ccl_coll_last_regular = ccl_coll_reduce_scatter,

    ccl_coll_partial,
    ccl_coll_undefined,

    ccl_coll_last_value
};

const char* ccl_coll_type_to_str(ccl_coll_type type);

void ccl_get_segment_sizes(size_t dtype_size,
                           size_t elem_count,
                           size_t requested_seg_size,
                           std::vector<size_t>& seg_sizes);
