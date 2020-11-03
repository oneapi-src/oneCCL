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

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll.hpp"
#include "common/global/global.hpp"

#include <map>
#include <string>

#define CCL_ALLGATHERV_SHORT_MSG_SIZE 32768
#define CCL_ALLREDUCE_SHORT_MSG_SIZE  8192
#define CCL_ALLREDUCE_MEDIUM_MSG_SIZE (1024 * 1024)
#define CCL_ALLTOALL_MEDIUM_MSG_SIZE  (1024 * 1024)
#define CCL_BCAST_SHORT_MSG_SIZE      8192
#define CCL_REDUCE_SHORT_MSG_SIZE     8192

enum ccl_selection_border_type {
    ccl_selection_border_left,
    ccl_selection_border_right,
    ccl_selection_border_both
};

struct ccl_selector_param {
    ccl_coll_type ctype;
    size_t count;
    ccl_datatype dtype;
    ccl_comm* comm;

    const size_t* send_counts;
    const size_t* recv_counts;
    int vector_buf;

    /* tmp fields to avoid selection of algorithms which don't support all coalesce modes or alloc_fn */
    ccl::sparse_coalesce_mode sparse_coalesce_mode;
    ccl::sparse_allreduce_alloc_fn sparse_allreduce_alloc_fn;
};

template <ccl_coll_type coll_id>
struct ccl_algorithm_selector;

template <typename algo_group_type>
using ccl_selection_table_t =
    std::map<size_t, std::pair<algo_group_type, ccl_selection_border_type>>;

template <typename algo_group_type>
using ccl_selection_table_iter_t = typename ccl_selection_table_t<algo_group_type>::const_iterator;

#define CCL_SELECTION_DECLARE_ALGO_SELECTOR_BASE() \
    template <typename algo_group_type> \
    struct ccl_algorithm_selector_base { \
        ccl_selection_table_t<algo_group_type> main_table{}; \
        ccl_selection_table_t<algo_group_type> fallback_table{}; \
        ccl_algorithm_selector_base(){}; \
        void init(); \
        void print() const; \
        algo_group_type get(const ccl_selector_param& param) const; \
        void insert(ccl_selection_table_t<algo_group_type>& table, \
                    size_t left, \
                    size_t right, \
                    algo_group_type algo_id); \
        bool is_direct(const ccl_selector_param& param) const; \
    };

#define CCL_SELECTION_DECLARE_ALGO_SELECTOR(coll_id, algo_group_type) \
    template <> \
    struct ccl_algorithm_selector<coll_id> : public ccl_algorithm_selector_base<algo_group_type> { \
        using type = algo_group_type; \
        ccl_algorithm_selector(); \
    };

CCL_SELECTION_DECLARE_ALGO_SELECTOR_BASE();

CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_allgatherv, ccl_coll_allgatherv_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_allreduce, ccl_coll_allreduce_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_alltoall, ccl_coll_alltoall_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_alltoallv, ccl_coll_alltoallv_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_barrier, ccl_coll_barrier_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_bcast, ccl_coll_bcast_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_reduce, ccl_coll_reduce_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_reduce_scatter, ccl_coll_reduce_scatter_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_sparse_allreduce, ccl_coll_sparse_allreduce_algo);

#include "coll/selection/selector_impl.hpp"
