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
#include "coll/selection/selection.hpp"

template <>
std::map<ccl_coll_sparse_allreduce_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_sparse_allreduce_algo>::algo_names = {
        std::make_pair(ccl_coll_sparse_allreduce_ring, "ring"),
        std::make_pair(ccl_coll_sparse_allreduce_mask, "mask"),
        std::make_pair(ccl_coll_sparse_allreduce_3_allgatherv, "allgatherv")
    };

ccl_algorithm_selector<ccl_coll_sparse_allreduce>::ccl_algorithm_selector() {
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_sparse_allreduce_3_allgatherv);
        insert(
            fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_sparse_allreduce_3_allgatherv);
    }
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_sparse_allreduce_ring);
        insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_sparse_allreduce_ring);
    }
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_sparse_allreduce_algo>::is_direct(
    ccl_coll_sparse_allreduce_algo algo) {
    return false;
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_sparse_allreduce_algo>::can_use(
    ccl_coll_sparse_allreduce_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_sparse_allreduce_algo>& table) {
    CCL_THROW_IF_NOT(
        table.size() == 2,
        "sparse_allreduce doesn't support algorithm selection for multiple size ranges, ",
        " please specify the single algorithm for the whole range");

    bool can_use = true;

    if (ccl::global_data::env().atl_transport == ccl_atl_mpi &&
        algo != ccl_coll_sparse_allreduce_ring) {
        can_use = false;
    }
    else if (param.sparse_coalesce_mode == ccl_sparse_coalesce_disable &&
             algo != ccl_coll_sparse_allreduce_3_allgatherv) {
        can_use = false;
    }
    else if (param.sparse_allreduce_alloc_fn && algo != ccl_coll_sparse_allreduce_3_allgatherv) {
        can_use = false;
    }

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_sparse_allreduce_algo,
                                    ccl_coll_sparse_allreduce,
                                    ccl::global_data::env().sparse_allreduce_algo_raw,
                                    0);
