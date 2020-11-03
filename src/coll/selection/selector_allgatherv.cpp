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
std::map<ccl_coll_allgatherv_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_allgatherv_algo>::algo_names = {
        std::make_pair(ccl_coll_allgatherv_direct, "direct"),
        std::make_pair(ccl_coll_allgatherv_naive, "naive"),
        std::make_pair(ccl_coll_allgatherv_ring, "ring"),
        std::make_pair(ccl_coll_allgatherv_flat, "flat"),
        std::make_pair(ccl_coll_allgatherv_multi_bcast, "multi_bcast")
    };

ccl_algorithm_selector<ccl_coll_allgatherv>::ccl_algorithm_selector() {
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        insert(main_table, 0, CCL_ALLGATHERV_SHORT_MSG_SIZE, ccl_coll_allgatherv_naive);
        insert(main_table,
               CCL_ALLGATHERV_SHORT_MSG_SIZE + 1,
               CCL_SELECTION_MAX_COLL_SIZE,
               ccl_coll_allgatherv_ring);
    }
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allgatherv_direct);

    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allgatherv_flat);
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_allgatherv_algo>::is_direct(
    ccl_coll_allgatherv_algo algo) {
    return (algo == ccl_coll_allgatherv_direct) ? true : false;
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_allgatherv_algo>::can_use(
    ccl_coll_allgatherv_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_allgatherv_algo>& table) {
    bool can_use = true;

    if (param.vector_buf && algo != ccl_coll_allgatherv_flat &&
        algo != ccl_coll_allgatherv_multi_bcast)
        can_use = false;
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi &&
             algo == ccl_coll_allgatherv_multi_bcast)
        can_use = false;
    else if (algo == ccl_coll_allgatherv_direct &&
             (ccl::global_data::env().atl_transport == ccl_atl_ofi))
        can_use = false;

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_allgatherv_algo,
                                    ccl_coll_allgatherv,
                                    ccl::global_data::env().allgatherv_algo_raw,
                                    ({
                                        CCL_ASSERT(param.recv_counts);
                                        size_t count = 0;
                                        for (size_t idx = 0; idx < param.comm->size(); idx++) {
                                            count += param.recv_counts[idx];
                                        }
                                        count /= param.comm->size();
                                        count;
                                    }));
