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
std::map<ccl_coll_send_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_send_algo>::algo_names = {
        std::make_pair(ccl_coll_send_direct, "direct"),
        std::make_pair(ccl_coll_send_offload, "offload"),
#ifdef CCL_ENABLE_SYCL
        std::make_pair(ccl_coll_send_topo, "topo")
#endif // CCL_ENABLE_SYCL
    };

ccl_algorithm_selector<ccl_coll_send>::ccl_algorithm_selector() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_send_topo);
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_send_direct);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (!use_pt2pt_offload_algo()) {
        insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_send_direct);
    }
    else {
        insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_send_offload);
        if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
            set_offload_pt2pt_mpi_env();
        }
    }
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_send_direct);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    // send currently does not support scale-out selection, but the table
    // has to be defined, therefore duplicating main table
    scaleout_table = main_table;
    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_send_direct);
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_send_algo>::can_use(
    ccl_coll_send_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_send_algo>& table) {
    bool can_use = true;

    ccl_coll_algo algo_param;
    algo_param.send = algo;
    can_use = ccl_can_use_datatype(algo_param, param);

    if (algo == ccl_coll_send_topo && !ccl_can_use_topo_algo(param)) {
        can_use = false;
    }
    else if (algo == ccl_coll_send_offload && !use_pt2pt_offload_algo()) {
        can_use = false;
    }

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_send_algo,
                                    ccl_coll_send,
                                    ccl::global_data::env().send_algo_raw,
                                    param.count,
                                    ccl::global_data::env().send_scaleout_algo_raw);
