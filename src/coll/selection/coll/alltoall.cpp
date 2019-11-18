/*
 Copyright 2016-2019 Intel Corporation
 
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

template<>
std::map<ccl_coll_alltoall_algo,
    std::string> ccl_algorithm_selector_helper<ccl_coll_alltoall_algo>::algo_names =
    {
        std::make_pair(ccl_coll_alltoall_direct, "direct"),
        std::make_pair(ccl_coll_alltoall_scatter, "scatter"),
        std::make_pair(ccl_coll_alltoall_scatter_message, "scatter_message"),
    };

ccl_algorithm_selector<ccl_coll_alltoall>::ccl_algorithm_selector()
{
    if (env_data.atl_transport == ccl_atl_ofi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoall_scatter);
    else if (env_data.atl_transport == ccl_atl_mpi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoall_direct);

    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoall_scatter);
}

template<>
bool ccl_algorithm_selector_helper<ccl_coll_alltoall_algo>::can_use(ccl_coll_alltoall_algo algo,
                                                                    const ccl_coll_param& param,
                                                                    const ccl_selection_table_t<ccl_coll_alltoall_algo>& table)
{
    bool can_use = true;

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_alltoall_algo, ccl_coll_alltoall, env_data.alltoall_algo_raw, param.count);
