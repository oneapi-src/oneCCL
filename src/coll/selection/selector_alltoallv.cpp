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

template<>
std::map<ccl_coll_alltoallv_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_alltoallv_algo>::algo_names =
  {
    std::make_pair(ccl_coll_alltoallv_direct, "direct"),
    std::make_pair(ccl_coll_alltoallv_naive, "naive")
  };

ccl_algorithm_selector<ccl_coll_alltoallv>::ccl_algorithm_selector()
{
    if (env_data.atl_transport == ccl_atl_ofi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_naive);
    else if (env_data.atl_transport == ccl_atl_mpi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_direct);

    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_naive);
}

template<>
bool ccl_algorithm_selector_helper<ccl_coll_alltoallv_algo>::is_direct(ccl_coll_alltoallv_algo algo)
{
    return (algo == ccl_coll_alltoallv_direct) ? true : false;
}

template<>
bool ccl_algorithm_selector_helper<ccl_coll_alltoallv_algo>::can_use(ccl_coll_alltoallv_algo algo,
                                                               const ccl_selector_param& param,
                                                               const ccl_selection_table_t<ccl_coll_alltoallv_algo>& table)
{
    return true;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_alltoallv_algo, ccl_coll_alltoallv,
                                    env_data.alltoallv_algo_raw, 0);
