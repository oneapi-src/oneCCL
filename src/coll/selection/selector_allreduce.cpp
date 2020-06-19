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
#include "exec/exec.hpp"

template<>
std::map<ccl_coll_allreduce_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_allreduce_algo>::algo_names =
  { 
    std::make_pair(ccl_coll_allreduce_direct, "direct"),
    std::make_pair(ccl_coll_allreduce_rabenseifner, "rabenseifner"),
    std::make_pair(ccl_coll_allreduce_starlike, "starlike"),
    std::make_pair(ccl_coll_allreduce_ring, "ring"),
    std::make_pair(ccl_coll_allreduce_ring_rma, "ring_rma"),
    std::make_pair(ccl_coll_allreduce_double_tree, "double_tree"),
    std::make_pair(ccl_coll_allreduce_recursive_doubling, "recursive_doubling"),
    std::make_pair(ccl_coll_allreduce_2d, "2d")
  };

ccl_algorithm_selector<ccl_coll_allreduce>::ccl_algorithm_selector()
{
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi)
    {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
        insert(main_table, 0, CCL_ALLREDUCE_SHORT_MSG_SIZE, ccl_coll_allreduce_recursive_doubling);
        insert(main_table, CCL_ALLREDUCE_SHORT_MSG_SIZE + 1, CCL_ALLREDUCE_MEDIUM_MSG_SIZE, ccl_coll_allreduce_starlike);
        insert(fallback_table, 0, CCL_ALLREDUCE_SHORT_MSG_SIZE, ccl_coll_allreduce_recursive_doubling);
    }
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_direct);

    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
}

template<>
bool ccl_algorithm_selector_helper<ccl_coll_allreduce_algo>::is_direct(ccl_coll_allreduce_algo algo)
{
    return (algo == ccl_coll_allreduce_direct) ? true : false;
}

template<>
bool ccl_algorithm_selector_helper<ccl_coll_allreduce_algo>::can_use(ccl_coll_allreduce_algo algo,
                                                              const ccl_selector_param& param,
                                                              const ccl_selection_table_t<ccl_coll_allreduce_algo>& table)
{
    bool can_use = true;

    if (algo == ccl_coll_allreduce_rabenseifner &&
        param.count < param.comm->pof2())
        can_use = false;
    else if (algo == ccl_coll_allreduce_ring_rma &&
             !ccl::global_data::get().executor->get_atl_attr().enable_rma)
        can_use = false;
    else if (algo == ccl_coll_allreduce_starlike &&
        !(param.count / param.comm->size()))
        can_use = false;
    else if (algo == ccl_coll_allreduce_2d &&
        (ccl::global_data::env().atl_transport == ccl_atl_mpi ||
         param.comm->size() != ccl::global_data::get().comm->size()))
        can_use = false;

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_allreduce_algo, ccl_coll_allreduce,
                                    ccl::global_data::env().allreduce_algo_raw, param.count);
