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

#pragma once

#include <iterator>
#include <sstream>

#include "coll/algorithms/algorithms.hpp"

template<typename algo_group_type>
struct ccl_algorithm_selector_helper
{
    static bool can_use(algo_group_type algo, const ccl_coll_param& param,
                        const ccl_selection_table_t<algo_group_type>& table);
    static const std::string& get_str_to_parse();
    static ccl_coll_type get_coll_id();
    static size_t get_count(const ccl_coll_param& param);
    static algo_group_type algo_from_str(const std::string& str);
    static const std::string& algo_to_str(algo_group_type algo);

    static std::map<algo_group_type, std::string> algo_names;
};

template<typename algo_group_type>
const std::string& ccl_coll_algorithm_to_str(algo_group_type algo)
{
    return ccl_algorithm_selector_helper<algo_group_type>::algo_to_str(algo);
}

#define CCL_SELECTION_DEFINE_HELPER_METHODS(algo_group_type, coll_id,                             \
                                            env_str, count_expr)                                  \
    template<>                                                                                    \
    const std::string& ccl_algorithm_selector_helper<algo_group_type>::get_str_to_parse()         \
    {                                                                                             \
        return env_str;                                                                           \
    }                                                                                             \
    template<>                                                                                    \
    ccl_coll_type ccl_algorithm_selector_helper<algo_group_type>::get_coll_id()                   \
    {                                                                                             \
        return coll_id;                                                                           \
    }                                                                                             \
    template<>                                                                                    \
    size_t ccl_algorithm_selector_helper<algo_group_type>::get_count(const ccl_coll_param& param) \
    {                                                                                             \
        return count_expr;                                                                        \
    }                                                                                             \
    template<> algo_group_type                                                                    \
    ccl_algorithm_selector_helper<algo_group_type>::algo_from_str(const std::string& str)         \
    {                                                                                             \
        for (const auto& name: algo_names)                                                        \
        {                                                                                         \
            if (!str.compare(name.second))                                                        \
            {                                                                                     \
                return name.first;                                                                \
            }                                                                                     \
        }                                                                                         \
        std::stringstream sstream;                                                                \
        std::for_each(algo_names.begin(), algo_names.end(),                                       \
            [&](const std::pair<algo_group_type, std::string>& p)                                 \
            { sstream << p.second << "\n"; });                                                    \
        CCL_THROW("unknown algorithm name '", str, "'\n",                                         \
                  "supported algorithms:\n", sstream.str());                                      \
    }                                                                                             \
    template<> const std::string&                                                                 \
    ccl_algorithm_selector_helper<algo_group_type>::algo_to_str(algo_group_type algo)             \
    {                                                                                             \
        auto it = algo_names.find(algo);                                                          \
        if (it != algo_names.end())                                                               \
            return it->second;                                                                    \
        static const std::string unknown("unknown");                                              \
        return unknown;                                                                           \
    }

template<>
std::map<ccl_coll_allgatherv_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_allgatherv_algo>::algo_names;

template<>
std::map<ccl_coll_allreduce_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_allreduce_algo>::algo_names;

template<>
std::map<ccl_coll_barrier_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_barrier_algo>::algo_names;

template<>
std::map<ccl_coll_bcast_algo,
        std::string> ccl_algorithm_selector_helper<ccl_coll_bcast_algo>::algo_names;

template<>
std::map<ccl_coll_reduce_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_reduce_algo>::algo_names;

template<>
std::map<ccl_coll_sparse_allreduce_algo,
         std::string> ccl_algorithm_selector_helper<ccl_coll_sparse_allreduce_algo>::algo_names;
