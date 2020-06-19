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
#include <tuple>

#include "sparse_test_algo.hpp"

template<ccl_datatype_t ccl_idx_type>
struct sparse_algo_iterator
{
    template<size_t i, typename v_type>
    void invoke()
    {
        sparse_test_run<ccl_idx_type, v_type::ccl_type::value>();
    }
};

template<ccl_datatype_t ...ccl_value_type>
struct sparse_value_type_iterator
{
    using types = std::tuple<ccl::type_info<ccl_value_type>...>;
    template<size_t index, typename i_type>
    void invoke()
    {
        ccl_tuple_for_each_indexed<types>(sparse_algo_iterator<i_type::ccl_type::value>());
    }
};

template<ccl_datatype_t ...ccl_index_type>
struct sparce_index_types
{
    using types = std::tuple<ccl::type_info<ccl_index_type>...>;
};

int main()
{
    test_init();

    using supported_sparce_index_types = sparce_index_types<ccl_dtype_char,
                                                            ccl_dtype_int,
                                                            ccl_dtype_int64,
                                                            ccl_dtype_uint64>::types;
#ifdef CCL_BFP16_COMPILER
    using supported_sparce_value_types = sparse_value_type_iterator<ccl_dtype_char,
                                                            ccl_dtype_int,
                                                            ccl_dtype_bfp16,
                                                            ccl_dtype_float,
                                                            ccl_dtype_double,
                                                            ccl_dtype_int64,
                                                            ccl_dtype_uint64>;
#else
    using supported_sparce_value_types = sparse_value_type_iterator<ccl_dtype_char,
                                                            ccl_dtype_int,
                                                            ccl_dtype_float,
                                                            ccl_dtype_double,
                                                            ccl_dtype_int64,
                                                            ccl_dtype_uint64>;
#endif /* CCL_BFP16_COMPILER */

    // run test for each combination of supported indexes and values
    ccl_tuple_for_each_indexed<supported_sparce_index_types>(supported_sparce_value_types());

    test_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
