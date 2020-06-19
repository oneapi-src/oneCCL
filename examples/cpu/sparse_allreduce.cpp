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
    void invoke(ccl_sparse_coalesce_mode_t coalesce_mode,
                sparse_test_callback_mode_t callback_mode)
    {
        sparse_test_run<ccl_idx_type, v_type::ccl_type::value>(coalesce_mode, callback_mode);
    }
};

template<ccl_datatype_t ...ccl_value_type>
struct sparse_value_type_iterator
{
    using types = std::tuple<ccl::type_info<ccl_value_type>...>;
    template<size_t index, typename i_type>
    void invoke(ccl_sparse_coalesce_mode_t coalesce_mode,
                sparse_test_callback_mode_t callback_mode)
    {
        ccl_tuple_for_each_indexed<types>(sparse_algo_iterator<i_type::ccl_type::value>(),
                                          coalesce_mode, callback_mode);
    }
};

template<ccl_datatype_t ...ccl_index_type>
struct sparce_index_types
{
    using types = std::tuple<ccl::type_info<ccl_index_type>...>;
};

// use -f command line parameter to run example with flag=0 and then with flag=1
int main(int argc, char** argv)
{
    test_init();

    ccl_sparse_coalesce_mode_t coalesce_mode = ccl_sparse_coalesce_regular;
    sparse_test_callback_mode_t callback_mode = sparse_test_callback_completion;

    if (argc >= 3)
    {
        for (int i = 1; i < argc; i++)
        {
            if ((i + 1) >= argc)
                break;

            if (std::string(argv[i]) == "-coalesce")
            {
                if (std::string(argv[i + 1]) == "regular")
                    coalesce_mode = ccl_sparse_coalesce_regular;
                else if (std::string(argv[i + 1]) == "disable")
                    coalesce_mode = ccl_sparse_coalesce_disable;
                else if (std::string(argv[i + 1]) == "keep_precision")
                    coalesce_mode = ccl_sparse_coalesce_keep_precision;
                else
                {
                    printf("unexpected coalesce option '%s'\n", argv[i + 1]);
                    printf("FAILED\n");
                    return -1;
                }
                i++;
            }

            if (std::string(argv[i]) == "-callback")
            {
                if (std::string(argv[i + 1]) == "completion")
                    callback_mode = sparse_test_callback_completion;
                else if (std::string(argv[i + 1]) == "alloc")
                    callback_mode = sparse_test_callback_alloc;
                else
                {
                    printf("unexpected callback option '%s'\n", argv[i + 1]);
                    printf("FAILED\n");
                    return -1;
                }
                i++;
            }
        }
    }

    PRINT_BY_ROOT("\ncoalesce_mode = %d, callback_mode = %d\n", coalesce_mode, callback_mode);

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
    ccl_tuple_for_each_indexed<supported_sparce_index_types>(supported_sparce_value_types(),
                                                             coalesce_mode, callback_mode);

    test_finalize();

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}
