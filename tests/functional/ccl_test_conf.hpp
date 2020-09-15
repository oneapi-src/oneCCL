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
#include <map>
#include <vector>

#define POST_AND_PRE_INCREMENTS(EnumName, LAST_ELEM) \
    EnumName& operator++(EnumName& orig) { \
        if (orig != LAST_ELEM) \
            orig = static_cast<EnumName>(orig + 1); \
        return orig; \
    } \
    EnumName operator++(EnumName& orig, int) { \
        EnumName rVal = orig; \
        ++orig; \
        return rVal; \
    }

#define GET_ELEMENT_BEFORE_LAST(EnumName, LAST_ELEM) \
    EnumName& operator++(EnumName& orig) { \
        if (orig != LAST_ELEM) \
            orig = static_cast<EnumName>(orig + 1); \
        return orig; \
    } \
    EnumName operator++(EnumName& orig, int) { \
        EnumName rVal = orig; \
        ++orig; \
        return rVal; \
    }

#define SOME_VALUE (0xdeadbeef)

#define ROOT_PROCESS_IDX (0)

typedef enum { PT_OOP = 0, PT_IN = 1, PT_LAST } ccl_place_type;
ccl_place_type first_ccl_place_type = PT_OOP;
ccl_place_type last_ccl_place_type = PT_LAST;

std::map<int, const char*> ccl_place_type_str = { { PT_OOP, "PT_OOP" }, { PT_IN, "PT_IN" } };

typedef enum { ST_SMALL = 0, ST_MEDIUM = 1, ST_LARGE = 2, ST_LAST } ccl_size_type;
ccl_size_type first_ccl_size_type = ST_SMALL;
ccl_size_type last_ccl_size_type = ST_LAST;

std::map<int, const char*> ccl_size_type_str = { { ST_SMALL, "ST_SMALL" },
                                                 { ST_MEDIUM, "ST_MEDIUM" },
                                                 { ST_LARGE, "ST_LARGE" } };

std::map<int, size_t> ccl_size_type_values = { { ST_SMALL, 16 },
                                               { ST_MEDIUM, 32769 },
                                               { ST_LARGE, 524288 } };

typedef enum { BC_SMALL = 0, BC_MEDIUM = 1, BC_LARGE = 2, BC_LAST } ccl_buffer_count;
ccl_buffer_count first_ccl_buffer_count = BC_SMALL;
ccl_buffer_count last_ccl_buffer_count = BC_LAST;

std::map<int, const char*> ccl_buffer_count_str = { { BC_SMALL, "BC_SMALL" },
                                                    { BC_MEDIUM, "BC_MEDIUM" },
                                                    { BC_LARGE, "BC_LARGE" } };

std::map<int, size_t> ccl_buffer_count_values = { { BC_SMALL, 1 },
                                                  { BC_MEDIUM, 2 },
                                                  { BC_LARGE, 4 } };

typedef enum { CMPT_WAIT = 0, CMPT_TEST = 1, CMPT_LAST } ccl_completion_type;
ccl_completion_type first_ccl_completion_type = CMPT_WAIT;
ccl_completion_type last_ccl_completion_type = CMPT_LAST;

std::map<int, const char*> ccl_completion_type_str = { { CMPT_WAIT, "CMPT_WAIT" },
                                                       { CMPT_TEST, "CMPT_TEST" } };

typedef enum {
    PTYPE_NULL = 0,
#ifdef TEST_CCL_CUSTOM_PROLOG
    PTYPE_T_TO_2X = 1,
    PTYPE_T_TO_CHAR = 2,
#endif
    PTYPE_LAST
} ccl_prolog_type;
ccl_prolog_type first_ccl_prolog_type = PTYPE_NULL;
ccl_prolog_type last_ccl_prolog_type = PTYPE_LAST;

std::map<int, const char*> ccl_prolog_type_str = { { PTYPE_NULL, "PTYPE_NULL" },
#ifdef TEST_CCL_CUSTOM_PROLOG
                                                   { PTYPE_T_TO_2X, "PTYPE_T_TO_2X" },
                                                   { PTYPE_T_TO_CHAR, "PTYPE_T_TO_CHAR" }
#endif
};

typedef enum {
    ETYPE_NULL = 0,
#ifdef TEST_CCL_CUSTOM_EPILOG
    ETYPE_T_TO_2X = 1,
    ETYPE_CHAR_TO_T = 2,
#endif
    ETYPE_LAST
} ccl_epilog_type;
ccl_epilog_type first_ccl_epilog_type = ETYPE_NULL;
ccl_epilog_type last_ccl_epilog_type = ETYPE_LAST;

std::map<int, const char*> ccl_epilog_type_str = { { ETYPE_NULL, "ETYPE_NULL" },
#ifdef TEST_CCL_CUSTOM_EPILOG
                                                   { ETYPE_T_TO_2X, "ETYPE_T_TO_2X" },
                                                   { ETYPE_CHAR_TO_T, "ETYPE_CHAR_TO_T" }
#endif
};

typedef enum {
    DT_CHAR = ccl_dtype_char,
    DT_INT = ccl_dtype_int,
    DT_BFP16 = ccl_dtype_bfp16,
    DT_FLOAT = ccl_dtype_float,
    DT_DOUBLE = ccl_dtype_double,
    // DT_INT64 = ccl_dtype_int64,
    // DT_UINT64 = ccl_dtype_uint64,
    DT_LAST
} ccl_data_type;
ccl_data_type first_ccl_data_type = DT_CHAR;
ccl_data_type last_ccl_data_type = DT_LAST;

std::map<int, const char*> ccl_data_type_str = {
    { DT_CHAR, "DT_CHAR" },
    { DT_INT, "DT_INT" },
    { DT_BFP16, "DT_BFP16" },
    { DT_FLOAT, "DT_FLOAT" },
    { DT_DOUBLE, "DT_DOUBLE" }
    // { DT_INT64, "INT64" },
    // { DT_UINT64, "UINT64" }
};

typedef enum {
    RT_SUM = 0,
#ifdef TEST_CCL_REDUCE
    RT_PROD = 1,
    RT_MIN = 2,
    RT_MAX = 3,
#ifdef TEST_CCL_CUSTOM_REDUCE
    RT_CUSTOM = 4,
    RT_CUSTOM_NULL = 5,
#endif
#endif
    RT_LAST
} ccl_reduction_type;
ccl_reduction_type first_ccl_reduction_type = RT_SUM;
ccl_reduction_type last_ccl_reduction_type = RT_LAST;

std::map<int, const char*> ccl_reduction_type_str = {
    { RT_SUM, "RT_SUM" },
#ifdef TEST_CCL_REDUCE
    { RT_PROD, "RT_PROD" },     { RT_MIN, "RT_MIN" },
    { RT_MAX, "RT_MAX" },
#ifdef TEST_CCL_CUSTOM_REDUCE
    { RT_CUSTOM, "RT_CUSTOM" }, { RT_CUSTOM_NULL, "RT_CUSTOM_NULL" }
#endif
#endif
};

std::map<int, ccl_reduction_t> ccl_reduction_type_values = {
    { RT_SUM, ccl_reduction_sum },
#ifdef TEST_CCL_REDUCE
    { RT_PROD, ccl_reduction_prod },     { RT_MIN, ccl_reduction_min },
    { RT_MAX, ccl_reduction_max },
#ifdef TEST_CCL_CUSTOM_REDUCE
    { RT_CUSTOM, ccl_reduction_custom }, { RT_CUSTOM_NULL, ccl_reduction_custom }
#endif
#endif
};

typedef enum { CT_CACHE_0 = 0, CT_CACHE_1 = 1, CT_LAST } ccl_cache_type;
ccl_cache_type first_ccl_cache_type = CT_CACHE_0;
ccl_cache_type last_ccl_cache_type = CT_LAST;

std::map<int, const char*> ccl_cache_type_str = { { CT_CACHE_0, "CT_CACHE_0" },
                                                  { CT_CACHE_1, "CT_CACHE_1" } };

std::map<int, int> ccl_cache_type_values = { { CT_CACHE_0, 0 }, { CT_CACHE_1, 1 } };

typedef enum { SNCT_SYNC_0 = 0, SNCT_SYNC_1 = 1, SNCT_LAST } ccl_sync_type;
ccl_sync_type first_ccl_sync_type = SNCT_SYNC_0;
ccl_sync_type last_ccl_sync_type = SNCT_LAST;

std::map<int, const char*> ccl_sync_type_str = { { SNCT_SYNC_0, "SNCT_SYNC_0" },
                                                 { SNCT_SYNC_1, "SNCT_SYNC_1" } };

std::map<int, int> ccl_sync_type_values = { { SNCT_SYNC_0, 0 }, { SNCT_SYNC_1, 1 } };

typedef enum {
    ORDER_DISABLE = 0,
    ORDER_DIRECT = 1,
    ORDER_INDIRECT = 2,
    ORDER_RANDOM = 3,
    ORDER_LAST
} ccl_order_type;
ccl_order_type first_ccl_order_type = ORDER_DISABLE;
ccl_order_type last_ccl_order_type = ORDER_LAST;

std::map<int, const char*> ccl_order_type_str = { { ORDER_DISABLE, "ORDER_DISABLE" },
                                                  { ORDER_DIRECT, "ORDER_DIRECT" },
                                                  { ORDER_INDIRECT, "ORDER_INDIRECT" },
                                                  { ORDER_RANDOM, "ORDER_RANDOM" } };

std::map<int, int> ccl_order_type_values = { { ORDER_DISABLE, 0 },
                                             { ORDER_DIRECT, 1 },
                                             { ORDER_INDIRECT, 2 },
                                             { ORDER_RANDOM, 3 } };

POST_AND_PRE_INCREMENTS(ccl_place_type, PT_LAST);
POST_AND_PRE_INCREMENTS(ccl_size_type, ST_LAST);
POST_AND_PRE_INCREMENTS(ccl_completion_type, CMPT_LAST);
POST_AND_PRE_INCREMENTS(ccl_data_type, DT_LAST);
POST_AND_PRE_INCREMENTS(ccl_reduction_type, RT_LAST);
POST_AND_PRE_INCREMENTS(ccl_cache_type, CT_LAST);
POST_AND_PRE_INCREMENTS(ccl_sync_type, SNCT_LAST);
POST_AND_PRE_INCREMENTS(ccl_order_type, ORDER_LAST);
POST_AND_PRE_INCREMENTS(ccl_buffer_count, BC_LAST);
POST_AND_PRE_INCREMENTS(ccl_prolog_type, PTYPE_LAST);
POST_AND_PRE_INCREMENTS(ccl_epilog_type, ETYPE_LAST);

struct ccl_test_conf {
    ccl_place_type place_type;
    ccl_cache_type cache_type;
    ccl_sync_type sync_type;
    ccl_size_type size_type;
    ccl_completion_type completion_type;
    ccl_reduction_type reduction_type;
    ccl_data_type data_type;
    ccl_order_type complete_order_type;
    ccl_order_type start_order_type;
    ccl_buffer_count buffer_count;
    ccl_prolog_type prolog_type;
    ccl_epilog_type epilog_type;
};

size_t get_ccl_elem_count(ccl_test_conf& test_conf) {
    return ccl_size_type_values[test_conf.size_type];
}

size_t get_ccl_buffer_count(ccl_test_conf& test_conf) {
    return ccl_buffer_count_values[test_conf.buffer_count];
}

ccl_reduction_t get_ccl_lib_reduction_type(const ccl_test_conf& test_conf) {
    return ccl_reduction_type_values[test_conf.reduction_type];
}

size_t calculate_test_count() {
    size_t test_count = ORDER_LAST * ORDER_LAST * CMPT_LAST * SNCT_LAST * DT_LAST * ST_LAST *
                        RT_LAST * BC_LAST * CT_LAST * PT_LAST * PTYPE_LAST * ETYPE_LAST;

    // CCL_TEST_EPILOG_TYPE=0 CCL_TEST_PROLOG_TYPE=0 CCL_TEST_PLACE_TYPE=0 CCL_TEST_CACHE_TYPE=0 CCL_TEST_BUFFER_COUNT=0 CCL_TEST_SIZE_TYPE=0 CCL_TEST_PRIORITY_TYPE=1 CCL_TEST_COMPLETION_TYPE=0 CCL_TEST_SYNC_TYPE=0 CCL_TEST_REDUCTION_TYPE=0 CCL_TEST_DATA_TYPE=0
    char* test_data_type_enabled = getenv("CCL_TEST_DATA_TYPE");
    char* test_reduction_enabled = getenv("CCL_TEST_REDUCTION_TYPE");
    char* test_sync_enabled = getenv("CCL_TEST_SYNC_TYPE");
    char* test_completion_enabled = getenv("CCL_TEST_COMPLETION_TYPE");
    char* test_order_type_enabled = getenv("CCL_TEST_PRIORITY_TYPE");
    char* test_size_type_enabled = getenv("CCL_TEST_SIZE_TYPE");
    char* test_buffer_count_enabled = getenv("CCL_TEST_BUFFER_COUNT");
    char* test_cache_type_enabled = getenv("CCL_TEST_CACHE_TYPE");
    char* test_place_type_enabled = getenv("CCL_TEST_PLACE_TYPE");
    char* test_prolog_enabled = getenv("CCL_TEST_PROLOG_TYPE");
    char* test_epilog_enabled = getenv("CCL_TEST_EPILOG_TYPE");

    if (test_data_type_enabled && atoi(test_data_type_enabled) == 0) {
        test_count /= last_ccl_data_type;
        first_ccl_data_type = static_cast<ccl_data_type>(DT_FLOAT);
        last_ccl_data_type = static_cast<ccl_data_type>(first_ccl_data_type + 1);
    }

    if (test_reduction_enabled && atoi(test_reduction_enabled) == 0) {
        test_count /= last_ccl_reduction_type;
        first_ccl_reduction_type = static_cast<ccl_reduction_type>(RT_SUM);
        last_ccl_reduction_type = static_cast<ccl_reduction_type>(first_ccl_reduction_type + 1);
    }

    if (test_sync_enabled && atoi(test_sync_enabled) == 0) {
        test_count /= last_ccl_sync_type;
        first_ccl_sync_type = static_cast<ccl_sync_type>(SNCT_SYNC_1);
        last_ccl_sync_type = static_cast<ccl_sync_type>(first_ccl_sync_type + 1);
    }

    if (test_completion_enabled && atoi(test_completion_enabled) == 0) {
        test_count /= last_ccl_completion_type;
        first_ccl_completion_type = static_cast<ccl_completion_type>(CMPT_WAIT);
        last_ccl_completion_type = static_cast<ccl_completion_type>(first_ccl_completion_type + 1);
    }

    if (test_order_type_enabled && atoi(test_order_type_enabled) == 0) {
        test_count /= (last_ccl_order_type * last_ccl_order_type);
        first_ccl_order_type = static_cast<ccl_order_type>(ORDER_DISABLE);
        last_ccl_order_type = static_cast<ccl_order_type>(first_ccl_order_type + 1);
    }

    if (test_size_type_enabled && atoi(test_size_type_enabled) == 0) {
        test_count /= last_ccl_size_type;
        first_ccl_size_type = static_cast<ccl_size_type>(ST_MEDIUM);
        last_ccl_size_type = static_cast<ccl_size_type>(first_ccl_size_type + 1);
    }

    if (test_buffer_count_enabled && atoi(test_buffer_count_enabled) == 0) {
        test_count /= last_ccl_buffer_count;
        first_ccl_buffer_count = static_cast<ccl_buffer_count>(BC_MEDIUM);
        last_ccl_buffer_count = static_cast<ccl_buffer_count>(first_ccl_buffer_count + 1);
    }

    if (test_cache_type_enabled && atoi(test_cache_type_enabled) == 0) {
        test_count /= last_ccl_cache_type;
        first_ccl_cache_type = static_cast<ccl_cache_type>(CT_CACHE_1);
        last_ccl_cache_type = static_cast<ccl_cache_type>(first_ccl_cache_type + 1);
    }

    if (test_place_type_enabled && atoi(test_place_type_enabled) == 0) {
        test_count /= last_ccl_place_type;
        first_ccl_place_type = static_cast<ccl_place_type>(PT_IN);
        last_ccl_place_type = static_cast<ccl_place_type>(first_ccl_place_type + 1);
    }

    if (test_prolog_enabled && atoi(test_prolog_enabled) == 0) {
        test_count /= last_ccl_prolog_type;
        first_ccl_prolog_type = static_cast<ccl_prolog_type>(PTYPE_NULL);
        last_ccl_prolog_type = static_cast<ccl_prolog_type>(first_ccl_prolog_type + 1);
    }

    if (test_epilog_enabled && atoi(test_epilog_enabled) == 0) {
        test_count /= last_ccl_epilog_type;
        first_ccl_epilog_type = static_cast<ccl_epilog_type>(ETYPE_NULL);
        last_ccl_epilog_type = static_cast<ccl_epilog_type>(first_ccl_epilog_type + 1);
    }

    return test_count;
}

int is_bfp16_enabled() {
#ifdef CCL_BFP16_COMPILER
    int is_avx512f_enabled = 0;
    uint32_t reg[4];

    __asm__ __volatile__("cpuid"
                         : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                         : "a"(7), "c"(0));
    is_avx512f_enabled =
        ((reg[1] & (1 << 16)) >> 16) & ((reg[1] & (1 << 30)) >> 30) & ((reg[1] & (1 << 31)) >> 31);

    return (is_avx512f_enabled) ? 1 : 0;
#else
    return 0;
#endif
}

std::vector<ccl_test_conf> test_params(calculate_test_count());

void init_test_params() {
    size_t idx = 0;
    for (ccl_prolog_type prolog_type = first_ccl_prolog_type; prolog_type < last_ccl_prolog_type;
         prolog_type++) {
        for (ccl_epilog_type epilog_type = first_ccl_epilog_type;
             epilog_type < last_ccl_epilog_type;
             epilog_type++) {
            // if ((epilog_type != ETYPE_CHAR_TO_T && prolog_type == PTYPE_T_TO_CHAR)||(epilog_type == ETYPE_CHAR_TO_T && prolog_type != PTYPE_T_TO_CHAR))
            //      // TODO: remove skipped data type
            //      continue;
            for (ccl_reduction_type reduction_type = first_ccl_reduction_type;
                 reduction_type < last_ccl_reduction_type;
                 reduction_type++) {
                for (ccl_sync_type sync_type = first_ccl_sync_type; sync_type < last_ccl_sync_type;
                     sync_type++) {
                    for (ccl_cache_type cache_type = first_ccl_cache_type;
                         cache_type < last_ccl_cache_type;
                         cache_type++) {
                        for (ccl_size_type size_type = first_ccl_size_type;
                             size_type < last_ccl_size_type;
                             size_type++) {
                            for (ccl_data_type data_type = first_ccl_data_type;
                                 data_type < last_ccl_data_type;
                                 data_type++) {
                                if (data_type == DT_BFP16 && !is_bfp16_enabled())
                                    continue;

                                for (ccl_completion_type completion_type =
                                         first_ccl_completion_type;
                                     completion_type < last_ccl_completion_type;
                                     completion_type++) {
                                    for (ccl_place_type place_type = first_ccl_place_type;
                                         place_type < last_ccl_place_type;
                                         place_type++) {
#ifdef TEST_CCL_BCAST
                                        if (place_type == PT_OOP)
                                            continue;
#endif
                                        for (ccl_order_type start_order_type = first_ccl_order_type;
                                             start_order_type < last_ccl_order_type;
                                             start_order_type++) {
                                            for (ccl_order_type complete_order_type =
                                                     first_ccl_order_type;
                                                 complete_order_type < last_ccl_order_type;
                                                 complete_order_type++) {
                                                for (ccl_buffer_count buffer_count =
                                                         first_ccl_buffer_count;
                                                     buffer_count < last_ccl_buffer_count;
                                                     buffer_count++) {
                                                    test_params[idx].place_type = place_type;
                                                    test_params[idx].size_type = size_type;
                                                    test_params[idx].data_type = data_type;
                                                    test_params[idx].cache_type = cache_type;
                                                    test_params[idx].sync_type = sync_type;
                                                    test_params[idx].completion_type =
                                                        completion_type;
                                                    test_params[idx].reduction_type =
                                                        reduction_type;
                                                    test_params[idx].buffer_count = buffer_count;
                                                    test_params[idx].start_order_type =
                                                        start_order_type;
                                                    test_params[idx].complete_order_type =
                                                        complete_order_type;
                                                    test_params[idx].prolog_type = prolog_type;
                                                    test_params[idx].epilog_type = epilog_type;
                                                    idx++;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    test_params.resize(idx);
}
