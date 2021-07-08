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
#include <assert.h>
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "atl.h"
#include "common/global/global.hpp"
#include "comp/bf16/bf16_intrisics.hpp"
#include "comp/bf16/bf16_utils.hpp"
#include "comp/fp16/fp16_intrisics.hpp"
#include "comp/fp16/fp16_utils.hpp"

#define ATL_MPI_PM_KEY "atl-mpi"

#define EP_IDX_KEY "ep_idx"

#define GLOBAL_NIC_IDX_KEY   "pref_nic"
#define GLOBAL_NIC_COUNT_KEY "num_nics"
#define LOCAL_NIC_IDX_KEY    "pref_close_nic"
#define LOCAL_NIC_COUNT_KEY  "num_close_nics"

#define RET2ATL(ret) (ret != MPI_SUCCESS) ? ATL_STATUS_FAILURE : ATL_STATUS_SUCCESS

typedef enum { ATL_MPI_LIB_IMPI, ATL_MPI_LIB_MPICH, ATL_MPI_LIB_NONE } atl_mpi_lib_type_t;

typedef struct {
    atl_mpi_lib_type_t type;
    int device_buf;
} atl_mpi_lib_attr_t;

typedef struct {
    atl_mpi_lib_type_t type;
    const char* name;

    /* string prefix before numerical version of library, mandatory */
    const char* version_prefix_1;

    /* string prefix before numerical version of library, following prefix_1, optional */
    const char* version_prefix_2;

    /* minimal expected version of library, mandatory */
    int min_version_value;

    /* minimal expected version of library with device_buf support, mandatory */
    int min_device_buf_version_value;

    /* string prefix before library kind, optional */
    const char* kind_prefix;

    /* library kind, optional */
    const char* kind_value;
} atl_mpi_lib_info_t;

#define MPI_LIB_INFO_MAX_COUNT 3

static atl_mpi_lib_info_t mpi_lib_infos[MPI_LIB_INFO_MAX_COUNT] = {
    { ATL_MPI_LIB_IMPI,
      "impi",
      "Intel(R) MPI Library",
      NULL,
      2019,
      2021,
      "library kind:",
      "release_mt" },
    { ATL_MPI_LIB_MPICH, "mpich", "MPICH Custom Information:", "drop", 34, -1, NULL, NULL },
    { ATL_MPI_LIB_NONE, "none", "", NULL, 0, -1, NULL, NULL },
};

#ifdef CCL_BF16_COMPILER
#define ATL_MPI_BF16
#endif /* CCL_BF16_COMPILER */

#ifdef CCL_FP16_COMPILER
#define ATL_MPI_FP16
#endif /* CCL_FP16_COMPILER */

typedef struct {
    // custom MPI operations for BF16
    MPI_Op sum_op;
    MPI_Op prod_op;
    MPI_Op min_op;
    MPI_Op max_op;
    // custom MPI dtype for BF16
    MPI_Datatype dtype;
} atl_mpi_bf16_data_t;

typedef struct {
    // custom MPI operations for FP16
    MPI_Op sum_op;
    MPI_Op prod_op;
    MPI_Op min_op;
    MPI_Op max_op;
    // custom MPI dtype for FP16
    MPI_Datatype dtype;
} atl_mpi_fp16_data_t;

typedef struct atl_mpi_global_data {
    int is_external_init;
    size_t ctx_count;
    int extra_ep;
    atl_mnic_t mnic_type;
    size_t mnic_count;
    atl_mpi_lib_attr_t mpi_lib_attr;
    atl_mpi_bf16_data_t bf16;
    atl_mpi_fp16_data_t fp16;

    atl_mpi_global_data()
            : is_external_init(0),
              ctx_count(0),
              extra_ep(0),
              mnic_type(ATL_MNIC_NONE),
              mnic_count(1) {
        mpi_lib_attr.type = ATL_MPI_LIB_NONE;
        mpi_lib_attr.device_buf = 0;

        bf16.dtype = MPI_DATATYPE_NULL;
        bf16.sum_op = MPI_OP_NULL;
        bf16.prod_op = MPI_OP_NULL;
        bf16.min_op = MPI_OP_NULL;
        bf16.max_op = MPI_OP_NULL;

        fp16.dtype = MPI_DATATYPE_NULL;
        fp16.sum_op = MPI_OP_NULL;
        fp16.prod_op = MPI_OP_NULL;
        fp16.min_op = MPI_OP_NULL;
        fp16.max_op = MPI_OP_NULL;
    }

} atl_mpi_global_data_t;

static atl_mpi_global_data_t global_data;

typedef enum { ATL_MPI_COMP_POSTED, ATL_MPI_COMP_COMPLETED } atl_mpi_comp_state_t;

typedef struct {
    MPI_Request native_req;
    atl_mpi_comp_state_t comp_state;
} atl_mpi_req_t;

typedef struct {
    atl_ctx_t ctx;
    int sync_coll;
    atl_progress_mode_t progress_mode;
} atl_mpi_ctx_t;

typedef struct {
    atl_ep_t ep;
    MPI_Comm mpi_comm;

    /* dummy recv operation to ensure progress in atl_poll */
    atl_mpi_req_t dummy_req;
    MPI_Comm dummy_comm;
} atl_mpi_ep_t;

typedef struct atl_mpi_comm_info {
    int found;
    MPI_Comm comm;
    char key[MPI_MAX_INFO_KEY];
    char value[MPI_MAX_INFO_VAL];

    atl_mpi_comm_info() {
        found = 0;
        comm = MPI_COMM_WORLD;
        memset(key, 0, MPI_MAX_INFO_KEY);
        memset(value, 0, MPI_MAX_INFO_VAL);
    }
} atl_mpi_comm_info_t;

#define MPI_BFLOAT16 \
    ({ \
        CCL_THROW_IF_NOT(global_data.bf16.dtype != MPI_DATATYPE_NULL, \
                         "unsupported datatype: ATL_DTYPE_BF16"); \
        global_data.bf16.dtype; \
    })

#define MPI_FLOAT16 \
    ({ \
        CCL_THROW_IF_NOT(global_data.fp16.dtype != MPI_DATATYPE_NULL, \
                         "unsupported datatype: ATL_DTYPE_FP16"); \
        global_data.fp16.dtype; \
    })

// helpers: check contract
static inline void atl_mpi_check_op_params(void* in_buf,
                                           void* inout_buf,
                                           int* length,
                                           MPI_Datatype* datatype,
                                           const char* caller_func_name) {
    (void)datatype;
    CCL_THROW_IF_NOT(in_buf && inout_buf && length,
                     caller_func_name,
                     " requested, bad arguments: ",
                     in_buf,
                     " ",
                     inout_buf,
                     " ",
                     length);
}

static void atl_mpi_print_error(int error) __attribute__((unused));
static void atl_mpi_print_error(int error) {
    char str_error[MPI_MAX_ERROR_STRING];
    int result_len = MPI_MAX_ERROR_STRING;

    MPI_Error_string(error, str_error, &result_len);

    if (result_len > MPI_MAX_ERROR_STRING) {
        result_len = MPI_MAX_ERROR_STRING;
    }
    str_error[result_len - 1] = '\0';

    ccl_logger::format(std::cout, "MPI error: %s (%d)", str_error, error);
}

#ifdef ATL_MPI_BF16

static void BF16_INLINE_TARGET_ATTRIBUTE_ALL atl_mpi_bf16_base_op(void* in,
                                                                  void* inout,
                                                                  int* length,
                                                                  ccl::reduction op) {
    unsigned short* in_buf = (unsigned short*)in;
    unsigned short* inout_buf = (unsigned short*)inout;

    size_t len = *length;
    ccl_bf16_reduce_impl(in_buf, inout_buf, len, op);
}

static void BF16_TARGET_ATTRIBUTE_ALL atl_mpi_bf16_sum_op(void* in,
                                                          void* inout,
                                                          int* length,
                                                          MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bf16_base_op(in, inout, length, ccl::reduction::sum);
}

static void BF16_TARGET_ATTRIBUTE_ALL atl_mpi_bf16_prod_op(void* in,
                                                           void* inout,
                                                           int* length,
                                                           MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bf16_base_op(in, inout, length, ccl::reduction::prod);
}

static void BF16_TARGET_ATTRIBUTE_ALL atl_mpi_bf16_min_op(void* in,
                                                          void* inout,
                                                          int* length,
                                                          MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bf16_base_op(in, inout, length, ccl::reduction::min);
}

static void BF16_TARGET_ATTRIBUTE_ALL atl_mpi_bf16_max_op(void* in,
                                                          void* inout,
                                                          int* length,
                                                          MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bf16_base_op(in, inout, length, ccl::reduction::max);
}
#endif /* ATL_MPI_BF16 */

#ifdef ATL_MPI_FP16

static void FP16_INLINE_TARGET_ATTRIBUTE_ALL atl_mpi_fp16_base_op(void* in,
                                                                  void* inout,
                                                                  int* length,
                                                                  ccl::reduction op) {
    unsigned short* in_buf = (unsigned short*)in;
    unsigned short* inout_buf = (unsigned short*)inout;

    size_t len = *length;
    ccl_fp16_reduce_impl(in_buf, inout_buf, len, op);
}

static void FP16_TARGET_ATTRIBUTE_ALL atl_mpi_fp16_sum_op(void* in,
                                                          void* inout,
                                                          int* length,
                                                          MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_fp16_base_op(in, inout, length, ccl::reduction::sum);
}

static void FP16_TARGET_ATTRIBUTE_ALL atl_mpi_fp16_prod_op(void* in,
                                                           void* inout,
                                                           int* length,
                                                           MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_fp16_base_op(in, inout, length, ccl::reduction::prod);
}

static void FP16_TARGET_ATTRIBUTE_ALL atl_mpi_fp16_min_op(void* in,
                                                          void* inout,
                                                          int* length,
                                                          MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_fp16_base_op(in, inout, length, ccl::reduction::min);
}

static void FP16_TARGET_ATTRIBUTE_ALL atl_mpi_fp16_max_op(void* in,
                                                          void* inout,
                                                          int* length,
                                                          MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_fp16_base_op(in, inout, length, ccl::reduction::max);
}
#endif /* ATL_MPI_FP16 */

static int atl_mpi_bf16_init() {
    int ret = MPI_SUCCESS;

    if (ccl::global_data::env().bf16_impl_type <= ccl_bf16_no_hardware_support) {
        return RET2ATL(ret);
    }

#ifdef ATL_MPI_BF16

    // create custom MPI BF16 dtype
    ret = MPI_Type_contiguous(2, MPI_BYTE, &global_data.bf16.dtype);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI BF16 dtype");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    ret = MPI_Type_commit(&global_data.bf16.dtype);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot commit MPI BF16 type");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BF16 summation op
    ret = MPI_Op_create(&atl_mpi_bf16_sum_op, 1, &global_data.bf16.sum_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI BF16 sum op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BF16 production op
    ret = MPI_Op_create(&atl_mpi_bf16_prod_op, 1, &global_data.bf16.prod_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI BF16 prod op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BF16 min op
    ret = MPI_Op_create(&atl_mpi_bf16_min_op, 1, &global_data.bf16.min_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI BF16 min op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BF16 max op
    ret = MPI_Op_create(&atl_mpi_bf16_max_op, 1, &global_data.bf16.max_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI BF16 max op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

#endif /* ATL_MPI_BF16 */

    return RET2ATL(ret);
}

static void atl_mpi_bf16_finalize() {
    if (global_data.bf16.dtype != MPI_DATATYPE_NULL) {
        MPI_Type_free(&global_data.bf16.dtype);
    }

    if (global_data.bf16.sum_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bf16.sum_op);
    }

    if (global_data.bf16.prod_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bf16.prod_op);
    }

    if (global_data.bf16.min_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bf16.min_op);
    }

    if (global_data.bf16.max_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bf16.max_op);
    }
}

static int atl_mpi_fp16_init() {
    int ret = MPI_SUCCESS;

    if (ccl::global_data::env().fp16_impl_type <= ccl_fp16_no_hardware_support) {
        return RET2ATL(ret);
    }

#ifdef ATL_MPI_FP16

    // create custom MPI FP16 dtype
    ret = MPI_Type_contiguous(2, MPI_BYTE, &global_data.fp16.dtype);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI FP16 dtype");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    ret = MPI_Type_commit(&global_data.fp16.dtype);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot commit MPI FP16 type");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI FP16 summation op
    ret = MPI_Op_create(&atl_mpi_fp16_sum_op, 1, &global_data.fp16.sum_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI FP16 sum op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI FP16 production op
    ret = MPI_Op_create(&atl_mpi_fp16_prod_op, 1, &global_data.fp16.prod_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI FP16 prod op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI FP16 min op
    ret = MPI_Op_create(&atl_mpi_fp16_min_op, 1, &global_data.fp16.min_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI FP16 min op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI FP16 max op
    ret = MPI_Op_create(&atl_mpi_fp16_max_op, 1, &global_data.fp16.max_op);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR("cannot create MPI FP16 max op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

#endif /* ATL_MPI_FP16 */

    return RET2ATL(ret);
}

static void atl_mpi_fp16_finalize() {
    if (global_data.fp16.dtype != MPI_DATATYPE_NULL) {
        MPI_Type_free(&global_data.fp16.dtype);
    }

    if (global_data.fp16.sum_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.fp16.sum_op);
    }

    if (global_data.fp16.prod_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.fp16.prod_op);
    }

    if (global_data.fp16.min_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.fp16.min_op);
    }

    if (global_data.fp16.max_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.fp16.max_op);
    }
}

static MPI_Datatype atl2mpi_dtype(atl_datatype_t dtype) {
    switch (dtype) {
        case ATL_DTYPE_INT8: return MPI_CHAR;
        case ATL_DTYPE_UINT8: return MPI_UNSIGNED_CHAR;
        case ATL_DTYPE_INT16: return MPI_INT16_T;
        case ATL_DTYPE_UINT16: return MPI_UINT16_T;
        case ATL_DTYPE_INT32: return MPI_INT;
        case ATL_DTYPE_UINT32: return MPI_UINT32_T;
        case ATL_DTYPE_INT64: return MPI_LONG_LONG;
        case ATL_DTYPE_UINT64: return MPI_UNSIGNED_LONG_LONG;
        case ATL_DTYPE_FLOAT16: return MPI_FLOAT16;
        case ATL_DTYPE_FLOAT32: return MPI_FLOAT;
        case ATL_DTYPE_FLOAT64: return MPI_DOUBLE;
        case ATL_DTYPE_BFLOAT16: return MPI_BFLOAT16;
        default: printf("unknown datatype: %d\n", dtype); exit(1);
    }
}

#ifdef ATL_MPI_BF16
static MPI_Op atl2mpi_op_bf16(atl_reduction_t rtype) {
    switch (rtype) {
        case ATL_REDUCTION_SUM: return global_data.bf16.sum_op;
        case ATL_REDUCTION_PROD: return global_data.bf16.prod_op;
        case ATL_REDUCTION_MIN: return global_data.bf16.min_op;
        case ATL_REDUCTION_MAX: return global_data.bf16.max_op;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
}
#endif /* ATL_MPI_BF16 */

#ifdef ATL_MPI_FP16
static MPI_Op atl2mpi_op_fp16(atl_reduction_t rtype) {
    switch (rtype) {
        case ATL_REDUCTION_SUM: return global_data.fp16.sum_op;
        case ATL_REDUCTION_PROD: return global_data.fp16.prod_op;
        case ATL_REDUCTION_MIN: return global_data.fp16.min_op;
        case ATL_REDUCTION_MAX: return global_data.fp16.max_op;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
}
#endif /* ATL_MPI_FP16 */

static MPI_Op atl2mpi_op(atl_reduction_t rtype, MPI_Datatype dtype) {
#ifdef ATL_MPI_BF16
    if (dtype == global_data.bf16.dtype)
        return atl2mpi_op_bf16(rtype);
#endif /* ATL_MPI_BF16 */

#ifdef ATL_MPI_FP16
    if (dtype == global_data.fp16.dtype)
        return atl2mpi_op_fp16(rtype);
#endif /* ATL_MPI_FP16 */

    (void)dtype;
    switch (rtype) {
        case ATL_REDUCTION_SUM: return MPI_SUM;
        case ATL_REDUCTION_PROD: return MPI_PROD;
        case ATL_REDUCTION_MIN: return MPI_MIN;
        case ATL_REDUCTION_MAX: return MPI_MAX;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
}

atl_mpi_lib_attr_t atl_mpi_get_lib_attr() {
    atl_mpi_lib_attr_t lib_attr = { ATL_MPI_LIB_NONE, 0 };

    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING] = { 0 };
    int mpi_version_len = -1, i;
    atl_mpi_lib_info_t* final_info = NULL;

    /* can be called before MPI_Init */
    int ret = MPI_Get_library_version(mpi_version, &mpi_version_len);

    if ((ret != MPI_SUCCESS) || (mpi_version_len < 0) ||
        (mpi_version_len > MPI_MAX_LIBRARY_VERSION_STRING)) {
        LOG_WARN("can not retrieve MPI version, mpi_version_len ", mpi_version_len, ", ret", ret);
        return lib_attr;
    }

    /* remove trailing spaces at the end for more compact log */
    while (strlen(mpi_version) && isspace(mpi_version[strlen(mpi_version) - 1]))
        mpi_version[strlen(mpi_version) - 1] = '\0';

    LOG_DEBUG("MPI version: ", mpi_version);

    /* for filtering */
    char* lib_type_env = getenv("CCL_ATL_MPI");

    for (i = 0; i < MPI_LIB_INFO_MAX_COUNT; i++) {
        atl_mpi_lib_info_t* info = &(mpi_lib_infos[i]);

        if (info->type == ATL_MPI_LIB_NONE)
            continue;

        if (lib_type_env) {
            if (strcmp(lib_type_env, info->name)) {
                LOG_DEBUG("library ", info->name, " is filtered out by user input ", lib_type_env);
                continue;
            }
            else {
                LOG_DEBUG("use lib_type = ", lib_type_env, " because it is requested explicitly");
            }
        }

        CCL_THROW_IF_NOT(info->version_prefix_1, "empty version_prefix_1");
        CCL_THROW_IF_NOT(info->min_version_value >= 0, "unexpected minimal version");

        const char* version_substr = NULL;
        if ((version_substr = strstr(mpi_version, info->version_prefix_1))) {
            version_substr += strlen(info->version_prefix_1);
            LOG_DEBUG("version_substr: ", version_substr);

            if (info->version_prefix_2) {
                version_substr = strstr(version_substr, info->version_prefix_2);
                if (!version_substr) {
                    LOG_DEBUG("can't find version_prefix_2 ", info->version_prefix_2);
                    continue;
                }
                version_substr += strlen(info->version_prefix_2);
                LOG_DEBUG("version_substr: ", version_substr);
            }

            int version_value = (version_substr) ? atoi(version_substr) : -1;
            LOG_DEBUG("MPI numerical version: ", version_value);

            if (version_value < info->min_version_value) {
                LOG_WARN("loaded MPI doesn't match with expected version, "
                         "consider to switch to ",
                         info->version_prefix_1,
                         " ",
                         (info->version_prefix_2 ? info->version_prefix_2 : ""),
                         info->min_version_value,
                         " (min) ",
                         (info->kind_value ? info->kind_value : ""),
                         "\n");
                continue;
            }

            if (info->kind_prefix && info->kind_value) {
                const char* kind_substr = mpi_version;

                if ((kind_substr = strstr(kind_substr, info->kind_prefix))) {
                    kind_substr += strlen(info->kind_prefix);
                    while ((isspace(*kind_substr)) &&
                           (kind_substr < (mpi_version + mpi_version_len)))
                        kind_substr++;

                    LOG_DEBUG("kind_substr: ", kind_substr);

                    if (strncmp(kind_substr, info->kind_value, strlen(info->kind_value))) {
                        LOG_WARN("loaded MPI version (",
                                 version_value,
                                 ") ",
                                 "is higher or equal to minimal expected version (",
                                 info->min_version_value,
                                 ") ",
                                 "but kind (",
                                 kind_substr,
                                 ") doesn't match with expected kind (",
                                 info->kind_value,
                                 "), "
                                 "consider to switch to ",
                                 info->version_prefix_1,
                                 " ",
                                 (info->version_prefix_2 ? info->version_prefix_2 : ""),
                                 info->min_version_value,
                                 " (min version) ",
                                 (info->kind_value ? info->kind_value : ""),
                                 "\n");
                    }
                }
                else {
                    LOG_DEBUG("MPI version is high enough, but kind_prefix (",
                              info->kind_prefix,
                              ") can not be found",
                              " treat this like expected kind (",
                              info->kind_value,
                              ") was found");
                }
            }

            final_info = info;
            LOG_DEBUG("set lib_type = ",
                      info->name,
                      " because "
                      "version (",
                      version_value,
                      ") is higher or equal to minimal expected version (",
                      info->min_version_value,
                      ")");

            lib_attr.type = final_info->type;
            lib_attr.device_buf =
                (final_info->min_device_buf_version_value >= version_value) ? 1 : 0;

            break;
        }
    }

    if (final_info) {
        LOG_DEBUG("MPI library type: ", final_info->name);
    }
    else {
        LOG_DEBUG("MPI library type: none");
    }

    return lib_attr;
}

size_t atl_mpi_get_ep_count(const atl_attr_t& attr) {
    size_t mpi_ep_count = attr.in.ep_count;
    if (attr.in.enable_extra_ep)
        mpi_ep_count += attr.in.enable_extra_ep;
    return mpi_ep_count;
}

size_t atl_mpi_get_ep_idx(size_t ep_idx) {
    size_t mpi_ep_idx = ep_idx;
    if (global_data.extra_ep)
        mpi_ep_idx += global_data.extra_ep;
    return mpi_ep_idx;
}

/* set these knobs without detection of MPI library type */
atl_status_t atl_mpi_set_base_env(const atl_attr_t& attr) {
    setenv("PSM2_MULTI_EP", "1", 0);
    setenv("FI_OFI_RXM_USE_HASH", "0", 0);

#ifdef CCL_ENABLE_SYCL
    setenv("FI_SHM_DISABLE_CMA", "1", 0);
#endif /* CCL_ENABLE_SYCL */

    setenv("MPIR_CVAR_DEFAULT_THREAD_LEVEL", "MPI_THREAD_MULTIPLE", 0);

    /* request IMPI level append library kind into MPI_Get_library_version output */
    setenv("I_MPI_INFO_LIBRARY_KIND", "1", 0);

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_set_impi_env(const atl_attr_t& attr, const atl_mpi_lib_attr_t& lib_attr) {
    char ep_count_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(ep_count_str, MPI_MAX_INFO_VAL, "%zu", atl_mpi_get_ep_count(attr));

    if (attr.in.ep_count)
        setenv("I_MPI_OFI_ISEND_INJECT_THRESHOLD", "0", 0);

#ifdef CCL_ENABLE_SYCL
    setenv("I_MPI_SHM_CMA", "0", 0);
    if (attr.in.enable_device_buf && lib_attr.device_buf) {
        setenv("I_MPI_OFFLOAD", "2", 0);
        setenv("I_MPI_OFFLOAD_TOPOLIB", "l0", 0);
        setenv("I_MPI_OFFLOAD_QUEUE_CACHE", "1", 0);
        setenv("I_MPI_OFFLOAD_LIST_CACHE", "1", 0);
        if (attr.in.ep_count > 1) {
            /* try to set global lock level before vci level
               because setenv is invoked with overwrite=0 */
            setenv("I_MPI_THREAD_LOCK_LEVEL", "global", 0);
        }
    }
#endif /* CCL_ENABLE_SYCL */

    setenv("I_MPI_THREAD_SPLIT", "1", 0);
    setenv("I_MPI_THREAD_RUNTIME", "generic", 0);
    setenv("I_MPI_THREAD_MAX", ep_count_str, 0);
    setenv("I_MPI_THREAD_ID_KEY", EP_IDX_KEY, 0);
    setenv("I_MPI_THREAD_LOCK_LEVEL", "vci", 0);

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_check_impi_env(const atl_attr_t& attr) {
    char* ep_count_env = getenv("I_MPI_THREAD_MAX");
    if (!ep_count_env)
        return ATL_STATUS_FAILURE;
    if (atoi(ep_count_env) != (int)(atl_mpi_get_ep_count(attr)))
        return ATL_STATUS_FAILURE;

    if (!getenv("I_MPI_ROOT")) {
        atl_mpi_lib_type_t type = ATL_MPI_LIB_IMPI;
        LOG_ERROR("CCL/MPI uses ",
                  mpi_lib_infos[type].version_prefix_1,
                  " but I_MPI_ROOT is not set. ",
                  "Please source ",
                  mpi_lib_infos[type].kind_value,
                  " version of ",
                  mpi_lib_infos[type].version_prefix_1,
                  " (",
                  mpi_lib_infos[type].min_version_value,
                  " or higher version).");
        return ATL_STATUS_FAILURE;
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_set_mpich_env(const atl_attr_t& attr) {
    char ep_count_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(ep_count_str, MPI_MAX_INFO_VAL, "%zu", atl_mpi_get_ep_count(attr));

    setenv("MPIR_CVAR_CH4_MT_MODEL", "direct", 0);
    setenv("MPIR_CVAR_CH4_NUM_VCIS", ep_count_str, 0);
    setenv("MPIR_CVAR_CH4_OFI_MAX_VCIS", ep_count_str, 0);
    setenv("MPIR_CVAR_CH4_ASYNC_PROGRESS_ID_KEY", EP_IDX_KEY, 0);
    setenv("MPIR_CVAR_CH4_OFI_ENABLE_SCALABLE_ENDPOINTS", "1", 0);

    if (attr.in.mnic_type != ATL_MNIC_NONE) {
        setenv("MPIR_CVAR_CH4_OFI_ENABLE_NIC_SELECTION", "1", 0);
        auto& env = ccl::global_data::env();
        if (env.log_level >= ccl_log_level::info) {
            setenv("MPIR_CVAR_CH4_OFI_DUMP_NIC_SETTINGS", "1", 0);
        }
    }

    setenv("FI_PSM2_DELAY", "0", 0);
    setenv("FI_PSM2_TIMEOUT", "0", 0);
    setenv("FI_PSM2_NAME_SERVER", "0", 0);
    setenv("HFI_NO_CPUAFFINITY", "1", 0);

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_check_mpich_env(const atl_attr_t& attr) {
    char* ep_count_env = getenv("MPIR_CVAR_CH4_OFI_MAX_VCIS");
    if (!ep_count_env)
        return ATL_STATUS_FAILURE;
    if (atoi(ep_count_env) != (int)(atl_mpi_get_ep_count(attr)))
        return ATL_STATUS_FAILURE;
    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_set_env(const atl_attr_t& attr) {
    if (global_data.mpi_lib_attr.type != ATL_MPI_LIB_NONE) {
        /* library type was already detected and env was set, make sanity check */
        if (global_data.mpi_lib_attr.type == ATL_MPI_LIB_IMPI) {
            return atl_mpi_check_impi_env(attr);
        }
        else if (global_data.mpi_lib_attr.type == ATL_MPI_LIB_MPICH) {
            return atl_mpi_check_mpich_env(attr);
        }
        return ATL_STATUS_SUCCESS;
    }

    atl_mpi_set_base_env(attr);

    atl_mpi_lib_attr_t mpi_lib_attr = atl_mpi_get_lib_attr();

    if (mpi_lib_attr.type == ATL_MPI_LIB_NONE) {
        return ATL_STATUS_SUCCESS;
    }

    if (mpi_lib_attr.type == ATL_MPI_LIB_IMPI) {
        atl_mpi_set_impi_env(attr, mpi_lib_attr);
        atl_mpi_check_impi_env(attr);
    }
    else if (mpi_lib_attr.type == ATL_MPI_LIB_MPICH) {
        atl_mpi_set_mpich_env(attr);
        atl_mpi_check_mpich_env(attr);
    }

    int is_mpi_inited = 0;
    MPI_Initialized(&is_mpi_inited);
    if (is_mpi_inited) {
        LOG_WARN("MPI was initialized externally, CCL-MPI specific environment is ignored");
    }
    else {
        LOG_DEBUG("set CCL-MPI specific environment");
    }

    global_data.mpi_lib_attr = mpi_lib_attr;

    return ATL_STATUS_SUCCESS;
}

atl_mpi_comm_info_t atl_mpi_get_comm_info(MPI_Comm comm, const char* key) {
    MPI_Info info;
    atl_mpi_comm_info_t res;
    res.comm = comm;
    snprintf(res.key, MPI_MAX_INFO_KEY, "%s", key);

    MPI_Comm_get_info(res.comm, &info);
    MPI_Info_get(info, key, MPI_MAX_INFO_VAL, res.value, &res.found);
    MPI_Info_free(&info);

    return res;
}

size_t atl_mpi_get_nic_count(const char* nic_count_key) {
    size_t count = 1;
    atl_mpi_comm_info_t info = atl_mpi_get_comm_info(MPI_COMM_WORLD, nic_count_key);
    CCL_THROW_IF_NOT(info.found, "MPI comm key ", nic_count_key, " was not set");

    count = atoi(info.value);
    if (count <= 0) {
        count = 1;
    }

    return count;
}

void atl_mpi_check_comm_info(MPI_Comm comm, const char* key, const char* expected_value) {
    atl_mpi_comm_info_t info = atl_mpi_get_comm_info(comm, key);

    CCL_THROW_IF_NOT(info.found, "MPI comm key ", key, " was not set");
    CCL_THROW_IF_NOT(!strcmp(info.value, expected_value),
                     "MPI comm key ",
                     key,
                     ": expected: ",
                     expected_value,
                     ", read: ",
                     info.value);
}

void atl_mpi_check_comm_ep_idx(MPI_Comm comm, size_t expected_idx) {
    if (global_data.mpi_lib_attr.type == ATL_MPI_LIB_NONE)
        return;

    char expected_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(expected_idx_str, MPI_MAX_INFO_VAL, "%zu", expected_idx);
    atl_mpi_check_comm_info(comm, EP_IDX_KEY, expected_idx_str);
}

void atl_mpi_check_comm_nic_idx(MPI_Comm comm, size_t expected_idx, const char* nic_idx_key) {
    char expected_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(expected_idx_str, MPI_MAX_INFO_VAL, "%zu", expected_idx);
    atl_mpi_check_comm_info(comm, nic_idx_key, expected_idx_str);
}

#ifdef ENABLE_DEBUG
inline void atl_mpi_check_ep(atl_ep_t* ep) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_check_comm_ep_idx(mpi_ep->mpi_comm, atl_mpi_get_ep_idx(ep->idx));
}
#else
#define atl_mpi_check_ep(ep)
#endif

static atl_status_t atl_mpi_finalize(atl_ctx_t* ctx) {
    int ret = MPI_SUCCESS;
    atl_mpi_ctx_t* mpi_ctx = container_of(ctx, atl_mpi_ctx_t, ctx);
    atl_ep_t** eps = ctx->eps;

    global_data.ctx_count--;
    if (ctx->coord.global_idx == 0) {
        LOG_INFO("finalize atl-mpi ctx, remaining ctx_count ", global_data.ctx_count);
    }

    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);

    if (!is_mpi_finalized) {
        for (size_t i = 0; i < ctx->ep_count; i++) {
            atl_mpi_ep_t* mpi_ep = container_of(eps[i], atl_mpi_ep_t, ep);

            if (mpi_ep) {
                if (mpi_ctx->progress_mode == ATL_PROGRESS_POLL) {
                    MPI_Cancel(&(mpi_ep->dummy_req.native_req));
                    MPI_Comm_free(&mpi_ep->dummy_comm);
                }
                MPI_Comm_free(&mpi_ep->mpi_comm);
                free(mpi_ep);
            }
        }

        if (global_data.ctx_count == 0) {
            atl_mpi_bf16_finalize();
            atl_mpi_fp16_finalize();
            if (!global_data.is_external_init) {
                ret = MPI_Finalize();
            }
            else {
                LOG_DEBUG("MPI_Init has been called externally, skip MPI_Finalize");
            }

            if (ctx->coord.global_idx == 0) {
                LOG_INFO("finalized last atl-mpi ctx");
            }
        }
    }
    else {
        for (size_t i = 0; i < ctx->ep_count; i++) {
            atl_mpi_ep_t* mpi_ep = container_of(eps[i], atl_mpi_ep_t, ep);
            free(mpi_ep);
        }
        if ((global_data.ctx_count == 0) && (ctx->coord.global_idx == 0)) {
            LOG_WARN("MPI_Finalize has been called");
        }
    }

    free(eps);
    free(mpi_ctx);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_mr_reg(atl_ctx_t* ctx, const void* buf, size_t len, atl_mr_t** mr) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_mr_dereg(atl_ctx_t* ctx, atl_mr_t* mr) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_ep_send(atl_ep_t* ep,
                                    const void* buf,
                                    size_t len,
                                    int dst_proc_idx,
                                    uint64_t tag,
                                    atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Isend(
        buf, len, MPI_CHAR, dst_proc_idx, (int)tag, mpi_ep->mpi_comm, &mpi_req->native_req);

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_recv(atl_ep_t* ep,
                                    void* buf,
                                    size_t len,
                                    int src_proc_idx,
                                    uint64_t tag,
                                    atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Irecv(
        buf, len, MPI_CHAR, src_proc_idx, (int)tag, mpi_ep->mpi_comm, &mpi_req->native_req);

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_probe(atl_ep_t* ep,
                                     int src_proc_idx,
                                     uint64_t tag,
                                     int* found,
                                     size_t* recv_len) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    int flag = 0, len = 0, ret;
    MPI_Status status;

    ret = MPI_Iprobe(src_proc_idx, tag, mpi_ep->mpi_comm, &flag, &status);
    if (flag) {
        MPI_Get_count(&status, MPI_BYTE, &len);
    }

    if (found)
        *found = flag;
    if (recv_len)
        *recv_len = len;

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_allgatherv(atl_ep_t* ep,
                                          const void* send_buf,
                                          size_t send_len,
                                          void* recv_buf,
                                          const int* recv_lens,
                                          const int* offsets,
                                          atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Allgatherv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             send_len,
                             MPI_CHAR,
                             recv_buf,
                             recv_lens,
                             offsets,
                             MPI_CHAR,
                             mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Iallgatherv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                              send_len,
                              MPI_CHAR,
                              recv_buf,
                              recv_lens,
                              offsets,
                              MPI_CHAR,
                              mpi_ep->mpi_comm,
                              &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_allreduce(atl_ep_t* ep,
                                         const void* send_buf,
                                         void* recv_buf,
                                         size_t count,
                                         atl_datatype_t dtype,
                                         atl_reduction_t op,
                                         atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Allreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            recv_buf,
                            count,
                            mpi_dtype,
                            mpi_op,
                            mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        //printf("atl_mpi: send_buf %p, recv_buf %p\n", send_buf, recv_buf);
        ret = MPI_Iallreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             recv_buf,
                             count,
                             mpi_dtype,
                             mpi_op,
                             mpi_ep->mpi_comm,
                             &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_alltoall(atl_ep_t* ep,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t len,
                                        atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Alltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                           len,
                           MPI_CHAR,
                           recv_buf,
                           len,
                           MPI_CHAR,
                           mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Ialltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            len,
                            MPI_CHAR,
                            recv_buf,
                            len,
                            MPI_CHAR,
                            mpi_ep->mpi_comm,
                            &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_alltoallv(atl_ep_t* ep,
                                         const void* send_buf,
                                         const int* send_lens,
                                         const int* send_offsets,
                                         void* recv_buf,
                                         const int* recv_lens,
                                         const int* recv_offsets,
                                         atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Alltoallv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            send_lens,
                            send_offsets,
                            MPI_CHAR,
                            recv_buf,
                            recv_lens,
                            recv_offsets,
                            MPI_CHAR,
                            mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Ialltoallv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             send_lens,
                             send_offsets,
                             MPI_CHAR,
                             recv_buf,
                             recv_lens,
                             recv_offsets,
                             MPI_CHAR,
                             mpi_ep->mpi_comm,
                             &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_barrier(atl_ep_t* ep, atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Barrier(mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Ibarrier(mpi_ep->mpi_comm, &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_bcast(atl_ep_t* ep,
                                     void* buf,
                                     size_t len,
                                     int root,
                                     atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Bcast(buf, len, MPI_CHAR, root, mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Ibcast(buf, len, MPI_CHAR, root, mpi_ep->mpi_comm, &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_reduce(atl_ep_t* ep,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      int root,
                                      atl_datatype_t dtype,
                                      atl_reduction_t op,
                                      atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    int my_proc_idx = ep->ctx->coord.global_idx;
    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    if (mpi_ctx->sync_coll) {
        ret = MPI_Reduce(
            (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            count,
            mpi_dtype,
            mpi_op,
            root,
            mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Ireduce(
            (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            count,
            mpi_dtype,
            mpi_op,
            root,
            mpi_ep->mpi_comm,
            &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_reduce_scatter(atl_ep_t* ep,
                                              const void* send_buf,
                                              void* recv_buf,
                                              size_t recv_count,
                                              atl_datatype_t dtype,
                                              atl_reduction_t op,
                                              atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    if (mpi_ctx->sync_coll) {
        ret =
            MPI_Reduce_scatter_block((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                                     recv_buf,
                                     recv_count,
                                     mpi_dtype,
                                     mpi_op,
                                     mpi_ep->mpi_comm);
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
        mpi_req->native_req = MPI_REQUEST_NULL;
    }
    else {
        ret = MPI_Ireduce_scatter_block(
            (send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            recv_count,
            mpi_dtype,
            mpi_op,
            mpi_ep->mpi_comm,
            &mpi_req->native_req);
        mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    }

    atl_mpi_check_ep(ep);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_read(atl_ep_t* ep,
                                    void* buf,
                                    size_t len,
                                    atl_mr_t* mr,
                                    uint64_t addr,
                                    uintptr_t r_key,
                                    int dst_proc_idx,
                                    atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_ep_write(atl_ep_t* ep,
                                     const void* buf,
                                     size_t len,
                                     atl_mr_t* mr,
                                     uint64_t addr,
                                     uintptr_t r_key,
                                     int dst_proc_idx,
                                     atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_ep_wait(atl_ep_t* ep, atl_req_t* req) {
    int ret;
    MPI_Status status;
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    ret = MPI_Wait(&mpi_req->native_req, &status);
    mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_wait_all(atl_ep_t* ep, atl_req_t* reqs, size_t count) {
    return ATL_STATUS_UNSUPPORTED;
}

static inline atl_status_t atl_mpi_ep_progress(atl_ep_t* ep, atl_mpi_req_t* req) {
    int flag = 0;
    int ret = MPI_Test(&req->native_req, &flag, MPI_STATUS_IGNORE);

    if (flag) {
        req->comp_state = ATL_MPI_COMP_COMPLETED;
    }

    return RET2ATL(ret);
}

static inline atl_status_t atl_mpi_ep_poll(atl_ep_t* ep) {
    atl_mpi_ctx_t* mpi_ctx = container_of(ep->ctx, atl_mpi_ctx_t, ctx);
    if (mpi_ctx->progress_mode == ATL_PROGRESS_POLL) {
        atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
        atl_mpi_ep_progress(ep, &(mpi_ep->dummy_req));
    }

    return ATL_STATUS_SUCCESS;
}

static atl_status_t atl_mpi_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) {
    CCL_THROW_IF_NOT(is_completed);

    atl_status_t status = ATL_STATUS_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    *is_completed = (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED);
    if (*is_completed) {
        return ATL_STATUS_SUCCESS;
    }

    status = atl_mpi_ep_progress(ep, mpi_req);
    *is_completed = (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED);

    return status;
}

static atl_ops_t atl_mpi_ops = {
    .finalize = atl_mpi_finalize,
};

static atl_mr_ops_t atl_mpi_mr_ops = {
    .mr_reg = atl_mpi_mr_reg,
    .mr_dereg = atl_mpi_mr_dereg,
};

static atl_p2p_ops_t atl_mpi_ep_p2p_ops = {
    .send = atl_mpi_ep_send,
    .recv = atl_mpi_ep_recv,
    .probe = atl_mpi_ep_probe,
};

static atl_coll_ops_t atl_mpi_ep_coll_ops = { .allgatherv = atl_mpi_ep_allgatherv,
                                              .allreduce = atl_mpi_ep_allreduce,
                                              .alltoall = atl_mpi_ep_alltoall,
                                              .alltoallv = atl_mpi_ep_alltoallv,
                                              .barrier = atl_mpi_ep_barrier,
                                              .bcast = atl_mpi_ep_bcast,
                                              .reduce = atl_mpi_ep_reduce,
                                              .reduce_scatter = atl_mpi_ep_reduce_scatter };

static atl_rma_ops_t atl_mpi_ep_rma_ops = {
    .read = atl_mpi_ep_read,
    .write = atl_mpi_ep_write,
};

static atl_comp_ops_t atl_mpi_ep_comp_ops = { .wait = atl_mpi_ep_wait,
                                              .wait_all = atl_mpi_ep_wait_all,
                                              .cancel = NULL,
                                              .poll = atl_mpi_ep_poll,
                                              .check = atl_mpi_ep_check };

static atl_status_t atl_mpi_ep_init(atl_mpi_ctx_t* mpi_ctx, size_t idx, atl_ep_t** ep) {
    int ret;

    ssize_t mpi_ep_idx = atl_mpi_get_ep_idx(idx);
    char mpi_ep_idx_str[MPI_MAX_INFO_VAL] = { 0 };

    size_t nic_idx = 0;
    char nic_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    const char* nic_idx_key =
        (global_data.mnic_type == ATL_MNIC_GLOBAL) ? GLOBAL_NIC_IDX_KEY : LOCAL_NIC_IDX_KEY;

    atl_mpi_ep_t* mpi_ep = (atl_mpi_ep_t*)calloc(1, sizeof(atl_mpi_ep_t));
    if (!mpi_ep)
        return ATL_STATUS_FAILURE;

    ret = MPI_Comm_dup(MPI_COMM_WORLD, &mpi_ep->mpi_comm);
    if (ret)
        goto err_ep_dup;

    MPI_Info info;
    MPI_Info_create(&info);

    /* set EP index */
    snprintf(mpi_ep_idx_str, MPI_MAX_INFO_VAL, "%zu", mpi_ep_idx);
    MPI_Info_set(info, EP_IDX_KEY, mpi_ep_idx_str);

    if (global_data.mnic_type != ATL_MNIC_NONE) {
        /* set NIC index */
        nic_idx = (idx % global_data.mnic_count);
        snprintf(nic_idx_str, MPI_MAX_INFO_VAL, "%zu", nic_idx);
        MPI_Info_set(info, nic_idx_key, nic_idx_str);
    }

    MPI_Comm_set_info(mpi_ep->mpi_comm, info);

    if (mpi_ctx->progress_mode == ATL_PROGRESS_POLL) {
        ret = MPI_Comm_dup(MPI_COMM_WORLD, &mpi_ep->dummy_comm);
        if (ret)
            goto err_ep_dup;
        MPI_Comm_set_info(mpi_ep->dummy_comm, info);
        MPI_Irecv(NULL, 0, MPI_CHAR, 0, 0, mpi_ep->dummy_comm, &(mpi_ep->dummy_req.native_req));

        atl_mpi_check_comm_ep_idx(mpi_ep->dummy_comm, mpi_ep_idx);
        if (global_data.mnic_type != ATL_MNIC_NONE) {
            atl_mpi_check_comm_nic_idx(mpi_ep->dummy_comm, nic_idx, nic_idx_key);
        }
    }

    MPI_Info_free(&info);

    atl_mpi_check_comm_ep_idx(mpi_ep->mpi_comm, mpi_ep_idx);
    if (global_data.mnic_type != ATL_MNIC_NONE) {
        atl_mpi_check_comm_nic_idx(mpi_ep->mpi_comm, nic_idx, nic_idx_key);
    }

    LOG_DEBUG("atl-mpi-ep: ", idx, ", ep_idx ", mpi_ep_idx, ", nic_idx ", nic_idx);

    *ep = &mpi_ep->ep;
    (*ep)->idx = idx;
    (*ep)->ctx = &mpi_ctx->ctx;
    (*ep)->p2p_ops = &atl_mpi_ep_p2p_ops;
    (*ep)->coll_ops = &atl_mpi_ep_coll_ops;
    (*ep)->rma_ops = &atl_mpi_ep_rma_ops;
    (*ep)->comp_ops = &atl_mpi_ep_comp_ops;

    return ATL_STATUS_SUCCESS;

err_ep_dup:
    free(mpi_ep);
    return RET2ATL(ret);
}

static atl_status_t atl_mpi_init(int* argc,
                                 char*** argv,
                                 atl_attr_t* attr,
                                 atl_ctx_t** out_ctx,
                                 const char* main_addr,
                                 ipmi* pmi) {
    CCL_THROW_IF_NOT((sizeof(atl_mpi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal)),
                     "unexpected offset: atl_mpi_request size ",
                     sizeof(atl_mpi_req_t),
                     ", atl_request size ",
                     sizeof(atl_req_t),
                     ", expected offset ",
                     offsetof(atl_req_t, internal));

    int ret = MPI_SUCCESS;
    size_t i;
    int is_tag_ub_set = 0;
    void* tag_ub_ptr = NULL;
    int required_thread_level = MPI_THREAD_MULTIPLE, provided_thread_level;

    atl_mpi_ctx_t* mpi_ctx = (atl_mpi_ctx_t*)calloc(1, sizeof(atl_mpi_ctx_t));
    if (!mpi_ctx)
        return ATL_STATUS_FAILURE;

    atl_ctx_t* ctx = &(mpi_ctx->ctx);

    if (global_data.ctx_count == 0) {
        if (atl_mpi_set_env(*attr)) {
            goto err_init;
        }

        MPI_Initialized(&global_data.is_external_init);

        if (!global_data.is_external_init) {
            ret = MPI_Init_thread(argc, argv, required_thread_level, &provided_thread_level);
            if (provided_thread_level < required_thread_level) {
                LOG_ERROR("unexpected MPI thread level: required ",
                          required_thread_level,
                          ", provided ",
                          provided_thread_level);
                goto err_init;
            }
        }
        else {
            LOG_DEBUG("MPI was initialized externaly");
            MPI_Query_thread(&provided_thread_level);
            if (provided_thread_level < required_thread_level) {
                LOG_WARN("MPI was initialized externaly but with unexpected thread level: "
                         "required ",
                         required_thread_level,
                         ", provided ",
                         provided_thread_level);
            }
        }

        if (ret)
            goto err_init;

        if (global_data.mpi_lib_attr.type == ATL_MPI_LIB_NONE)
            global_data.mpi_lib_attr = atl_mpi_get_lib_attr();

        global_data.extra_ep = attr->in.enable_extra_ep;

        global_data.mnic_type = attr->in.mnic_type;
        if (global_data.mpi_lib_attr.type != ATL_MPI_LIB_MPICH) {
            /* only MPICH supports multi-NIC */
            global_data.mnic_type = ATL_MNIC_NONE;
        }

        if (global_data.mnic_type == ATL_MNIC_LOCAL) {
            global_data.mnic_count = atl_mpi_get_nic_count(LOCAL_NIC_COUNT_KEY);
        }
        else if (global_data.mnic_type == ATL_MNIC_GLOBAL) {
            global_data.mnic_count = atl_mpi_get_nic_count(GLOBAL_NIC_IDX_KEY);
        }
        else if (global_data.mnic_type == ATL_MNIC_NONE) {
            global_data.mnic_count = 1;
        }
        global_data.mnic_count = std::min(global_data.mnic_count, attr->in.mnic_count);
        global_data.mnic_count = std::min(global_data.mnic_count, attr->in.ep_count);
        global_data.mnic_count = std::max(global_data.mnic_count, (size_t)(1));

        if (atl_mpi_bf16_init() == ATL_STATUS_FAILURE) {
            atl_mpi_bf16_finalize();
            goto err_init;
        }

        if (atl_mpi_fp16_init() == ATL_STATUS_FAILURE) {
            atl_mpi_fp16_finalize();
            goto err_init;
        }
    }
    global_data.ctx_count++;

    atl_proc_coord_t* coord;
    coord = &(ctx->coord);

    MPI_Comm_rank(MPI_COMM_WORLD, (int*)&(coord->global_idx));
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&(coord->global_count));

    MPI_Comm local_comm;
    MPI_Comm_split_type(
        MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, coord->global_count, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, (int*)&(coord->local_idx));
    MPI_Comm_size(local_comm, (int*)&(coord->local_count));
    MPI_Comm_free(&local_comm);

    ctx->ops = &atl_mpi_ops;
    ctx->mr_ops = &atl_mpi_mr_ops;
    ctx->ep_count = attr->in.ep_count;
    ctx->eps = (atl_ep_t**)calloc(1, sizeof(void*) * attr->in.ep_count);
    if (!ctx->eps)
        goto err_after_init;

    char* progress_mode_env;
    progress_mode_env = getenv(ATL_PROGRESS_MODE_ENV);
    if (progress_mode_env) {
        mpi_ctx->progress_mode = (atl_progress_mode_t)atoi(progress_mode_env);
    }
    else {
        mpi_ctx->progress_mode = ATL_PROGRESS_CHECK;
    }
    mpi_ctx->sync_coll = attr->in.enable_sync_coll;

    if (coord->global_idx == 0) {
        if (global_data.ctx_count == 1) {
            LOG_INFO("atl-mpi-global:")
            LOG_INFO("  is_external_init: ", global_data.is_external_init);
            LOG_INFO("  mpi_lib_attr.type: ", mpi_lib_infos[global_data.mpi_lib_attr.type].name);
            LOG_INFO("  mpi_lib_attr.device_buf: ", global_data.mpi_lib_attr.device_buf);
            LOG_INFO("  extra_ep: ", global_data.extra_ep);
            LOG_INFO("  mnic_type: ", global_data.mnic_type);
            if (global_data.mnic_type != ATL_MNIC_NONE)
                LOG_INFO("  mnic_count: ", global_data.mnic_count);
        }
        LOG_INFO("atl-mpi-ctx: ", (global_data.ctx_count - 1));
        LOG_INFO("  progress_mode: ", mpi_ctx->progress_mode);
        LOG_INFO("  sync_coll: ", mpi_ctx->sync_coll);
    }

    for (i = 0; i < attr->in.ep_count; i++) {
        ret = atl_mpi_ep_init(mpi_ctx, i, &(ctx->eps[i]));
        if (ret)
            goto err_ep_dup;
    }

    *out_ctx = &mpi_ctx->ctx;

    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &is_tag_ub_set);

    /* report actual attributes back to upper level */
    attr->out.enable_shm = 0;
    attr->out.enable_rma = 0;
    attr->out.enable_device_buf = attr->in.enable_device_buf & global_data.mpi_lib_attr.device_buf;
    attr->out.mnic_type = global_data.mnic_type;
    attr->out.mnic_count = global_data.mnic_count;
    attr->out.tag_bits = 32;
    attr->out.max_tag = (is_tag_ub_set) ? *((int*)tag_ub_ptr) : 0;
    attr->out.max_order_waw_size = 0;

    return ATL_STATUS_SUCCESS;

err_ep_dup:
    for (i = 0; i < attr->in.ep_count; i++) {
        atl_mpi_ep_t* mpi_ep = container_of(ctx->eps[i], atl_mpi_ep_t, ep);

        if (ctx->eps[i] && mpi_ep) {
            if (mpi_ctx->progress_mode == ATL_PROGRESS_POLL) {
                MPI_Cancel(&(mpi_ep->dummy_req.native_req));
                MPI_Comm_free(&mpi_ep->dummy_comm);
            }
            MPI_Comm_free(&mpi_ep->mpi_comm);
        }
    }
    free(ctx->eps);

err_after_init:
    global_data.ctx_count--;
    if (global_data.ctx_count == 0) {
        atl_mpi_bf16_finalize();
        atl_mpi_fp16_finalize();
        if (!global_data.is_external_init) {
            MPI_Finalize();
        }
    }

err_init:
    free(mpi_ctx);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_mpi_main_addr_reserve(char* main_addr) {
    return ATL_STATUS_UNSUPPORTED;
}
