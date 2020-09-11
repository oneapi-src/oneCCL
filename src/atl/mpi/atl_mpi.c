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
#include "comp/bfp16/bfp16_intrisics.h"
#include "comp/bfp16/bfp16_utils.h"

#define ATL_MPI_PM_KEY     "atl-mpi"
#define EP_IDX_MAX_STR_LEN 4
#define EP_IDX_KEY         "ep_idx"

#define ATL_MPI_PRINT(s, ...) \
    do { \
        pid_t tid = gettid(); \
        char hoststr[32]; \
        gethostname(hoststr, sizeof(hoststr)); \
        fprintf(stdout, \
                "(%d): %s: @ %s:%d:%s() " s "\n", \
                tid, \
                hoststr, \
                FILENAME, \
                __LINE__, \
                __func__, \
                ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

#define ATL_MPI_ASSERT(cond, ...) \
    do { \
        if (!(cond)) { \
            ATL_MPI_PRINT("ASSERT failed, cond: " #cond " " __VA_ARGS__); \
            exit(0); \
        } \
    } while (0)

#define ATL_MPI_PRINT_ROOT(s, ...) \
    if (coord->global_idx == 0) \
        ATL_MPI_PRINT(s, ##__VA_ARGS__);

#ifdef ENABLE_DEBUG
#define ATL_MPI_DEBUG_PRINT(s, ...)      ATL_MPI_PRINT(s, ##__VA_ARGS__)
#define ATL_MPI_DEBUG_PRINT_ROOT(s, ...) ATL_MPI_PRINT_ROOT(s, ##__VA_ARGS__)
#else
#define ATL_MPI_DEBUG_PRINT(s, ...)
#define ATL_MPI_DEBUG_PRINT_ROOT(s, ...)
#endif

#define RET2ATL(ret) (ret != MPI_SUCCESS) ? ATL_STATUS_FAILURE : ATL_STATUS_SUCCESS

typedef enum { ATL_MPI_LIB_NONE, ATL_MPI_LIB_IMPI } atl_mpi_lib_type_t;

typedef struct {
    atl_mpi_lib_type_t type;
    const char* name;

    /* string prefix before numerical version of library, mandatory */
    const char* version_prefix_1;

    /* string prefix before numerical version of library, following prefix_1, optional */
    const char* version_prefix_2;

    /* minimal expected version of library, mandatory */
    int min_version_value;

    /* string prefix before library kind, optional */
    const char* kind_prefix;

    /* library kind, optional */
    const char* kind_value;
} atl_mpi_lib_info_t;

#define MPI_LIB_INFO_MAX_COUNT 1

static atl_mpi_lib_info_t mpi_lib_infos[MPI_LIB_INFO_MAX_COUNT] = {
    { ATL_MPI_LIB_IMPI, "impi", "Intel(R) MPI Library", "", 2019, "library kind:", "release_mt" }
};

typedef struct {
#ifdef CCL_BFP16_COMPILER

#ifndef ATL_MPI_BFP16

// BFP16 type support
#define ATL_MPI_BFP16 /* more strict than CCL_BFP16_COMPILER */

    // custom MPI operations for BFP16
    MPI_Op sum_op;
    MPI_Op prod_op;
    MPI_Op min_op;
    MPI_Op max_op;
#else /* ATL_MPI_BFP16 */
#error "MPI_BFP16 is already defined, unsupported case"
#endif /* ATL_MPI_BFP16 */
#endif /* CCL_BFP16_COMPILER */

    // custom MPI dtype for BFP16
    MPI_Datatype dtype;

    ccl_bfp16_impl_type impl_type;

} atl_mpi_bfp16_data_t;

typedef struct {
    atl_mpi_lib_type_t mpi_lib_type;
    atl_mpi_bfp16_data_t bfp16;
} atl_mpi_global_data_t;

static atl_mpi_global_data_t global_data;
static int is_external_init = 0;

typedef enum { ATL_MPI_COMP_POSTED, ATL_MPI_COMP_COMPLETED } atl_mpi_comp_state_t;

typedef struct {
    MPI_Request native_req;
    atl_mpi_comp_state_t comp_state;
} atl_mpi_req_t;

typedef struct {
    atl_ctx_t ctx;
    atl_progress_mode_t progress_mode;
} atl_mpi_ctx_t;

typedef struct {
    atl_ep_t ep;
    MPI_Comm mpi_comm;

    /* dummy recv operation to ensure progress in atl_poll */
    atl_mpi_req_t dummy_req;
    MPI_Comm dummy_comm;
} atl_mpi_ep_t;

#define MPI_BFP16 \
    ({ \
        ATL_MPI_ASSERT(global_data.bfp16.dtype != MPI_DATATYPE_NULL, \
                       "unsupported datatype: ATL_DTYPE_BFP16"); \
        global_data.bfp16.dtype; \
    })

#ifdef ATL_MPI_BFP16

// helpers: check contract
static inline void atl_mpi_check_op_params(void* in_buf,
                                           void* inout_buf,
                                           int* length,
                                           MPI_Datatype* datatype,
                                           const char* caller_func_name) {
    (void)datatype;
    ATL_MPI_ASSERT(in_buf && inout_buf && length,
                   "%s requested, bad arguments: %p, %p, %p",
                   caller_func_name,
                   in_buf,
                   inout_buf,
                   length);
}

static void INLINE_TARGET_ATTRIBUTE_ALL atl_mpi_bfp16_base_op(void* in,
                                                              void* inout,
                                                              int* length,
                                                              ccl_bfp16_reduction_func_ptr op) {
    unsigned short* in_buf = (unsigned short*)in;
    unsigned short* inout_buf = (unsigned short*)inout;

    size_t len = *length;
    ccl_bfp16_reduce_impl(in_buf, inout_buf, len, op, global_data.bfp16.impl_type);
}

// MPI BFP16 operation definitions
static void TARGET_ATTRIBUTE_ALL atl_mpi_bfp16_sum_op(void* in,
                                                      void* inout,
                                                      int* length,
                                                      MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bfp16_base_op(in, inout, length, &sum_wrap);
}

static void TARGET_ATTRIBUTE_ALL atl_mpi_bfp16_prod_op(void* in,
                                                       void* inout,
                                                       int* length,
                                                       MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bfp16_base_op(in, inout, length, &prod_wrap);
}

static void TARGET_ATTRIBUTE_ALL atl_mpi_bfp16_min_op(void* in,
                                                      void* inout,
                                                      int* length,
                                                      MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bfp16_base_op(in, inout, length, &min_wrap);
}

static void TARGET_ATTRIBUTE_ALL atl_mpi_bfp16_max_op(void* in,
                                                      void* inout,
                                                      int* length,
                                                      MPI_Datatype* datatype) {
    atl_mpi_check_op_params(in, inout, length, datatype, __FUNCTION__);
    atl_mpi_bfp16_base_op(in, inout, length, &max_wrap);
}

static void atl_mpi_print_error(int error) {
    char str_error[MPI_MAX_ERROR_STRING];
    int result_len = MPI_MAX_ERROR_STRING;

    MPI_Error_string(error, str_error, &result_len);

    if (result_len > MPI_MAX_ERROR_STRING) {
        result_len = MPI_MAX_ERROR_STRING;
    }
    str_error[result_len - 1] = '\0';

    ATL_MPI_PRINT("MPI error: %s(%d)", str_error, error);
}
#endif /* ATL_MPI_BFP16 */

static int atl_mpi_bfp16_init() {
    int ret = MPI_SUCCESS;

    global_data.bfp16.dtype = MPI_DATATYPE_NULL;

#ifdef ATL_MPI_BFP16

    global_data.bfp16.sum_op = MPI_OP_NULL;
    global_data.bfp16.prod_op = MPI_OP_NULL;
    global_data.bfp16.min_op = MPI_OP_NULL;
    global_data.bfp16.max_op = MPI_OP_NULL;

    global_data.bfp16.impl_type = ccl_bfp16_get_impl_type();

    if (global_data.bfp16.impl_type == ccl_bfp16_none) {
        ATL_MPI_DEBUG_PRINT("%s: success - BFP16 is not supported on current arch", __FUNCTION__);
        return RET2ATL(ret);
    }

    // create custom MPI BFP16 dtype
    ret = MPI_Type_contiguous(2, MPI_BYTE, &global_data.bfp16.dtype);
    if (ret != MPI_SUCCESS) {
        ATL_MPI_DEBUG_PRINT("cannot create MPI BFP16 dtype");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    ret = MPI_Type_commit(&global_data.bfp16.dtype);
    if (ret != MPI_SUCCESS) {
        ATL_MPI_DEBUG_PRINT("cannot commit MPI BFP16 type");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BFP16 summation op
    ret = MPI_Op_create(&atl_mpi_bfp16_sum_op, 1, &global_data.bfp16.sum_op);
    if (ret != MPI_SUCCESS) {
        ATL_MPI_DEBUG_PRINT("cannot create MPI BFP16 sum op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BFP16 production op
    ret = MPI_Op_create(&atl_mpi_bfp16_prod_op, 1, &global_data.bfp16.prod_op);
    if (ret != MPI_SUCCESS) {
        ATL_MPI_DEBUG_PRINT("cannot create MPI BFP16 prod op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BFP16 min op
    ret = MPI_Op_create(&atl_mpi_bfp16_min_op, 1, &global_data.bfp16.min_op);
    if (ret != MPI_SUCCESS) {
        ATL_MPI_DEBUG_PRINT("cannot create MPI BFP16 min op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

    // create custom MPI BFP16 max op
    ret = MPI_Op_create(&atl_mpi_bfp16_max_op, 1, &global_data.bfp16.max_op);
    if (ret != MPI_SUCCESS) {
        ATL_MPI_DEBUG_PRINT("cannot create MPI BFP16 max op");
        atl_mpi_print_error(ret);
        return RET2ATL(ret);
    }

#endif /* ATL_MPI_BFP16 */

    return RET2ATL(ret);
}

static void atl_mpi_bfp16_finalize() {
#ifdef ATL_MPI_BFP16

    if (global_data.bfp16.dtype != MPI_DATATYPE_NULL) {
        MPI_Type_free(&global_data.bfp16.dtype);
    }

    if (global_data.bfp16.sum_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bfp16.sum_op);
    }

    if (global_data.bfp16.prod_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bfp16.prod_op);
    }

    if (global_data.bfp16.min_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bfp16.min_op);
    }

    if (global_data.bfp16.max_op != MPI_OP_NULL) {
        MPI_Op_free(&global_data.bfp16.max_op);
    }

#endif /* ATL_MPI_BFP16 */
}

static MPI_Datatype atl2mpi_dtype(atl_datatype_t dtype) {
    switch (dtype) {
        case ATL_DTYPE_CHAR: return MPI_CHAR;
        case ATL_DTYPE_INT: return MPI_INT;
        case ATL_DTYPE_BFP16: return MPI_BFP16;
        case ATL_DTYPE_FLOAT: return MPI_FLOAT;
        case ATL_DTYPE_DOUBLE: return MPI_DOUBLE;
        case ATL_DTYPE_INT64: return MPI_LONG_LONG;
        case ATL_DTYPE_UINT64: return MPI_UNSIGNED_LONG_LONG;
        default: printf("unknown datatype: %d\n", dtype); exit(1);
    }
}

static MPI_Op atl2mpi_op(atl_reduction_t rtype, MPI_Datatype dtype) {
#ifdef ATL_MPI_BFP16
    switch (rtype) {
        case ATL_REDUCTION_SUM:
            return dtype == global_data.bfp16.dtype ? global_data.bfp16.sum_op : MPI_SUM;
        case ATL_REDUCTION_PROD:
            return dtype == global_data.bfp16.dtype ? global_data.bfp16.prod_op : MPI_PROD;
        case ATL_REDUCTION_MIN:
            return dtype == global_data.bfp16.dtype ? global_data.bfp16.min_op : MPI_MIN;
        case ATL_REDUCTION_MAX:
            return dtype == global_data.bfp16.dtype ? global_data.bfp16.max_op : MPI_MAX;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
#else /* ATL_MPI_BFP16 */
    (void)dtype;
    switch (rtype) {
        case ATL_REDUCTION_SUM: return MPI_SUM;
        case ATL_REDUCTION_PROD: return MPI_PROD;
        case ATL_REDUCTION_MIN: return MPI_MIN;
        case ATL_REDUCTION_MAX: return MPI_MAX;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
#endif /* ATL_MPI_BFP16 */
}

atl_mpi_lib_type_t atl_mpi_get_lib_type() {
    atl_mpi_lib_type_t lib_type = ATL_MPI_LIB_NONE;
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING] = { 0 };
    int mpi_version_len = -1, i;
    atl_mpi_lib_info_t* final_info = NULL;

    /* request IMPI level append library kind into MPI_Get_library_version output */
    setenv("I_MPI_INFO_LIBRARY_KIND", "1", 0);

    int ret = MPI_Get_library_version(mpi_version, &mpi_version_len);
    if ((ret != MPI_SUCCESS) || (mpi_version_len < 0) ||
        (mpi_version_len > MPI_MAX_LIBRARY_VERSION_STRING)) {
        ATL_MPI_PRINT(
            "can't retrieve MPI version, mpi_version_len %d, ret %d", mpi_version_len, ret);
        return ATL_MPI_LIB_NONE;
    }

    /* remove trailing spaces at the end for more compact log */
    while (strlen(mpi_version) && isspace(mpi_version[strlen(mpi_version) - 1]))
        mpi_version[strlen(mpi_version) - 1] = '\0';

    ATL_MPI_DEBUG_PRINT("\nMPI version:\n%s\n", mpi_version);

    for (i = 0; i < MPI_LIB_INFO_MAX_COUNT; i++) {
        atl_mpi_lib_info_t* info = &(mpi_lib_infos[i]);

        ATL_MPI_ASSERT(info->version_prefix_1, "empty version_prefix_1");
        ATL_MPI_ASSERT(info->min_version_value >= 0, "unexpected minimal version");

        const char* version_substr = NULL;
        if ((version_substr = strstr(mpi_version, info->version_prefix_1))) {
            version_substr += strlen(info->version_prefix_1);
            ATL_MPI_DEBUG_PRINT("\nversion_substr:\n%s\n", version_substr);

            if (info->version_prefix_2) {
                version_substr = strstr(version_substr, info->version_prefix_2);
                if (!version_substr) {
                    ATL_MPI_DEBUG_PRINT("can't find version_prefix_2 %s\n", info->version_prefix_2);
                    continue;
                }
                version_substr += strlen(info->version_prefix_2);
                ATL_MPI_DEBUG_PRINT("\nversion_substr:\n%s\n", version_substr);
            }

            int version_value = (version_substr) ? atoi(version_substr) : -1;
            ATL_MPI_DEBUG_PRINT("\nMPI numerical version:\n%d\n", version_value);

            if (version_value < info->min_version_value) {
                ATL_MPI_PRINT("\n\nWARNING !!!\n\n"
                              "Loaded MPI doesn't match with expected version, "
                              "consider to switch to %s %s%d (min) %s\n",
                              info->version_prefix_1,
                              (info->version_prefix_2 ? info->version_prefix_2 : ""),
                              info->min_version_value,
                              (info->kind_value ? info->kind_value : ""));
                continue;
            }
            else if (info->kind_prefix && info->kind_value) {
                const char* kind_substr = mpi_version;

                /* skip WARNING for IMPI which has high enough version but doesn't support library kind yet */
                if ((kind_substr = strstr(kind_substr, info->kind_prefix))) {
                    kind_substr += strlen(info->kind_prefix);
                    while ((isspace(*kind_substr)) &&
                           (kind_substr < (mpi_version + mpi_version_len)))
                        kind_substr++;

                    ATL_MPI_DEBUG_PRINT("\nkind_substr:\n%s\n", kind_substr);

                    if (strncmp(kind_substr, info->kind_value, strlen(info->kind_value))) {
                        ATL_MPI_PRINT("\n\nWARNING !!!\n\n"
                                      "Loaded MPI version (%d) "
                                      "is higher or equal to minimal expected version (%d) "
                                      "but kind (%s) doesn't match with expected kind (%s), "
                                      "consider to switch to %s %s%d (min version) %s\n",
                                      version_value,
                                      info->min_version_value,
                                      kind_substr,
                                      info->kind_value,
                                      info->version_prefix_1,
                                      (info->version_prefix_2 ? info->version_prefix_2 : ""),
                                      info->min_version_value,
                                      (info->kind_value ? info->kind_value : ""));
                        continue;
                    }
                }

                final_info = info;
                ATL_MPI_DEBUG_PRINT(
                    "set lib_type = %s because "
                    "version (%d) is higher or equal to minimal expected version (%d) "
                    "and kind matches with expected kind",
                    info->name,
                    version_value,
                    info->min_version_value);
                break;
            }
        }
    }

    /* user input has higher priority */
    char* lib_type_env = NULL;
    if ((lib_type_env = getenv("CCL_ATL_MPI_LIB_TYPE")) != NULL) {
        final_info = NULL;
        for (i = 0; i < MPI_LIB_INFO_MAX_COUNT; i++) {
            atl_mpi_lib_info_t* info = &(mpi_lib_infos[i]);

            if (!strcmp(lib_type_env, info->name)) {
                final_info = info;
                ATL_MPI_DEBUG_PRINT("set lib_type = %s because it is requested explicitly",
                                    lib_type_env);
                break;
            }
        }
    }

    if (final_info) {
        ATL_MPI_DEBUG_PRINT("use lib_type = %s", final_info->name);
        lib_type = final_info->type;
    }
    else {
        ATL_MPI_DEBUG_PRINT("use lib_type none");
        lib_type = ATL_MPI_LIB_NONE;
    }

    return lib_type;
}

atl_status_t atl_mpi_set_lib_environment(const atl_attr_t* attr) {
    char ep_count_str[EP_IDX_MAX_STR_LEN] = { 0 };
    snprintf(ep_count_str, EP_IDX_MAX_STR_LEN, "%zu", attr->ep_count);

    if (global_data.mpi_lib_type == ATL_MPI_LIB_IMPI) {
        setenv("I_MPI_THREAD_SPLIT", "1", 0);
        setenv("I_MPI_THREAD_RUNTIME", "generic", 0);
        setenv("I_MPI_THREAD_MAX", ep_count_str, 0);
        setenv("I_MPI_THREAD_ID_KEY", EP_IDX_KEY, 0);
        setenv("I_MPI_THREAD_LOCK_LEVEL", "vci", 0);

        if (attr->enable_shm)
            setenv("I_MPI_FABRICS", "shm:ofi", 0);
        else
            setenv("I_MPI_FABRICS", "ofi", 0);

        if (attr->ep_count)
            setenv("I_MPI_OFI_ISEND_INJECT_THRESHOLD", "0", 0);
    }

    return ATL_STATUS_SUCCESS;
}

static atl_status_t atl_mpi_finalize(atl_ctx_t* ctx) {
    int ret = MPI_SUCCESS;

    atl_mpi_ctx_t* mpi_ctx = container_of(ctx, atl_mpi_ctx_t, ctx);

    atl_ep_t** eps = ctx->eps;

    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);

    if (!is_mpi_finalized) {
        for (int i = 0; i < ctx->ep_count; i++) {
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

        if (!is_external_init) {
            atl_mpi_bfp16_finalize();
            ret = MPI_Finalize();
        }
        else
            ATL_MPI_DEBUG_PRINT("MPI_Init has been called externally, skip MPI_Finalize");
    }
    else
        ATL_MPI_DEBUG_PRINT("MPI_Finalize has been already called");

    free(eps);
    free(mpi_ctx);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_update(atl_ctx_t* ctx) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_wait_notification(atl_ctx_t* ctx) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_set_resize_function(atl_resize_fn_t fn) {
    return ATL_STATUS_UNSUPPORTED;
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
                                    size_t dest_proc_idx,
                                    uint64_t tag,
                                    atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Isend(
        buf, len, MPI_CHAR, dest_proc_idx, (int)tag, mpi_ep->mpi_comm, &mpi_req->native_req);

#if 0
//#ifdef ENABLE_DEBUG
    if (global_data.mpi_lib_type != ATL_MPI_LIB_NONE)
    {
        MPI_Info info_out;
        char buf[MPI_MAX_INFO_VAL];
        int flag;
        MPI_Comm_get_info(mpi_ep->mpi_comm, &info_out);
        MPI_Info_get(info_out, EP_IDX_KEY, MPI_MAX_INFO_VAL, buf, &flag);
        
        if (!flag)
        {
            ATL_MPI_PRINT("unexpected ep_idx_key %s", EP_IDX_KEY);
            return ATL_STATUS_FAILURE;
        }

        if (ep->idx != atoi(buf))
        {
            ATL_MPI_PRINT("unexpected ep_idx: expected %zu, got %d\n", ep->idx, atoi(buf));
        }

        ATL_MPI_DEBUG_PRINT("len %zu, comm_key %s, comm %p", len, buf, &(mpi_ep->mpi_comm));

        MPI_Info_free(&info_out);
    }
#endif

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_recv(atl_ep_t* ep,
                                    void* buf,
                                    size_t len,
                                    size_t src_proc_idx,
                                    uint64_t tag,
                                    atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Irecv(
        buf, len, MPI_CHAR, src_proc_idx, (int)tag, mpi_ep->mpi_comm, &mpi_req->native_req);

#if 0
//#ifdef ENABLE_DEBUG
    if (global_data.mpi_lib_type != ATL_MPI_LIB_NONE)
    {
        MPI_Info info_out;
        char buf[MPI_MAX_INFO_VAL];
        int flag;
        MPI_Comm_get_info(mpi_ep->mpi_comm, &info_out);
        MPI_Info_get(info_out, EP_IDX_KEY, MPI_MAX_INFO_VAL, buf, &flag);
        
        if (!flag)
        {
            ATL_MPI_PRINT("unexpected ep_idx_key %s", EP_IDX_KEY);
            return ATL_STATUS_FAILURE;
        }

        if (ep->idx != atoi(buf))
        {
            ATL_MPI_PRINT("unexpected ep_idx: expected %zu, got %d\n", ep->idx, atoi(buf));
        }

        ATL_MPI_DEBUG_PRINT("len %zu, comm_key %s, comm %p", len, buf, &(mpi_ep->mpi_comm));

        MPI_Info_free(&info_out);
    }
#endif

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_probe(atl_ep_t* ep,
                                     size_t src_proc_idx,
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

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_allgatherv(atl_ep_t* ep,
                                          const void* send_buf,
                                          size_t send_len,
                                          void* recv_buf,
                                          const int* recv_lens,
                                          const int* offsets,
                                          atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    int ret = MPI_SUCCESS;

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

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_allreduce(atl_ep_t* ep,
                                         const void* send_buf,
                                         void* recv_buf,
                                         size_t count,
                                         atl_datatype_t dtype,
                                         atl_reduction_t op,
                                         atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);
    int ret = MPI_Iallreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             recv_buf,
                             count,
                             mpi_dtype,
                             mpi_op,
                             mpi_ep->mpi_comm,
                             &mpi_req->native_req);

#if 0
//#ifdef ENABLE_DEBUG
    if (global_data.mpi_lib_type != ATL_MPI_LIB_NONE)
    {
        MPI_Info info_out;
        char buf[MPI_MAX_INFO_VAL];
        int flag;
        MPI_Comm_get_info(mpi_ep->mpi_comm, &info_out);
        MPI_Info_get(info_out, EP_IDX_KEY, MPI_MAX_INFO_VAL, buf, &flag);
        
        if (!flag)
        {
            ATL_MPI_PRINT("unexpected ep_idx_key %s", EP_IDX_KEY);
            return ATL_STATUS_FAILURE;
        }

        if (ep->idx != atoi(buf))
        {
            ATL_MPI_PRINT("unexpected ep_idx: expected %zu, got %d\n", ep->idx, atoi(buf));
        }

        ATL_MPI_DEBUG_PRINT("count %zu, comm_key %s, comm %p", count, buf, &(mpi_ep->mpi_comm));

        MPI_Info_free(&info_out);
    }
#endif

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_alltoall(atl_ep_t* ep,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t len,
                                        atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    int ret = MPI_SUCCESS;

    ret = MPI_Ialltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                        len,
                        MPI_CHAR,
                        recv_buf,
                        len,
                        MPI_CHAR,
                        mpi_ep->mpi_comm,
                        &mpi_req->native_req);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

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
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    int ret = MPI_SUCCESS;

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

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_barrier(atl_ep_t* ep, atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Ibarrier(mpi_ep->mpi_comm, &mpi_req->native_req);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_bcast(atl_ep_t* ep,
                                     void* buf,
                                     size_t len,
                                     size_t root,
                                     atl_req_t* req) {
    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Ibcast(buf, len, MPI_CHAR, root, mpi_ep->mpi_comm, &mpi_req->native_req);
    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_reduce(atl_ep_t* ep,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      size_t root,
                                      atl_datatype_t dtype,
                                      atl_reduction_t op,
                                      atl_req_t* req) {
    size_t my_proc_idx = ep->ctx->coord.global_idx;

    atl_mpi_ep_t* mpi_ep = container_of(ep, atl_mpi_ep_t, ep);

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);
    int ret = MPI_Ireduce(
        (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
        recv_buf,
        count,
        mpi_dtype,
        mpi_op,
        root,
        mpi_ep->mpi_comm,
        &mpi_req->native_req);

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_ep_read(atl_ep_t* ep,
                                    void* buf,
                                    size_t len,
                                    atl_mr_t* mr,
                                    uint64_t addr,
                                    uintptr_t r_key,
                                    size_t dest_proc_idx,
                                    atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_ep_write(atl_ep_t* ep,
                                     const void* buf,
                                     size_t len,
                                     atl_mr_t* mr,
                                     uint64_t addr,
                                     uintptr_t r_key,
                                     size_t dest_proc_idx,
                                     atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

static atl_status_t atl_mpi_ep_wait(atl_ep_t* ep, atl_req_t* req) {
    atl_status_t ret;
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
    ATL_MPI_ASSERT(is_completed);

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
    .update = atl_mpi_update,
    .wait_notification = atl_mpi_wait_notification,
    .set_resize_function = atl_mpi_set_resize_function,
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
                                              .reduce = atl_mpi_ep_reduce };

static atl_rma_ops_t atl_mpi_ep_rma_ops = {
    .read = atl_mpi_ep_read,
    .write = atl_mpi_ep_write,
};

static atl_comp_ops_t atl_mpi_ep_comp_ops = { .wait = atl_mpi_ep_wait,
                                              .wait_all = atl_mpi_ep_wait_all,
                                              .poll = atl_mpi_ep_poll,
                                              .check = atl_mpi_ep_check };

static atl_status_t atl_mpi_ep_init(atl_mpi_ctx_t* mpi_ctx, size_t idx, atl_ep_t** ep) {
    int ret;
    atl_mpi_ep_t* mpi_ep = calloc(1, sizeof(atl_mpi_ep_t));
    if (!mpi_ep)
        return ATL_STATUS_FAILURE;

    ret = MPI_Comm_dup(MPI_COMM_WORLD, &mpi_ep->mpi_comm);
    if (ret)
        goto err_ep_dup;

    MPI_Info info;
    MPI_Info_create(&info);
    char ep_idx_str[EP_IDX_MAX_STR_LEN] = { 0 };
    snprintf(ep_idx_str, EP_IDX_MAX_STR_LEN, "%zu", idx);
    MPI_Info_set(info, EP_IDX_KEY, ep_idx_str);
    MPI_Comm_set_info(mpi_ep->mpi_comm, info);

    if (mpi_ctx->progress_mode == ATL_PROGRESS_POLL) {
        ret = MPI_Comm_dup(MPI_COMM_WORLD, &mpi_ep->dummy_comm);
        if (ret)
            goto err_ep_dup;
        MPI_Comm_set_info(mpi_ep->dummy_comm, info);
        MPI_Irecv(NULL, 0, MPI_CHAR, 0, 0, mpi_ep->dummy_comm, &(mpi_ep->dummy_req.native_req));
    }

    MPI_Info_free(&info);

    ATL_MPI_DEBUG_PRINT("idx %zu, ep_idx_str %s", idx, ep_idx_str);

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
                                 const char* main_addr) {
    ATL_MPI_ASSERT(
        (sizeof(atl_mpi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal)),
        "unexpected offset: atl_mpi_request size %zu, atl_request size %zu, expected offset %zu",
        sizeof(atl_mpi_req_t),
        sizeof(atl_req_t),
        offsetof(atl_req_t, internal));

    int ret = MPI_SUCCESS;
    size_t i;
    int is_tag_ub_set = 0;
    void* tag_ub_ptr = NULL;
    int required_thread_level = MPI_THREAD_MULTIPLE, provided_thread_level;

    atl_mpi_ctx_t* mpi_ctx = calloc(1, sizeof(atl_mpi_ctx_t));
    if (!mpi_ctx)
        return ATL_STATUS_FAILURE;

    atl_ctx_t* ctx = &(mpi_ctx->ctx);

    global_data.mpi_lib_type = atl_mpi_get_lib_type();

    if (global_data.mpi_lib_type != ATL_MPI_LIB_NONE) {
        atl_mpi_set_lib_environment(attr);
    }

    int is_mpi_inited = 0;
    MPI_Initialized(&is_mpi_inited);

    if (!is_mpi_inited) {
        ret = MPI_Init_thread(argc, argv, required_thread_level, &provided_thread_level);
        if (provided_thread_level < required_thread_level)
            goto err_init;

        if (atl_mpi_bfp16_init() == ATL_STATUS_FAILURE) {
            atl_mpi_bfp16_finalize();
            goto err_init;
        }
    }
    else {
        is_external_init = 1;
        ATL_MPI_DEBUG_PRINT("MPI was initialized externaly");
    }

    if (ret)
        goto err_init;

    atl_proc_coord_t* coord = &(ctx->coord);

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
    ctx->ep_count = attr->ep_count;
    ctx->eps = calloc(1, sizeof(void*) * attr->ep_count);
    if (!ctx->eps)
        goto err_after_init;
    ctx->is_resize_enabled = 0;

    mpi_ctx->progress_mode = ATL_PROGRESS_CHECK;
    char* progress_mode_env = getenv(ATL_PROGRESS_MODE_ENV);
    if (progress_mode_env) {
        mpi_ctx->progress_mode = atoi(progress_mode_env);
    }
    ATL_MPI_DEBUG_PRINT_ROOT("progress_mode %d", mpi_ctx->progress_mode);

    for (i = 0; i < attr->ep_count; i++) {
        ret = atl_mpi_ep_init(mpi_ctx, i, &(ctx->eps[i]));
        if (ret)
            goto err_ep_dup;
    }

    *out_ctx = &mpi_ctx->ctx;

    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &is_tag_ub_set);

    attr->tag_bits = 32;
    attr->max_tag = (is_tag_ub_set) ? *((int*)tag_ub_ptr) : 0;
    attr->enable_rma = 0;
    attr->max_order_waw_size = 0;

    return ATL_STATUS_SUCCESS;

err_ep_dup:
    for (i = 0; i < attr->ep_count; i++) {
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
    if (!is_external_init) {
        atl_mpi_bfp16_finalize();
        MPI_Finalize();
    }

err_init:
    free(mpi_ctx);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_mpi_main_addr_reserv(char* main_addr) {
    return ATL_STATUS_UNSUPPORTED;
}

ATL_MPI_INI {
    atl_transport->name = "mpi";
    atl_transport->init = atl_mpi_init;
    atl_transport->main_addr_reserv = atl_mpi_main_addr_reserv;
    return ATL_STATUS_SUCCESS;
}
