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
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string>

#ifndef container_of
#define container_of(ptr, type, field) ((type*)((char*)ptr - offsetof(type, field)))
#endif

#ifndef gettid
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)
#endif

#define SIZEOFARR(arr) (sizeof(arr) / sizeof(arr[0]))

#define ATL_CACHELINE_LEN     64
#define ATL_REQ_SIZE          16
#define ATL_PROGRESS_MODE_ENV "ATL_PROGRESS_MODE"
#define ATL_MAX_HOSTNAME_LEN  64

#define DIR_SEP  '/'
#define FILENAME (strrchr(__FILE__, DIR_SEP) ? strrchr(__FILE__, DIR_SEP) + 1 : __FILE__)

/*
 * Dynamically loaded transports must export the following entry point.
 * This is invoked by the ATL framework when the transport library is loaded.
 */

#define ATL_CALL(func, err_action) \
    do { \
        atl_status_t status = func; \
        if (status != FI_SUCCESS) { \
            CCL_THROW(#func "\n fails with status: ", status); \
            err_action; \
        } \
    } while (0)

class ipmi;

typedef struct atl_ctx atl_ctx_t;
typedef struct atl_ep atl_ep_t;

typedef enum { ATL_PROGRESS_POLL, ATL_PROGRESS_CHECK } atl_progress_mode_t;

typedef enum { ATL_RA_WAIT, ATL_RA_RUN, ATL_RA_FINALIZE } atl_resize_action_t;

typedef atl_resize_action_t (*atl_resize_fn_t)(int size);

typedef enum {
    ATL_STATUS_SUCCESS,
    ATL_STATUS_FAILURE,
    ATL_STATUS_AGAIN,
    ATL_STATUS_UNSUPPORTED
} atl_status_t;

inline const char* atl_status_to_str(atl_status_t status) {
    switch (status) {
        case ATL_STATUS_SUCCESS: return "SUCCESS";
        case ATL_STATUS_FAILURE: return "FAILURE";
        case ATL_STATUS_UNSUPPORTED: return "UNSUPPORTED";
        default: return "UNKNOWN";
    }
}

typedef enum {
    ATL_DTYPE_INT8,
    ATL_DTYPE_UINT8,
    ATL_DTYPE_INT16,
    ATL_DTYPE_UINT16,
    ATL_DTYPE_INT32,
    ATL_DTYPE_UINT32,
    ATL_DTYPE_INT64,
    ATL_DTYPE_UINT64,

    ATL_DTYPE_FLOAT16,
    ATL_DTYPE_FLOAT32,
    ATL_DTYPE_FLOAT64,

    ATL_DTYPE_BFLOAT16
} atl_datatype_t;

typedef enum {
    ATL_REDUCTION_SUM,
    ATL_REDUCTION_PROD,
    ATL_REDUCTION_MIN,
    ATL_REDUCTION_MAX,
    ATL_REDUCTION_CUSTOM
} atl_reduction_t;

typedef enum { ATL_MNIC_NONE, ATL_MNIC_LOCAL, ATL_MNIC_GLOBAL } atl_mnic_t;

typedef struct {
    struct {
        int enable_shm;
        int enable_rma;
        int enable_hmem;
        int enable_sync_coll;
        int enable_extra_ep;
        size_t ep_count;
        atl_mnic_t mnic_type;
        std::string mnic_name;
        size_t mnic_count;
    } in;
    struct {
        int enable_shm;
        int enable_rma;
        int enable_hmem;
        atl_mnic_t mnic_type;
        size_t mnic_count;
        size_t tag_bits;
        uint64_t max_tag;
        size_t max_order_waw_size;
    } out;
} atl_attr_t;

typedef struct {
    void* buf;
    size_t len;
    uintptr_t local_key;
    uintptr_t remote_key;
} atl_mr_t;

typedef struct {
    int global_idx;
    int global_count;
    int local_idx;
    int local_count;
    size_t hostname_hash;
} atl_proc_coord_t;

typedef struct {
    uint64_t tag;
    size_t remote_proc_idx;
    void* internal[ATL_REQ_SIZE];
} atl_req_t __attribute__((aligned(ATL_CACHELINE_LEN)));

struct atl_ctx {
    atl_proc_coord_t coord;

    size_t ep_count;
    atl_ep_t** eps;
};

/*
   name convention
   len - for bytes
   count - for iov and for dtype-arrays like in reduce/allreduce
*/

struct atl_ep {
    size_t idx;
    atl_ctx_t* ctx;
};
