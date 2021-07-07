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
#define ATL_REQ_SIZE          8
#define ATL_PROGRESS_MODE_ENV "ATL_PROGRESS_MODE"

#define DIR_SEP  '/'
#define FILENAME (strrchr(__FILE__, DIR_SEP) ? strrchr(__FILE__, DIR_SEP) + 1 : __FILE__)

/*
 * Dynamically loaded transports must export the following entry point.
 * This is invoked by the ATL framework when the transport library is loaded.
 */

#define ATL_EXT_INI atl_status_t atl_ini(atl_transport_t* atl_transport)

#define ATL_OFI_INI ATL_EXT_INI
#define ATL_MPI_INI ATL_EXT_INI

#define ATL_CALL(func, err_action) \
    do { \
        atl_status_t status = func; \
        if (status != FI_SUCCESS) { \
            LOG_ERROR(#func "\n fails with status: ", status); \
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
        int enable_device_buf;
        int enable_sync_coll;
        int enable_extra_ep;
        size_t ep_count;
        atl_mnic_t mnic_type;
        size_t mnic_count;
    } in;
    struct {
        int enable_shm;
        int enable_rma;
        int enable_device_buf;
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
} atl_proc_coord_t;

typedef struct {
    uint64_t tag;
    size_t remote_proc_idx;
    void* internal[ATL_REQ_SIZE];
} atl_req_t __attribute__((aligned(ATL_CACHELINE_LEN)));

typedef struct {
    const char* name;
    atl_status_t (*init)(int* argc,
                         char*** argv,
                         atl_attr_t* attr,
                         atl_ctx_t** ctx,
                         const char* main_addr,
                         ipmi* pmi);
    atl_status_t (*reserve_addr)(char* main_addr);
} atl_transport_t;

typedef struct {
    atl_status_t (*finalize)(atl_ctx_t* ctx);
} atl_ops_t;

typedef struct {
    atl_status_t (*mr_reg)(atl_ctx_t* ctx, const void* buf, size_t len, atl_mr_t** mr);
    atl_status_t (*mr_dereg)(atl_ctx_t* ctx, atl_mr_t* mr);
} atl_mr_ops_t;

struct atl_ctx {
    atl_ops_t* ops;
    atl_mr_ops_t* mr_ops;
    atl_proc_coord_t coord;

    size_t ep_count;
    atl_ep_t** eps;
};

/*
   name convention
   len - for bytes
   count - for iov and for dtype-arrays like in reduce/allreduce
*/

typedef struct {
    atl_status_t (*send)(atl_ep_t* ep,
                         const void* buf,
                         size_t len,
                         int dst_proc_idx,
                         uint64_t tag,
                         atl_req_t* req);
    atl_status_t (
        *recv)(atl_ep_t* ep, void* buf, size_t len, int src_proc_idx, uint64_t tag, atl_req_t* req);
    atl_status_t (
        *probe)(atl_ep_t* ep, int src_proc_idx, uint64_t tag, int* found, size_t* recv_len);
} atl_p2p_ops_t;

typedef struct {
    /* order convention - keep alphabetical order */
    atl_status_t (*allgatherv)(atl_ep_t* ep,
                               const void* send_buf,
                               size_t send_len,
                               void* recv_buf,
                               const int* recv_lens,
                               const int* offsets,
                               atl_req_t* req);
    atl_status_t (*allreduce)(atl_ep_t* ep,
                              const void* send_buf,
                              void* recv_buf,
                              size_t count,
                              atl_datatype_t dtype,
                              atl_reduction_t op,
                              atl_req_t* req);
    atl_status_t (
        *alltoall)(atl_ep_t* ep, const void* send_buf, void* recv_buf, size_t len, atl_req_t* req);
    atl_status_t (*alltoallv)(atl_ep_t* ep,
                              const void* send_buf,
                              const int* send_lens,
                              const int* send_offsets,
                              void* recv_buf,
                              const int* recv_lens,
                              const int* recv_offsets,
                              atl_req_t* req);
    atl_status_t (*barrier)(atl_ep_t* ep, atl_req_t* req);
    atl_status_t (*bcast)(atl_ep_t* ep, void* buf, size_t len, int root, atl_req_t* req);
    atl_status_t (*reduce)(atl_ep_t* ep,
                           const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           int root,
                           atl_datatype_t dtype,
                           atl_reduction_t op,
                           atl_req_t* req);
    atl_status_t (*reduce_scatter)(atl_ep_t* ep,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t recv_count,
                                   atl_datatype_t dtype,
                                   atl_reduction_t op,
                                   atl_req_t* req);
} atl_coll_ops_t;

typedef struct {
    atl_status_t (*read)(atl_ep_t* ep,
                         void* buf,
                         size_t len,
                         atl_mr_t* mr,
                         uint64_t addr,
                         uintptr_t remote_key,
                         int dst_proc_idx,
                         atl_req_t* req);
    atl_status_t (*write)(atl_ep_t* ep,
                          const void* buf,
                          size_t len,
                          atl_mr_t* mr,
                          uint64_t addr,
                          uintptr_t remote_key,
                          int dst_proc_idx,
                          atl_req_t* req);
} atl_rma_ops_t;

typedef struct {
    atl_status_t (*wait)(atl_ep_t* ep, atl_req_t* req);
    atl_status_t (*wait_all)(atl_ep_t* ep, atl_req_t* reqs, size_t count);
    atl_status_t (*cancel)(atl_ep_t* ep, atl_req_t* req);
    atl_status_t (*poll)(atl_ep_t* ep);
    atl_status_t (*check)(atl_ep_t* ep, int* is_completed, atl_req_t* req);
} atl_comp_ops_t;

struct atl_ep {
    size_t idx;
    atl_ctx_t* ctx;
    atl_p2p_ops_t* p2p_ops;
    atl_coll_ops_t* coll_ops;
    atl_rma_ops_t* rma_ops;
    atl_comp_ops_t* comp_ops;
};
