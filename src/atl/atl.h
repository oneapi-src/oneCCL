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
#define ATL_EXT_INI \
    __attribute__((visibility("default"))) atl_status_t atl_ini(atl_transport_t* atl_transport)

#define ATL_OFI_INI ATL_EXT_INI
#define ATL_MPI_INI ATL_EXT_INI

typedef struct atl_ctx atl_ctx_t;
typedef struct atl_ep atl_ep_t;

typedef enum { ATL_PROGRESS_POLL, ATL_PROGRESS_CHECK } atl_progress_mode_t;

typedef enum { ATL_RA_WAIT, ATL_RA_RUN, ATL_RA_FINALIZE } atl_resize_action_t;

typedef atl_resize_action_t (*atl_resize_fn_t)(size_t size);

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
    ATL_DTYPE_CHAR,
    ATL_DTYPE_INT,
    ATL_DTYPE_BFP16,
    ATL_DTYPE_FLOAT,
    ATL_DTYPE_DOUBLE,
    ATL_DTYPE_INT64,
    ATL_DTYPE_UINT64
} atl_datatype_t;

typedef enum {
    ATL_REDUCTION_SUM,
    ATL_REDUCTION_PROD,
    ATL_REDUCTION_MIN,
    ATL_REDUCTION_MAX,
    ATL_REDUCTION_CUSTOM
} atl_reduction_t;

typedef struct {
    size_t ep_count;
    int enable_shm;
    size_t tag_bits;
    uint64_t max_tag;
    int enable_rma;
    size_t max_order_waw_size;
} atl_attr_t;

typedef struct {
    void* buf;
    size_t len;
    uintptr_t local_key;
    uintptr_t remote_key;
} atl_mr_t;

typedef struct {
    size_t global_idx;
    size_t global_count;
    size_t local_idx;
    size_t local_count;
} atl_proc_coord_t;

typedef struct {
    uint64_t tag;
    size_t remote_proc_idx;
    void* internal[ATL_REQ_SIZE];
} atl_req_t __attribute__((aligned(ATL_CACHELINE_LEN)));

typedef struct {
    const char* name;
    atl_status_t (
        *init)(int* argc, char*** argv, atl_attr_t* attr, atl_ctx_t** ctx, const char* main_addr);
    atl_status_t (*main_addr_reserv)(char* main_addr);
} atl_transport_t;

typedef struct {
    atl_status_t (*finalize)(atl_ctx_t* ctx);
    atl_status_t (*update)(atl_ctx_t* ctx);
    atl_status_t (*wait_notification)(atl_ctx_t* ctx);
    atl_status_t (*set_resize_function)(atl_resize_fn_t fn);
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

    int is_resize_enabled;
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
                         size_t dst_proc_idx,
                         uint64_t tag,
                         atl_req_t* req);
    atl_status_t (*recv)(atl_ep_t* ep,
                         void* buf,
                         size_t len,
                         size_t src_proc_idx,
                         uint64_t tag,
                         atl_req_t* req);
    atl_status_t (
        *probe)(atl_ep_t* ep, size_t src_proc_idx, uint64_t tag, int* found, size_t* recv_len);
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
    atl_status_t (*bcast)(atl_ep_t* ep, void* buf, size_t len, size_t root, atl_req_t* req);
    atl_status_t (*reduce)(atl_ep_t* ep,
                           const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           size_t root,
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
                         size_t dst_proc_idx,
                         atl_req_t* req);
    atl_status_t (*write)(atl_ep_t* ep,
                          const void* buf,
                          size_t len,
                          atl_mr_t* mr,
                          uint64_t addr,
                          uintptr_t remote_key,
                          size_t dst_proc_idx,
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

atl_status_t atl_init(const char* transport_name,
                      int* argc,
                      char*** argv,
                      atl_attr_t* att,
                      atl_ctx_t** ctx,
                      const char* main_addr);

void atl_main_addr_reserv(char* main_addr);

static inline atl_status_t atl_finalize(atl_ctx_t* ctx) {
    return ctx->ops->finalize(ctx);
}

static inline atl_status_t atl_update(atl_ctx_t* ctx) {
    return ctx->ops->update(ctx);
}

static inline atl_status_t atl_wait_notification(atl_ctx_t* ctx) {
    return ctx->ops->wait_notification(ctx);
}

static inline atl_status_t atl_set_resize_function(atl_ctx_t* ctx, atl_resize_fn_t fn) {
    return ctx->ops->set_resize_function(fn);
}

static inline atl_ep_t** atl_get_eps(atl_ctx_t* ctx) {
    return ctx->eps;
}

static inline atl_proc_coord_t* atl_get_proc_coord(atl_ctx_t* ctx) {
    return &(ctx->coord);
}

static inline int atl_is_resize_enabled(atl_ctx_t* ctx) {
    return ctx->is_resize_enabled;
}

static inline atl_status_t atl_mr_reg(atl_ctx_t* ctx, const void* buf, size_t len, atl_mr_t** mr) {
    return ctx->mr_ops->mr_reg(ctx, buf, len, mr);
}

static inline atl_status_t atl_mr_dereg(atl_ctx_t* ctx, atl_mr_t* mr) {
    return ctx->mr_ops->mr_dereg(ctx, mr);
}

static inline atl_status_t atl_ep_send(atl_ep_t* ep,
                                       const void* buf,
                                       size_t len,
                                       size_t dst_proc_idx,
                                       uint64_t tag,
                                       atl_req_t* req) {
    return ep->p2p_ops->send(ep, buf, len, dst_proc_idx, tag, req);
}

static inline atl_status_t atl_ep_recv(atl_ep_t* ep,
                                       void* buf,
                                       size_t len,
                                       size_t src_proc_idx,
                                       uint64_t tag,
                                       atl_req_t* req) {
    return ep->p2p_ops->recv(ep, buf, len, src_proc_idx, tag, req);
}

static inline atl_status_t atl_ep_probe(atl_ep_t* ep,
                                        size_t src_proc_idx,
                                        uint64_t tag,
                                        int* found,
                                        size_t* recv_len) {
    return ep->p2p_ops->probe(ep, src_proc_idx, tag, found, recv_len);
}

static inline atl_status_t atl_ep_allgatherv(atl_ep_t* ep,
                                             const void* send_buf,
                                             size_t send_len,
                                             void* recv_buf,
                                             const int* recv_lens,
                                             const int* offsets,
                                             atl_req_t* req) {
    return ep->coll_ops->allgatherv(ep, send_buf, send_len, recv_buf, recv_lens, offsets, req);
}

static inline atl_status_t atl_ep_allreduce(atl_ep_t* ep,
                                            const void* send_buf,
                                            void* recv_buf,
                                            size_t len,
                                            atl_datatype_t dtype,
                                            atl_reduction_t op,
                                            atl_req_t* req) {
    return ep->coll_ops->allreduce(ep, send_buf, recv_buf, len, dtype, op, req);
}

static inline atl_status_t atl_ep_alltoall(atl_ep_t* ep,
                                           const void* send_buf,
                                           void* recv_buf,
                                           int len,
                                           atl_req_t* req) {
    return ep->coll_ops->alltoall(ep, send_buf, recv_buf, len, req);
}

static inline atl_status_t atl_ep_alltoallv(atl_ep_t* ep,
                                            const void* send_buf,
                                            const int* send_lens,
                                            const int* send_offsets,
                                            void* recv_buf,
                                            const int* recv_lens,
                                            const int* recv_offsets,
                                            atl_req_t* req) {
    return ep->coll_ops->alltoallv(
        ep, send_buf, send_lens, send_offsets, recv_buf, recv_lens, recv_offsets, req);
}

static inline atl_status_t atl_ep_barrier(atl_ep_t* ep, atl_req_t* req) {
    return ep->coll_ops->barrier(ep, req);
}

static inline atl_status_t atl_ep_bcast(atl_ep_t* ep,
                                        void* buf,
                                        size_t len,
                                        size_t root,
                                        atl_req_t* req) {
    return ep->coll_ops->bcast(ep, buf, len, root, req);
}

static inline atl_status_t atl_ep_reduce(atl_ep_t* ep,
                                         const void* send_buf,
                                         void* recv_buf,
                                         size_t len,
                                         size_t root,
                                         atl_datatype_t dtype,
                                         atl_reduction_t op,
                                         atl_req_t* req) {
    return ep->coll_ops->reduce(ep, send_buf, recv_buf, len, root, dtype, op, req);
}

static inline atl_status_t atl_ep_read(atl_ep_t* ep,
                                       void* buf,
                                       size_t len,
                                       atl_mr_t* mr,
                                       uint64_t addr,
                                       uintptr_t remote_key,
                                       size_t dst_proc_idx,
                                       atl_req_t* req) {
    return ep->rma_ops->read(ep, buf, len, mr, addr, remote_key, dst_proc_idx, req);
}

static inline atl_status_t atl_ep_write(atl_ep_t* ep,
                                        const void* buf,
                                        size_t len,
                                        atl_mr_t* mr,
                                        uint64_t addr,
                                        uintptr_t remote_key,
                                        size_t dst_proc_idx,
                                        atl_req_t* req) {
    return ep->rma_ops->write(ep, buf, len, mr, addr, remote_key, dst_proc_idx, req);
}

static inline atl_status_t atl_ep_wait(atl_ep_t* ep, atl_req_t* req) {
    return ep->comp_ops->wait(ep, req);
}

static inline atl_status_t atl_ep_wait_all(atl_ep_t* ep, atl_req_t* req, size_t count) {
    return ep->comp_ops->wait_all(ep, req, count);
}

static inline atl_status_t atl_ep_cancel(atl_ep_t* ep, atl_req_t* req) {
    return ep->comp_ops->cancel(ep, req);
}

static inline atl_status_t atl_ep_poll(atl_ep_t* ep) {
    return ep->comp_ops->poll(ep);
}

static inline atl_status_t atl_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) {
    return ep->comp_ops->check(ep, is_completed, req);
}
