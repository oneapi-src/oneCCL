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
#include <sys/uio.h>

// TODO: remove:
#define HAVE_OFI    1
#define HAVE_OFI_DL 1
#define HAVE_MPI    0
#define HAVE_MPI_DL 0

/* TODO move to utility code */
#ifndef container_of
#define container_of(ptr, type, field) \
	((type *) ((char *)ptr - offsetof(type, field)))
#endif

#define ATL_CACHELINE_LEN 64
#define ATL_REQ_SIZE      8

typedef enum atl_resize_action
{
    ATL_RA_WAIT     = 0,
    ATL_RA_RUN      = 1,
    ATL_RA_FINALIZE = 2,
} atl_resize_action_t;

typedef atl_resize_action_t (*atl_resize_fn_t)(size_t comm_size);

typedef enum {
    atl_status_success,
    atl_status_failure,
    atl_status_again,

    atl_status_unsupported,
} atl_status_t;

inline const char* atl_status_to_str(atl_status_t status)
{
    switch (status)
    {
        case atl_status_success:
            return "SUCCESS";
        case atl_status_failure:
            return "FAILURE";
        case atl_status_unsupported:
            return "UNSUPPORTED";
        default:
            return "UNKNOWN";
    }
}

typedef enum
{
    atl_dtype_char   = 0,
    atl_dtype_int    = 1,
    atl_dtype_bfp16  = 2,
    atl_dtype_float  = 3,
    atl_dtype_double = 4,
    atl_dtype_int64  = 5,
    atl_dtype_uint64 = 6,
    atl_dtype_custom = 7
} atl_datatype_t;

typedef enum
{
    atl_reduction_sum    = 0,
    atl_reduction_prod   = 1,
    atl_reduction_min    = 2,
    atl_reduction_max    = 3,
    atl_reduction_custom = 4
} atl_reduction_t;

typedef struct atl_desc atl_desc_t;
typedef struct atl_comm atl_comm_t;

typedef struct atl_comm_attr {
} atl_comm_attr_t;

typedef struct atl_attr {
    size_t comm_count;
    int enable_shm;
    int is_tagged_coll_enabled;
    size_t tag_bits;
    uint64_t max_tag;
    int enable_rma;
    size_t max_order_waw_size;
    atl_comm_attr_t comm_attr;
} atl_attr_t;

typedef struct atl_mr {
    void *buf;
    size_t len;
    uintptr_t l_key;
    uintptr_t r_key;
} atl_mr_t;

typedef struct atl_proc_coord {
    size_t global_idx;
    size_t global_count;
    size_t local_idx;
    size_t local_count;
} atl_proc_coord_t;

typedef struct atl_ops {
    void (*global_proc_idx)(atl_desc_t *desc, size_t *global_proc_idx);
    void (*global_proc_count)(atl_desc_t *desc, size_t *global_proc_count);
    atl_status_t (*finalize)(atl_desc_t *desc, atl_comm_t **comms);
    atl_status_t (*update)(atl_proc_coord_t *proc_coord,
                           atl_desc_t *desc, atl_comm_t** atl_comms);
    atl_status_t (*wait_notification)(atl_desc_t *desc);
    atl_status_t (*set_resize_function)(atl_resize_fn_t user_checker);
    size_t is_resize_enabled;
} atl_ops_t;

typedef struct atl_mr_ops {
    atl_status_t (*mr_reg)(atl_desc_t *atl_desc, const void *buf, size_t len,
                           atl_mr_t **atl_mr);
    atl_status_t (*mr_dereg)(atl_desc_t *atl_desc, atl_mr_t *atl_mr);
} atl_mr_ops_t;

struct atl_desc {
    atl_ops_t *ops;
    atl_mr_ops_t *mr_ops;
};

typedef struct atl_transport {
    const char *name;
    atl_status_t (*init)(int *argc, char ***argv, atl_proc_coord_t *proc_coord,
                         atl_attr_t *attr, atl_comm_t ***atl_comms, atl_desc_t **atl_desc);
} atl_transport_t;

typedef struct atl_req {
    uint64_t tag;
    size_t remote_proc_idx;

    void *internal[ATL_REQ_SIZE];
} atl_req_t __attribute__ ((aligned (ATL_CACHELINE_LEN)));


/* len - for bytes */
/* count - for iov and for dtype-arrays like in reduce/allreduce */

typedef struct atl_coll_ops {
    atl_status_t (*allgatherv)(atl_comm_t *comm, const void *send_buf, size_t send_len,
                               void *recv_buf, const int recv_lens[], int  displs[], atl_req_t *req);
    atl_status_t (*allreduce)(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t count,
                              atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req);
    atl_status_t (*alltoall)(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t len,
                             atl_req_t *req);
    atl_status_t (*barrier)(atl_comm_t *comm, atl_req_t *req);
    atl_status_t (*bcast)(atl_comm_t *comm, void *buf, size_t len, size_t root,
                          atl_req_t *req);
    atl_status_t (*reduce)(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t count, size_t root,
                           atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req);
} atl_coll_ops_t;

typedef struct atl_pt2pt_ops {
    /* Non-blocking I/O vector pt2pt ops */
    atl_status_t (*sendv)(atl_comm_t *comm, const struct iovec *iov, size_t count,
                          size_t dest_proc_idx, uint64_t tag, atl_req_t *req);
    atl_status_t (*recvv)(atl_comm_t *comm, struct iovec *iov, size_t count,
                          size_t src_proc_idx, uint64_t tag, atl_req_t *req);
    /* Non-blocking pt2pt ops */
    atl_status_t (*send)(atl_comm_t *comm, const void *buf, size_t len,
                         size_t dest_proc_idx, uint64_t tag, atl_req_t *req);
    atl_status_t (*recv)(atl_comm_t *comm, void *buf, size_t len,
                         size_t src_proc_idx, uint64_t tag, atl_req_t *req);
    atl_status_t (*probe)(atl_comm_t *comm, size_t src_proc_idx, uint64_t tag,
                          int *found, size_t *recv_len);
} atl_pt2pt_ops_t;

typedef struct atl_rma_ops {
    atl_status_t (*read)(atl_comm_t *comm, void *buf, size_t len, atl_mr_t *atl_mr,
                         uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req);
    atl_status_t (*write)(atl_comm_t *comm, const void *buf, size_t len, atl_mr_t *atl_mr,
       uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req);
} atl_rma_ops_t;

typedef struct atl_comp_ops {
    atl_status_t (*wait)(atl_comm_t *comm, atl_req_t *req);
    atl_status_t (*wait_all)(atl_comm_t *comm, atl_req_t *reqs, size_t count);
    atl_status_t (*cancel)(atl_comm_t *comm, atl_req_t *req);
    atl_status_t (*poll)(atl_comm_t *comm);
    atl_status_t (*check)(atl_comm_t *comm, int *status, atl_req_t *req);
} atl_comp_ops_t;

struct atl_comm {
    atl_desc_t *atl_desc;
    atl_coll_ops_t *coll_ops;
    atl_pt2pt_ops_t *pt2pt_ops;
    atl_rma_ops_t *rma_ops;
    atl_comp_ops_t *comp_ops;
};

/*
 * Dynamically loaded transports must export the following entry point.
 * This is invoked by the ATL framework when the transport library
 * is loaded.
 */
#define ATL_EXT_INI                                        \
__attribute__((visibility ("default"))) \
atl_status_t atl_ini(atl_transport_t *atl_transport)

/* Transport initialization function signature that built-in transportss
 * must specify. */
#define INI_SIG(name)                            \
atl_status_t name(atl_transport_t *atl_transport)

/* for each transport defines for three scenarios:
 * dl: externally visible ctor with known name
 * built-in: ctor function def, don't export symbols
 * not built: no-op call for ctor
 */

#if (HAVE_OFI) && (HAVE_OFI_DL)
#  define ATL_OFI_INI ATL_EXT_INI
#  define ATL_MPI_INI ATL_EXT_INI
#  define ATL_OFI_INIT atl_noop_init
#elif (HAVE_OFI)
#  define ATL_OFI_INI INI_SIG(atl_ofi_ini)
#  define ATL_OFI_INIT atl_ofi_init
ATL_OFI_INI ;
#else
#  define ATL_OFI_INIT atl_noop_init
#endif

static inline INI_SIG(atl_noop_init)
{
    return atl_status_success;
}

/* FIXME: use ccl_atl_transport enum instead of char* after better integration of CCL core and ATL codes */
atl_status_t atl_init(const char *transport_name, int *argc, char ***argv, atl_proc_coord_t *proc_coord,
                      atl_attr_t *attr, atl_comm_t ***atl_comms, atl_desc_t **atl_desc);

static inline size_t is_ft_enabled(atl_desc_t *desc)
{
    return desc->ops->is_resize_enabled;
}

static inline void atl_global_proc_idx(atl_desc_t *desc, size_t *global_proc_idx)
{
    desc->ops->global_proc_idx(desc, global_proc_idx);
}

static inline void atl_global_proc_count(atl_desc_t *desc, size_t *global_proc_count)
{
    desc->ops->global_proc_count(desc, global_proc_count);
}

static inline atl_status_t atl_update(atl_proc_coord_t *proc_coord,
                                      atl_desc_t *desc, atl_comm_t** atl_comms)
{
    return desc->ops->update(proc_coord, desc, atl_comms);
}

static inline atl_status_t atl_wait_notification(atl_desc_t *desc)
{
    return desc->ops->wait_notification(desc);
}

static inline atl_status_t atl_finalize(atl_desc_t *desc, atl_comm_t **comms)
{
    return desc->ops->finalize(desc, comms);
}

static inline atl_status_t atl_set_resize_function(atl_desc_t *desc, atl_resize_fn_t user_checker)
{
    return desc->ops->set_resize_function(user_checker);
}

static inline atl_status_t atl_comm_send(atl_comm_t *comm, const void *buf, size_t len,
                                         size_t dest_proc_idx, uint64_t tag, atl_req_t *req)
{
    return comm->pt2pt_ops->send(comm, buf, len, dest_proc_idx, tag, req);
}

static inline atl_status_t atl_comm_recv(atl_comm_t *comm, void *buf, size_t len,
                                         size_t src_proc_idx, uint64_t tag, atl_req_t *req)
{
    return comm->pt2pt_ops->recv(comm, buf, len, src_proc_idx, tag, req);
}

static inline atl_status_t atl_comm_sendv(atl_comm_t *comm, const struct iovec *iov, size_t count,
                                          size_t dest_proc_idx, uint64_t tag, atl_req_t *req)
{
    return comm->pt2pt_ops->sendv(comm, iov, count, dest_proc_idx, tag, req);
}

static inline atl_status_t atl_comm_recvv(atl_comm_t *comm, struct iovec *iov, size_t count,
                                          size_t src_proc_idx, uint64_t tag, atl_req_t *req)
{
    return comm->pt2pt_ops->recvv(comm, iov, count, src_proc_idx, tag, req);
}

static inline atl_status_t atl_mr_reg(atl_desc_t *atl_desc, const void *buf, size_t len,
                                      atl_mr_t **atl_mr)
{
    return atl_desc->mr_ops->mr_reg(atl_desc, buf, len, atl_mr);
}

static inline atl_status_t atl_mr_dereg(atl_desc_t *atl_desc, atl_mr_t *atl_mr)
{
    return atl_desc->mr_ops->mr_dereg(atl_desc, atl_mr);
}

static inline atl_status_t atl_comm_read(atl_comm_t *comm, void *buf, size_t len, atl_mr_t *atl_mr,
                                         uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req)
{
    return comm->rma_ops->read(comm, buf, len, atl_mr, addr, r_key, dest_proc_idx, req);
}

static inline atl_status_t atl_comm_write(atl_comm_t *comm, const void *buf, size_t len, atl_mr_t *atl_mr,
                                          uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req)
{
    return comm->rma_ops->write(comm, buf, len, atl_mr, addr, r_key, dest_proc_idx, req);
}

static inline atl_status_t atl_comm_probe(atl_comm_t *comm, size_t src_proc_idx,
                                          uint64_t tag, int *found, size_t *recv_len)
{
    return comm->pt2pt_ops->probe(comm, src_proc_idx, tag, found, recv_len);
}

static inline atl_status_t atl_comm_cancel(atl_comm_t *comm, atl_req_t *req)
{
    return comm->comp_ops->cancel(comm, req);
}

static inline atl_status_t atl_comm_wait(atl_comm_t *comm, atl_req_t *req)
{
    return comm->comp_ops->wait(comm, req);
}

static inline atl_status_t atl_comm_wait_all(atl_comm_t *comm, atl_req_t *req, size_t count)
{
    return comm->comp_ops->wait_all(comm, req, count);
}

static inline atl_status_t atl_comm_poll(atl_comm_t *comm)
{
    return comm->comp_ops->poll(comm);
}

static inline atl_status_t atl_comm_check(atl_comm_t *comm, int *status, atl_req_t *req)
{
    return comm->comp_ops->check(comm, status, req);
}

static inline atl_status_t
atl_comm_allreduce(atl_comm_t *comm, const void *s_buf, void *r_buf, size_t len,
                   atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req)
{
    return comm->coll_ops->allreduce(comm, s_buf, r_buf, len, dtype, op, req);
}

static inline atl_status_t
atl_comm_reduce(atl_comm_t *comm, const void *s_buf, void *r_buf, size_t len, size_t root,
                atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req)
{
    return comm->coll_ops->reduce(comm, s_buf, r_buf, len, root,  dtype, op,req);
}

static inline atl_status_t
atl_comm_allgatherv(atl_comm_t *comm, const void *s_buf, size_t s_len,
                    void *r_buf, int r_lens[], int displs[], atl_req_t *req)
{
    return comm->coll_ops->allgatherv(comm, s_buf, s_len, r_buf, r_lens, displs, req);
}

static inline atl_status_t
atl_comm_alltoall(atl_comm_t *comm, const void *s_buf, void *r_buf,
                  int lens, atl_req_t *req)
{
    return comm->coll_ops->alltoall(comm, s_buf, r_buf, lens, req);
}

static inline atl_status_t
atl_comm_bcast(atl_comm_t *comm, void *buf, size_t len, size_t root, atl_req_t *req)
{
    return comm->coll_ops->bcast(comm, buf, len, root, req);
}

static inline atl_status_t
atl_comm_barrier(atl_comm_t *comm, atl_req_t *req)
{
    return comm->coll_ops->barrier(comm, req);
}
