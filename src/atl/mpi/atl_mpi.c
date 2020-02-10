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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "atl.h"
#include <unistd.h>
#include <sys/syscall.h>

#include <mpi.h>

#include <inttypes.h>

#ifndef gettid
#define gettid() syscall(SYS_gettid)
#endif

#define SIZEOFARR(arr) (sizeof(arr) / sizeof(arr[0]))

#define COMM_IDX_MAX_STR_LEN 4
#define COMM_IDX_KEY         "comm_idx"

#define OPTIMIZED_MPI_VERSION_PREFIX "Intel(R) MPI Library"

int optimized_impi_versions[] = { 2019, 2020, 2021 };
int is_external_init = 0;

#define ATL_MPI_PRINT(s, ...)                             \
    do {                                                  \
        pid_t tid = gettid();                             \
        char hoststr[32];                                 \
        gethostname(hoststr, sizeof(hoststr));            \
        fprintf(stdout, "(%d): %s: @ %s:%d:%s() " s "\n", \
                tid, hoststr,                             \
                __FILE__, __LINE__,                       \
                __func__, ##__VA_ARGS__);                 \
        fflush(stdout);                                   \
    } while (0)

#ifdef ENABLE_DEBUG
#define ATL_MPI_DEBUG_PRINT(s, ...) ATL_MPI_PRINT(s, ##__VA_ARGS__)
#else
#define ATL_MPI_DEBUG_PRINT(s, ...)
#endif

#define ATL_MPI_PM_KEY "atl-mpi"

#define RET2ATL(ret) (ret != MPI_SUCCESS) ? atl_status_failure : atl_status_success

static const char *atl_mpi_name = "mpi";

typedef struct atl_mpi_context {
    atl_desc_t atl_desc;
    atl_proc_coord_t proc_coord;
    size_t comm_ref_count;
} atl_mpi_context_t;

typedef struct atl_mpi_comm_context {
    atl_comm_t atl_comm;
    size_t idx;
    MPI_Comm mpi_comm;
} atl_mpi_comm_context_t;

typedef enum atl_mpi_comp_state {
    ATL_MPI_COMP_POSTED,
    ATL_MPI_COMP_COMPLETED
} atl_mpi_comp_state_t;

typedef struct atl_mpi_req {
    MPI_Request mpi_context;
    atl_mpi_comp_state_t comp_state;
} atl_mpi_req_t;

static MPI_Datatype
atl2mpi_dtype(atl_datatype_t dtype)
{
    switch (dtype)
    {
        case atl_dtype_char:
            return MPI_CHAR;
        case atl_dtype_int:
            return MPI_INT;
        case atl_dtype_float:
            return MPI_FLOAT;
        case atl_dtype_double:
            return MPI_DOUBLE;
        case atl_dtype_int64:
            return MPI_LONG_LONG;
        case atl_dtype_uint64:
            return MPI_UNSIGNED_LONG_LONG;
        default:
            printf("Unknown datatype: %d\n", dtype);
            exit(1);
    }
}

static MPI_Op
atl2mpi_op(atl_reduction_t rtype)
{
    switch (rtype)
    {
        case atl_reduction_sum:
            return MPI_SUM;
        case atl_reduction_prod:
            return MPI_PROD;
        case atl_reduction_min:
            return MPI_MIN;
        case atl_reduction_max:
            return MPI_MAX;
        default:
            printf("Unknown reduction type: %d\n", rtype);
            exit(1);
    }
}

static atl_status_t atl_mpi_finalize(atl_desc_t *atl_desc, atl_comm_t **atl_comms)
{
    int ret = MPI_SUCCESS;

    atl_mpi_context_t *atl_mpi_context =
        container_of(atl_desc, atl_mpi_context_t, atl_desc);

    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);

    if (!is_mpi_finalized)
    {
        for (int i = 0; i < atl_mpi_context->comm_ref_count; i++)
        {
            atl_mpi_comm_context_t *comm_context =
                container_of(atl_comms[i], atl_mpi_comm_context_t, atl_comm);
            MPI_Comm_free(&comm_context->mpi_comm);
        }

        if (!is_external_init)
            ret = MPI_Finalize();
        else
            ATL_MPI_DEBUG_PRINT("MPI_Init has been called externally, skip MPI_Finalize");
    }
    else
        ATL_MPI_DEBUG_PRINT("MPI_Finalize has been already called");
        

    free(atl_comms);
    free(atl_mpi_context);

    return RET2ATL(ret);
}


static inline atl_status_t atl_mpi_comm_poll(atl_comm_t *comm)
{
    return atl_status_success;
}

/* Non-blocking I/O vector pt2pt ops */
static atl_status_t
atl_mpi_comm_sendv(atl_comm_t *comm, const struct iovec *iov, size_t count,
                   size_t dest_proc_idx, uint64_t tag, atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_mpi_comm_recvv(atl_comm_t *comm, struct iovec *iov, size_t count,
                   size_t src_proc_idx, uint64_t tag, atl_req_t *req)
{
    return atl_status_unsupported;
}

/* Non-blocking pt2pt ops */
static atl_status_t
atl_mpi_comm_send(atl_comm_t *comm, const void *buf, size_t len,
                  size_t dest_proc_idx, uint64_t tag, atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
            container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Isend(buf, len, MPI_CHAR, dest_proc_idx,
                        (int)tag, comm_context->mpi_comm, &mpi_req->mpi_context);
    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_recv(atl_comm_t *comm, void *buf, size_t len,
                  size_t src_proc_idx, uint64_t tag, atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Irecv(buf, len, MPI_CHAR, src_proc_idx,
                        (int)tag, comm_context->mpi_comm, &mpi_req->mpi_context);
    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_allgatherv(atl_comm_t *comm, const void *send_buf, size_t send_len,
                        void *recv_buf, const int recv_lens[], int displs[], atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    int ret = MPI_SUCCESS;

    ret = MPI_Iallgatherv((send_buf == recv_buf) ? MPI_IN_PLACE : send_buf, send_len, MPI_CHAR,
                          recv_buf, recv_lens, displs, MPI_CHAR,
                          comm_context->mpi_comm, &mpi_req->mpi_context);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    
    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_allreduce(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t count,
                       atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op);
    int ret = MPI_Iallreduce((send_buf == recv_buf) ? MPI_IN_PLACE : send_buf,
                             recv_buf, count, mpi_dtype, mpi_op,
                             comm_context->mpi_comm, &mpi_req->mpi_context);

#ifdef ENABLE_DEBUG
    MPI_Info info_out;
    char buf[MPI_MAX_INFO_VAL];
    int flag;
    MPI_Comm_get_info(comm_context->mpi_comm, &info_out);
    MPI_Info_get(info_out, COMM_IDX_KEY, MPI_MAX_INFO_VAL, buf, &flag);
    if (!flag) {
        printf("unexpected key %s", COMM_IDX_KEY);
        return atl_status_failure;
    }
    else {
        //ATL_MPI_DEBUG_PRINT("allreduce: count %zu, comm_key %s, comm %p", count, buf, comm);
    }
    MPI_Info_free(&info_out);
#endif

    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_alltoall(atl_comm_t *comm, const void *send_buf, void *recv_buf,
                      size_t len, atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    int ret = MPI_SUCCESS;

    ret = MPI_Ialltoall((send_buf == recv_buf) ? MPI_IN_PLACE : send_buf, len,  MPI_CHAR,
                        recv_buf, len, MPI_CHAR,
                        comm_context->mpi_comm, &mpi_req->mpi_context);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_barrier(atl_comm_t *comm, atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Ibarrier(comm_context->mpi_comm, &mpi_req->mpi_context);
    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_bcast(atl_comm_t *comm, void *buf, size_t len, size_t root,
                   atl_req_t *req)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    int ret = MPI_Ibcast(buf, len, MPI_CHAR, root,
                         comm_context->mpi_comm, &mpi_req->mpi_context);
    return RET2ATL(ret);
}

static void atl_mpi_global_proc_idx(atl_desc_t *atl_desc, size_t *global_proc_idx);

static atl_status_t
atl_mpi_comm_reduce(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t count, size_t root,
                    atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req)
{
    size_t my_proc_idx;
    atl_mpi_global_proc_idx(comm->atl_desc, &my_proc_idx);
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op);
    int ret = MPI_Ireduce(((send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
                          recv_buf, count, mpi_dtype, mpi_op, root,
                          comm_context->mpi_comm, &mpi_req->mpi_context);
    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_read(atl_comm_t *comm, void *buf, size_t len, atl_mr_t *atl_mr,
                  uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_mpi_comm_write(atl_comm_t *comm, const void *buf, size_t len, atl_mr_t *atl_mr,
                   uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_mpi_comm_probe(atl_comm_t *comm, size_t src_proc_idx, uint64_t tag, int *found, size_t *recv_len)
{
    atl_mpi_comm_context_t *comm_context =
        container_of(comm, atl_mpi_comm_context_t, atl_comm);
    int flag = 0, len = 0, ret;
    MPI_Status status;

    ret = MPI_Iprobe(src_proc_idx, tag, comm_context->mpi_comm, &flag, &status);
    if (flag)
    {
        MPI_Get_count(&status, MPI_BYTE, &len);
    }

    if (found) *found = flag;
    if (recv_len) *recv_len = len;

    return RET2ATL(ret);
}

static atl_status_t atl_mpi_comm_wait(atl_comm_t *comm, atl_req_t *req)
{
    atl_status_t ret;
    MPI_Status status;
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    ret = MPI_Wait(&mpi_req->mpi_context, &status);
    mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    return RET2ATL(ret);
}

static atl_status_t
atl_mpi_comm_wait_all(atl_comm_t *comm, atl_req_t *reqs, size_t count)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_mpi_comm_check(atl_comm_t *comm, int *status, atl_req_t *req)
{
    atl_mpi_req_t *mpi_req = ((atl_mpi_req_t *)req->internal);
    if (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED)
    {
        *status = 1;
        return atl_status_success;
    }

    int flag = 0;
    MPI_Status mpi_status;
    int ret = MPI_Test(&mpi_req->mpi_context, &flag, &mpi_status);
    if (flag)
    {
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    }

    if (status) *status = flag;

    return RET2ATL(ret);
}

static atl_coll_ops_t atl_mpi_comm_coll_ops = {
    .allreduce = atl_mpi_comm_allreduce,
    .reduce = atl_mpi_comm_reduce,
    .allgatherv = atl_mpi_comm_allgatherv,
    .alltoall = atl_mpi_comm_alltoall,
    .bcast = atl_mpi_comm_bcast,
    .barrier = atl_mpi_comm_barrier,
};

static atl_pt2pt_ops_t atl_mpi_comm_pt2pt_ops = {
    .sendv = atl_mpi_comm_sendv,
    .recvv = atl_mpi_comm_recvv,
    .send = atl_mpi_comm_send,
    .recv = atl_mpi_comm_recv,
    .probe = atl_mpi_comm_probe,
};

static atl_rma_ops_t atl_mpi_comm_rma_ops = {
    .read = atl_mpi_comm_read,
    .write = atl_mpi_comm_write,
};

static atl_comp_ops_t atl_mpi_comm_comp_ops = {
    .wait = atl_mpi_comm_wait,
    .wait_all = atl_mpi_comm_wait_all,
    .poll = atl_mpi_comm_poll,
    .check = atl_mpi_comm_check
};

static void atl_mpi_global_proc_idx(atl_desc_t *atl_desc, size_t *global_proc_idx)
{
    atl_mpi_context_t *atl_mpi_context =
        container_of(atl_desc, atl_mpi_context_t, atl_desc);
    *global_proc_idx = atl_mpi_context->proc_coord.global_idx;
}

static void atl_mpi_global_proc_count(atl_desc_t *atl_desc, size_t *global_proc_count)
{
    atl_mpi_context_t *atl_mpi_context =
        container_of(atl_desc, atl_mpi_context_t, atl_desc);
    *global_proc_count = atl_mpi_context->proc_coord.global_count;
}

static atl_status_t
atl_mpi_update(atl_proc_coord_t *proc_coord, atl_desc_t *atl_desc, atl_comm_t** atl_comms)
{
    return atl_status_unsupported;
}

static atl_status_t atl_mpi_mr_reg(atl_desc_t *atl_desc, const void *buf, size_t len,
                                   atl_mr_t **atl_mr)
{
    return atl_status_unsupported;
}

static atl_status_t atl_mpi_mr_dereg(atl_desc_t *atl_desc, atl_mr_t *atl_mr)
{
    return atl_status_unsupported;
}

static atl_status_t atl_mpi_set_resize_function(atl_resize_fn_t resize_fn)
{
    return atl_status_unsupported;
}

atl_ops_t atl_mpi_ops = {
    .global_proc_idx        = atl_mpi_global_proc_idx,
    .global_proc_count      = atl_mpi_global_proc_count,
    .finalize               = atl_mpi_finalize,
    .update                 = atl_mpi_update,
    .set_resize_function = atl_mpi_set_resize_function,
};

static atl_mr_ops_t atl_mpi_mr_ops = {
    .mr_reg = atl_mpi_mr_reg,
    .mr_dereg = atl_mpi_mr_dereg,
};

static atl_status_t
atl_mpi_comm_init(atl_mpi_context_t *atl_mpi_context, size_t index, atl_comm_t **comm)
{
    int ret;
    atl_mpi_comm_context_t *atl_mpi_comm_context;

    atl_mpi_comm_context = calloc(1, sizeof(*atl_mpi_comm_context));
    if (!atl_mpi_comm_context)
        return atl_status_failure;

    ret = MPI_Comm_dup(MPI_COMM_WORLD, &atl_mpi_comm_context->mpi_comm);
    if (ret)
        goto err_comm_dup;

    MPI_Info info;
    MPI_Info_create(&info);
    char comm_idx_str[COMM_IDX_MAX_STR_LEN] = { 0 };
    snprintf(comm_idx_str, COMM_IDX_MAX_STR_LEN, "%zu", index);
    MPI_Info_set(info, COMM_IDX_KEY, comm_idx_str);
    MPI_Comm_set_info(atl_mpi_comm_context->mpi_comm, info);
    MPI_Info_free(&info);

    ATL_MPI_DEBUG_PRINT("idx %zu, comm_id_str %s", index, comm_idx_str);

    atl_mpi_comm_context->idx = index;
    *comm = &atl_mpi_comm_context->atl_comm;
    (*comm)->atl_desc = &atl_mpi_context->atl_desc;
    (*comm)->coll_ops = &atl_mpi_comm_coll_ops;
    (*comm)->pt2pt_ops = &atl_mpi_comm_pt2pt_ops;
    (*comm)->rma_ops = &atl_mpi_comm_rma_ops;
    (*comm)->comp_ops = &atl_mpi_comm_comp_ops;

    return atl_status_success;
err_comm_dup:
    free(atl_mpi_comm_context);
    return RET2ATL(ret);
}

int atl_mpi_use_optimized_mpi(atl_attr_t *attr)
{
    int i;
    int use_optimized_mpi = 0;
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
    int mpi_version_len;
    char* use_opt_env = NULL;

    MPI_Get_library_version(mpi_version, &mpi_version_len);
    ATL_MPI_DEBUG_PRINT("initial use_optimized_mpi = %d", use_optimized_mpi);
    ATL_MPI_DEBUG_PRINT("MPI version %s", mpi_version);

    if (strncmp(mpi_version, OPTIMIZED_MPI_VERSION_PREFIX, strlen(OPTIMIZED_MPI_VERSION_PREFIX)) == 0)
    {
        int impi_version = atoi(mpi_version + strlen(OPTIMIZED_MPI_VERSION_PREFIX));
        ATL_MPI_DEBUG_PRINT("IMPI version %d", impi_version);
        for (i = 0; i < SIZEOFARR(optimized_impi_versions); i++)
        {
            if (impi_version == optimized_impi_versions[i])
            {
                ATL_MPI_DEBUG_PRINT("set use_optimized_mpi = 1 because IMPI version matches with expected one");
                use_optimized_mpi = 1;
                break;
            }
        }
    }

    if (attr->comm_count == 1)
    {
        ATL_MPI_DEBUG_PRINT("set use_optimized_mpi = 0 because single ATL comm is requested");
        use_optimized_mpi = 0;
    }

    if ((use_opt_env = getenv("CCL_ATL_MPI_OPT")) != NULL)
    {
        use_optimized_mpi = atoi(use_opt_env);
        ATL_MPI_DEBUG_PRINT("set use_optimized_mpi = %d because optimized MPI environment is requested explicitly",
            use_optimized_mpi);
    }

    ATL_MPI_DEBUG_PRINT("final use_optimized_mpi = %d", use_optimized_mpi);

    return use_optimized_mpi;
}

atl_status_t atl_mpi_set_optimized_mpi_environment(atl_attr_t *attr)
{
    char comm_count_str[COMM_IDX_MAX_STR_LEN] = { 0 };
    snprintf(comm_count_str, COMM_IDX_MAX_STR_LEN, "%zu", attr->comm_count);

    setenv("I_MPI_THREAD_SPLIT", "1", 0);
    setenv("I_MPI_THREAD_RUNTIME", "generic", 0);
    setenv("I_MPI_THREAD_MAX", comm_count_str, 0);
    setenv("I_MPI_THREAD_ID_KEY", COMM_IDX_KEY, 0);
    setenv("I_MPI_THREAD_LOCK_LEVEL", "vci", 0);

    return atl_status_success;
}

atl_status_t atl_mpi_init(int *argc, char ***argv, atl_proc_coord_t *proc_coord,
                          atl_attr_t *attr, atl_comm_t ***atl_comms, atl_desc_t **atl_desc)
{
    assert(sizeof(atl_mpi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal));

    int ret = MPI_SUCCESS;
    size_t i;
    int comm_attr_flag;
    atl_mpi_context_t *atl_mpi_context;
    int required_thread_level = MPI_THREAD_MULTIPLE, provided_thread_level;

    atl_mpi_context = calloc(1, sizeof(*atl_mpi_context));
    if (!atl_mpi_context)
        return atl_status_failure;

    if (atl_mpi_use_optimized_mpi(attr))
    {
        atl_mpi_set_optimized_mpi_environment(attr);
    }
    
    int is_mpi_inited = 0;
    MPI_Initialized(&is_mpi_inited);
    
    if (!is_mpi_inited)
    {
        ret = MPI_Init_thread(argc, argv, required_thread_level, &provided_thread_level);
        if (provided_thread_level < required_thread_level)
            goto err_init;
    }
    else
    {
        is_external_init = 1;
        ATL_MPI_DEBUG_PRINT("MPI was initialized externaly");
    }  

    if (ret)
        goto err_init;

    memset(proc_coord, 0, sizeof(atl_proc_coord_t));
    
    MPI_Comm_rank(MPI_COMM_WORLD, (int*)&(proc_coord->global_idx));
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&(proc_coord->global_count));

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        proc_coord->global_count, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, (int*)&(proc_coord->local_idx));
    MPI_Comm_size(local_comm, (int*)&(proc_coord->local_count));
    MPI_Comm_free(&local_comm);

    atl_mpi_context->proc_coord = *proc_coord;
    atl_mpi_context->comm_ref_count = 0;

    *atl_comms = calloc(1, sizeof(atl_comm_t**) * attr->comm_count);
    if (!*atl_comms)
        goto err_comm;

    for (i = 0; i < attr->comm_count; i++)
    {
        ret = atl_mpi_comm_init(atl_mpi_context, i, &(*atl_comms)[i]);
        if (ret)
            goto err_comm_dup;
        atl_mpi_context->comm_ref_count++;
    }

    atl_mpi_context->atl_desc.ops = &atl_mpi_ops;
    atl_mpi_context->atl_desc.ops->is_resize_enabled = 0;
    atl_mpi_context->atl_desc.mr_ops = &atl_mpi_mr_ops;
    *atl_desc = &atl_mpi_context->atl_desc;

    attr->is_tagged_coll_enabled = 0;
    attr->tag_bits = 32;
    attr->max_tag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &(attr->max_tag), &comm_attr_flag);

    return atl_status_success;

err_comm_dup:
    for (i = 0; i < atl_mpi_context->comm_ref_count; i++)
    {
        atl_mpi_comm_context_t *comm_context =
            container_of((*atl_comms)[i], atl_mpi_comm_context_t, atl_comm);
        MPI_Comm_free(&comm_context->mpi_comm);
    }
    free(*atl_comms);
err_comm:
    if (!is_external_init)
        MPI_Finalize();
err_init:
    free(atl_mpi_context);
    return atl_status_failure;
}

ATL_MPI_INI
{
    atl_transport->name = atl_mpi_name;
    atl_transport->init = atl_mpi_init;

    return atl_status_success;
}
