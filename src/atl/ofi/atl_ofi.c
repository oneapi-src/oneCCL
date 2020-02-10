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
#include <time.h>

#include <unistd.h>
#include <sys/syscall.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>

#include <inttypes.h>
#include "pm_rt.h"

#ifndef gettid
#define gettid() syscall(SYS_gettid)
#endif

#define ATL_OFI_PRINT(s, ...)                             \
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

#define ATL_OFI_ASSERT(cond, ...)                 \
    do {                                          \
        if (!(cond))                              \
        {                                         \
            ATL_OFI_PRINT("ASSERT failed, cond: " \
                          #cond " " __VA_ARGS__); \
            exit(0);                              \
        }                                         \
    } while(0)

#ifdef ENABLE_DEBUG
#define ATL_OFI_DEBUG_PRINT(s, ...) ATL_OFI_PRINT(s, ##__VA_ARGS__)
#else
#define ATL_OFI_DEBUG_PRINT(s, ...)
#endif

#define ATL_OFI_TIMEOUT_SEC_ENV     "ATL_OFI_TIMEOUT_SEC"

#define ATL_OFI_DEFAULT_TIMEOUT_SEC 60
#define ATL_OFI_MAX_RETRY_COUNT     10000
#define ATL_OFI_MAX_HOSTNAME_LEN    256
#define ATL_OFI_WAIT_SEC            10
#define ATL_OFI_CQ_READ_ITERS       10000
#define ATL_OFI_CQ_BUNCH_SIZE       8
#define ATL_OFI_PMI_PROC_MULTIPLIER 100

#define ATL_OFI_PM_KEY              "atl-ofi"

typedef struct
{
    size_t timeout_sec;
    int is_resize_enabled;
} atl_ofi_global_data_t;

static atl_ofi_global_data_t global_data;

#define ATL_OFI_RETRY(func, comm, ret_val)                                 \
    do {                                                                   \
        time_t start = 0, end = 0;                                         \
        do {                                                               \
            size_t retry_count = 0;                                        \
            if (global_data.is_resize_enabled)                             \
            {                                                              \
                start = time(NULL);                                        \
            }                                                              \
            do {                                                           \
                ret_val = func;                                            \
                if (ret_val == FI_SUCCESS)                                 \
                    break;                                                 \
                if (ret_val != -FI_EAGAIN) {                               \
                    ATL_OFI_PRINT(#func "\n fails with ret %"PRId64", %s", \
                                        ret_val, fi_strerror(ret_val));    \
                    ATL_OFI_ASSERT(0, "OFI function error");               \
                    break;                                                 \
                }                                                          \
                (void) atl_ofi_comm_poll(comm);                            \
                retry_count += 1;                                          \
            } while (ret_val == -FI_EAGAIN &&                              \
                     retry_count < ATL_OFI_MAX_RETRY_COUNT);               \
            if (global_data.is_resize_enabled)                             \
            {                                                              \
                end = time(NULL);                                          \
            }                                                              \
        } while (ret_val == -FI_EAGAIN &&                                  \
                (end - start) < global_data.timeout_sec);                  \
    } while (0)

/* OFI returns 0 or -errno */
#define RET2ATL(ret)                                                \
    ({                                                              \
        atl_status_t res;                                           \
        if (ret == -FI_EAGAIN)                                      \
            res = atl_status_again;                                 \
        else                                                        \
            res = (ret) ? atl_status_failure : atl_status_success;  \
        res;                                                        \
    })

static const char *atl_ofi_name = "ofi";

typedef struct atl_ofi_context {
    atl_desc_t atl_desc;

    atl_proc_coord_t proc_coord;

    pm_rt_desc_t *pm_rt;

    struct fi_info *prov;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *av;

    int is_shm_enabled;
    struct fi_info *shm_prov;
    struct fid_fabric *shm_fabric;
    struct fid_domain *shm_domain;
    struct fid_av *shm_av;

    /* This is used only in case of SEP supported */
    struct fid_ep *sep;
    int rx_ctx_bits;
    size_t comm_ref_count;
    struct {
        fi_addr_t *table;
        size_t ep_num; /* table[][0..ep_num], this is 1 for SEP */
        /* table[0..global_proc_count][] */
    } addr_table;
    size_t addr_len;
} atl_ofi_context_t;

typedef struct atl_ofi_comm_name {
    void *addr;
    size_t len;
} atl_ofi_comm_name_t;

typedef struct atl_ofi_comm_context {
    atl_comm_t atl_comm;
    size_t idx;
    struct fid_ep *tx_ctx;
    struct fid_ep *rx_ctx;
    struct fid_cq *cq;
    atl_ofi_comm_name_t *name;
} atl_ofi_comm_context_t;

typedef enum atl_ofi_comp_state {
    ATL_OFI_COMP_POSTED,
    ATL_OFI_COMP_COMPLETED,
    ATL_OFI_COMP_PEEK_STARTED,
    ATL_OFI_COMP_PEEK_FOUND,
    ATL_OFI_COMP_PEEK_NOT_FOUND,
} atl_ofi_comp_state_t;

typedef struct atl_ofi_req {
    struct fi_context ofi_context;

    atl_ofi_comm_context_t *comm;
    struct fid_ep *last_ctx;
    atl_ofi_comp_state_t comp_state;
    size_t recv_len;
} atl_ofi_req_t;

typedef struct atl_ofi_mr {
    atl_mr_t atl_mr;
    struct fid_mr *mr;
} atl_ofi_mr_t;

static inline fi_addr_t
atl_ofi_comm_atl_addr_2_fi_addr(atl_ofi_context_t *atl, size_t proc_idx, size_t ep_idx)
{
    if (atl->sep)
        return fi_rx_addr(*(atl->addr_table.table + ((atl->addr_table.ep_num * proc_idx))),
                          ep_idx, atl->rx_ctx_bits);
    else
        return *(atl->addr_table.table + ((atl->addr_table.ep_num * proc_idx) + ep_idx));
}

static atl_status_t
atl_ofi_get_local_proc_coord(atl_ofi_context_t *atl_ofi_context)
{
    ATL_OFI_ASSERT(atl_ofi_context, "atl_ofi_context is null");

    int ret = atl_status_success, i;
    size_t local_idx = 0, local_count = 0;
    char *all_hostnames = NULL;
    char my_hostname[ATL_OFI_MAX_HOSTNAME_LEN] = { 0 };
    size_t my_hostname_len = 0;
    size_t my_global_proc_idx = atl_ofi_context->proc_coord.global_idx;

    gethostname(my_hostname, ATL_OFI_MAX_HOSTNAME_LEN - 1);
    my_hostname_len = strlen(my_hostname);

    ATL_OFI_ASSERT(my_hostname_len < ATL_OFI_MAX_HOSTNAME_LEN,
                   "unexpected my_hostname_len %zu, expected max %zu",
                   my_hostname_len, (size_t)(ATL_OFI_MAX_HOSTNAME_LEN));

    if (ATL_OFI_MAX_HOSTNAME_LEN - my_hostname_len <= 10) {
        ATL_OFI_PRINT("WARNING: hostname is quite long (len %zu, name %s)",
                      my_hostname_len, my_hostname);
    }

    snprintf(my_hostname + my_hostname_len,
             ATL_OFI_MAX_HOSTNAME_LEN - my_hostname_len,
             "-%zu", my_global_proc_idx);

    ret = pmrt_kvs_put(atl_ofi_context->pm_rt, ATL_OFI_PM_KEY,
                       my_global_proc_idx * ATL_OFI_PMI_PROC_MULTIPLIER,
                       my_hostname,
                       ATL_OFI_MAX_HOSTNAME_LEN);

    if (ret) {
        ATL_OFI_PRINT("pmrt_kvs_put: ret: %d", ret);
        goto fn_err;
    }

    pmrt_barrier(atl_ofi_context->pm_rt);

    all_hostnames = calloc(1, atl_ofi_context->proc_coord.global_count * ATL_OFI_MAX_HOSTNAME_LEN);
    if (!all_hostnames) {
        ATL_OFI_PRINT("can't allocate all_hostnames");
        goto fn_err;
    }

    for (i = 0; i < atl_ofi_context->proc_coord.global_count; i++) {
        ret = pmrt_kvs_get(atl_ofi_context->pm_rt, ATL_OFI_PM_KEY,
                           i * ATL_OFI_PMI_PROC_MULTIPLIER,
                           all_hostnames + i * ATL_OFI_MAX_HOSTNAME_LEN,
                           ATL_OFI_MAX_HOSTNAME_LEN);
        if (ret) {
            ATL_OFI_PRINT("pmrt_kvs_put: ret: %d", ret);
            goto fn_err;
        }
    }

    for (i = 0; i < atl_ofi_context->proc_coord.global_count; i++) {
        if (!strncmp(my_hostname,
                     all_hostnames + i * ATL_OFI_MAX_HOSTNAME_LEN,
                     my_hostname_len + 1 /* including "-" at the end */)) {
            local_count++;
            size_t peer_global_proc_idx;
            sscanf(all_hostnames + i * ATL_OFI_MAX_HOSTNAME_LEN + my_hostname_len + 1,
                   "%zu", &peer_global_proc_idx);
            if (my_global_proc_idx > peer_global_proc_idx)
                local_idx++;
        }
    }

    atl_ofi_context->proc_coord.local_idx = local_idx;
    atl_ofi_context->proc_coord.local_count = local_count;


fn_exit:
    free(all_hostnames);
    return ret;

fn_err:
    ret = atl_status_failure;
    goto fn_exit;
}

static atl_status_t
atl_ofi_update_addr_table(atl_ofi_context_t *atl_ofi_context)
{
    int ret;
    size_t i, j;
    char *epnames_table;
    size_t epnames_table_len;

    /* Allocate OFI EP names table that will contain all published names */
    epnames_table_len = atl_ofi_context->addr_len *
                        atl_ofi_context->addr_table.ep_num *
                        atl_ofi_context->proc_coord.global_count;

    if (epnames_table_len == 0) {
        ATL_OFI_PRINT("epnames_table_len == 0");
        return atl_status_failure;
    }

    epnames_table = (char *)calloc(1, epnames_table_len);
    if (!epnames_table) {
        ATL_OFI_PRINT("Can't allocate epnames_table");
        return atl_status_failure;
    }
    pmrt_barrier(atl_ofi_context->pm_rt);

    /* Retrieve all the OFI EP names in order */
    for (i = 0; i < atl_ofi_context->proc_coord.global_count; i++) {
        for (j = 0; j < atl_ofi_context->addr_table.ep_num; j++) {
            ret = pmrt_kvs_get(atl_ofi_context->pm_rt, ATL_OFI_PM_KEY,
                               i * ATL_OFI_PMI_PROC_MULTIPLIER + j,
                               epnames_table +
                               (/* iterate by comms(EPs) per process */ j +
                                /* iterate by processes */
                                (i * atl_ofi_context->addr_table.ep_num)) *
                               atl_ofi_context->addr_len,
                               atl_ofi_context->addr_len);
            if (ret)
                goto err_ep_names;
        }
    }

    if (atl_ofi_context->addr_table.table != NULL)
        free(atl_ofi_context->addr_table.table);

    atl_ofi_context->addr_table.table =
        calloc(1, atl_ofi_context->addr_table.ep_num *
                  atl_ofi_context->proc_coord.global_count *
                  sizeof(fi_addr_t));
    if (!atl_ofi_context->addr_table.table)
        goto err_ep_names;

    /* Insert all the EP names into the AV */
    ret = fi_av_insert(atl_ofi_context->av, epnames_table,
                       atl_ofi_context->addr_table.ep_num *
                       atl_ofi_context->proc_coord.global_count,
                       atl_ofi_context->addr_table.table, 0, NULL);

    ATL_OFI_DEBUG_PRINT("av_insert: ep_num %zu, global_proc_count %zu, inserted %d",
        atl_ofi_context->addr_table.ep_num, atl_ofi_context->proc_coord.global_count, ret);

    if (ret != atl_ofi_context->addr_table.ep_num * atl_ofi_context->proc_coord.global_count) {
        ret = atl_status_failure;
        goto err_addr_table;
    } else {
        ret = atl_status_success;
    }

    /* Normal end of execution */
    free(epnames_table);
    return ret;

    /*Abnormal end of execution */
err_addr_table:
    free(atl_ofi_context->addr_table.table);
    atl_ofi_context->addr_table.ep_num = 0;
err_ep_names:
    free(epnames_table);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comms_connect(atl_ofi_context_t *atl_ofi_context,
                      atl_comm_t **comms, size_t comm_count)
{
    int ret;
    size_t i;

    atl_ofi_context->addr_table.ep_num =
        (atl_ofi_context->sep ? 1 : comm_count);

    for (i = 0; i < atl_ofi_context->addr_table.ep_num; i++) {
        atl_ofi_comm_context_t *comm_context =
            container_of(comms[i], atl_ofi_comm_context_t, atl_comm);
        ret = pmrt_kvs_put(atl_ofi_context->pm_rt, ATL_OFI_PM_KEY,
                           atl_ofi_context->proc_coord.global_idx * ATL_OFI_PMI_PROC_MULTIPLIER + i,
                           comm_context->name->addr,
                           comm_context->name->len);
        if (ret) {
            ATL_OFI_PRINT("pmrt_kvs_put: ret: %d", ret);
            return atl_status_failure;
        }
    }

    ret = atl_ofi_update_addr_table(atl_ofi_context);

    return ret;
}

static atl_status_t atl_ofi_comms_destroy_conns(atl_ofi_context_t *atl_ofi_context)
{
    int ret = atl_status_success;

    /* disabled as it leads to unstable fails */
//     ret = fi_av_remove(atl_ofi_context->av, atl_ofi_context->addr_table.table,
//                        atl_ofi_context->addr_table.ep_num *
//                        atl_ofi_context->proc_coord.global_count, 0);
//
//    if (ret)
//        ATL_OFI_DEBUG_PRINT("AV remove failed (%d)", ret);

    free(atl_ofi_context->addr_table.table);
    atl_ofi_context->addr_table.ep_num = 0;

    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_get_epname(struct fid_ep *ep, atl_ofi_comm_name_t **name)
{
    int ret;

    *name = calloc(1, sizeof(atl_ofi_comm_name_t));
    if (!*name) {
        ATL_OFI_PRINT("can't allocate name");
        return atl_status_failure;
    }

    ret = fi_getname(&ep->fid, (*name)->addr, &(*name)->len);
    if ((ret != -FI_ETOOSMALL) || ((*name)->len <= 0))
        (*name)->len = FI_NAME_MAX;

    (*name)->addr = calloc(1, (*name)->len);
    if (!(*name)->addr) {
        ATL_OFI_PRINT("can't allocate addr");
        ret = atl_status_failure;
        goto err_addr;
    }

    ret = fi_getname(&ep->fid, (*name)->addr, &(*name)->len);
    if (ret)
        goto err_getname;

    return atl_status_success;
err_getname:
    free((*name)->addr);
    (*name)->len = 0;
err_addr:
    free(*name);
    *name = NULL;
    return RET2ATL(ret);
}

static void atl_ofi_comms_set_epname(atl_comm_t **comms, size_t comm_count,
                                     atl_ofi_comm_name_t *name)
{
    atl_ofi_comm_context_t *comm_context;
    int i;

    for (i = 0; i < comm_count; i++) {
        comm_context = container_of(comms[i], atl_ofi_comm_context_t, atl_comm);
        comm_context->name = name;
    }
}

static void atl_ofi_comm_ep_destroy(atl_ofi_context_t *atl_ofi_context,
                                    atl_ofi_comm_context_t *atl_ofi_comm_context)
{
    if (atl_ofi_context->sep) {
        if (atl_ofi_comm_context->rx_ctx)
            fi_close(&atl_ofi_comm_context->rx_ctx->fid);
        if (atl_ofi_comm_context->tx_ctx)
            fi_close(&atl_ofi_comm_context->tx_ctx->fid);
    } else {
        if (atl_ofi_comm_context->rx_ctx && atl_ofi_comm_context->tx_ctx)
            fi_close(&atl_ofi_comm_context->rx_ctx->fid);
        atl_ofi_comm_context->rx_ctx = atl_ofi_comm_context->tx_ctx = NULL;
    }
}

static void atl_ofi_comm_conn_destroy(atl_ofi_context_t *atl_ofi_context,
                                 atl_comm_t *comm)
{
    atl_ofi_comm_context_t *atl_ofi_comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);

    atl_ofi_comm_ep_destroy(atl_ofi_context, atl_ofi_comm_context);
    if (atl_ofi_comm_context->name) {
        free(atl_ofi_comm_context->name->addr);
        atl_ofi_comm_context->name->addr = NULL;
        atl_ofi_comm_context->name->len = 0;
        free(atl_ofi_comm_context->name);
        atl_ofi_comm_context->name = NULL;
    }
}

static void atl_ofi_comms_conn_destroy(atl_ofi_context_t *atl_ofi_context,
                                       atl_comm_t **comms)
{
    size_t i;

    if (atl_ofi_context->sep) {
        atl_ofi_comm_context_t *atl_ofi_comm_context =
            container_of(comms[0], atl_ofi_comm_context_t, atl_comm);
        /* Free memory for one of the COMM contexts, for other - just set to NULL*/
        free(atl_ofi_comm_context->name->addr);
        atl_ofi_comm_context->name->addr = NULL;
        atl_ofi_comm_context->name->len = 0;
        free(atl_ofi_comm_context->name);
        atl_ofi_comms_set_epname(comms, atl_ofi_context->comm_ref_count, NULL);
    }

    for (i = 0; i < atl_ofi_context->comm_ref_count; i++)
        atl_ofi_comm_conn_destroy(atl_ofi_context, comms[i]);

    if (atl_ofi_context->sep)
        fi_close(&atl_ofi_context->sep->fid);
}

static atl_status_t atl_ofi_finalize(atl_desc_t *atl_desc, atl_comm_t **atl_comms)
{
    int ret;
    size_t i;
    atl_ofi_context_t *atl_ofi_context =
        container_of(atl_desc, atl_ofi_context_t, atl_desc);

    ret = atl_ofi_comms_destroy_conns(atl_ofi_context);
    atl_ofi_comms_conn_destroy(atl_ofi_context, atl_comms);
    for (i = 0; i < atl_ofi_context->comm_ref_count; i++) {
        atl_ofi_comm_context_t *atl_ofi_comm_context =
            container_of(atl_comms[i], atl_ofi_comm_context_t, atl_comm);
        if (atl_ofi_comm_context->cq)
            fi_close(&atl_ofi_comm_context->cq->fid);
        free(atl_ofi_comm_context);
    }
    free(atl_comms);
    fi_close(&atl_ofi_context->av->fid);
    fi_close(&atl_ofi_context->domain->fid);
    fi_close(&atl_ofi_context->fabric->fid);
    fi_freeinfo(atl_ofi_context->prov);
    pmrt_finalize(atl_ofi_context->pm_rt);
    free(atl_ofi_context);

    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_ep_create(atl_ofi_context_t *atl_ofi_context,
                       atl_ofi_comm_context_t *atl_ofi_comm_context,
                       size_t index)
{
    int ret;
    struct fi_cq_attr cq_attr = {
        .format = FI_CQ_FORMAT_TAGGED,
    };
    struct fi_tx_attr tx_attr;
    struct fi_rx_attr rx_attr;

    ret = fi_cq_open(atl_ofi_context->domain, &cq_attr,
                     &atl_ofi_comm_context->cq, NULL);
    if (ret)
        return atl_status_failure;

    if (atl_ofi_context->sep) {
        rx_attr = *atl_ofi_context->prov->rx_attr;
        rx_attr.caps |= FI_TAGGED;
        ret = fi_rx_context(atl_ofi_context->sep, index, &rx_attr,
                            &atl_ofi_comm_context->rx_ctx, NULL);
        if (ret)
            goto err;
        ret = fi_ep_bind(atl_ofi_comm_context->rx_ctx,
                         &atl_ofi_comm_context->cq->fid, FI_RECV);
        if (ret)
            goto err;

        tx_attr = *atl_ofi_context->prov->tx_attr;
        tx_attr.caps |= FI_TAGGED;
        ret = fi_tx_context(atl_ofi_context->sep, index, &tx_attr,
                            &atl_ofi_comm_context->tx_ctx, NULL);
        if (ret)
            goto err;
        ret = fi_ep_bind(atl_ofi_comm_context->tx_ctx,
                         &atl_ofi_comm_context->cq->fid, FI_SEND);
        if (ret)
            goto err;

        fi_enable(atl_ofi_comm_context->rx_ctx);
        fi_enable(atl_ofi_comm_context->tx_ctx);
    } else {
        struct fid_ep *ep;

        ret = fi_endpoint(atl_ofi_context->domain, atl_ofi_context->prov, &ep, NULL);
        if (ret)
            goto err;
        atl_ofi_comm_context->tx_ctx = atl_ofi_comm_context->rx_ctx = ep;
        ret = fi_ep_bind(ep, &atl_ofi_comm_context->cq->fid, FI_SEND | FI_RECV);
        if (ret)
            goto err;
        ret = fi_ep_bind(ep, &atl_ofi_context->av->fid, 0);
        if (ret)
            goto err;

        fi_enable(ep);

        ret = atl_ofi_comm_get_epname(ep, &atl_ofi_comm_context->name);
        if (ret)
            goto err;
        atl_ofi_context->addr_len = atl_ofi_comm_context->name->len;
        if (ret)
            goto err;
    }

    return atl_status_success;
err:
    atl_ofi_comm_ep_destroy(atl_ofi_context, atl_ofi_comm_context);
    return RET2ATL(ret);
}

static atl_status_t atl_ofi_comm_handle_cq_err(atl_ofi_comm_context_t *comm_context)
{
    struct fi_cq_err_entry err_entry;
    atl_ofi_req_t *ofi_req;
    int ret = fi_cq_readerr(comm_context->cq, &err_entry, 0);
    if (ret != 1) {
        ATL_OFI_ASSERT(0, "unable to read error from cq");
        return atl_status_failure;
    } else {
        ofi_req = container_of(err_entry.op_context, atl_ofi_req_t, ofi_context);
        if (err_entry.err == FI_ECANCELED) {
            return atl_status_success;
        }
        if (err_entry.err == FI_ENOMSG &&
            ofi_req->comp_state == ATL_OFI_COMP_PEEK_STARTED) {
            ofi_req->comp_state = ATL_OFI_COMP_PEEK_NOT_FOUND;
        }
        else {
            ATL_OFI_PRINT("fi_cq_readerr: err: %d, prov_err: %s (%d)",
                          err_entry.err,
                          fi_cq_strerror(comm_context->cq,
                                         err_entry.prov_errno,
                                         err_entry.err_data,
                                         NULL, 0),
                          err_entry.prov_errno);
            return atl_status_failure;
        }
        return atl_status_success;
    }
}

static inline void atl_ofi_process_comps(struct fi_cq_tagged_entry *entries,
                                         ssize_t ret)
{
    size_t idx;
    atl_ofi_req_t *comp_ofi_req;
    for (idx = 0; idx < ret; idx++) {
        comp_ofi_req = container_of(entries[idx].op_context, atl_ofi_req_t,
                                    ofi_context);
        switch (comp_ofi_req->comp_state) {
            case ATL_OFI_COMP_POSTED:
                comp_ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;
                break;
            case ATL_OFI_COMP_COMPLETED:
                break;
            case ATL_OFI_COMP_PEEK_STARTED:
                comp_ofi_req->comp_state = ATL_OFI_COMP_PEEK_FOUND;
                break;
            default:
                ATL_OFI_ASSERT(0, "unexpected completion state %d", comp_ofi_req->comp_state);
                break;
        }
        if (entries[idx].flags & FI_RECV) {
            comp_ofi_req->recv_len = entries[idx].len;
        }
    }
}

static inline atl_status_t atl_ofi_comm_poll(atl_comm_t *comm)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];
    ssize_t ret;

    do {
        ret = fi_cq_read(comm_context->cq, entries, ATL_OFI_CQ_BUNCH_SIZE);
        if (ret > 0) {
            atl_ofi_process_comps(entries, ret);
        } else if (ret == -FI_EAGAIN) {
            break;
        } else {
            return atl_ofi_comm_handle_cq_err(comm_context);
        }
    } while (ret > 0);

    return atl_status_success;
}

/* Non-blocking I/O vector pt2pt ops */
static atl_status_t
atl_ofi_comm_sendv(atl_comm_t *comm, const struct iovec *iov, size_t count,
                   size_t dest_proc_idx, uint64_t tag, atl_req_t *req)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    ssize_t ret;

    req->remote_proc_idx = dest_proc_idx;
    req->tag = tag;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->last_ctx = comm_context->tx_ctx;
    ofi_req->comm = comm_context;

    ATL_OFI_RETRY(fi_tsendv(comm_context->tx_ctx, iov, NULL, count,
                             atl_ofi_comm_atl_addr_2_fi_addr(
                             container_of(comm->atl_desc,
                                          atl_ofi_context_t,
                                          atl_desc),
                             dest_proc_idx, comm_context->idx),
                             tag, &ofi_req->ofi_context),
                  comm, ret);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_recvv(atl_comm_t *comm, struct iovec *iov, size_t count,
                   size_t src_proc_idx, uint64_t tag, atl_req_t *req)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    ssize_t ret;

    req->remote_proc_idx = src_proc_idx;
    req->tag = tag;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->last_ctx = comm_context->rx_ctx;
    ofi_req->comm = comm_context;

    ATL_OFI_RETRY(fi_trecvv(comm_context->rx_ctx, iov, NULL, count,
                            atl_ofi_comm_atl_addr_2_fi_addr(
                                 container_of(comm->atl_desc,
                                              atl_ofi_context_t,
                                              atl_desc),
                                 src_proc_idx, comm_context->idx),
                            tag, 0, &ofi_req->ofi_context),
                  comm, ret);
    return RET2ATL(ret);
}

/* Non-blocking pt2pt ops */
static atl_status_t
atl_ofi_comm_send(atl_comm_t *comm, const void *buf, size_t len,
                  size_t dest_proc_idx, uint64_t tag, atl_req_t *req)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    ssize_t ret;

    req->remote_proc_idx = dest_proc_idx;
    req->tag = tag;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->last_ctx = comm_context->tx_ctx;
    ofi_req->comm = comm_context;

    ATL_OFI_RETRY(fi_tsend(comm_context->tx_ctx, buf, len, NULL,
                           atl_ofi_comm_atl_addr_2_fi_addr(
                                container_of(comm->atl_desc,
                                             atl_ofi_context_t,
                                             atl_desc),
                                dest_proc_idx, comm_context->idx),
                           tag, &ofi_req->ofi_context),
                  comm, ret);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_recv(atl_comm_t *comm, void *buf, size_t len,
                  size_t src_proc_idx, uint64_t tag, atl_req_t *req)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    ssize_t ret;

    req->remote_proc_idx = src_proc_idx;
    req->tag = tag;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->last_ctx = comm_context->rx_ctx;
    ofi_req->comm = comm_context;

    ATL_OFI_RETRY(fi_trecv(comm_context->rx_ctx, buf, len, NULL,
                           atl_ofi_comm_atl_addr_2_fi_addr(
                                container_of(comm->atl_desc,
                                             atl_ofi_context_t,
                                             atl_desc),
                                src_proc_idx, comm_context->idx),
                           tag, 0, &ofi_req->ofi_context),
                  comm, ret);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_read(atl_comm_t *comm, void *buf, size_t len, atl_mr_t *atl_mr,
                  uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    ssize_t ret;

    req->remote_proc_idx = dest_proc_idx;
    req->tag = 0;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->last_ctx = comm_context->tx_ctx;
    ofi_req->comm = comm_context;

    ATL_OFI_RETRY(fi_read(comm_context->tx_ctx, buf, len, (void*)atl_mr->l_key,
                          atl_ofi_comm_atl_addr_2_fi_addr(
                              container_of(comm->atl_desc,
                                           atl_ofi_context_t,
                                           atl_desc),
                              dest_proc_idx, comm_context->idx),
                          addr, r_key, &ofi_req->ofi_context),
                  comm, ret);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_write(atl_comm_t *comm, const void *buf, size_t len, atl_mr_t *atl_mr,
                   uint64_t addr, uintptr_t r_key, size_t dest_proc_idx, atl_req_t *req)
{
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    ssize_t ret;

    req->remote_proc_idx = dest_proc_idx;
    req->tag = 0;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->last_ctx = comm_context->tx_ctx;
    ofi_req->comm = comm_context;

    ATL_OFI_RETRY(fi_write(comm_context->tx_ctx, buf, len, (void*)atl_mr->l_key,
                           atl_ofi_comm_atl_addr_2_fi_addr(
                               container_of(comm->atl_desc,
                                            atl_ofi_context_t,
                                            atl_desc),
                               dest_proc_idx, comm_context->idx),
                           addr, r_key, &ofi_req->ofi_context),
                   comm, ret);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comm_probe(atl_comm_t *comm, size_t src_proc_idx, uint64_t tag,
                   int *found, size_t *recv_len)
{
    atl_status_t ret = atl_status_success;
    ssize_t ofi_ret;
    atl_ofi_comm_context_t *comm_context =
        container_of(comm, atl_ofi_comm_context_t, atl_comm);
    atl_ofi_req_t req;
    int flag = 0, len = 0;

    struct fi_msg_tagged msg = {
        .msg_iov = NULL,
        .desc = NULL,
        .iov_count = 0,
        .addr = atl_ofi_comm_atl_addr_2_fi_addr(
                    container_of(comm->atl_desc,
                                 atl_ofi_context_t,
                                 atl_desc),
                    src_proc_idx, comm_context->idx),
        .tag = tag,
        .ignore = 0,
        .context = &(req.ofi_context),
        .data = 0,
    };

    req.comp_state = ATL_OFI_COMP_PEEK_STARTED;
    ATL_OFI_RETRY(fi_trecvmsg(comm_context->rx_ctx, &msg,
                              FI_PEEK | FI_COMPLETION),
                  comm, ofi_ret);

    while (req.comp_state == ATL_OFI_COMP_PEEK_STARTED)
    {
        ret = atl_ofi_comm_poll(comm);
        if (ret != atl_status_success)
            return ret;
    }

    if (req.comp_state == ATL_OFI_COMP_PEEK_FOUND)
    {
        flag = 1;
        len = req.recv_len;
    }
    else if (req.comp_state != ATL_OFI_COMP_PEEK_NOT_FOUND)
    {
        ATL_OFI_ASSERT(0, "unexpected completion state %d", req.comp_state);
    }

    if (found) *found = flag;
    if (recv_len) *recv_len = len;

    return RET2ATL(ofi_ret);
}

static int
atl_ofi_wait_cancel_cq(struct fid_cq *cq)
{
    struct fi_cq_err_entry err_entry;
    int ret, i;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];

    double time = 0;
    clock_t start, end;

    while (time < ATL_OFI_WAIT_SEC)
    {
        for (i = 0; i < ATL_OFI_CQ_READ_ITERS; i++)
        {
            start = clock();
            ret = fi_cq_read(cq, entries, ATL_OFI_CQ_BUNCH_SIZE);

            if (ret < 0 && ret != -FI_EAGAIN)
            {
                ret = fi_cq_readerr(cq, &err_entry, 0);

                if (err_entry.err != FI_ECANCELED)
                {
                    ATL_OFI_PRINT("fi_cq_readerr: err: %d, prov_err: %s (%d)",
                                  err_entry.err, fi_cq_strerror(cq,
                                                                err_entry.prov_errno,
                                                                err_entry.err_data,
                                                                NULL, 0),
                                  err_entry.prov_errno);
                    return atl_status_failure;
                }
                return atl_status_success;
            }
        }
        end = clock();
        time += (double) (end - start) / CLOCKS_PER_SEC;
    }
    ATL_OFI_PRINT("too long for cancel");
    return atl_status_failure;
}


static atl_status_t atl_ofi_comm_cancel(atl_comm_t *comm, atl_req_t *req)
{
    atl_status_t ret = atl_status_success;
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);

    ret = fi_cancel(&ofi_req->last_ctx->fid, &ofi_req->ofi_context);
    if (ret == 0)
    {
        return atl_ofi_wait_cancel_cq(ofi_req->comm->cq);
    }

    return atl_status_success;
}

static atl_status_t atl_ofi_comm_wait(atl_comm_t *comm, atl_req_t *req)
{
    atl_status_t ret = atl_status_success;
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);

    while ((ofi_req->comp_state != ATL_OFI_COMP_COMPLETED) &&
           ((ret = atl_ofi_comm_poll(comm)) == atl_status_success));

    return ret;
}

static atl_status_t
atl_ofi_comm_wait_all(atl_comm_t *comm, atl_req_t *reqs, size_t count)
{
    size_t i;
    atl_status_t ret;

    for (i = 0; i < count; i++) {
        ret = atl_ofi_comm_wait(comm, &reqs[i]);
        if (ret != atl_status_success)
            return ret;
    }

    return atl_status_success;
}

static atl_status_t
atl_ofi_comm_check(atl_comm_t *comm, int *status, atl_req_t *req)
{
    atl_ofi_req_t *ofi_req = ((atl_ofi_req_t *)req->internal);
    *status = (ofi_req->comp_state == ATL_OFI_COMP_COMPLETED);
    return atl_status_success;
}

static atl_status_t
atl_ofi_comm_allgatherv(atl_comm_t *comm, const void *send_buf, size_t send_len,
                        void *recv_buf, const int recv_lens[], int displs[], atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_ofi_comm_allreduce(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t count,
                       atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_ofi_comm_barrier(atl_comm_t *comm, atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_ofi_comm_bcast(atl_comm_t *comm, void *buf, size_t len, size_t root,
                   atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_status_t
atl_ofi_comm_reduce(atl_comm_t *comm, const void *send_buf, void *recv_buf, size_t count, size_t root,
                    atl_datatype_t dtype, atl_reduction_t op, atl_req_t *req)
{
    return atl_status_unsupported;
}

static atl_coll_ops_t atl_ofi_comm_coll_ops = {
    .allgatherv = atl_ofi_comm_allgatherv,
    .allreduce = atl_ofi_comm_allreduce,
    .barrier = atl_ofi_comm_barrier,
    .bcast = atl_ofi_comm_bcast,
    .reduce = atl_ofi_comm_reduce,
};

static atl_pt2pt_ops_t atl_ofi_comm_pt2pt_ops = {
    .sendv = atl_ofi_comm_sendv,
    .recvv = atl_ofi_comm_recvv,
    .send = atl_ofi_comm_send,
    .recv = atl_ofi_comm_recv,
    .probe = atl_ofi_comm_probe,
};

static atl_rma_ops_t atl_ofi_comm_rma_ops = {
    .read = atl_ofi_comm_read,
    .write = atl_ofi_comm_write,
};

static atl_comp_ops_t atl_ofi_comm_comp_ops = {
    .wait = atl_ofi_comm_wait,
    .wait_all = atl_ofi_comm_wait_all,
    .cancel = atl_ofi_comm_cancel,
    .poll = atl_ofi_comm_poll,
    .check = atl_ofi_comm_check
};

static atl_status_t
atl_ofi_comm_init(atl_ofi_context_t *atl_ofi_context, atl_comm_attr_t *attr,
                  size_t index, atl_comm_t **comm)
{
    int ret;
    atl_ofi_comm_context_t *atl_ofi_comm_context;

    atl_ofi_comm_context = calloc(1, sizeof(*atl_ofi_comm_context));
    if (!atl_ofi_comm_context)
        return atl_status_failure;

    ret = atl_ofi_comm_ep_create(atl_ofi_context, atl_ofi_comm_context, index);
    if (ret)
        goto err_ep;

    atl_ofi_comm_context->idx = index;
    *comm = &atl_ofi_comm_context->atl_comm;
    (*comm)->atl_desc = &atl_ofi_context->atl_desc;
    (*comm)->coll_ops = &atl_ofi_comm_coll_ops;
    (*comm)->pt2pt_ops = &atl_ofi_comm_pt2pt_ops;
    (*comm)->rma_ops = &atl_ofi_comm_rma_ops;
    (*comm)->comp_ops = &atl_ofi_comm_comp_ops;

    return atl_status_success;
err_ep:
    free(atl_ofi_comm_context);
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_comms_init(atl_ofi_context_t *atl_ofi_context, size_t comm_count,
                   atl_comm_attr_t *attr, atl_comm_t ***comms)
{
    int i = 0, ret;

    *comms = calloc(1, sizeof(**comms) * comm_count);
    if (!*comms)
        return atl_status_failure;

    if (atl_ofi_context->prov->domain_attr->max_ep_tx_ctx > 1) {
        ret = fi_scalable_ep(atl_ofi_context->domain, atl_ofi_context->prov,
                             &atl_ofi_context->sep, NULL);
        if (ret)
            goto err_sep;
        ret = fi_scalable_ep_bind(atl_ofi_context->sep,
                                  &atl_ofi_context->av->fid, 0);
        if (ret) {
            fi_close(&atl_ofi_context->sep->fid);
            goto err_sep;
        }
    }

    for (i = 0; i < comm_count; i++) {
        ret = atl_ofi_comm_init(atl_ofi_context, attr, i, &(*comms)[i]);
        if (ret)
            goto err_comm;
        atl_ofi_context->comm_ref_count++;
    }

    if (atl_ofi_context->sep) {
        atl_ofi_comm_name_t *name;

        fi_enable(atl_ofi_context->sep);

        ret = atl_ofi_comm_get_epname(atl_ofi_context->sep, &name);
        if (ret)
            goto err_comm;
        atl_ofi_comms_set_epname(*comms, comm_count, name);
        atl_ofi_context->addr_len = name->len;
    }

    ret = atl_ofi_comms_connect(atl_ofi_context, *comms, comm_count);
    if (ret)
        goto err_comm;

    return atl_status_success;
err_comm:
    if (atl_ofi_context->sep)
        fi_close(&atl_ofi_context->sep->fid);
    while (atl_ofi_context->comm_ref_count) {
        atl_ofi_comm_context_t *atl_ofi_comm_context =
            container_of((*comms)[atl_ofi_context->comm_ref_count],
                         atl_ofi_comm_context_t, atl_comm);
        atl_ofi_comm_conn_destroy(atl_ofi_context,
                                  (*comms)[atl_ofi_context->comm_ref_count]);
        free(atl_ofi_comm_context);
        atl_ofi_context->comm_ref_count--;
    }
err_sep:
    free(*comms);
    return RET2ATL(ret);
}

static void atl_ofi_global_proc_idx(atl_desc_t *atl_desc, size_t *global_proc_idx)
{
    atl_ofi_context_t *atl_ofi_context =
        container_of(atl_desc, atl_ofi_context_t, atl_desc);
    *global_proc_idx = atl_ofi_context->proc_coord.global_idx;
}

static void atl_ofi_global_proc_count(atl_desc_t *atl_desc, size_t *global_proc_count)
{
    atl_ofi_context_t *atl_ofi_context =
        container_of(atl_desc, atl_ofi_context_t, atl_desc);
    *global_proc_count = atl_ofi_context->proc_coord.global_count;
}

static atl_status_t atl_ofi_try_to_drain_cq_err(struct fid_cq *cq)
{
    struct fi_cq_err_entry err_entry;
    int ret = fi_cq_readerr(cq, &err_entry, 0);
    if (ret != 1) {
        ATL_OFI_DEBUG_PRINT("Unable to fi_cq_readerr");
        return atl_status_failure;
    } else {
        if (err_entry.err != FI_ENOMSG
            && err_entry.err != FI_ECANCELED
            && err_entry.err != FI_ETRUNC) {
            ATL_OFI_PRINT("fi_cq_readerr: err: %d, prov_err: %s (%d)",
                          err_entry.err, fi_cq_strerror(cq,
                                                        err_entry.prov_errno,
                                                        err_entry.err_data,
                                                        NULL, 0),
                          err_entry.prov_errno);
            return atl_status_failure;
        }
        return atl_status_success;
    }
}

static int
atl_ofi_try_to_drain_cq(struct fid_cq *cq)
{
    int ret, i;
    double time = 0;
    clock_t start, end;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];

    while (time < ATL_OFI_WAIT_SEC)
    {
        start = clock();
        for (i = 0; i < ATL_OFI_CQ_READ_ITERS; i++)
        {
            ret = fi_cq_read(cq, entries, ATL_OFI_CQ_BUNCH_SIZE);

            if (ret < 0 && ret != -FI_EAGAIN)
            {
                atl_ofi_try_to_drain_cq_err(cq);
                return ret;
            }

            if (ret > 0)
                return ret;
        }
        end = clock();
        time += (double) (end - start) / CLOCKS_PER_SEC;
    }

    return ret;
}

static void
atl_ofi_reset(size_t comm_ref_count, atl_comm_t** atl_comms)
{
    int again = 1, i;
    int recv_buf_len = sizeof(char);
    char* recv_buf = malloc(recv_buf_len);
    struct fi_context ofi_context;

    for (i = 0; i < comm_ref_count; i++)
    {
        atl_ofi_comm_context_t *comm_context =
            container_of(atl_comms[i], atl_ofi_comm_context_t, atl_comm);
        // complete active sends and receives
        while (atl_ofi_try_to_drain_cq(comm_context->cq) != -FI_EAGAIN) {}

        // try to complete active incoming sends
        while (again)
        {
            again = 0;
            // post recv to complete incoming send
            while (fi_trecv(comm_context->rx_ctx, recv_buf, recv_buf_len, NULL,
                            FI_ADDR_UNSPEC, 0, UINTMAX_MAX, &ofi_context) ==
                   -FI_EAGAIN)
            {}

            // wait until recv will be completed or finished by timeout
            while (atl_ofi_try_to_drain_cq(comm_context->cq) != -FI_EAGAIN)
            {
                // Something is completed -> send queue not empty
                again = 1;
            }
        }

        // Nothing to recv -> cancel last recv
        fi_cancel(&comm_context->rx_ctx->fid, &ofi_context);

        atl_ofi_wait_cancel_cq(comm_context->cq);
    }

    free(recv_buf);
}

static atl_status_t
atl_ofi_update(atl_proc_coord_t *proc_coord, atl_desc_t *atl_desc, atl_comm_t** atl_comms)
{
    int ret;

    atl_ofi_context_t *atl_ofi_context =
        container_of(atl_desc, atl_ofi_context_t, atl_desc);

    pmrt_barrier(atl_ofi_context->pm_rt);

    atl_ofi_reset(atl_ofi_context->comm_ref_count, atl_comms);
    memset(proc_coord, 0, sizeof(atl_proc_coord_t));

    ret = pmrt_update(&atl_ofi_context->proc_coord.global_idx,
                      &atl_ofi_context->proc_coord.global_count,
                      atl_ofi_context->pm_rt);
    if (ret)
        goto err_pmrt_update;

    ret = atl_ofi_get_local_proc_coord(atl_ofi_context);
    if (ret)
        goto err_pmrt_update;

    ret = atl_ofi_update_addr_table(atl_ofi_context);
    if (ret)
        goto err_pmrt_update;

    *proc_coord = atl_ofi_context->proc_coord;

    /* Normal end of execution */
    return ret;

    /* Abnormal end of execution */
err_pmrt_update:
    return RET2ATL(ret);
}

static atl_status_t
atl_ofi_wait_notification(atl_desc_t *atl_desc)
{
    int ret;

    atl_ofi_context_t *atl_ofi_context =
        container_of(atl_desc, atl_ofi_context_t, atl_desc);

    ret = pmrt_wait_notification(atl_ofi_context->pm_rt);

    return RET2ATL(ret);
}

static atl_status_t atl_ofi_mr_reg(atl_desc_t *atl_desc, const void *buf, size_t len,
                                   atl_mr_t **atl_mr)
{
    int ret;
    atl_ofi_context_t *atl_context =
        container_of(atl_desc, atl_ofi_context_t, atl_desc);
    atl_ofi_mr_t *atl_ofi_mr = calloc(1, sizeof(*atl_ofi_mr));
    if (!atl_ofi_mr)
        return atl_status_failure;

    ret = fi_mr_reg(atl_context->domain, buf, len,
                    FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                    FI_REMOTE_READ | FI_REMOTE_WRITE, 0, 0, 0,
                    &atl_ofi_mr->mr, NULL);
    if (ret)
        goto mr_reg_err;

    atl_ofi_mr->atl_mr.buf = (void *)buf;
    atl_ofi_mr->atl_mr.len = len;
    atl_ofi_mr->atl_mr.r_key = (uintptr_t)fi_mr_key(atl_ofi_mr->mr);
    atl_ofi_mr->atl_mr.l_key = (uintptr_t)fi_mr_desc(atl_ofi_mr->mr);

    *atl_mr = &atl_ofi_mr->atl_mr;
    return atl_status_success;

mr_reg_err:
    free(atl_ofi_mr);
    return atl_status_failure;
}

static atl_status_t atl_ofi_mr_dereg(atl_desc_t *atl_desc, atl_mr_t *atl_mr)
{
    atl_ofi_mr_t *atl_ofi_mr = container_of(atl_mr, atl_ofi_mr_t, atl_mr);
    int ret = fi_close(&atl_ofi_mr->mr->fid);
    free(atl_ofi_mr);
    return RET2ATL(ret);
}


static atl_status_t atl_ofi_set_resize_function(atl_resize_fn_t resize_fn)
{
    return pmrt_set_resize_function(resize_fn);
}

atl_ops_t atl_ofi_ops = {
    .global_proc_idx        = atl_ofi_global_proc_idx,
    .global_proc_count      = atl_ofi_global_proc_count,
    .finalize               = atl_ofi_finalize,
    .update                 = atl_ofi_update,
    .wait_notification      = atl_ofi_wait_notification,
    .set_resize_function = atl_ofi_set_resize_function,
};

static atl_mr_ops_t atl_ofi_mr_ops = {
    .mr_reg = atl_ofi_mr_reg,
    .mr_dereg = atl_ofi_mr_dereg,
};

static void atl_ofi_tune(void)
{
    setenv("FI_PSM2_TIMEOUT", "1", 0);
    setenv("FI_PSM2_DELAY", "0", 0);
    setenv("FI_PSM2_LOCK_LEVEL", "1", 0);
    setenv("HFI_NO_CPUAFFINITY", "1", 0);

    setenv("FI_OFI_RXM_RX_SIZE", "8192", 0);
    setenv("FI_OFI_RXM_TX_SIZE", "8192", 0);
    setenv("FI_OFI_RXM_MSG_RX_SIZE", "128", 0);
    setenv("FI_OFI_RXM_MSG_TX_SIZE", "128", 0);
}

atl_status_t atl_ofi_init(int *argc, char ***argv, atl_proc_coord_t *proc_coord,
                          atl_attr_t *attr, atl_comm_t ***atl_comms, atl_desc_t **atl_desc)
{
    struct fi_info *providers, *hints;
    struct fi_av_attr av_attr = { (enum fi_av_type)0 };
    int fi_version, ret;
    atl_ofi_context_t *atl_ofi_context;

    ATL_OFI_ASSERT((sizeof(atl_ofi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal)),
                   "unexpected offset: atl_ofi_request size %zu, atl_request size %zu, expected offset %zu",
                   sizeof(atl_ofi_req_t), sizeof(atl_req_t), offsetof(atl_req_t, internal));

    atl_ofi_tune();

    atl_ofi_context = calloc(1, sizeof(*atl_ofi_context));
    if (!atl_ofi_context)
        return atl_status_failure;

    memset(proc_coord, 0, sizeof(atl_proc_coord_t));

    ret = pmrt_init(&atl_ofi_context->proc_coord.global_idx,
                    &atl_ofi_context->proc_coord.global_count,
                    &atl_ofi_context->pm_rt);

    if (ret)
        goto err_pmrt_init;

    ret = atl_ofi_get_local_proc_coord(atl_ofi_context);
    if (ret)
        goto err_pmrt_init;

    *proc_coord = atl_ofi_context->proc_coord;

    hints = fi_allocinfo();
    if (!hints)
        goto err_hints;

    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->caps = FI_TAGGED;

    fi_version = FI_VERSION(1, 0);

    ret = fi_getinfo(fi_version, NULL, NULL, 0ULL, hints, &providers);
    if (ret || !providers)
        goto err_getinfo;

    if (providers->domain_attr->max_ep_tx_ctx > 1)
    {
        hints->ep_attr->tx_ctx_cnt = attr->comm_count;
        hints->ep_attr->rx_ctx_cnt = attr->comm_count;
    }
    fi_freeinfo(providers);

    if (attr->enable_rma)
    {
        ATL_OFI_DEBUG_PRINT("try to enable RMA");
        hints->caps |= FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        hints->domain_attr->mr_mode = FI_MR_UNSPEC;
        // TODO:
        //hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR | FI_MR_LOCAL | FI_MR_BASIC;
    }

    ret = fi_getinfo(fi_version, NULL, NULL, 0ULL, hints, &providers);
    if (ret || !providers)
    {
        if (attr->enable_rma)
        {
            attr->enable_rma = 0;
            ATL_OFI_DEBUG_PRINT("try without RMA");
            hints->caps = FI_TAGGED;
            hints->domain_attr->mr_mode = FI_MR_UNSPEC;
            ret = fi_getinfo(fi_version, NULL, NULL, 0ULL, hints, &providers);
            if (ret || !providers)
                goto err_getinfo;
        }
        else
            goto err_getinfo;
    }

    /* Use first provider from the list of providers */
    atl_ofi_context->prov = fi_dupinfo(providers);
    if (!atl_ofi_context->prov)
        goto err_prov;

    attr->max_order_waw_size = atl_ofi_context->prov->ep_attr->max_order_waw_size;
    attr->is_tagged_coll_enabled = 0;
    attr->tag_bits = 64;
    attr->max_tag = 0xFFFFFFFFFFFFFFFF;

    ATL_OFI_DEBUG_PRINT("provider %s", atl_ofi_context->prov->fabric_attr->prov_name);
    ATL_OFI_DEBUG_PRINT("mr_mode %d", atl_ofi_context->prov->domain_attr->mr_mode);
    ATL_OFI_DEBUG_PRINT("threading %d", atl_ofi_context->prov->domain_attr->threading);
    ATL_OFI_DEBUG_PRINT("tx_ctx_cnt %zu", atl_ofi_context->prov->domain_attr->tx_ctx_cnt);
    ATL_OFI_DEBUG_PRINT("max_ep_tx_ctx %zu", atl_ofi_context->prov->domain_attr->max_ep_tx_ctx);

    ret = fi_fabric(atl_ofi_context->prov->fabric_attr,
                    &atl_ofi_context->fabric, NULL);
    if (ret)
        goto err_fab;

    ret = fi_domain(atl_ofi_context->fabric, atl_ofi_context->prov,
                    &atl_ofi_context->domain, NULL);
    if (ret)
        goto err_dom;

    av_attr.type = FI_AV_TABLE;
    av_attr.rx_ctx_bits = atl_ofi_context->rx_ctx_bits =
        (int)ceil(log2(hints->ep_attr->rx_ctx_cnt));
    ret = fi_av_open(atl_ofi_context->domain, &av_attr,
                     &atl_ofi_context->av, NULL);
    if (ret)
        goto err_av;

    atl_ofi_context->comm_ref_count = 0;

    ret = atl_ofi_comms_init(atl_ofi_context, attr->comm_count,
                             &attr->comm_attr, atl_comms);
    if (ret)
        goto err_comms_init;

    atl_ofi_context->atl_desc.ops = &atl_ofi_ops;
    global_data.is_resize_enabled = is_pm_resize_enabled();
    atl_ofi_context->atl_desc.ops->is_resize_enabled = global_data.is_resize_enabled;

    /* by default don't use timeout for retry operations and just break by retry_count */
    global_data.timeout_sec = 0;

    if (global_data.is_resize_enabled)
    {
        char* timeout_sec_env = getenv(ATL_OFI_TIMEOUT_SEC_ENV);
        if (timeout_sec_env)
        {
            global_data.timeout_sec = strtol(timeout_sec_env, NULL, 10);
        }
        else
        {
            global_data.timeout_sec = ATL_OFI_DEFAULT_TIMEOUT_SEC;
        }
    }

    atl_ofi_context->atl_desc.mr_ops = &atl_ofi_mr_ops;
    *atl_desc = &atl_ofi_context->atl_desc;

    fi_freeinfo(providers);
    fi_freeinfo(hints);

    return atl_status_success;

err_comms_init:
    fi_close(&atl_ofi_context->av->fid);
err_av:
    fi_close(&atl_ofi_context->domain->fid);
err_dom:
    fi_close(&atl_ofi_context->fabric->fid);
err_fab:
    fi_freeinfo(atl_ofi_context->prov);
err_prov:
    fi_freeinfo(providers);
err_getinfo:
    ATL_OFI_DEBUG_PRINT("can't find suitable provider");
    fi_freeinfo(hints);
err_hints:
    pmrt_finalize(atl_ofi_context->pm_rt);
err_pmrt_init:
    free(atl_ofi_context);
    return RET2ATL(ret);
}

ATL_OFI_INI
{
    atl_transport->name = atl_ofi_name;
    atl_transport->init = atl_ofi_init;

    return atl_status_success;
}
