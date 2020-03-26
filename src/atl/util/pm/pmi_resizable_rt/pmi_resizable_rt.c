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
#include "pm_rt_codec.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "pmi_resizable/resizable_pmi.h"

#include "pm_rt.h"

#define RESIZABLE_PMI_RT_KEY_FORMAT "%s-%zu"

typedef struct resizable_pm_rt_context {
    pm_rt_desc_t pmrt_desc;
    struct {
        size_t initialized;
        size_t ref_cnt;
        size_t max_keylen;
        size_t max_vallen;
        char *key_storage;
        char *val_storage;
        char *kvsname;
    } resizablert_main;
} resizable_pm_context_t;

/* Ensures that this is allocated/initialized only once per process */
static resizable_pm_context_t resizable_ctx_singleton;

static void resizable_pmirt_finalize(pm_rt_desc_t *pmrt_desc)
{
    resizable_pm_context_t *ctx =
        container_of(pmrt_desc, resizable_pm_context_t, pmrt_desc);
    if (!ctx->resizablert_main.initialized)
        return;

    if (--ctx->resizablert_main.ref_cnt)
        return;

    free(ctx->resizablert_main.kvsname);
    free(ctx->resizablert_main.key_storage);
    free(ctx->resizablert_main.val_storage);

    PMIR_Finalize();

    memset(ctx, 0, sizeof(*ctx));
}

static void resizable_pmirt_barrier(pm_rt_desc_t *pmrt_desc)
{
    resizable_pm_context_t *ctx =
        container_of(pmrt_desc, resizable_pm_context_t, pmrt_desc);

    if (!ctx->resizablert_main.initialized)
        return;

    PMIR_Barrier();
}

static atl_status_t
resizable_pmirt_kvs_put(pm_rt_desc_t *pmrt_desc, char *kvs_key, size_t proc_idx,
                        const void *kvs_val, size_t kvs_val_len)
{
    int ret;
    resizable_pm_context_t *ctx =
        container_of(pmrt_desc, resizable_pm_context_t, pmrt_desc);

    if (!ctx->resizablert_main.initialized)
        return ATL_STATUS_FAILURE;

    if (kvs_val_len > ctx->resizablert_main.max_vallen)
        return ATL_STATUS_FAILURE;

    ret = snprintf(ctx->resizablert_main.key_storage, ctx->resizablert_main.max_keylen - 1,
                   RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = encode(kvs_val, kvs_val_len, ctx->resizablert_main.val_storage,
                 ctx->resizablert_main.max_vallen);
    if (ret)
        return ATL_STATUS_FAILURE;

    ret = PMIR_KVS_Put(ctx->resizablert_main.kvsname,
                       ctx->resizablert_main.key_storage,
                       ctx->resizablert_main.val_storage);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMIR_KVS_Commit(ctx->resizablert_main.kvsname);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

static atl_status_t
resizable_pmirt_kvs_get(pm_rt_desc_t *pmrt_desc, char *kvs_key, size_t proc_idx,
                        void *kvs_val, size_t kvs_val_len)
{
    int ret;
    resizable_pm_context_t *ctx =
        container_of(pmrt_desc, resizable_pm_context_t, pmrt_desc);

    if (!ctx->resizablert_main.initialized)
        return ATL_STATUS_FAILURE;

    ret = snprintf(ctx->resizablert_main.key_storage, ctx->resizablert_main.max_keylen - 1,
                   RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = PMIR_KVS_Get(ctx->resizablert_main.kvsname, ctx->resizablert_main.key_storage,
                       ctx->resizablert_main.val_storage,
                       ctx->resizablert_main.max_vallen);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = decode(ctx->resizablert_main.val_storage, kvs_val, kvs_val_len);
    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

static atl_status_t
resizable_pmirt_update(size_t *proc_idx, size_t *proc_count)
{
    int ret;
    ret = PMIR_Update();
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    ret = PMIR_Get_size(proc_count);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    ret = PMIR_Get_rank(proc_idx);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    return ATL_STATUS_SUCCESS;

err_resizable:
    PMIR_Finalize();
    return ATL_STATUS_FAILURE;
}

atl_status_t resizable_pmirt_wait_notification()
{
    int ret;

    ret = PMIR_Wait_notification();

    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

pm_rt_ops_t resizable_ops = {
    .finalize = resizable_pmirt_finalize,
    .barrier  = resizable_pmirt_barrier,
    .update   = resizable_pmirt_update,
    .wait_notification = resizable_pmirt_wait_notification,
};

pm_rt_kvs_ops_t resizable_kvs_ops = {
    .put = resizable_pmirt_kvs_put,
    .get = resizable_pmirt_kvs_get,
};

atl_status_t resizable_pmirt_init(size_t *proc_idx, size_t *proc_count, pm_rt_desc_t **pmrt_desc, const char* main_addr)
{
    int ret;
    size_t max_kvsnamelen;

    if (resizable_ctx_singleton.resizablert_main.initialized) {
        PMIR_Get_size(proc_idx);
        PMIR_Get_rank(proc_count);
        *pmrt_desc = &resizable_ctx_singleton.pmrt_desc;
        resizable_ctx_singleton.resizablert_main.ref_cnt++;
        return ATL_STATUS_SUCCESS;
    }

    ret = PMIR_Init(main_addr);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMIR_Update();
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMIR_Get_size(proc_count);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;
    ret = PMIR_Get_rank(proc_idx);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    ret = PMIR_KVS_Get_name_length_max(&max_kvsnamelen);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    resizable_ctx_singleton.resizablert_main.kvsname = calloc(1, max_kvsnamelen);
    if (!resizable_ctx_singleton.resizablert_main.kvsname)
        goto err_resizable;

    ret = PMIR_KVS_Get_my_name(resizable_ctx_singleton.resizablert_main.kvsname,
                               max_kvsnamelen);
    if (ret != PMIR_SUCCESS)
        goto err_alloc_key;

    ret = PMIR_KVS_Get_key_length_max(&resizable_ctx_singleton.resizablert_main.max_keylen);
    if (ret != PMIR_SUCCESS)
        goto err_alloc_key;

    resizable_ctx_singleton.resizablert_main.key_storage =
        (char *)calloc(1, resizable_ctx_singleton.resizablert_main.max_keylen);
    if (!resizable_ctx_singleton.resizablert_main.key_storage)
        goto err_alloc_key;

    ret = PMIR_KVS_Get_value_length_max(&resizable_ctx_singleton.resizablert_main.max_vallen);
    if (ret != PMIR_SUCCESS)
        goto err_alloc_val;

    resizable_ctx_singleton.resizablert_main.val_storage =
        (char *)calloc(1, resizable_ctx_singleton.resizablert_main.max_vallen);
    if (!resizable_ctx_singleton.resizablert_main.val_storage)
        goto err_alloc_val;

    resizable_ctx_singleton.resizablert_main.initialized = 1;
    resizable_ctx_singleton.resizablert_main.ref_cnt = 1;
    resizable_ctx_singleton.pmrt_desc.ops = &resizable_ops;
    resizable_ctx_singleton.pmrt_desc.kvs_ops = &resizable_kvs_ops;
    *pmrt_desc = &resizable_ctx_singleton.pmrt_desc;

    return ATL_STATUS_SUCCESS;
err_alloc_val:
    free(resizable_ctx_singleton.resizablert_main.key_storage);
err_alloc_key:
    free(resizable_ctx_singleton.resizablert_main.kvsname);
err_resizable:
    PMIR_Finalize();
    return ATL_STATUS_FAILURE;
}

atl_status_t resizable_pmirt_main_addr_reserv(char* main_addr)
{
    int ret = PMIR_Main_Addr_Reserv(main_addr);

    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

atl_status_t resizable_pmirt_set_resize_function(atl_resize_fn_t resize_fn)
{
    int ret = PMIR_set_resize_function((pmir_resize_fn_t) resize_fn);

    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}
