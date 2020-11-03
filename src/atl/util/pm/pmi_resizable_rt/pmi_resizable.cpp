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

#include "util/pm/codec/pm_rt_codec.h"
#include "pmi_resizable.h"

#define RESIZABLE_PMI_RT_KEY_FORMAT "%s-%zu"

int pmi_resizable::is_pm_resize_enabled() {
    return true;
}

atl_status_t pmi_resizable::pmrt_init(const char *main_addr) {
    int ret;
    size_t max_kvsnamelen;

    ret = PMIR_Init(main_addr);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMIR_Update();
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMIR_Get_size(&size);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;
    ret = PMIR_Get_rank(&rank);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    ret = PMIR_KVS_Get_name_length_max(&max_kvsnamelen);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    kvsname = (char *)calloc(1, max_kvsnamelen);
    if (!kvsname)
        goto err_resizable;

    ret = PMIR_KVS_Get_my_name(kvsname, max_kvsnamelen);
    if (ret != PMIR_SUCCESS)
        goto err_alloc_key;

    ret = PMIR_KVS_Get_key_length_max(&max_keylen);
    if (ret != PMIR_SUCCESS)
        goto err_alloc_key;

    key_storage = (char *)calloc(1, max_keylen);
    if (!key_storage)
        goto err_alloc_key;

    ret = PMIR_KVS_Get_value_length_max(&max_vallen);
    if (ret != PMIR_SUCCESS)
        goto err_alloc_val;

    val_storage = (char *)calloc(1, max_vallen);
    if (!val_storage)
        goto err_alloc_val;

    initialized = true;

    return ATL_STATUS_SUCCESS;
err_alloc_val:
    free(key_storage);
err_alloc_key:
    free(kvsname);
err_resizable:
    PMIR_Finalize();
    return ATL_STATUS_FAILURE;
}

atl_status_t pmi_resizable::pmrt_main_addr_reserv(char *main_addr) {
    int ret = PMIR_Main_Addr_Reserv(main_addr);

    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable::pmrt_set_resize_function(atl_resize_fn_t resize_fn) {
    int ret = PMIR_set_resize_function((pmir_resize_fn_t)resize_fn);

    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable::pmrt_update() {
    int ret;
    ret = PMIR_Update();
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    ret = PMIR_Get_size(&size);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    ret = PMIR_Get_rank(&rank);
    if (ret != PMIR_SUCCESS)
        goto err_resizable;

    return ATL_STATUS_SUCCESS;

err_resizable:
    PMIR_Finalize();
    return ATL_STATUS_FAILURE;
}

atl_status_t pmi_resizable::pmrt_wait_notification() {
    int ret;

    ret = PMIR_Wait_notification();

    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

void pmi_resizable::pmrt_finalize() {
    is_finalized = true;
    if (!initialized)
        return;

    free(kvsname);
    free(key_storage);
    free(val_storage);

    PMIR_Finalize();
}

void pmi_resizable::pmrt_barrier() {
    if (!initialized)
        return;

    PMIR_Barrier();
}

atl_status_t pmi_resizable::pmrt_kvs_put(char *kvs_key,
                                         size_t proc_idx,
                                         const void *kvs_val,
                                         size_t kvs_val_len) {
    int ret;

    if (!initialized)
        return ATL_STATUS_FAILURE;

    if (kvs_val_len > max_vallen)
        return ATL_STATUS_FAILURE;

    ret = snprintf(key_storage, max_keylen - 1, RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = encode(kvs_val, kvs_val_len, val_storage, max_vallen);
    if (ret)
        return ATL_STATUS_FAILURE;

    ret = PMIR_KVS_Put(kvsname, key_storage, val_storage);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMIR_KVS_Commit(kvsname);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable::pmrt_kvs_get(char *kvs_key,
                                         size_t proc_idx,
                                         void *kvs_val,
                                         size_t kvs_val_len) {
    int ret;

    if (!initialized)
        return ATL_STATUS_FAILURE;

    ret = snprintf(key_storage, max_keylen - 1, RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = PMIR_KVS_Get(kvsname, key_storage, val_storage, max_vallen);
    if (ret != PMIR_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = decode(val_storage, kvs_val, kvs_val_len);
    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}
