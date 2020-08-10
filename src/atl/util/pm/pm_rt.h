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
#ifndef PM_RT_H
#define PM_RT_H

#include "atl.h"

#define PM_TYPE "CCL_PM_TYPE"

#define PM_RT_VAL_SIMPLE    "simple"
#define PM_RT_VAL_RESIZABLE "resizable"

typedef struct pm_rt_desc pm_rt_desc_t;

/* PMI RT */
atl_status_t pmirt_init(size_t *proc_idx, size_t *procs_num, pm_rt_desc_t **pmrt_desc);
atl_status_t resizable_pmirt_init(size_t *proc_idx,
                                  size_t *proc_count,
                                  pm_rt_desc_t **pmrt_desc,
                                  const char *main_addr);
atl_status_t resizable_pmirt_set_resize_function(atl_resize_fn_t resize_fn);
atl_status_t resizable_pmirt_main_addr_reserv(char *main_addr);

typedef enum pm_rt_type {
    PM_RT_SIMPLE = 0,
    PM_RT_RESIZABLE = 1,
} pm_rt_type_t;

static pm_rt_type_t type = PM_RT_SIMPLE;

typedef struct pm_rt_ops {
    void (*finalize)(pm_rt_desc_t *pmrt_desc);
    void (*barrier)(pm_rt_desc_t *pmrt_desc);
    atl_status_t (*update)(size_t *proc_idx, size_t *proc_count);
    atl_status_t (*wait_notification)(void);
} pm_rt_ops_t;

typedef struct pm_rt_kvs_ops {
    atl_status_t (*put)(pm_rt_desc_t *pmrt_desc,
                        char *kvs_key,
                        size_t proc_idx,
                        const void *kvs_val,
                        size_t kvs_val_len);
    atl_status_t (*get)(pm_rt_desc_t *pmrt_desc,
                        char *kvs_key,
                        size_t proc_idx,
                        void *kvs_val,
                        size_t kvs_val_len);
} pm_rt_kvs_ops_t;

struct pm_rt_desc {
    pm_rt_ops_t *ops;
    pm_rt_kvs_ops_t *kvs_ops;
};

static inline int is_pm_resize_enabled() {
    if (type == PM_RT_RESIZABLE)
        return 1;
    return 0;
}

static inline atl_status_t pmrt_init(size_t *proc_idx,
                                     size_t *procs_num,
                                     pm_rt_desc_t **pmrt_desc,
                                     const char *main_addr) {
    char *type_str = getenv(PM_TYPE);

    if (type_str) {
        if (strstr(type_str, PM_RT_VAL_SIMPLE)) {
            type = PM_RT_SIMPLE;
        }
        else if (strstr(type_str, PM_RT_VAL_RESIZABLE)) {
            type = PM_RT_RESIZABLE;
        }
        else {
            printf("Unknown %s: %s\n", PM_TYPE, type_str);
            return ATL_STATUS_FAILURE;
        }
    }

    switch (type) {
        case PM_RT_SIMPLE: return pmirt_init(proc_idx, procs_num, pmrt_desc);
        case PM_RT_RESIZABLE:
            return resizable_pmirt_init(proc_idx, procs_num, pmrt_desc, main_addr);
        default: printf("Wrong CCL_PM_TYPE: %s", type_str); return ATL_STATUS_FAILURE;
    }
}

static inline atl_status_t pmrt_main_addr_reserv(char *main_addr) {
    return resizable_pmirt_main_addr_reserv(main_addr);
}

static inline atl_status_t pmrt_set_resize_function(atl_resize_fn_t user_checker) {
    switch (type) {
        case PM_RT_RESIZABLE: return resizable_pmirt_set_resize_function(user_checker);
        default: return ATL_STATUS_SUCCESS;
    }
}
static inline atl_status_t pmrt_update(size_t *proc_idx,
                                       size_t *proc_count,
                                       pm_rt_desc_t *pmrt_desc) {
    return pmrt_desc->ops->update(proc_idx, proc_count);
}
static inline atl_status_t pmrt_wait_notification(pm_rt_desc_t *pmrt_desc) {
    return pmrt_desc->ops->wait_notification();
}
static inline void pmrt_finalize(pm_rt_desc_t *pmrt_desc) {
    pmrt_desc->ops->finalize(pmrt_desc);
}
static inline void pmrt_barrier(pm_rt_desc_t *pmrt_desc) {
    pmrt_desc->ops->barrier(pmrt_desc);
}

static inline atl_status_t pmrt_kvs_put(pm_rt_desc_t *pmrt_desc,
                                        char *kvs_key,
                                        size_t proc_idx,
                                        const void *kvs_val,
                                        size_t kvs_val_len) {
    return pmrt_desc->kvs_ops->put(pmrt_desc, kvs_key, proc_idx, kvs_val, kvs_val_len);
}

static inline atl_status_t pmrt_kvs_get(pm_rt_desc_t *pmrt_desc,
                                        char *kvs_key,
                                        size_t proc_idx,
                                        void *kvs_val,
                                        size_t kvs_val_len) {
    return pmrt_desc->kvs_ops->get(pmrt_desc, kvs_key, proc_idx, kvs_val, kvs_val_len);
}

#endif /* PM_RT_H */
