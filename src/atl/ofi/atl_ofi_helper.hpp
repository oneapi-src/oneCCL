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
#include <algorithm>
#include <assert.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <math.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#include "atl.h"
#include "common/global/global.hpp"
#include "hwloc/hwloc_wrapper.hpp"
#ifdef CCL_ENABLE_OFI_HMEM
#include "sched/entry/gpu/ze_primitives.hpp"
#endif // CCL_ENABLE_OFI_HMEM

#define ATL_OFI_BASE_PM_KEY     "atl-ofi"
#define ATL_OFI_FI_ADDR_PM_KEY  ATL_OFI_BASE_PM_KEY "-fiaddr"
#define ATL_OFI_HOSTNAME_PM_KEY ATL_OFI_BASE_PM_KEY "-hostname"

#define ATL_OFI_MAJOR_VERSION       "CCL_ATL_OFI_MAJOR_VERSION"
#define ATL_OFI_MINOR_VERSION       "CCL_ATL_OFI_MINOR_VERSION"
#define ATL_OFI_TIMEOUT_SEC_ENV     "CCL_ATL_OFI_TIMEOUT_SEC"
#define ATL_OFI_MAX_RETRY_COUNT_ENV "CCL_ATL_OFI_MAX_RETRY_COUNT"

#define ATL_OFI_DEFAULT_TIMEOUT_SEC 60
#define ATL_OFI_MAX_RETRY_COUNT     10000
#define ATL_OFI_WAIT_SEC            10
#define ATL_OFI_CQ_READ_ITERS       10000
#define ATL_OFI_CQ_BUNCH_SIZE       8

#define ATL_OFI_MAX_PROV_ENV_LEN    128
#define ATL_OFI_PMI_PROV_MULTIPLIER 100
#define ATL_OFI_PMI_PROC_MULTIPLIER (ATL_OFI_PMI_PROV_MULTIPLIER * 10)
#define ATL_OFI_MAX_NW_PROV_COUNT   1024
#define ATL_OFI_MAX_PROV_COUNT      (ATL_OFI_MAX_NW_PROV_COUNT + 1) /* NW and SHM providers */
#define ATL_OFI_MAX_ACTIVE_PROV_COUNT \
    2 /* by current scheme each EP may use only SHM and 1 NW prov */
#define ATL_OFI_SHM_PROV_NAME "shm"

#define ATL_OFI_MAX_ZE_DEV_COUNT 1024

#ifndef PRId64
#define PRId64 "lld"
#endif

#define MAX(a, b) \
    ({ \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b; \
    })

#define MIN(a, b) \
    ({ \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a < _b ? _a : _b; \
    })

#define ATL_OFI_CALL(func, ret_val, err_action) \
    do { \
        (ret_val) = func; \
        if ((ret_val) != FI_SUCCESS) { \
            LOG_ERROR( \
                #func "\n fails with ret: ", ret_val, ", strerror: ", fi_strerror(-(ret_val))); \
            err_action; \
        } \
    } while (0)

#define ATL_OFI_RETRY(func, ep, ret_val) \
    do { \
        atl_ctx_t* ctx_local = (ep)->ctx; \
        atl_ofi_ctx_t* ofi_ctx_local = container_of(ctx_local, atl_ofi_ctx_t, ctx); \
        size_t max_retry_count = ofi_ctx_local->max_retry_count; \
        size_t retry_count = 0; \
        do { \
            (ret_val) = func; \
            if ((ret_val) == FI_SUCCESS) \
                break; \
            if ((ret_val) != -FI_EAGAIN) { \
                LOG_ERROR(#func "\n fails with ret: ", \
                          (ret_val), \
                          ", strerror: ", \
                          fi_strerror(-(ret_val))); \
                CCL_THROW("OFI function error"); \
                break; \
            } \
            (void)atl_ep_poll(ep); \
            retry_count++; \
        } while (((ret_val) == -FI_EAGAIN) && (retry_count < max_retry_count)); \
    } while (0)

/* OFI returns 0 or -errno */
#define RET2ATL(ret) \
    ({ \
        atl_status_t res; \
        if ((ret) == -FI_EAGAIN) \
            res = ATL_STATUS_AGAIN; \
        else \
            res = (ret) ? ATL_STATUS_FAILURE : ATL_STATUS_SUCCESS; \
        res; \
    })

inline long int safe_c_strtol(const char* str, char** endptr, int base) {
    long int val = strtol(str, endptr, base);
    if (val == 0) {
        /* if a conversion error occurred, display a message and exit */
        if (errno == EINVAL) {
            LOG_ERROR("conversion error occurred for string: ", str);
        }
        /* if the value provided was out of range, display a error message */
        if (errno == ERANGE) {
            LOG_ERROR("the value provided was out of range, string: ", str);
        }
    }
    return val;
}

typedef enum {
    ATL_OFI_COMP_POSTED,
    ATL_OFI_COMP_COMPLETED,
    ATL_OFI_COMP_PEEK_STARTED,
    ATL_OFI_COMP_PEEK_FOUND,
    ATL_OFI_COMP_PEEK_NOT_FOUND,
} atl_ofi_comp_state_t;

typedef struct {
    atl_mr_t mr;
    struct fid_mr* fi_mr;
} atl_ofi_mr_t;

typedef struct {
    void* addr;
    size_t len;
} atl_ofi_prov_ep_name_t;

typedef struct {
    struct fid_ep* tx;
    struct fid_ep* rx;
    struct fid_cq* cq;
    atl_ofi_prov_ep_name_t name;
} atl_ofi_prov_ep_t;

typedef struct {
    size_t idx;
    struct fi_info* info;
    struct fid_fabric* fabric;
    struct fid_domain* domain;
    struct fid_av* av;
    atl_ofi_prov_ep_t* eps;

    int is_shm;
    size_t max_msg_size;

    /* used only in case of SEP supported */
    struct fid_ep* sep;
    int rx_ctx_bits;

    /* table[0..proc_count][0..ep_count] */
    fi_addr_t* addr_table;
    size_t addr_len;
    int first_proc_idx;
} atl_ofi_prov_t;

typedef struct {
    atl_ep_t ep;

    /* used to make progressing only for really used providers */
    size_t active_prov_count;
    size_t active_prov_idxs[ATL_OFI_MAX_ACTIVE_PROV_COUNT];

} atl_ofi_ep_t;

typedef struct {
    atl_ctx_t ctx;
    pm_rt_desc_t* pm_rt;
    atl_ofi_prov_t provs[ATL_OFI_MAX_PROV_COUNT];
    size_t prov_count;
    size_t nw_prov_count;
    size_t nw_prov_first_idx;
    size_t shm_prov_idx;
    size_t max_retry_count;
    atl_progress_mode_t progress_mode;
    atl_mnic_t mnic_type;
    std::vector<std::string> mnic_include_names;
    std::vector<std::string> mnic_exclude_names;
    size_t mnic_count;
    int enable_hmem;
} atl_ofi_ctx_t;

typedef struct {
    struct fi_context fi_ctx;
    atl_ofi_prov_ep_t* prov_ep;
    struct fid_ep* fi_ep;
    atl_ofi_comp_state_t comp_state;
    size_t recv_len;
    struct fid_mr* mr;
} atl_ofi_req_t;

#ifdef CCL_ENABLE_OFI_HMEM
typedef struct atl_ofi_ze_data {
    ze_driver_handle_t driver;
    ze_context_handle_t context;
    uint32_t device_count;
    ze_device_handle_t devices[ATL_OFI_MAX_ZE_DEV_COUNT];

    atl_ofi_ze_data() : driver(nullptr), context(nullptr), device_count(0) {}

} atl_ofi_ze_data_t;
#endif // CCL_ENABLE_OFI_HMEM

typedef struct atl_ofi_global_data {
    size_t ctx_count;
    int is_env_inited;
    void* dlhandle;
    char prov_env_copy[ATL_OFI_MAX_PROV_ENV_LEN];

    int fi_major_version;
    int fi_minor_version;

#ifdef CCL_ENABLE_OFI_HMEM
    atl_ofi_ze_data ze_data;
#endif // CCL_ENABLE_OFI_HMEM

    atl_ofi_global_data()
            : ctx_count(0),
              is_env_inited(0),
              dlhandle(nullptr),
              prov_env_copy(),
              fi_major_version(1),
              fi_minor_version(10) {
        memset(prov_env_copy, 0, sizeof(prov_env_copy));
    }
} atl_ofi_global_data_t;

extern atl_ofi_global_data_t global_data;

template <class Container>
std::string vec_to_string(Container& elems);

#ifdef CCL_ENABLE_OFI_HMEM
void atl_ofi_init_ze_data();
#endif // CCL_ENABLE_OFI_HMEM

void atl_ofi_print_coord(atl_proc_coord_t* coord);
std::string atl_ofi_get_short_nic_name(const struct fi_info* prov);
std::string atl_ofi_get_nic_name(const struct fi_info* prov);
atl_ofi_prov_t* atl_ofi_get_prov(atl_ep_t* ep, int peer_proc_idx, size_t msg_size);
fi_addr_t atl_ofi_get_addr(atl_ctx_t* ctx, atl_ofi_prov_t* prov, int proc_idx, size_t ep_idx);
atl_status_t atl_ofi_get_local_proc_coord(atl_ofi_ctx_t* ofi_ctx, std::unique_ptr<ipmi>& pmi);
atl_status_t atl_ofi_prov_update_addr_table(atl_ofi_ctx_t* ofi_ctx,
                                            size_t prov_idx,
                                            std::unique_ptr<ipmi>& pmi);
atl_status_t atl_ofi_prov_ep_get_name(atl_ofi_prov_t* prov, size_t ep_idx);
atl_status_t atl_ofi_prov_eps_connect(atl_ofi_ctx_t* ofi_ctx,
                                      size_t prov_idx,
                                      std::unique_ptr<ipmi>& pmi);
void atl_ofi_prov_ep_destroy(atl_ofi_prov_t* prov, atl_ofi_prov_ep_t* ep);
void atl_ofi_prov_destroy(atl_ctx_t* ctx, atl_ofi_prov_t* prov);
int atl_ofi_wait_cancel_cq(struct fid_cq* cq);
atl_status_t atl_ofi_prov_ep_init(atl_ofi_prov_t* prov, size_t ep_idx);
atl_status_t atl_ofi_try_to_drain_cq_err(struct fid_cq* cq);
int atl_ofi_try_to_drain_cq(struct fid_cq* cq);
void atl_ofi_reset(atl_ctx_t* ctx);
atl_status_t atl_ofi_adjust_env(const atl_attr_t& attr);
atl_status_t atl_ofi_set_env(const atl_attr_t& attr);
atl_status_t atl_ofi_get_prov_list(atl_ctx_t* ctx,
                                   const char* prov_name,
                                   struct fi_info* base_hints,
                                   struct fi_info** out_prov_list);
atl_status_t atl_ofi_prov_init(atl_ctx_t* ctx,
                               struct fi_info* info,
                               atl_ofi_prov_t* prov,
                               atl_attr_t* attr,
                               std::unique_ptr<ipmi>& pmi);
atl_status_t atl_ofi_adjust_out_tag(atl_ofi_prov_t* prov, atl_attr_t* attr);
int atl_ofi_nic_already_used(const struct fi_info* prov, struct fi_info** others, size_t nic_count);
int atl_ofi_is_nic_local(struct fi_info* info);
atl_status_t atl_ofi_parse_mnic_name(atl_ctx_t* ctx, std::string str_to_parse);
int atl_ofi_is_allowed_nic_name(atl_ofi_ctx_t* ofi_ctx, struct fi_info* info);
atl_status_t atl_ofi_open_nw_provs(atl_ctx_t* ctx,
                                   struct fi_info* base_hints,
                                   atl_attr_t* attr,
                                   std::unique_ptr<ipmi>& pmi);
