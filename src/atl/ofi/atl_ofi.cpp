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
#include "atl_ofi.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

// cache

void atl_ofi::fi_cache::clear() {
    for (auto& instance : memory_regions) {
        instance.clear();
    }
}

void atl_ofi::fi_cache::init(size_t instance_count, int enable_hmem) {
    this->enable_hmem = enable_hmem;
    memory_regions.resize(instance_count);
}

atl_ofi::fi_cache::~fi_cache() {
    clear();
}

void atl_ofi::fi_cache::get(size_t idx, fid_domain* domain, void* buf, size_t bytes, fid_mr** mr) {
    CCL_THROW_IF_NOT(mr);
    *mr = nullptr;
#ifdef CCL_ENABLE_OFI_HMEM
    if (enable_hmem) {
        memory_regions.at(idx % memory_regions.size()).get(domain, buf, bytes, mr);
    }
#endif // CCL_ENABLE_OFI_HMEM
}

void atl_ofi::fi_cache::push(size_t idx, fid_mr* mr) {
#ifdef CCL_ENABLE_OFI_HMEM
    if (mr)
        memory_regions.at(idx % memory_regions.size()).push(mr);
#endif // CCL_ENABLE_OFI_HMEM
}

atl_ofi::mr_cache::~mr_cache() {
    if (!cache.empty()) {
        LOG_WARN("mr cache is not empty, size: ", cache.size());
        clear();
    }
}

void atl_ofi::mr_cache::clear() {
    LOG_DEBUG("mr cache size: ", cache.size());
    for (auto& key_value : cache) {
        fi_close(&key_value.second->fid);
    }
    cache.clear();
}

void atl_ofi::mr_cache::get(fid_domain* domain, void* buf, size_t bytes, fid_mr** mr) {
    CCL_THROW_IF_NOT(domain);
    CCL_THROW_IF_NOT(mr);

    if (ccl::global_data::env().enable_atl_cache) {
        key_t key(domain, buf, bytes);
        auto key_value = cache.find(key);
        if (key_value != cache.end()) {
            *mr = key_value->second;
            LOG_DEBUG("loaded from mr cache: buf: ", buf, ", bytes: ", bytes);
            return;
        }
    }

    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {};

    iov.iov_base = buf;
    iov.iov_len = bytes;
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE;
    mr_attr.requested_key = mr_key++;

#ifdef CCL_ENABLE_OFI_HMEM

    mr_attr.iface = FI_HMEM_SYSTEM;
    mr_attr.device.ze = 0;

    atl_ofi_ze_data& ze_data = global_data.ze_data;
    ze_memory_allocation_properties_t alloc_props = ccl::ze::default_alloc_props;
    ze_device_handle_t alloc_dev = nullptr;
    ZE_CALL(zeMemGetAllocProperties, (ze_data.context, buf, &alloc_props, &alloc_dev));

    LOG_DEBUG("alloc_props: dev ", alloc_dev, ", type ", alloc_props.type);

    if (alloc_props.type == ZE_MEMORY_TYPE_HOST || alloc_props.type == ZE_MEMORY_TYPE_DEVICE ||
        alloc_props.type == ZE_MEMORY_TYPE_SHARED) {
        mr_attr.iface = FI_HMEM_ZE;
    }

    if (alloc_dev) {
        ze_device_properties_t alloc_dev_props = ccl::ze::default_device_props;
        ZE_CALL(zeDeviceGetProperties, (alloc_dev, &alloc_dev_props));

        int dev_idx = -1;
        for (int idx = 0; idx < ze_data.device_count; idx++) {
            ze_device_properties_t dev_props = ccl::ze::default_device_props;
            ZE_CALL(zeDeviceGetProperties, (ze_data.devices[idx], &dev_props));

            if (!std::memcmp(&dev_props.uuid, &alloc_dev_props.uuid, sizeof(ze_device_uuid_t))) {
                dev_idx = idx;
                LOG_DEBUG("buf ", buf, " corresponds to ze device idx ", dev_idx);
                break;
            }
        }
        CCL_THROW_IF_NOT(dev_idx != -1);
        mr_attr.device.ze = dev_idx;
    }
#endif // CCL_ENABLE_OFI_HMEM

    int ofi_ret;
    ATL_OFI_CALL(fi_mr_regattr(domain, &mr_attr, 0, mr),
                 ofi_ret,
                 CCL_THROW("failed to register mr, ret: ",
                           ofi_ret,
                           ", buf: ",
                           buf,
                           ", bytes: ",
                           bytes,
                           ", iface: ",
                           mr_attr.iface));

    if (ccl::global_data::env().enable_atl_cache) {
        key_t key(domain, buf, bytes);
        LOG_DEBUG("inserted to mr cache: buf: ", buf, ", bytes: ", bytes);
        cache.insert({ std::move(key), *mr });
    }
}

void atl_ofi::mr_cache::push(fid_mr* mr) {
    CCL_THROW_IF_NOT(mr);
    if (ccl::global_data::env().enable_atl_cache) {
        /* do nothing, all mem regions will be closed in clear() */
        return;
    }
    fi_close(&mr->fid);
}

// atl_ofi

atl_status_t atl_ofi::atl_set_env(const atl_attr_t& attr) {
    return atl_ofi_set_env(attr);
}

atl_status_t atl_ofi::atl_init(int* argc,
                               char*** argv,
                               atl_attr_t* attr,
                               const char* main_addr,
                               std::unique_ptr<ipmi>& pmi) {
    inited = true;
    struct fi_info *prov_list = nullptr, *base_hints = nullptr, *prov_hints = nullptr;
    int fi_version;
    ssize_t ret = 0;
    size_t idx = 0, ep_idx = 0, prov_idx = 0;
    char* prov_name = nullptr;
    char* prov_env = nullptr;
    char* fi_version_env = nullptr;
    atl_ofi_prov_t* prov = nullptr;
    char *max_retry_count_env = nullptr, *progress_mode_env = nullptr;
    int open_nw_provs = 1;
    int enable_shm = 0;

    CCL_THROW_IF_NOT((sizeof(atl_ofi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal)),
                     "unexpected offset: atl_ofi_request size ",
                     sizeof(atl_ofi_req_t),
                     ", atl_request size ",
                     sizeof(atl_req_t),
                     ", expected offset ",
                     offsetof(atl_req_t, internal));

    if (global_data.ctx_count == 0) {
        ret = atl_ofi_set_env(*attr);
        if (ret != ATL_STATUS_SUCCESS) {
            LOG_ERROR("atl_ofi_set_env error");
            return ATL_STATUS_FAILURE;
        }

        fi_version_env = getenv(ATL_OFI_MAJOR_VERSION);
        if (fi_version_env) {
            global_data.fi_major_version = safe_c_strtol(fi_version_env, nullptr, 10);
        }

        fi_version_env = getenv(ATL_OFI_MINOR_VERSION);
        if (fi_version_env) {
            global_data.fi_minor_version = safe_c_strtol(fi_version_env, nullptr, 10);
        }

        LOG_INFO("fi_version: ", global_data.fi_major_version, ".", global_data.fi_minor_version);

#ifdef CCL_ENABLE_OFI_HMEM
        atl_ofi_init_ze_data();
#endif // CCL_ENABLE_OFI_HMEM
    }
    global_data.ctx_count++;

    atl_ofi_ctx_t* ofi_ctx;
    ofi_ctx = (atl_ofi_ctx_t*)calloc(1, sizeof(atl_ofi_ctx_t));
    if (!ofi_ctx)
        return ATL_STATUS_FAILURE;

    ctx = &(ofi_ctx->ctx);

    ctx->ep_count = attr->in.ep_count;
    ctx->eps = (atl_ep**)calloc(1, sizeof(void*) * attr->in.ep_count);
    if (!ctx->eps)
        goto err;

    ctx->coord.global_count = pmi->get_size();
    ctx->coord.global_idx = pmi->get_rank();

    ret = atl_ofi_get_local_proc_coord(ofi_ctx, pmi);
    if (ret) {
        LOG_ERROR("atl_ofi_get_local_proc_coord error");
        goto err;
    }

    atl_proc_coord_t* coord;
    coord = &(ctx->coord);

    base_hints = fi_allocinfo();
    if (!base_hints) {
        LOG_ERROR("can't alloc base_hints");
        goto err;
    }

    base_hints->mode = FI_CONTEXT;
    base_hints->ep_attr->type = FI_EP_RDM;
    base_hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
    base_hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    base_hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    base_hints->caps = FI_TAGGED;

    prov_env = getenv("FI_PROVIDER");

    ofi_ctx->enable_hmem = 0;

#ifdef CCL_ENABLE_OFI_HMEM
    if (prov_env && strstr(prov_env, "verbs") && attr->in.enable_hmem) {
        ofi_ctx->enable_hmem = 1;
    }

    if (ofi_ctx->enable_hmem) {
        base_hints->caps |= FI_HMEM;
        base_hints->domain_attr->mr_mode =
            (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR | FI_MR_LOCAL | FI_MR_HMEM);

        /* TODO: enable shm with HMEM */
        attr->in.enable_shm = 0;

        /* TODO: implement fallback logic if HMEM can't be enabled */
    }
#endif // CCL_ENABLE_OFI_HMEM

    cache.init(attr->in.ep_count, ofi_ctx->enable_hmem);

    fi_version = FI_VERSION(global_data.fi_major_version, global_data.fi_minor_version);

    if (coord->global_idx == 0)
        LOG_INFO("libfabric version: ", fi_tostr("1" /* ignored */, FI_TYPE_VERSION));

    if (prov_env && !strcmp(prov_env, ATL_OFI_SHM_PROV_NAME)) {
        if (coord->global_count != coord->local_count) {
            LOG_ERROR("shm provider is requested as primary provider but global_count (",
                      coord->global_count,
                      ") != local_count (",
                      coord->local_count,
                      ")");
            goto err;
        }

        if (!attr->in.enable_shm) {
            LOG_ERROR(
                "shm provider is requested through FI_PROVIDER but not requested from CCL level");
            goto err;
        }
    }

    atl_ofi_print_coord(coord);

    enable_shm = attr->in.enable_shm;
    if (enable_shm) {
        prov_hints = fi_dupinfo(base_hints);
        prov_hints->fabric_attr->prov_name = strdup(ATL_OFI_SHM_PROV_NAME);
        ret = fi_getinfo(fi_version, nullptr, nullptr, 0ULL, prov_hints, &prov_list);
        if (ret || !prov_list) {
            enable_shm = 0;
            LOG_INFO("shm provider is requested but not available");
        }
        else {
            LOG_INFO("shm provider is requested and available");
        }

        fi_freeinfo(prov_list);
        prov_list = nullptr;

        fi_freeinfo(prov_hints);
        prov_hints = nullptr;
    }

    ofi_ctx->prov_count = 0;
    ofi_ctx->nw_prov_count = 0;
    ofi_ctx->shm_prov_idx = 0;
    ofi_ctx->nw_prov_first_idx = (enable_shm) ? 1 : 0;
    ofi_ctx->mnic_type = attr->in.mnic_type;
    ATL_CALL(atl_ofi_parse_mnic_name(ctx, attr->in.mnic_name), goto err);
    ofi_ctx->mnic_count = std::min(attr->in.mnic_count, (size_t)(ATL_OFI_MAX_NW_PROV_COUNT));
    ofi_ctx->mnic_count = std::min(ofi_ctx->mnic_count, attr->in.ep_count);
    ofi_ctx->mnic_count = std::max(ofi_ctx->mnic_count, (size_t)(1));

    if ((ofi_ctx->mnic_type != ATL_MNIC_NONE) &&
        !ccl::global_data::get().hwloc_wrapper->is_initialized()) {
        ofi_ctx->mnic_type = ATL_MNIC_NONE;
        LOG_WARN("hwloc is not initialized, disable multi-nic")
    }

    if (ofi_ctx->mnic_type == ATL_MNIC_NONE)
        ofi_ctx->mnic_count = 1;

    attr->out.tag_bits = 64;
    attr->out.max_tag = 0xFFFFFFFFFFFFFFFF;

    /* open SHM provider */
    if (enable_shm) {
        prov_idx = ofi_ctx->shm_prov_idx;
        prov_name = strdup(ATL_OFI_SHM_PROV_NAME);
        prov = &ofi_ctx->provs[prov_idx];
        prov->idx = prov_idx;
        prov->is_shm = 1;
        ATL_CALL(atl_ofi_get_prov_list(ctx, prov_name, base_hints, &prov_list), goto err);
        ATL_CALL(atl_ofi_prov_init(ctx, prov_list, prov, attr, pmi), goto err);
        free(prov_name);
        fi_freeinfo(prov_list);
        ofi_ctx->prov_count++;
    }

    /* open NW provider(s) */
    if (prov_env && !strcmp(prov_env, ATL_OFI_SHM_PROV_NAME) && enable_shm) {
        open_nw_provs = 0;
    }

    if (open_nw_provs) {
        ATL_CALL(atl_ofi_open_nw_provs(ctx, base_hints, attr, pmi), goto err);
        ofi_ctx->mnic_count = ofi_ctx->nw_prov_count;
    }

    for (ep_idx = 0; ep_idx < ctx->ep_count; ep_idx++) {
        atl_ofi_ep_t* ofi_ep;
        ofi_ep = (atl_ofi_ep_t*)calloc(1, sizeof(atl_ofi_ep_t));
        if (!ofi_ep) {
            LOG_ERROR("can't alloc ofi_ep, idx ", ep_idx);
            goto err;
        }

        atl_ep_t* ep;
        ep = &(ofi_ep->ep);
        ep->idx = ep_idx;
        ep->ctx = ctx;

        ofi_ep->active_prov_count = 0;
        if (enable_shm) {
            ofi_ep->active_prov_idxs[ofi_ep->active_prov_count] = ofi_ctx->shm_prov_idx;
            ofi_ep->active_prov_count++;
        }
        if (open_nw_provs) {
            ofi_ep->active_prov_idxs[ofi_ep->active_prov_count] =
                ofi_ctx->nw_prov_first_idx + ep_idx % ofi_ctx->nw_prov_count;
            ofi_ep->active_prov_count++;
        }
        CCL_THROW_IF_NOT(ofi_ep->active_prov_count, "no active providers for ep_idx ", ep_idx);

        if (coord->global_idx == 0) {
            std::stringstream ss;
            for (idx = 0; idx < ofi_ep->active_prov_count; idx++) {
                ss << ofi_ep->active_prov_idxs[idx] << " ";
            }
            LOG_INFO("ep_idx: ", ep_idx, ", active_prov_idxs: ", ss.str());
        }

        ctx->eps[ep_idx] = ep;
    }

    pmi->pmrt_barrier();

    max_retry_count_env = getenv(ATL_OFI_MAX_RETRY_COUNT_ENV);
    if (max_retry_count_env) {
        ofi_ctx->max_retry_count = safe_c_strtol(max_retry_count_env, nullptr, 10);
    }
    else {
        ofi_ctx->max_retry_count = ATL_OFI_MAX_RETRY_COUNT;
    }

    if ((coord->global_count == coord->local_count) && (coord->global_count <= 4)) {
        ofi_ctx->progress_mode = ATL_PROGRESS_CHECK;
    }
    else {
        ofi_ctx->progress_mode = ATL_PROGRESS_POLL;
    }

    progress_mode_env = getenv(ATL_PROGRESS_MODE_ENV);
    if (progress_mode_env) {
        ofi_ctx->progress_mode = static_cast<atl_progress_mode_t>(atoi(progress_mode_env));
    }

    if (coord->global_idx == 0) {
        LOG_INFO("atl-ofi-ctx:");
        LOG_INFO("  new ctx_count: ", global_data.ctx_count);
        LOG_INFO("  prov_count: ", ofi_ctx->prov_count);
        LOG_INFO("  nw_prov_count: ", ofi_ctx->nw_prov_count);
        LOG_INFO("  nw_prov_first_idx: ", ofi_ctx->nw_prov_first_idx);
        LOG_INFO("  mnic_type: ", ofi_ctx->mnic_type);
        LOG_INFO("  mnic_include_names: ", vec_to_string(ofi_ctx->mnic_include_names));
        LOG_INFO("  mnic_exclude_names: ", vec_to_string(ofi_ctx->mnic_exclude_names));
        LOG_INFO("  mnic_count: ", ofi_ctx->mnic_count);
        LOG_INFO("  max_retry_count: ", ofi_ctx->max_retry_count);
        LOG_INFO("  progress_mode: ", ofi_ctx->progress_mode);
#ifdef CCL_ENABLE_OFI_HMEM
        LOG_INFO("  hmem: ", ofi_ctx->enable_hmem);
#endif // CCL_ENABLE_OFI_HMEM
    }

    fi_freeinfo(base_hints);
    base_hints = nullptr;

    /* report actual attributes back to upper level */
    attr->out.enable_shm = enable_shm;
    attr->out.enable_rma = 0;
    attr->out.enable_hmem = ofi_ctx->enable_hmem;
    attr->out.mnic_type = ofi_ctx->mnic_type;
    attr->out.mnic_count = ofi_ctx->mnic_count;
    attr->out.max_order_waw_size = 0;

    return ATL_STATUS_SUCCESS;

err:
    LOG_ERROR("can't find suitable provider");

    if (prov_list) {
        fi_freeinfo(prov_list);
    }

    if (base_hints) {
        fi_freeinfo(base_hints);
    }

    if (prov_hints) {
        fi_freeinfo(prov_hints);
    }

    if (ctx != nullptr)
        atl_finalize();

    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi::atl_finalize() {
    is_finalized = true;
    int ret = 0;
    size_t idx;

    atl_ofi_ctx_t* ofi_ctx = container_of(ctx, atl_ofi_ctx_t, ctx);

    global_data.ctx_count--;
    if (ctx->coord.global_idx == 0) {
        LOG_INFO("finalize atl-ofi ctx, remaining ctx_count ", global_data.ctx_count);
    }

    cache.clear();

    for (idx = 0; idx < ofi_ctx->prov_count; idx++) {
        atl_ofi_prov_t* prov = &ofi_ctx->provs[idx];
        atl_ofi_prov_destroy(ctx, prov);
    }

    for (idx = 0; idx < ctx->ep_count; idx++) {
        atl_ofi_ep_t* ofi_ep = container_of(ctx->eps[idx], atl_ofi_ep_t, ep);
        free(ofi_ep);
    }

    if (global_data.ctx_count == 0) {
        if (global_data.dlhandle) {
            dlclose(global_data.dlhandle);
        }

        if (ctx->coord.global_idx == 0) {
            LOG_INFO("finalized last atl-ofi ctx");
        }
    }

    free(ctx->eps);
    free(ofi_ctx);

    return RET2ATL(ret);
}

atl_status_t atl_ofi::atl_update(std::unique_ptr<ipmi>& pmi) {
    int ret;
    size_t prov_idx;

    atl_ofi_ctx_t* ofi_ctx;
    ofi_ctx = container_of(ctx, atl_ofi_ctx_t, ctx);

    pmi->pmrt_barrier();

    atl_ofi_reset(ctx);
    memset(&(ctx->coord), 0, sizeof(atl_proc_coord_t));

    ret = pmi->pmrt_update();
    if (ret)
        return RET2ATL(ret);

    ctx->coord.global_count = pmi->get_size();
    ctx->coord.global_idx = pmi->get_rank();

    ret = atl_ofi_get_local_proc_coord(ofi_ctx, pmi);
    if (ret)
        return RET2ATL(ret);

    atl_proc_coord_t* coord;
    coord = &(ctx->coord);

    if (ofi_ctx->prov_count == 1 && ofi_ctx->provs[0].is_shm) {
        CCL_THROW_IF_NOT(coord->global_count == coord->local_count,
                         "unexpected coord after update: global_count ",
                         coord->global_count,
                         ", local_count ",
                         coord->local_count);
        /* TODO: recreate providers */
    }
    atl_ofi_print_coord(coord);

    for (prov_idx = 0; prov_idx < ofi_ctx->prov_count; prov_idx++) {
        ret = atl_ofi_prov_eps_connect(ofi_ctx, prov_idx, pmi);
        if (ret)
            return RET2ATL(ret);
    }

    pmi->pmrt_barrier();

    /* normal end of execution */
    return RET2ATL(ret);
}

atl_ep_t** atl_ofi::atl_get_eps() {
    return ctx->eps;
}

atl_proc_coord_t* atl_ofi::atl_get_proc_coord() {
    return &(ctx->coord);
}

atl_status_t atl_ofi::atl_mr_reg(const void* buf, size_t len, atl_mr_t** mr) {
    int ret;
    atl_ofi_ctx_t* ofi_ctx;
    ofi_ctx = container_of(ctx, atl_ofi_ctx_t, ctx);
    atl_ofi_prov_t* prov = &(ofi_ctx->provs[0]);

    atl_ofi_mr_t* ofi_mr;
    ofi_mr = (atl_ofi_mr_t*)calloc(1, sizeof(atl_ofi_mr_t));
    if (!ofi_mr)
        return ATL_STATUS_FAILURE;

    ret = fi_mr_reg(prov->domain,
                    buf,
                    len,
                    FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
                    0,
                    0,
                    0,
                    &ofi_mr->fi_mr,
                    nullptr);
    if (ret)
        goto mr_reg_err;

    ofi_mr->mr.buf = (void*)buf;
    ofi_mr->mr.len = len;
    ofi_mr->mr.remote_key = (uintptr_t)fi_mr_key(ofi_mr->fi_mr);
    ofi_mr->mr.local_key = (uintptr_t)fi_mr_desc(ofi_mr->fi_mr);

    *mr = &ofi_mr->mr;
    return ATL_STATUS_SUCCESS;

mr_reg_err:
    free(ofi_mr);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi::atl_mr_dereg(atl_mr_t* mr) {
    atl_ofi_mr_t* ofi_mr;
    ofi_mr = container_of(mr, atl_ofi_mr_t, mr);
    int ret = fi_close(&ofi_mr->fi_mr->fid);
    free(ofi_mr);
    return RET2ATL(ret);
}

atl_status_t atl_ofi::atl_ep_send(atl_ep_t* ep,
                                  const void* buf,
                                  size_t len,
                                  int dst_proc_idx,
                                  uint64_t tag,
                                  atl_req_t* req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ep, dst_proc_idx, len);
    prov_ep = &(prov->eps[ep->idx]);
    ofi_req = ((atl_ofi_req_t*)req->internal);

    req->tag = tag;
    req->remote_proc_idx = dst_proc_idx;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->prov_ep = prov_ep;
    ofi_req->fi_ep = prov_ep->tx;

    cache.get(ep->idx, prov->domain, const_cast<void*>(buf), len, &ofi_req->mr);
    void* desc = (ofi_req->mr) ? fi_mr_desc(ofi_req->mr) : nullptr;

    struct iovec iov;
    iov.iov_base = const_cast<void*>(buf);
    iov.iov_len = len;

    struct fi_msg_tagged msg;
    msg.desc = &desc;
    msg.msg_iov = &iov;
    msg.iov_count = 1;
    msg.tag = tag;
    msg.ignore = 0;
    msg.addr = atl_ofi_get_addr(ep->ctx, prov, dst_proc_idx, ep->idx);
    msg.context = &ofi_req->fi_ctx;
    msg.data = 0;

    ATL_OFI_RETRY(fi_tsendmsg(prov_ep->tx, &msg, 0), ep, ret);

    return RET2ATL(ret);
}

atl_status_t atl_ofi::atl_ep_recv(atl_ep_t* ep,
                                  void* buf,
                                  size_t len,
                                  int src_proc_idx,
                                  uint64_t tag,
                                  atl_req_t* req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ep, src_proc_idx, len);
    prov_ep = &(prov->eps[ep->idx]);
    ofi_req = ((atl_ofi_req_t*)req->internal);

    req->tag = tag;
    req->remote_proc_idx = src_proc_idx;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->prov_ep = prov_ep;
    ofi_req->fi_ep = prov_ep->rx;

    cache.get(ep->idx, prov->domain, const_cast<void*>(buf), len, &ofi_req->mr);
    void* desc = (ofi_req->mr) ? fi_mr_desc(ofi_req->mr) : nullptr;

    struct iovec iov;
    iov.iov_base = buf;
    iov.iov_len = len;

    struct fi_msg_tagged msg;
    msg.desc = &desc;
    msg.msg_iov = &iov;
    msg.iov_count = 1;
    msg.tag = tag;
    msg.ignore = 0;
    msg.addr = atl_ofi_get_addr(ep->ctx, prov, src_proc_idx, ep->idx);
    msg.context = &ofi_req->fi_ctx;
    msg.data = 0;

    ATL_OFI_RETRY(fi_trecvmsg(prov_ep->rx, &msg, 0), ep, ret);

    return RET2ATL(ret);
}

atl_status_t atl_ofi::atl_ep_probe(atl_ep_t* ep,
                                   int src_proc_idx,
                                   uint64_t tag,
                                   int* found,
                                   size_t* recv_len) {
    CCL_THROW("unexpected path");

    atl_status_t ret;
    atl_ofi_req_t reqs[ATL_OFI_MAX_PROV_COUNT];
    struct fi_msg_tagged msgs[ATL_OFI_MAX_PROV_COUNT];
    int flag, len;
    ssize_t ofi_ret;
    size_t idx;
    int do_poll;

    atl_ofi_ctx_t* ofi_ctx;

    ret = ATL_STATUS_SUCCESS;
    flag = 0;
    len = 0;
    ofi_ret = FI_SUCCESS;
    do_poll = 1;

    ofi_ctx = container_of(ep->ctx, atl_ofi_ctx_t, ctx);

    for (idx = 0; idx < ofi_ctx->prov_count; idx++) {
        atl_ofi_prov_t* prov;
        atl_ofi_prov_ep_t* prov_ep;
        atl_ofi_req_t* req;
        struct fi_msg_tagged* msg;

        prov = &(ofi_ctx->provs[idx]);
        prov_ep = &(prov->eps[ep->idx]);
        req = &(reqs[idx]);
        msg = &(msgs[idx]);

        if (prov->is_shm &&
            ((src_proc_idx < prov->first_proc_idx) ||
             (src_proc_idx >= (prov->first_proc_idx + ep->ctx->coord.local_count)))) {
            req->prov_ep = nullptr;
            continue;
        }

        req->comp_state = ATL_OFI_COMP_PEEK_STARTED;
        req->prov_ep = prov_ep;
        req->fi_ep = prov_ep->rx;

        msg->msg_iov = nullptr;
        msg->desc = nullptr;
        msg->iov_count = 0;
        msg->addr = atl_ofi_get_addr(ep->ctx, prov, src_proc_idx, ep->idx);
        msg->tag = tag;
        msg->ignore = 0;
        msg->context = &(req->fi_ctx);
        msg->data = 0;

        ATL_OFI_RETRY(fi_trecvmsg(prov_ep->rx, msg, FI_PEEK | FI_COMPLETION), ep, ofi_ret);
    }

    do {
        ret = atl_ep_poll(ep);
        if (ret != ATL_STATUS_SUCCESS)
            return ret;

        for (idx = 0; idx < ofi_ctx->prov_count; idx++) {
            atl_ofi_req_t* req;
            req = &(reqs[idx]);

            if (!req->prov_ep)
                continue;

            if (req->comp_state != ATL_OFI_COMP_PEEK_STARTED) {
                do_poll = 0;

                if (req->comp_state == ATL_OFI_COMP_PEEK_FOUND) {
                    flag = 1;
                    len = req->recv_len;
                    req->prov_ep = nullptr;
                }
                else if (req->comp_state == ATL_OFI_COMP_PEEK_NOT_FOUND) {
                    req->prov_ep = nullptr;
                }
                else {
                    CCL_THROW("unexpected completion state ", req->comp_state);
                }

                break;
            }
        }
    } while (do_poll);

    for (idx = 0; idx < ofi_ctx->prov_count; idx++) {
        atl_ofi_req_t* req;
        req = &(reqs[idx]);

        if (!req->prov_ep)
            continue;

        if (fi_cancel(&req->fi_ep->fid, &req->fi_ctx) == 0) {
            atl_ofi_wait_cancel_cq(req->prov_ep->cq);
        }
    }

    if (found)
        *found = flag;
    if (recv_len)
        *recv_len = len;

    return RET2ATL(ofi_ret);
}

atl_status_t atl_ofi::atl_ep_allgatherv(atl_ep_t* ep,
                                        const void* send_buf,
                                        size_t send_len,
                                        void* recv_buf,
                                        const int* recv_lens,
                                        const int* offsets,
                                        atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_allreduce(atl_ep_t* ep,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t len,
                                       atl_datatype_t dtype,
                                       atl_reduction_t op,
                                       atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_alltoall(atl_ep_t* ep,
                                      const void* send_buf,
                                      void* recv_buf,
                                      int len,
                                      atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_alltoallv(atl_ep_t* ep,
                                       const void* send_buf,
                                       const int* send_lens,
                                       const int* send_offsets,
                                       void* recv_buf,
                                       const int* recv_lens,
                                       const int* recv_offsets,
                                       atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_barrier(atl_ep_t* ep, atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_bcast(atl_ep_t* ep, void* buf, size_t len, int root, atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_reduce(atl_ep_t* ep,
                                    const void* send_buf,
                                    void* recv_buf,
                                    size_t len,
                                    int root,
                                    atl_datatype_t dtype,
                                    atl_reduction_t op,
                                    atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_reduce_scatter(atl_ep_t* ep,
                                            const void* send_buf,
                                            void* recv_buf,
                                            size_t recv_len,
                                            atl_datatype_t dtype,
                                            atl_reduction_t op,
                                            atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_ofi::atl_ep_read(atl_ep_t* ep,
                                  void* buf,
                                  size_t len,
                                  atl_mr_t* mr,
                                  uint64_t addr,
                                  uintptr_t remote_key,
                                  int dst_proc_idx,
                                  atl_req_t* req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ep, dst_proc_idx, len);
    prov_ep = &(prov->eps[ep->idx]);
    ofi_req = ((atl_ofi_req_t*)req->internal);

    req->tag = 0;
    req->remote_proc_idx = dst_proc_idx;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->prov_ep = prov_ep;
    ofi_req->fi_ep = prov_ep->tx;

    ATL_OFI_RETRY(fi_read(prov_ep->tx,
                          buf,
                          len,
                          (void*)mr->local_key,
                          atl_ofi_get_addr(ep->ctx, prov, dst_proc_idx, ep->idx),
                          addr,
                          remote_key,
                          &ofi_req->fi_ctx),
                  ep,
                  ret);
    return RET2ATL(ret);
}

atl_status_t atl_ofi::atl_ep_write(atl_ep_t* ep,
                                   const void* buf,
                                   size_t len,
                                   atl_mr_t* mr,
                                   uint64_t addr,
                                   uintptr_t remote_key,
                                   int dst_proc_idx,
                                   atl_req_t* req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ep, dst_proc_idx, len);
    prov_ep = &(prov->eps[ep->idx]);
    ofi_req = ((atl_ofi_req_t*)req->internal);

    req->tag = 0;
    req->remote_proc_idx = dst_proc_idx;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;

    ofi_req->prov_ep = prov_ep;
    ofi_req->fi_ep = prov_ep->tx;

    ATL_OFI_RETRY(fi_write(prov_ep->tx,
                           buf,
                           len,
                           (void*)mr->local_key,
                           atl_ofi_get_addr(ep->ctx, prov, dst_proc_idx, ep->idx),
                           addr,
                           remote_key,
                           &ofi_req->fi_ctx),
                  ep,
                  ret);
    return RET2ATL(ret);
}

atl_status_t atl_ofi::atl_ep_wait(atl_ep_t* ep, atl_req_t* req) {
    atl_status_t ret;
    atl_ofi_req_t* ofi_req;

    ret = ATL_STATUS_SUCCESS;
    ofi_req = ((atl_ofi_req_t*)req->internal);

    while ((ofi_req->comp_state != ATL_OFI_COMP_COMPLETED) &&
           ((ret = atl_ep_poll(ep)) == ATL_STATUS_SUCCESS))
        ;

    return ret;
}

atl_status_t atl_ofi::atl_ep_wait_all(atl_ep_t* ep, atl_req_t* reqs, size_t count) {
    size_t i;
    atl_status_t ret;

    for (i = 0; i < count; i++) {
        ret = atl_ep_wait(ep, &reqs[i]);
        if (ret != ATL_STATUS_SUCCESS)
            return ret;
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::atl_ep_cancel(atl_ep_t* ep, atl_req_t* req) {
    int ret;
    atl_ofi_req_t* ofi_req;

    ret = ATL_STATUS_SUCCESS;
    ofi_req = ((atl_ofi_req_t*)req->internal);

    ret = fi_cancel(&ofi_req->fi_ep->fid, &ofi_req->fi_ctx);
    if (ret == 0) {
        return RET2ATL(atl_ofi_wait_cancel_cq(ofi_req->prov_ep->cq));
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::atl_ep_poll(atl_ep_t* ep) {
    atl_ofi_ctx_t* ofi_ctx = container_of(ep->ctx, atl_ofi_ctx_t, ctx);
    if (ofi_ctx->progress_mode == ATL_PROGRESS_POLL) {
        atl_ep_progress(ep);
    }
    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::atl_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) {
    CCL_THROW_IF_NOT(is_completed);

    atl_status_t status;
    atl_ofi_req_t* ofi_req;
    atl_ofi_ctx_t* ofi_ctx = container_of(ep->ctx, atl_ofi_ctx_t, ctx);

    status = ATL_STATUS_SUCCESS;
    ofi_req = ((atl_ofi_req_t*)req->internal);

    *is_completed = (ofi_req->comp_state == ATL_OFI_COMP_COMPLETED);
    if (*is_completed) {
        return ATL_STATUS_SUCCESS;
    }

    if (ofi_ctx->progress_mode == ATL_PROGRESS_CHECK) {
        status = atl_ep_progress(ep);
        *is_completed = (ofi_req->comp_state == ATL_OFI_COMP_COMPLETED);
    }

    return status;
}

atl_ofi::~atl_ofi() {
    if (!is_finalized) {
        atl_finalize();
    }
}

atl_status_t atl_ofi::atl_ep_progress(atl_ep_t* ep) {
    ssize_t ret;
    size_t idx;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];
    atl_ofi_ep_t* ofi_ep = container_of(ep, atl_ofi_ep_t, ep);
    atl_ofi_ctx_t* ofi_ctx = container_of(ep->ctx, atl_ofi_ctx_t, ctx);
    size_t ep_idx = ep->idx;

    /* ensure progress for all active providers */
    for (idx = 0; idx < ofi_ep->active_prov_count; idx++) {
        atl_ofi_prov_ep_t* prov_ep;
        prov_ep = &(ofi_ctx->provs[ofi_ep->active_prov_idxs[idx]].eps[ep_idx]);
        do {
            ret = fi_cq_read(prov_ep->cq, entries, ATL_OFI_CQ_BUNCH_SIZE);
            if (ret > 0)
                atl_process_comps(ep, entries, ret);
            else if (ret == -FI_EAGAIN)
                break;
            else
                return atl_prov_ep_handle_cq_err(prov_ep);
        } while (ret > 0);
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::atl_prov_ep_handle_cq_err(atl_ofi_prov_ep_t* ep) {
    struct fi_cq_err_entry err_entry;
    atl_ofi_req_t* ofi_req;

    int ret = fi_cq_readerr(ep->cq, &err_entry, 0);
    if (ret != 1) {
        CCL_THROW("unable to read error from cq");
        return ATL_STATUS_FAILURE;
    }
    else {
        ofi_req = container_of(err_entry.op_context, atl_ofi_req_t, fi_ctx);

        if (err_entry.err == FI_ECANCELED) {
            return ATL_STATUS_SUCCESS;
        }

        if (err_entry.err == FI_ENOMSG && ofi_req->comp_state == ATL_OFI_COMP_PEEK_STARTED) {
            ofi_req->comp_state = ATL_OFI_COMP_PEEK_NOT_FOUND;
        }
        else {
            LOG_ERROR("fi_cq_readerr: err: ",
                      err_entry.err,
                      ", prov_err: ",
                      fi_cq_strerror(ep->cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0),
                      "(",
                      err_entry.prov_errno,
                      ")");
            return ATL_STATUS_FAILURE;
        }
        return ATL_STATUS_SUCCESS;
    }
}

void atl_ofi::atl_process_comps(atl_ep_t* ep, struct fi_cq_tagged_entry* entries, ssize_t ret) {
    ssize_t idx;
    atl_ofi_req_t* comp_ofi_req;
    for (idx = 0; idx < ret; idx++) {
        comp_ofi_req = container_of(entries[idx].op_context, atl_ofi_req_t, fi_ctx);
        switch (comp_ofi_req->comp_state) {
            case ATL_OFI_COMP_POSTED:
                comp_ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;
                cache.push(ep->idx, comp_ofi_req->mr);
                break;
            case ATL_OFI_COMP_COMPLETED: break;
            case ATL_OFI_COMP_PEEK_STARTED:
                comp_ofi_req->comp_state = ATL_OFI_COMP_PEEK_FOUND;
                break;
            default: CCL_THROW("unexpected completion state ", comp_ofi_req->comp_state); break;
        }

        if (entries[idx].flags & FI_RECV) {
            comp_ofi_req->recv_len = entries[idx].len;
        }
    }
}
