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

// atl_ofi

atl_ofi::~atl_ofi() {
    if (!is_finalized) {
        finalize();
    }
}

atl_status_t atl_ofi::init(int* argc,
                           char*** argv,
                           atl_attr_t* attr,
                           const char* main_addr,
                           std::shared_ptr<ipmi> pmi) {
    CCL_THROW_IF_NOT(!inited, "atl_ofi reinit is not expected");
    inited = true;

    struct fi_info* base_hints = nullptr;
    int fi_version;
    ssize_t ret = 0;
    size_t idx = 0, ep_idx = 0;
    char* prov_env = nullptr;
    char* fi_version_env = nullptr;
    char *max_retry_count_env = nullptr, *progress_mode_env = nullptr;
    int open_nw_provs = 1;
    int enable_shm = 0;
    bool should_open_provs = true;

    CCL_THROW_IF_NOT((sizeof(atl_ofi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal)),
                     "unexpected offset: atl_ofi_request size ",
                     sizeof(atl_ofi_req_t),
                     ", atl_request size ",
                     sizeof(atl_req_t),
                     ", expected offset ",
                     offsetof(atl_req_t, internal));

    ret = atl_ofi_set_env(*attr);
    ATL_CHECK_STATUS(ret, "atl_ofi_set_env error");

    fi_version_env = getenv(ATL_OFI_MAJOR_VERSION);
    if (fi_version_env) {
        global_data.fi_major_version = safe_c_strtol(fi_version_env, nullptr, 10);
    }

    fi_version_env = getenv(ATL_OFI_MINOR_VERSION);
    if (fi_version_env) {
        global_data.fi_minor_version = safe_c_strtol(fi_version_env, nullptr, 10);
    }

    LOG_INFO("fi_version: ", global_data.fi_major_version, ".", global_data.fi_minor_version);

    ctx.ep_count = attr->in.ep_count;

    coord.global_count = pmi->get_size();
    coord.global_idx = pmi->get_rank();

    ret = atl_ofi_get_local_proc_coord(coord, pmi);
    if (ret) {
        LOG_ERROR("atl_ofi_get_local_proc_coord error");
        goto err;
    }

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

    ctx.enable_hmem = 0;

    fi_version = FI_VERSION(global_data.fi_major_version, global_data.fi_minor_version);

    if (coord.global_idx == 0)
        LOG_INFO("libfabric version: ", fi_tostr("1" /* ignored */, FI_TYPE_VERSION));

    LOG_INFO(::to_string(coord));
    coord.validate();

    ctx.prov_count = 0;
    ctx.nw_prov_count = 0;
    ctx.shm_prov_idx = 0;
    ctx.mnic_type = attr->in.mnic_type;
    ATL_CALL(atl_ofi_parse_mnic_name(ctx, attr->in.mnic_name), goto err);
    ctx.mnic_count = std::min(attr->in.mnic_count, (size_t)(ATL_OFI_MAX_NW_PROV_COUNT));
    ctx.mnic_count = std::max(ctx.mnic_count, (size_t)(1));

    if ((ctx.mnic_type != ATL_MNIC_NONE) &&
        !ccl::global_data::get().hwloc_wrapper->is_initialized()) {
        ctx.mnic_type = ATL_MNIC_NONE;
        LOG_WARN("hwloc is not initialized, disable multi-nic")
    }

    if (ctx.mnic_type == ATL_MNIC_NONE)
        ctx.mnic_count = 1;

    ctx.mnic_offset = attr->in.mnic_offset;

    attr->out.tag_bits = 64;
    attr->out.max_tag = 0xFFFFFFFFFFFFFFFF;

#ifdef CCL_ENABLE_OFI_HMEM
    if (prov_env && strstr(prov_env, "verbs") && attr->in.enable_hmem) {
        struct fi_info* hmem_hints = fi_dupinfo(base_hints);
        atl_attr_t hmem_attr = *attr;

        hmem_hints->caps |= FI_HMEM;
        hmem_hints->domain_attr->mr_mode =
            (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR | FI_MR_LOCAL | FI_MR_HMEM);

        /* TODO: enable shm with HMEM */
        hmem_attr.in.enable_shm = 0;
        if (open_providers(prov_env,
                           coord,
                           &hmem_attr,
                           hmem_hints,
                           open_nw_provs,
                           fi_version,
                           pmi,
                           false /* log_on_error */) == ATL_STATUS_SUCCESS) {
            ctx.enable_hmem = 1;
            should_open_provs = false;
        }
        else {
            LOG_WARN("can not open network providers with hmem support");
        }
        fi_freeinfo(hmem_hints);
    }
#endif // CCL_ENABLE_OFI_HMEM
    if (should_open_provs) {
        ATL_CALL(open_providers(prov_env,
                                coord,
                                attr,
                                base_hints,
                                open_nw_provs,
                                fi_version,
                                pmi,
                                true /* log_on_error */),
                 goto err);
    }

    cache.init(attr->in.ep_count, ctx.enable_hmem);

    for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
        atl_ep_t ep;

        atl_ofi_ep_t* ofi_ep = ((atl_ofi_ep_t*)ep.internal);

        ofi_ep->active_prov_count = 0;
        if (enable_shm) {
            ofi_ep->active_prov_idxs[ofi_ep->active_prov_count] = ctx.shm_prov_idx;
            ofi_ep->active_prov_count++;
        }
        if (open_nw_provs) {
            ofi_ep->active_prov_idxs[ofi_ep->active_prov_count] =
                ctx.nw_prov_first_idx + ep_idx % ctx.nw_prov_count;
            ofi_ep->active_prov_count++;
        }
        CCL_THROW_IF_NOT(ofi_ep->active_prov_count, "no active providers for ep_idx ", ep_idx);

        if (coord.global_idx == 0) {
            std::stringstream ss;
            for (idx = 0; idx < ofi_ep->active_prov_count; idx++) {
                ss << ofi_ep->active_prov_idxs[idx] << " ";
            }
            LOG_INFO("ep_idx: ", ep_idx, ", active_prov_idxs: ", ss.str());
        }

        ep.idx = ep_idx;
        eps.push_back(ep);
    }

    ATL_CHECK_STATUS(pmi->pmrt_barrier(), "barrier failed");

    max_retry_count_env = getenv(ATL_OFI_MAX_RETRY_COUNT_ENV);
    if (max_retry_count_env) {
        ctx.max_retry_count = safe_c_strtol(max_retry_count_env, nullptr, 10);
    }
    else {
        ctx.max_retry_count = ATL_OFI_MAX_RETRY_COUNT;
    }

    if ((coord.global_count == coord.local_count) && (coord.global_count <= 4)) {
        ctx.progress_mode = ATL_PROGRESS_CHECK;
    }
    else {
        ctx.progress_mode = ATL_PROGRESS_POLL;
    }

    progress_mode_env = getenv(ATL_PROGRESS_MODE_ENV);
    if (progress_mode_env) {
        ctx.progress_mode = static_cast<atl_progress_mode_t>(atoi(progress_mode_env));
    }

    fi_freeinfo(base_hints);
    base_hints = nullptr;

    /* report actual attributes back to upper level */
    attr->out.enable_shm = enable_shm;
    attr->out.enable_rma = 0;
    attr->out.enable_hmem = ctx.enable_hmem;
    attr->out.mnic_type = ctx.mnic_type;
    attr->out.mnic_count = ctx.mnic_count;
    attr->out.max_order_waw_size = 0;

    return ATL_STATUS_SUCCESS;

err:
    LOG_ERROR("can't find suitable provider");

    if (base_hints) {
        fi_freeinfo(base_hints);
    }

    finalize();

    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi::update(std::shared_ptr<ipmi> pmi) {
    int ret;
    size_t prov_idx;

    ATL_CHECK_STATUS(pmi->pmrt_barrier(), "barrier failed");

    atl_ofi_reset(ctx);

    coord.reset();

    ret = pmi->pmrt_update();
    if (ret)
        return ATL_OFI_RET(ret);

    coord.global_count = pmi->get_size();
    coord.global_idx = pmi->get_rank();

    ret = atl_ofi_get_local_proc_coord(coord, pmi);
    if (ret)
        return ATL_OFI_RET(ret);

    if (ctx.prov_count == 1 && ctx.provs[0].is_shm) {
        CCL_THROW_IF_NOT(coord.global_count == coord.local_count,
                         "unexpected coord after update: global_count ",
                         coord.global_count,
                         ", local_count ",
                         coord.local_count);
        /* TODO: recreate providers */
    }

    LOG_INFO(::to_string(coord));
    coord.validate();

    for (prov_idx = 0; prov_idx < ctx.prov_count; prov_idx++) {
        ret = atl_ofi_prov_eps_connect(ctx, coord, prov_idx, pmi, ep_names);
        if (ret)
            return ATL_OFI_RET(ret);
    }

    ATL_CHECK_STATUS(pmi->pmrt_barrier(), "barrier failed");

    /* normal end of execution */
    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi::mr_reg(const void* buf, size_t len, atl_mr_t** mr) {
    int ret;
    atl_ofi_prov_t* prov = &(ctx.provs[0]);

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

atl_status_t atl_ofi::mr_dereg(atl_mr_t* mr) {
    atl_ofi_mr_t* ofi_mr;
    ofi_mr = container_of(mr, atl_ofi_mr_t, mr);
    int ret = fi_close(&ofi_mr->fi_mr->fid);
    free(ofi_mr);
    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi::send(atl_ep_t& ep,
                           const void* buf,
                           size_t len,
                           int dst_proc_idx,
                           uint64_t tag,
                           atl_req_t& req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ctx, coord, ep, dst_proc_idx, len);
    prov_ep = &(prov->eps[ep.idx]);

    atl_ofi_init_req(req, prov_ep, prov_ep->tx);

    ofi_req = ((atl_ofi_req_t*)req.internal);

    cache.get(ep.idx, prov->domain, const_cast<void*>(buf), len, &ofi_req->mr);
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
    msg.addr = atl_ofi_get_addr(ctx, prov, dst_proc_idx, ep.idx);
    msg.context = &ofi_req->fi_ctx;
    msg.data = 0;

    ATL_OFI_RETRY(fi_tsendmsg(prov_ep->tx, &msg, 0), ep, ret);

    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi::recv(atl_ep_t& ep,
                           void* buf,
                           size_t len,
                           int src_proc_idx,
                           uint64_t tag,
                           atl_req_t& req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ctx, coord, ep, src_proc_idx, len);
    prov_ep = &(prov->eps[ep.idx]);

    atl_ofi_init_req(req, prov_ep, prov_ep->rx);

    ofi_req = ((atl_ofi_req_t*)req.internal);

    cache.get(ep.idx, prov->domain, const_cast<void*>(buf), len, &ofi_req->mr);
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
    msg.addr = atl_ofi_get_addr(ctx, prov, src_proc_idx, ep.idx);
    msg.context = &ofi_req->fi_ctx;
    msg.data = 0;

    ATL_OFI_RETRY(fi_trecvmsg(prov_ep->rx, &msg, 0), ep, ret);

    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi::probe(atl_ep_t& ep,
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

    ret = ATL_STATUS_SUCCESS;
    flag = 0;
    len = 0;
    ofi_ret = FI_SUCCESS;
    do_poll = 1;

    for (idx = 0; idx < ctx.prov_count; idx++) {
        atl_ofi_prov_t* prov;
        atl_ofi_prov_ep_t* prov_ep;
        atl_ofi_req_t* req;
        struct fi_msg_tagged* msg;

        prov = &(ctx.provs[idx]);
        prov_ep = &(prov->eps[ep.idx]);
        req = &(reqs[idx]);
        msg = &(msgs[idx]);

        if (prov->is_shm && ((src_proc_idx < prov->first_proc_idx) ||
                             (src_proc_idx >= (prov->first_proc_idx + coord.local_count)))) {
            req->prov_ep = nullptr;
            continue;
        }

        req->comp_state = ATL_OFI_COMP_PEEK_STARTED;
        req->prov_ep = prov_ep;
        req->fi_ep = prov_ep->rx;

        msg->msg_iov = nullptr;
        msg->desc = nullptr;
        msg->iov_count = 0;
        msg->addr = atl_ofi_get_addr(ctx, prov, src_proc_idx, ep.idx);
        msg->tag = tag;
        msg->ignore = 0;
        msg->context = &(req->fi_ctx);
        msg->data = 0;

        ATL_OFI_RETRY(fi_trecvmsg(prov_ep->rx, msg, FI_PEEK | FI_COMPLETION), ep, ofi_ret);
    }

    do {
        ret = poll(ep);
        if (ret != ATL_STATUS_SUCCESS)
            return ret;

        for (idx = 0; idx < ctx.prov_count; idx++) {
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

    for (idx = 0; idx < ctx.prov_count; idx++) {
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

    return ATL_OFI_RET(ofi_ret);
}

atl_status_t atl_ofi::read(atl_ep_t& ep,
                           void* buf,
                           size_t len,
                           atl_mr_t* mr,
                           uint64_t addr,
                           uintptr_t remote_key,
                           int dst_proc_idx,
                           atl_req_t& req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ctx, coord, ep, dst_proc_idx, len);
    prov_ep = &(prov->eps[ep.idx]);

    atl_ofi_init_req(req, prov_ep, prov_ep->tx);

    ofi_req = ((atl_ofi_req_t*)req.internal);

    ATL_OFI_RETRY(fi_read(prov_ep->tx,
                          buf,
                          len,
                          (void*)mr->local_key,
                          atl_ofi_get_addr(ctx, prov, dst_proc_idx, ep.idx),
                          addr,
                          remote_key,
                          &ofi_req->fi_ctx),
                  ep,
                  ret);
    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi::write(atl_ep_t& ep,
                            const void* buf,
                            size_t len,
                            atl_mr_t* mr,
                            uint64_t addr,
                            uintptr_t remote_key,
                            int dst_proc_idx,
                            atl_req_t& req) {
    ssize_t ret;

    atl_ofi_prov_t* prov;
    atl_ofi_prov_ep_t* prov_ep;
    atl_ofi_req_t* ofi_req;

    prov = atl_ofi_get_prov(ctx, coord, ep, dst_proc_idx, len);
    prov_ep = &(prov->eps[ep.idx]);

    atl_ofi_init_req(req, prov_ep, prov_ep->tx);

    ofi_req = ((atl_ofi_req_t*)req.internal);

    ATL_OFI_RETRY(fi_write(prov_ep->tx,
                           buf,
                           len,
                           (void*)mr->local_key,
                           atl_ofi_get_addr(ctx, prov, dst_proc_idx, ep.idx),
                           addr,
                           remote_key,
                           &ofi_req->fi_ctx),
                  ep,
                  ret);
    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi::wait(atl_ep_t& ep, atl_req_t& req) {
    atl_status_t ret;
    atl_ofi_req_t* ofi_req;

    ret = ATL_STATUS_SUCCESS;
    ofi_req = ((atl_ofi_req_t*)req.internal);

    while ((ofi_req->comp_state != ATL_OFI_COMP_COMPLETED) &&
           ((ret = poll(ep)) == ATL_STATUS_SUCCESS)) {
    }

    req.is_completed = 1;

    return ret;
}

atl_status_t atl_ofi::wait_all(atl_ep_t& ep, std::vector<atl_req_t>& reqs, size_t count) {
    size_t i;
    atl_status_t ret;

    for (i = 0; i < count; i++) {
        ret = wait(ep, reqs[i]);
        if (ret != ATL_STATUS_SUCCESS)
            return ret;
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::cancel(atl_ep_t& ep, atl_req_t& req) {
    int ret;
    atl_ofi_req_t* ofi_req;

    ret = ATL_STATUS_SUCCESS;
    ofi_req = ((atl_ofi_req_t*)req.internal);

    ret = fi_cancel(&ofi_req->fi_ep->fid, &ofi_req->fi_ctx);
    if (ret == 0) {
        return ATL_OFI_RET(atl_ofi_wait_cancel_cq(ofi_req->prov_ep->cq));
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::poll(atl_ep_t& ep) {
    if (ctx.progress_mode == ATL_PROGRESS_POLL) {
        progress_ep(ep);
    }
    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi::check(atl_ep_t& ep, atl_req_t& req) {
    atl_status_t status;
    atl_ofi_req_t* ofi_req;

    status = ATL_STATUS_SUCCESS;
    ofi_req = ((atl_ofi_req_t*)req.internal);

    CCL_THROW_IF_NOT(!req.is_completed, "request is already completed");

    req.is_completed = (ofi_req->comp_state == ATL_OFI_COMP_COMPLETED);
    if (req.is_completed) {
        return ATL_STATUS_SUCCESS;
    }

    if (ctx.progress_mode == ATL_PROGRESS_CHECK) {
        status = progress_ep(ep);
        req.is_completed = (ofi_req->comp_state == ATL_OFI_COMP_COMPLETED);
    }

    return status;
}

atl_status_t atl_ofi::get_rank2rank_map(std::shared_ptr<ipmi> pmi,
                                        std::vector<int>& rank2rank_map) {
    int ret;
    //TODO: add support for multi-provs
    atl_ofi_prov_t* prov =
        atl_ofi_get_prov(ctx, coord, eps[0], 0 /* peer_proc_idx */, 0 /* msg_size */);

    std::vector<char> addr_name(prov->addr_len, '\0');

    size_t local_rank = pmi->get_rank();
    size_t local_size = pmi->get_size();
    // TODO: uncomment after AV insert update
    //    for (size_t ep_idx = 0; ep_idx < eps.size(); ep_idx++) {
    ret = pmi->pmrt_kvs_put(
        (char*)ATL_OFI_FI_ADDR_UPDATE_PM_KEY,
        local_rank * ATL_OFI_PMI_PROC_MULTIPLIER, // +
        //                                    prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + ep_idx,
        prov->eps[0].name.addr,
        //                            addr_name.data(),
        prov->addr_len);
    if (ret) {
        LOG_ERROR("pmrt_kvs_put: ret: ", ret);
        return ATL_STATUS_FAILURE;
    }
    //    }
    pmi->pmrt_barrier();
    for (size_t i = 0; i < local_size; ++i) {
        ret = pmi->pmrt_kvs_get(
            (char*)ATL_OFI_FI_ADDR_UPDATE_PM_KEY,
            i * ATL_OFI_PMI_PROC_MULTIPLIER, // +
            //                                    prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + ep_idx,
            (void*)addr_name.data(),
            prov->addr_len);
        if (ret) {
            LOG_ERROR("pmrt_kvs_get: ret: ", ret);
            return ATL_STATUS_FAILURE;
        }

        auto it = std::find(ep_names.begin(), ep_names.end(), addr_name);
        if (it == ep_names.end()) {
            LOG_ERROR("not found addr_name: ", i);
            return ATL_STATUS_FAILURE;
        }

        size_t named_ep_count = (prov->sep ? 1 : ctx.ep_count);
        rank2rank_map[i] = std::distance(ep_names.begin(), it) / named_ep_count;
    }

    LOG_DEBUG("transport: rank2rank_map:", ccl::utils::vec_to_string(rank2rank_map));

    return ATL_STATUS_SUCCESS;
}

std::string atl_ofi::to_string() {
    std::stringstream ss;
    ss << "atl-ofi:\n{\n"
       << "  prov_count: " << ctx.prov_count << "\n"
       << "  nw_prov_count: " << ctx.nw_prov_count << "\n"
       << "  nw_prov_first_idx: " << ctx.nw_prov_first_idx << "\n"
       << "  mnic_type: " << ::to_string(ctx.mnic_type) << "\n"
       << "  mnic_include_names: " << ccl::utils::vec_to_string(ctx.mnic_include_names) << "\n"
       << "  mnic_exclude_names: " << ccl::utils::vec_to_string(ctx.mnic_exclude_names) << "\n"
       << "  mnic_count: " << ctx.mnic_count << "\n"
       << "  mnic_offset: " << ::to_string(ctx.mnic_offset) << "\n"
       << "  max_retry_count: " << ctx.max_retry_count << "\n"
       << "  progress_mode: " << ctx.progress_mode << "\n"
#ifdef CCL_ENABLE_OFI_HMEM
       << "  hmem: " << ctx.enable_hmem << "\n"
#endif // CCL_ENABLE_OFI_HMEM
       << "}";
    return ss.str();
}

atl_status_t atl_ofi::finalize(int global_idx) {
    CCL_THROW_IF_NOT(!is_finalized, "atl_ofi refinalize is not expected");
    is_finalized = true;
    inited = false;

    int ret = 0;
    size_t idx;

    if (coord.global_idx == 0) {
        LOG_INFO("finalizing atl-ofi");
    }

    cache.clear();

    for (idx = 0; idx < ctx.prov_count; idx++) {
        atl_ofi_prov_t* prov = &ctx.provs[idx];
        atl_ofi_prov_destroy(ctx, prov);
    }

    if (global_data.dlhandle) {
        dlclose(global_data.dlhandle);
        global_data.dlhandle = nullptr;
    }

    if (coord.global_idx == 0) {
        LOG_INFO("finalized atl-ofi");
    }

    return ATL_OFI_RET(ret);
}

void atl_ofi::set_env(const atl_attr_t& attr) {
    atl_ofi_set_env(attr);
}

atl_status_t atl_ofi::progress_ep(atl_ep_t& ep) {
    ssize_t ret;
    size_t idx;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];
    atl_ofi_ep_t* ofi_ep = ((atl_ofi_ep_t*)ep.internal);
    size_t ep_idx = ep.idx;

    /* ensure progress for all active providers */
    for (idx = 0; idx < ofi_ep->active_prov_count; idx++) {
        atl_ofi_prov_ep_t* prov_ep;
        prov_ep = &(ctx.provs[ofi_ep->active_prov_idxs[idx]].eps[ep_idx]);
        do {
            ret = fi_cq_read(prov_ep->cq, entries, ATL_OFI_CQ_BUNCH_SIZE);
            if (ret > 0)
                process_comps(ep, entries, ret);
            else if (ret == -FI_EAGAIN)
                break;
            else
                return prov_ep_handle_cq_err(prov_ep);
        } while (ret > 0);
    }

    return ATL_STATUS_SUCCESS;
}

void atl_ofi::process_comps(atl_ep_t& ep, struct fi_cq_tagged_entry* entries, ssize_t ret) {
    ssize_t idx;
    atl_ofi_req_t* comp_ofi_req;
    for (idx = 0; idx < ret; idx++) {
        comp_ofi_req = container_of(entries[idx].op_context, atl_ofi_req_t, fi_ctx);
        switch (comp_ofi_req->comp_state) {
            case ATL_OFI_COMP_POSTED:
                comp_ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;
                cache.push(ep.idx, comp_ofi_req->mr);
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

atl_status_t atl_ofi::prov_ep_handle_cq_err(atl_ofi_prov_ep_t* ep) {
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

atl_status_t atl_ofi::open_providers(char* prov_env,
                                     const atl_proc_coord_t& coord,
                                     atl_attr_t* attr,
                                     struct fi_info* base_hints,
                                     int& open_nw_provs,
                                     int fi_version,
                                     std::shared_ptr<ipmi> pmi,
                                     bool log_on_error) {
    size_t prov_idx = 0;
    int enable_shm = 0;
    ssize_t ret = 0;
    char* prov_name = nullptr;
    struct fi_info *prov_list = nullptr, *prov_hints = nullptr;
    atl_ofi_prov_t* prov = nullptr;

    if (prov_env && !strcmp(prov_env, ATL_OFI_SHM_PROV_NAME)) {
        if (coord.global_count != coord.local_count) {
            LOG_ERROR("shm provider is requested as primary provider but global_count (",
                      coord.global_count,
                      ") != local_count (",
                      coord.local_count,
                      ")");
            goto err;
        }

        if (!attr->in.enable_shm) {
            LOG_ERROR(
                "shm provider is requested through FI_PROVIDER but not requested from CCL level");
            goto err;
        }
    }

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

    ctx.nw_prov_first_idx = (enable_shm) ? 1 : 0;

    /* open SHM provider */
    if (enable_shm) {
        prov_idx = ctx.shm_prov_idx;
        prov_name = strdup(ATL_OFI_SHM_PROV_NAME);
        prov = &ctx.provs[prov_idx];
        prov->idx = prov_idx;
        prov->is_shm = 1;
        ATL_CALL(atl_ofi_get_prov_list(ctx, prov_name, base_hints, &prov_list), goto err);
        ATL_CALL(atl_ofi_prov_init(ctx, coord, prov_list, prov, attr, pmi, ep_names), goto err);
        free(prov_name);
        fi_freeinfo(prov_list);
        ctx.prov_count++;
    }

    /* open NW provider(s) */
    if (prov_env && !strcmp(prov_env, ATL_OFI_SHM_PROV_NAME) && enable_shm) {
        open_nw_provs = 0;
    }

    if (open_nw_provs) {
        ret = atl_ofi_open_nw_provs(ctx, coord, base_hints, attr, pmi, ep_names, log_on_error);
        if (ret != ATL_STATUS_SUCCESS) {
            if (log_on_error) {
                LOG_ERROR("atl_ofi_open_nw_provs failed with status: ", ret);
            }
            goto err;
        }
        ctx.mnic_count = ctx.nw_prov_count;
    }
    return ATL_STATUS_SUCCESS;
err:
    if (prov_list) {
        fi_freeinfo(prov_list);
    }

    if (prov_hints) {
        fi_freeinfo(prov_hints);
    }
    return ATL_STATUS_FAILURE;
}

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

    struct fi_mr_attr mr_attr;
    struct iovec iov;

    memset(&mr_attr, 0, sizeof(mr_attr));
    memset(&iov, 0, sizeof(iov));

    iov.iov_base = buf;
    iov.iov_len = bytes;
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE;
    mr_attr.requested_key = mr_key++;

#ifdef CCL_ENABLE_OFI_HMEM

    mr_attr.iface = FI_HMEM_SYSTEM;
    mr_attr.device.ze = 0;

    ze_context_handle_t context = ccl::global_data::get().ze_data->contexts[0];
    ze_memory_allocation_properties_t alloc_props = ccl::ze::default_alloc_props;
    ze_device_handle_t alloc_dev = nullptr;
    ZE_CALL(zeMemGetAllocProperties, (context, buf, &alloc_props, &alloc_dev));

    LOG_DEBUG("alloc_props: dev ", alloc_dev, ", type ", alloc_props.type);

    if (alloc_props.type == ZE_MEMORY_TYPE_HOST || alloc_props.type == ZE_MEMORY_TYPE_DEVICE ||
        alloc_props.type == ZE_MEMORY_TYPE_SHARED) {
        mr_attr.iface = FI_HMEM_ZE;
    }

    if (alloc_dev) {
        ze_device_properties_t alloc_dev_props = ccl::ze::default_device_props;
        ZE_CALL(zeDeviceGetProperties, (alloc_dev, &alloc_dev_props));

        int dev_idx = -1;
        for (const auto& dev : ccl::global_data::get().ze_data->devices) {
            if (ccl::ze::is_same_dev_uuid(dev.uuid, alloc_dev_props.uuid)) {
                dev_idx = dev.parent_idx;
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
