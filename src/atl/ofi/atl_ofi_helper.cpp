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
#include "atl_ofi_helper.hpp"

atl_ofi_global_data_t global_data;

std::string atl_ofi_get_short_nic_name(const struct fi_info* prov) {
    std::stringstream ss;
    ss << prov->domain_attr->name;
    return ss.str();
}

std::string atl_ofi_get_nic_name(const struct fi_info* prov) {
    std::stringstream ss;
    ss << prov->fabric_attr->prov_name << ":";
    // ss << prov->fabric_attr->name << ":";
    ss << atl_ofi_get_short_nic_name(prov);
    return ss.str();
}

const char* atl_ofi_link_state_str(enum fi_link_state state) {
    switch (state) {
        case FI_LINK_DOWN: return "down";
        case FI_LINK_UP: return "up";
        default: return "unknown";
    }
}

std::string atl_ofi_get_nic_info(const struct fi_info* prov) {
    std::stringstream ss;

    ss << "{ ";

    ss << "name " << atl_ofi_get_nic_name(prov);

    if (prov->nic && prov->nic->link_attr) {
        ss << ", state " << atl_ofi_link_state_str(prov->nic->link_attr->state);

        if (prov->nic->link_attr->mtu) {
            ss << ", mtu " << prov->nic->link_attr->mtu << " bytes";
        }

        if (prov->nic->link_attr->speed) {
            const float bits_to_gbytes_coef = 8.0 * 1000 * 1000 * 1000;
            ss << ", speed " << (float)prov->nic->link_attr->speed / bits_to_gbytes_coef << " GB/s";
        }

        if (prov->nic->link_attr->address) {
            ss << ", address " << prov->nic->link_attr->address;
        }

        if (prov->nic->link_attr->network_type) {
            ss << ", network_type " << prov->nic->link_attr->network_type;
        }
    }
    else {
        ss << ", no link attr";
    }

    ss << " }";

    return ss.str();
}

atl_ofi_prov_t* atl_ofi_get_prov(atl_ofi_ctx_t& ctx,
                                 const atl_proc_coord_t& coord,
                                 const atl_ep_t& ep,
                                 int peer_proc_idx,
                                 size_t msg_size) {
    size_t prov_idx;

    CCL_THROW_IF_NOT(
        ctx.prov_count <= ATL_OFI_MAX_PROV_COUNT, "unexpected prov_count ", ctx.prov_count);

    int my_node_idx = coord.global_idx / coord.local_count;
    int peer_node_idx = peer_proc_idx / coord.local_count;

    int has_shm = (ctx.prov_count == ctx.nw_prov_count + 1) ? 1 : 0;

    if (has_shm && (my_node_idx == peer_node_idx) &&
        (msg_size <= ctx.provs[ctx.shm_prov_idx].max_msg_size)) {
        prov_idx = ctx.shm_prov_idx;
    }
    else {
        size_t nw_prov_offset = ep.idx % ctx.nw_prov_count;
        prov_idx = ctx.nw_prov_first_idx + nw_prov_offset;
    }

    LOG_DEBUG("select nic: ep_idx ",
              ep.idx,
              ", local_proc_idx ",
              coord.local_idx,
              ", nic_idx ",
              prov_idx,
              ", my_node_idx ",
              my_node_idx,
              ", peer_node_idx ",
              peer_node_idx,
              ", msg_size ",
              msg_size,
              ", has_shm ",
              has_shm);

    /* TODO: add segmentation logic */
    CCL_THROW_IF_NOT(msg_size <= ctx.provs[prov_idx].max_msg_size,
                     "msg_size (",
                     msg_size,
                     ") is greater than max_msg_size (",
                     ctx.provs[prov_idx].max_msg_size,
                     "), prov_idx ",
                     prov_idx);

    return &(ctx.provs[prov_idx]);
}

atl_status_t atl_ofi_get_local_proc_coord(atl_proc_coord_t& coord, std::shared_ptr<ipmi> pmi) {
    atl_status_t ret = ATL_STATUS_SUCCESS;
    int i;
    int local_idx = 0, local_count = 0;
    char* all_hostnames = nullptr;
    char my_hostname[ATL_MAX_HOSTNAME_LEN] = { 0 };
    size_t my_hostname_len = 0;
    int my_global_proc_idx = coord.global_idx;

    gethostname(my_hostname, ATL_MAX_HOSTNAME_LEN - 1);
    my_hostname_len = strlen(my_hostname);
    coord.hostname_hash = std::hash<std::string>{}(my_hostname);

    CCL_THROW_IF_NOT(my_hostname_len < ATL_MAX_HOSTNAME_LEN,
                     "unexpected my_hostname_len ",
                     my_hostname_len,
                     ", expected max ",
                     (size_t)(ATL_MAX_HOSTNAME_LEN));

    if (ATL_MAX_HOSTNAME_LEN - my_hostname_len <= 10) {
        LOG_WARN("hostname is quite long, len: ", my_hostname_len, ", name: ", my_hostname);
    }

    snprintf(my_hostname + my_hostname_len,
             ATL_MAX_HOSTNAME_LEN - my_hostname_len,
             "-%d",
             my_global_proc_idx);

    ret = pmi->pmrt_kvs_put((char*)ATL_OFI_HOSTNAME_PM_KEY,
                            my_global_proc_idx * ATL_OFI_PMI_PROC_MULTIPLIER,
                            my_hostname,
                            ATL_MAX_HOSTNAME_LEN);

    if (ret) {
        LOG_ERROR("pmrt_kvs_put: ret: ", ret);
        goto fn_err;
    }

    ATL_CHECK_STATUS(pmi->pmrt_barrier(), "barrier failed");

    all_hostnames = (char*)calloc(1, coord.global_count * ATL_MAX_HOSTNAME_LEN);
    if (!all_hostnames) {
        LOG_ERROR("can't allocate all_hostnames");
        goto fn_err;
    }

    for (i = 0; i < coord.global_count; i++) {
        ret = pmi->pmrt_kvs_get((char*)ATL_OFI_HOSTNAME_PM_KEY,
                                i * ATL_OFI_PMI_PROC_MULTIPLIER,
                                all_hostnames + i * ATL_MAX_HOSTNAME_LEN,
                                ATL_MAX_HOSTNAME_LEN);
        if (ret) {
            LOG_ERROR("pmrt_kvs_get: ret: ", ret);
            goto fn_err;
        }
    }

    for (i = 0; i < coord.global_count; i++) {
        if (!strncmp(my_hostname,
                     all_hostnames + i * ATL_MAX_HOSTNAME_LEN,
                     my_hostname_len + 1 /* including "-" at the end */)) {
            local_count++;
            int peer_global_proc_idx;
            sscanf(all_hostnames + i * ATL_MAX_HOSTNAME_LEN + my_hostname_len + 1,
                   "%d",
                   &peer_global_proc_idx);
            if (my_global_proc_idx > peer_global_proc_idx)
                local_idx++;
        }
    }

    coord.local_idx = local_idx;
    coord.local_count = local_count;

fn_exit:
    free(all_hostnames);
    return ret;

fn_err:
    ret = ATL_STATUS_FAILURE;
    goto fn_exit;
}

atl_status_t atl_ofi_prov_update_addr_table(atl_ofi_ctx_t& ctx,
                                            const atl_proc_coord_t& coord,
                                            size_t prov_idx,
                                            std::shared_ptr<ipmi> pmi,
                                            ep_names_t& ep_names) {
    atl_ofi_prov_t* prov = &(ctx.provs[prov_idx]);

    atl_status_t ret = ATL_STATUS_SUCCESS;
    int i;
    size_t j;
    int insert_count;

    size_t addr_idx = 0;
    char* ep_names_table;
    size_t ep_names_table_len;

    size_t named_ep_count = (prov->sep ? 1 : ctx.ep_count);

    int local_count = coord.local_count;
    int node_idx = coord.global_idx / local_count;
    int shm_start_idx = node_idx * local_count;
    int shm_end_idx = (node_idx + 1) * local_count;

    LOG_DEBUG("shm_start_idx ", shm_start_idx, ", shm_end_idx ", shm_end_idx);

    int proc_count = prov->is_shm ? coord.local_count : coord.global_count;

    if (proc_count == 0)
        return ATL_STATUS_SUCCESS;

    LOG_DEBUG("name ",
              atl_ofi_get_nic_name(prov->info),
              ", is_shm ",
              prov->is_shm,
              ", addr_len ",
              prov->addr_len,
              ", local_count ",
              coord.local_count,
              ", global_count ",
              coord.global_count,
              ", proc_count ",
              proc_count);

    /* allocate OFI EP names table that will contain all published names */
    ep_names_table_len = prov->addr_len * named_ep_count * proc_count;

    if (ep_names_table_len == 0) {
        LOG_ERROR("ep_names_table_len == 0, addr_len ",
                  prov->addr_len,
                  ", named_ep_count ",
                  named_ep_count,
                  ", proc_count ",
                  proc_count);
        return ATL_STATUS_FAILURE;
    }

    ep_names_table = (char*)calloc(1, ep_names_table_len);
    if (!ep_names_table) {
        LOG_ERROR("can't allocate epnames_table");
        return ATL_STATUS_FAILURE;
    }

    ATL_CHECK_STATUS(pmi->pmrt_barrier(), "barrier failed");

    std::vector<char> ret_ep_name(prov->addr_len, '\0');
    /* retrieve all OFI EP names in order */
    for (i = 0; i < coord.global_count; i++) {
        if (prov->is_shm) {
            if (!(i >= shm_start_idx && i < shm_end_idx)) {
                continue;
            }
        }

        for (j = 0; j < named_ep_count; j++) {
            ret = pmi->pmrt_kvs_get(
                (char*)ATL_OFI_FI_ADDR_PM_KEY,
                i * ATL_OFI_PMI_PROC_MULTIPLIER + prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + j,
                (void*)ret_ep_name.data(),
                prov->addr_len);

            auto it = std::find(ep_names.begin(), ep_names.end(), ret_ep_name);
            if (it == ep_names.end()) {
                ep_names.push_back(ret_ep_name);
            }
            memcpy(ep_names_table + addr_idx * prov->addr_len, ret_ep_name.data(), prov->addr_len);
            if (ret) {
                LOG_ERROR("kvs_get error: ret ",
                          ret,
                          ", proc_idx ",
                          i,
                          ", ep_idx ",
                          j,
                          ", addr_idx ",
                          addr_idx);
                goto err_ep_names;
            }

            addr_idx++;
        }
    }

    LOG_DEBUG(
        "kvs_get: ep_count ", named_ep_count, ", proc_count ", proc_count, ", got ", addr_idx);

    if (addr_idx != named_ep_count * proc_count) {
        LOG_ERROR("unexpected kvs_get results: expected ",
                  named_ep_count * proc_count,
                  ", got ",
                  addr_idx);

        ret = ATL_STATUS_FAILURE;
        goto err_addr_table;
    }

    if (prov->addr_table != nullptr)
        free(prov->addr_table);

    prov->addr_table = (fi_addr_t*)calloc(1, ctx.ep_count * proc_count * sizeof(fi_addr_t));

    if (!prov->addr_table)
        goto err_ep_names;

    /* insert all the EP names into the AV */
    insert_count = fi_av_insert(
        prov->av, ep_names_table, named_ep_count * proc_count, prov->addr_table, 0, nullptr);

    LOG_DEBUG("av_insert: ep_count ",
              named_ep_count,
              ", proc_count ",
              proc_count,
              ", inserted ",
              insert_count);

    if (insert_count != (int)(named_ep_count * proc_count)) {
        LOG_ERROR("unexpected av_insert results: expected ",
                  named_ep_count * proc_count,
                  " got ",
                  insert_count);
        ret = ATL_STATUS_FAILURE;
        goto err_addr_table;
    }
    else {
        ret = ATL_STATUS_SUCCESS;
    }

    if (prov->sep) {
        if (named_ep_count != 1) {
            LOG_ERROR("unexpected named_ep_count ", named_ep_count);
            goto err_addr_table;
        }

        fi_addr_t* table;
        table = (fi_addr_t*)calloc(1, proc_count * sizeof(fi_addr_t));
        if (table == nullptr) {
            LOG_ERROR("memory allocaion failed");
            ret = ATL_STATUS_FAILURE;
            goto err_addr_table;
        }
        memcpy(table, prov->addr_table, proc_count * sizeof(fi_addr_t));

        for (i = 0; i < proc_count; i++) {
            for (j = 0; j < ctx.ep_count; j++) {
                prov->addr_table[i * ctx.ep_count + j] = fi_rx_addr(table[i], j, prov->rx_ctx_bits);
            }
        }
        free(table);
    }

    /* normal end of execution */
    free(ep_names_table);
    return ret;

    /* abnormal end of execution */
err_addr_table:
    free(prov->addr_table);

err_ep_names:
    free(ep_names_table);
    return ret;
}

atl_status_t atl_ofi_prov_ep_get_name(atl_ofi_prov_t* prov, size_t ep_idx) {
    int ret;

    atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);
    struct fid_ep* fi_ep = (prov->sep) ? prov->sep : ep->tx;

    ep->name.addr = nullptr;
    ep->name.len = 0;

    ret = fi_getname(&fi_ep->fid, ep->name.addr, &(ep->name.len));
    if ((ret != -FI_ETOOSMALL) || ep->name.len <= 0)
        ep->name.len = FI_NAME_MAX;

    if (ep->name.addr)
        free(ep->name.addr);

    ep->name.addr = calloc(1, ep->name.len);

    if (!(ep->name.addr)) {
        LOG_ERROR("can't allocate addr");
        ret = ATL_STATUS_FAILURE;
        goto err_addr;
    }

    ret = fi_getname(&fi_ep->fid, ep->name.addr, &(ep->name.len));
    if (ret) {
        LOG_ERROR("fi_getname error");
        goto err_getname;
    }

    prov->addr_len = MAX(prov->addr_len, ep->name.len);

    return ATL_STATUS_SUCCESS;

err_getname:
    free(ep->name.addr);
    ep->name.addr = nullptr;
    ep->name.len = 0;

err_addr:
    return ATL_OFI_RET(ret);
}

atl_status_t atl_ofi_prov_eps_connect(atl_ofi_ctx_t& ctx,
                                      const atl_proc_coord_t& coord,
                                      size_t prov_idx,
                                      std::shared_ptr<ipmi> pmi,
                                      ep_names_t& ep_names) {
    int ret;
    size_t ep_idx;

    atl_ofi_prov_t* prov = &(ctx.provs[prov_idx]);
    size_t named_ep_count = (prov->sep ? 1 : ctx.ep_count);

    prov->addr_len = 0;
    prov->first_proc_idx =
        (prov->is_shm) ? ((coord.global_idx / coord.local_count) * coord.local_count) : 0;

    for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
        ret = atl_ofi_prov_ep_get_name(prov, ep_idx);
        if (ret) {
            LOG_ERROR("atl_ofi_prov_ep_get_name error");
            return ATL_STATUS_FAILURE;
        }
    }

    for (ep_idx = 0; ep_idx < named_ep_count; ep_idx++) {
        atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);
        ret = pmi->pmrt_kvs_put((char*)ATL_OFI_FI_ADDR_PM_KEY,
                                coord.global_idx * ATL_OFI_PMI_PROC_MULTIPLIER +
                                    prov_idx * ATL_OFI_PMI_PROV_MULTIPLIER + ep_idx,
                                ep->name.addr,
                                ep->name.len);
        if (ret) {
            LOG_ERROR("pmrt_kvs_put: ret: ", ret);
            return ATL_STATUS_FAILURE;
        }
    }

    ret = atl_ofi_prov_update_addr_table(ctx, coord, prov_idx, pmi, ep_names);

    return ATL_OFI_RET(ret);
}

void atl_ofi_prov_ep_destroy(atl_ofi_prov_t* prov, atl_ofi_prov_ep_t* ep) {
    if (ep->rx)
        fi_close(&ep->rx->fid);

    if (prov->sep && ep->tx)
        fi_close(&ep->tx->fid);

    if (ep->cq)
        fi_close(&ep->cq->fid);

    if (ep->name.addr)
        free(ep->name.addr);

    ep->rx = ep->tx = nullptr;
    ep->cq = nullptr;
    ep->name.addr = nullptr;
    ep->name.len = 0;
}

void atl_ofi_prov_destroy(atl_ofi_ctx_t& ctx, atl_ofi_prov_t* prov) {
    size_t i;

    for (i = 0; i < ctx.ep_count; i++) {
        atl_ofi_prov_ep_destroy(prov, &(prov->eps[i]));
    }

    free(prov->eps);
    free(prov->addr_table);

    if (prov->sep)
        fi_close(&prov->sep->fid);

    if (prov->av)
        fi_close(&prov->av->fid);

    if (prov->domain)
        fi_close(&prov->domain->fid);

    if (prov->fabric)
        fi_close(&prov->fabric->fid);

    if (prov->info) {
        fi_freeinfo(prov->info);
    }
}

int atl_ofi_wait_cancel_cq(struct fid_cq* cq) {
    struct fi_cq_err_entry err_entry;
    int ret, i;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];

    double time = 0;
    clock_t start, end;

    while (time < ATL_OFI_WAIT_SEC) {
        for (i = 0; i < ATL_OFI_CQ_READ_ITERS; i++) {
            start = clock();
            ret = fi_cq_read(cq, entries, ATL_OFI_CQ_BUNCH_SIZE);

            if (ret < 0 && ret != -FI_EAGAIN) {
                ret = fi_cq_readerr(cq, &err_entry, 0);

                if (err_entry.err != FI_ECANCELED) {
                    LOG_ERROR(
                        "fi_cq_readerr: err: ",
                        err_entry.err,
                        ", prov_err: ",
                        fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0),
                        "(",
                        err_entry.prov_errno,
                        ")");
                    return ATL_STATUS_FAILURE;
                }
                return ATL_STATUS_SUCCESS;
            }
        }
        end = clock();
        time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    LOG_ERROR("too long for cancel");

    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_prov_ep_init(atl_ofi_prov_t* prov, size_t ep_idx) {
    ssize_t ret = 0;

    struct fi_cq_attr cq_attr;
    struct fi_tx_attr tx_attr;
    struct fi_rx_attr rx_attr;

    atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);

    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_TAGGED;

    ATL_OFI_CALL(
        fi_cq_open(prov->domain, &cq_attr, &ep->cq, nullptr), ret, return ATL_STATUS_FAILURE);

    if (prov->sep) {
        rx_attr = *prov->info->rx_attr;
        rx_attr.caps |= FI_TAGGED;

        ATL_OFI_CALL(fi_rx_context(prov->sep, ep_idx, &rx_attr, &ep->rx, nullptr), ret, goto err);

        ATL_OFI_CALL(fi_ep_bind(ep->rx, &ep->cq->fid, FI_RECV), ret, goto err);

        tx_attr = *prov->info->tx_attr;
        tx_attr.caps |= FI_TAGGED;

        ATL_OFI_CALL(fi_tx_context(prov->sep, ep_idx, &tx_attr, &ep->tx, nullptr), ret, goto err);

        ATL_OFI_CALL(fi_ep_bind(ep->tx, &ep->cq->fid, FI_SEND), ret, goto err);

        fi_enable(ep->rx);
        fi_enable(ep->tx);
    }
    else {
        struct fid_ep* endpoint;

        ATL_OFI_CALL(fi_endpoint(prov->domain, prov->info, &endpoint, nullptr), ret, goto err);

        ep->tx = ep->rx = endpoint;

        ATL_OFI_CALL(fi_ep_bind(endpoint, &ep->cq->fid, FI_SEND | FI_RECV), ret, goto err);

        ATL_OFI_CALL(fi_ep_bind(endpoint, &prov->av->fid, 0), ret, goto err);

        fi_enable(endpoint);
    }

    return ATL_STATUS_SUCCESS;

err:
    atl_ofi_prov_ep_destroy(prov, ep);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_try_to_drain_cq_err(struct fid_cq* cq) {
    struct fi_cq_err_entry err_entry;
    int ret = fi_cq_readerr(cq, &err_entry, 0);
    if (ret != 1) {
        LOG_DEBUG("unable to fi_cq_readerr");
        return ATL_STATUS_FAILURE;
    }
    else {
        if (err_entry.err != FI_ENOMSG && err_entry.err != FI_ECANCELED &&
            err_entry.err != FI_ETRUNC) {
            LOG_ERROR("fi_cq_readerr: err: ",
                      err_entry.err,
                      ", prov_err: ",
                      fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, nullptr, 0),
                      "(",
                      err_entry.prov_errno,
                      ")");
            return ATL_STATUS_FAILURE;
        }
        return ATL_STATUS_SUCCESS;
    }
}

int atl_ofi_try_to_drain_cq(struct fid_cq* cq) {
    int ret = -FI_EAGAIN, i;
    double time = 0;
    clock_t start, end;
    struct fi_cq_tagged_entry entries[ATL_OFI_CQ_BUNCH_SIZE];

    while (time < ATL_OFI_WAIT_SEC) {
        start = clock();
        for (i = 0; i < ATL_OFI_CQ_READ_ITERS; i++) {
            ret = fi_cq_read(cq, entries, ATL_OFI_CQ_BUNCH_SIZE);

            if (ret < 0 && ret != -FI_EAGAIN) {
                atl_ofi_try_to_drain_cq_err(cq);
                return ret;
            }

            if (ret > 0)
                return ret;
        }
        end = clock();
        time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    return ret;
}

void atl_ofi_reset(atl_ofi_ctx_t& ctx) {
    int again = 1;
    size_t prov_idx, ep_idx;
    int recv_buf_len = sizeof(char);
    char* recv_buf;
    struct fi_context fi_ctx;
    recv_buf = (char*)malloc(recv_buf_len);
    for (prov_idx = 0; prov_idx < ctx.prov_count; prov_idx++) {
        atl_ofi_prov_t* prov = &(ctx.provs[prov_idx]);

        for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
            atl_ofi_prov_ep_t* ep = &(prov->eps[ep_idx]);

            /* complete active sends and receives */
            while (atl_ofi_try_to_drain_cq(ep->cq) != -FI_EAGAIN) {
            }

            /* try to complete active incoming sends */
            while (again) {
                again = 0;
                /* post recv to complete incoming send */
                while (fi_trecv(ep->rx,
                                recv_buf,
                                recv_buf_len,
                                nullptr,
                                FI_ADDR_UNSPEC,
                                0,
                                UINTMAX_MAX,
                                &fi_ctx) == -FI_EAGAIN) {
                }

                /* wait until recv will be completed or finished by timeout */
                while (atl_ofi_try_to_drain_cq(ep->cq) != -FI_EAGAIN) {
                    /* something is completed -> send queue not empty */
                    again = 1;
                }
            }

            /* nothing to recv -> cancel last recv */
            fi_cancel(&ep->rx->fid, &fi_ctx);

            atl_ofi_wait_cancel_cq(ep->cq);
        }
    }

    free(recv_buf);
}

atl_status_t atl_ofi_adjust_env(const atl_attr_t& attr) {
    char* prov_env = getenv("FI_PROVIDER");

    if (prov_env && strlen(prov_env)) {
        CCL_THROW_IF_NOT(strlen(prov_env) < sizeof(global_data.prov_env_copy),
                         "too long FI_PROVIDER value, max expected length ",
                         sizeof(global_data.prov_env_copy));
        memcpy(global_data.prov_env_copy, prov_env, strlen(prov_env));
    }

    if (attr.in.enable_shm) {
        /* add shm provider in the list of allowed providers */
        if (prov_env && !strstr(prov_env, ATL_OFI_SHM_PROV_NAME)) {
            /* whether single provider will be in the final env variable */
            int single_prov = (strlen(prov_env) == 0) ? 1 : 0;

            size_t prov_env_new_size = strlen(prov_env) + strlen(ATL_OFI_SHM_PROV_NAME) +
                                       (single_prov ? 0 : 1) + /* for delimeter */
                                       1; /* for terminating null symbol */

            char* prov_env_new = (char*)calloc(prov_env_new_size, sizeof(char));
            if (prov_env_new == nullptr) {
                LOG_ERROR("memory allocaion failed");
                return ATL_STATUS_FAILURE;
            }

            if (single_prov)
                snprintf(prov_env_new, prov_env_new_size, "%s", ATL_OFI_SHM_PROV_NAME);
            else {
                snprintf(prov_env_new, prov_env_new_size, "%s,%s", prov_env, ATL_OFI_SHM_PROV_NAME);
            }

            LOG_INFO("atl-ofi-shm is requested, modify FI_PROVIDER: old value: ",
                     prov_env,
                     ", new value: ",
                     prov_env_new);

            setenv("FI_PROVIDER", prov_env_new, 1);

            free(prov_env_new);
        }
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_set_env(const atl_attr_t& attr) {
    if (global_data.is_env_inited) {
        return ATL_STATUS_SUCCESS;
    }

    setenv("FI_PSM2_DELAY", "0", 0);
    setenv("FI_PSM2_TIMEOUT", "0", 0);
    setenv("FI_PSM2_LOCK_LEVEL", "1", 0);
    setenv("FI_PSM2_NAME_SERVER", "0", 0);
    setenv("HFI_NO_CPUAFFINITY", "1", 0);
    setenv("PSM2_MULTI_EP", "1", 0);

    setenv("FI_PSM3_DELAY", "0", 0);
    setenv("FI_PSM3_TIMEOUT", "0", 0);
    setenv("FI_PSM3_LOCK_LEVEL", "1", 0);
    setenv("FI_PSM3_NAME_SERVER", "0", 0);
    setenv("PSM3_NO_CPUAFFINITY", "1", 0);
    setenv("PSM3_RDMA", "2", 0);
    setenv("PSM3_MR_CACHE_MODE", "0", 0); //TODO temporary
    setenv("PSM3_MULTI_EP", "1", 0);
    if (attr.in.mnic_type == ATL_MNIC_NONE)
        setenv("PSM3_NIC", "any", 0);

    char* hydra_uuid_env = getenv("I_MPI_HYDRA_UUID");
    if (hydra_uuid_env) {
        setenv("FI_PSM2_UUID", hydra_uuid_env, 0);
        setenv("FI_PSM3_UUID", hydra_uuid_env, 0);
    }

    setenv("FI_OFI_RXM_USE_HASH", "0", 0);
    setenv("FI_OFI_RXM_USE_SRX", "0", 0);
    setenv("FI_OFI_RXM_RX_SIZE", "8192", 0);
    setenv("FI_OFI_RXM_TX_SIZE", "8192", 0);
    setenv("FI_OFI_RXM_MSG_RX_SIZE", "128", 0);
    setenv("FI_OFI_RXM_MSG_TX_SIZE", "128", 0);

    setenv("FI_SHM_TX_SIZE", "8192", 0);
    setenv("FI_SHM_RX_SIZE", "8192", 0);

#ifdef CCL_ENABLE_SYCL
    setenv("FI_SHM_DISABLE_CMA", "1", 0);
#endif // CCL_ENABLE_SYCL

    setenv("FI_MLX_MULTI_EP", "1", 0);

    atl_ofi_adjust_env(attr);

#ifdef CCL_ENABLE_OFI_OOT_PROV
    /*
       load libfabric symbols into global namespace
       to workaround issue with undefined symbols
       in case of out-of-tree providers, like OFI/PSM3
    */
    global_data.dlhandle = dlopen("libfabric.so", RTLD_GLOBAL | RTLD_NOW);
    if (global_data.dlhandle == nullptr) {
        LOG_WARN("dlopen (libfabric.so): ", dlerror());
    }
#endif // CCL_ENABLE_OFI_OOT_PROV

    global_data.is_env_inited = 1;

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_ofi_get_prov_list(atl_ofi_ctx_t& ctx,
                                   const char* prov_name,
                                   struct fi_info* base_hints,
                                   struct fi_info** out_prov_list) {
    struct fi_info* hints = nullptr;
    struct fi_info* prov_list = nullptr;
    ssize_t ret = 0;
    int fi_version = FI_VERSION(global_data.fi_major_version, global_data.fi_minor_version);
    const char* prov_name_str = (prov_name) ? prov_name : "<default>";

    hints = fi_dupinfo(base_hints);
    if (!hints) {
        LOG_ERROR("fi_dupinfo error");
        goto err;
    }

    *out_prov_list = nullptr;

    LOG_DEBUG("request providers with name: ", prov_name_str);

    hints->fabric_attr->prov_name = (prov_name) ? strdup(prov_name) : nullptr;

    ret = fi_getinfo(fi_version, nullptr, nullptr, 0ULL, hints, &prov_list);

    if ((ret || !prov_list || !strcmp(prov_list->fabric_attr->prov_name, ATL_OFI_SHM_PROV_NAME)) &&
        prov_list->caps & FI_HMEM) {
        // skip OFI/SHM with HMEM capability
        fi_freeinfo(hints);
        fi_freeinfo(prov_list);
        return ATL_STATUS_FAILURE;
    }
    if (ret || !prov_list) {
        LOG_ERROR("fi_getinfo error: ret ", ret, ", providers ", (void*)prov_list);
        goto err;
    }

    if (prov_list->domain_attr->max_ep_tx_ctx > 1) {
        hints->ep_attr->tx_ctx_cnt = ctx.ep_count;
        hints->ep_attr->rx_ctx_cnt = ctx.ep_count;
    }
    else {
        hints->ep_attr->tx_ctx_cnt = 1;
        hints->ep_attr->rx_ctx_cnt = 1;
    }

    fi_freeinfo(prov_list);
    prov_list = nullptr;

    ret = fi_getinfo(fi_version, nullptr, nullptr, 0ULL, hints, &prov_list);
    if (ret || !prov_list) {
        LOG_ERROR("fi_getinfo error, prov_name ", prov_name_str);
        goto err;
    }

    fi_freeinfo(hints);
    hints = nullptr;

    *out_prov_list = prov_list;
    return ATL_STATUS_SUCCESS;

err:
    if (hints) {
        fi_freeinfo(hints);
    }
    if (prov_list) {
        fi_freeinfo(prov_list);
    }
    LOG_ERROR("can't create providers for name ", prov_name_str);
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_prov_init(atl_ofi_ctx_t& ctx,
                               const atl_proc_coord_t& coord,
                               struct fi_info* info,
                               atl_ofi_prov_t* prov,
                               atl_attr_t* attr,
                               std::shared_ptr<ipmi> pmi,
                               ep_names_t& ep_names) {
    struct fi_av_attr av_attr;
    size_t ep_idx = 0;
    ssize_t ret = 0;

    memset(&av_attr, 0, sizeof(av_attr));

    if (coord.global_idx == 0) {
        LOG_INFO("provider: ", info->fabric_attr->prov_name);
        LOG_INFO("  nic: ", atl_ofi_get_nic_info(info));
        LOG_INFO("  mr_mode: ", info->domain_attr->mr_mode);
        LOG_INFO("  threading: ", info->domain_attr->threading);
        LOG_INFO("  tx_ctx_cnt: ", info->domain_attr->tx_ctx_cnt);
        LOG_INFO("  max_ep_tx_ctx: ", info->domain_attr->max_ep_tx_ctx);
        LOG_INFO("  max_msg_size: ", info->ep_attr->max_msg_size);
    }

    prov->info = fi_dupinfo(info);

    if (!prov->info) {
        LOG_ERROR("fi_dupinfo error");
        goto err;
    }

    prov->max_msg_size = info->ep_attr->max_msg_size;

    ATL_OFI_CALL(fi_fabric(info->fabric_attr, &prov->fabric, nullptr), ret, goto err);

    ATL_OFI_CALL(fi_domain(prov->fabric, info, &prov->domain, nullptr), ret, goto err);

    av_attr.type = FI_AV_TABLE;
    av_attr.rx_ctx_bits = prov->rx_ctx_bits = (int)ceil(log2(prov->info->ep_attr->rx_ctx_cnt));

    ATL_OFI_CALL(fi_av_open(prov->domain, &av_attr, &prov->av, nullptr), ret, goto err);

    if (info->domain_attr->max_ep_tx_ctx > 1) {
        ATL_OFI_CALL(fi_scalable_ep(prov->domain, info, &prov->sep, nullptr), ret, goto err);
        ATL_OFI_CALL(fi_scalable_ep_bind(prov->sep, &prov->av->fid, 0), ret, goto err);
    }

    prov->eps = (atl_ofi_prov_ep_t*)calloc(1, sizeof(atl_ofi_prov_ep_t) * ctx.ep_count);
    if (!prov->eps) {
        LOG_ERROR("can't allocate prov->eps");
        goto err;
    }

    for (ep_idx = 0; ep_idx < ctx.ep_count; ep_idx++) {
        ret = atl_ofi_prov_ep_init(prov, ep_idx);
        if (ret) {
            LOG_ERROR("atl_ofi_prov_ep_init error");
            goto err;
        }
    }

    if (prov->sep) {
        fi_enable(prov->sep);
    }

    /* TODO: make separate function to be called on CCL comm creation */
    ret = atl_ofi_prov_eps_connect(ctx, coord, prov->idx, pmi, ep_names);
    if (ret) {
        LOG_ERROR("atl_ofi_prov_eps_connect error, prov_idx ", prov->idx);
        goto err;
    }

    ATL_CALL(atl_ofi_adjust_out_tag(prov, attr), goto err);

    return ATL_STATUS_SUCCESS;

err:
    if (prov->info) {
        fi_freeinfo(prov->info);
        prov->info = nullptr;
    }
    LOG_ERROR("can't init provider ", atl_ofi_get_nic_name(info));
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_ofi_adjust_out_tag(atl_ofi_prov_t* prov, atl_attr_t* attr) {
    size_t tag_bits = 64;
    uint64_t mem_tag_format = prov->info->ep_attr->mem_tag_format;
    while (tag_bits && !(mem_tag_format & ((uint64_t)1 << (tag_bits - 1)))) {
        tag_bits--;
    }

    attr->out.tag_bits = std::min(attr->out.tag_bits, tag_bits);

    if (attr->out.tag_bits == 64) {
        attr->out.max_tag = 0xFFFFFFFFFFFFFFFF;
    }
    else {
        attr->out.max_tag = (((uint64_t)1 << attr->out.tag_bits) - 1);
    }

    const char* prov_name = prov->info->fabric_attr->prov_name;

    if (!(attr->out.tag_bits > 0)) {
        LOG_ERROR("unexpected tag_bits ", attr->out.tag_bits, " for prov ", prov_name);
        return ATL_STATUS_FAILURE;
    }

    if (!(attr->out.max_tag > 0)) {
        LOG_ERROR("unexpected max_tag ", attr->out.max_tag, " for prov ", prov_name);
        return ATL_STATUS_FAILURE;
    }
    LOG_INFO(prov_name,
             " tag_bits: ",
             attr->out.tag_bits,
             ", max_tag: ",
             attr->out.max_tag,
             ", mem_tag_format: ",
             mem_tag_format);

    return ATL_STATUS_SUCCESS;
}

static bool atl_ofi_is_nic_down(struct fi_info* prov) {
    if (prov->nic && prov->nic->link_attr->state == FI_LINK_DOWN) {
        return true;
    }

    return false;
}

/* determine if NIC has already been included in others */
int atl_ofi_nic_already_used(const struct fi_info* prov,
                             const std::vector<struct fi_info*>& others,
                             bool check_pci = false) {
    for (size_t i = 0; i < others.size(); i++) {
        if (check_pci && prov->nic && others[i]->nic &&
            prov->nic->bus_attr->bus_type == FI_BUS_PCI &&
            others[i]->nic->bus_attr->bus_type == FI_BUS_PCI) {
            struct fi_pci_attr pci = prov->nic->bus_attr->attr.pci;
            struct fi_pci_attr other_pci = others[i]->nic->bus_attr->attr.pci;
            LOG_TRACE("compare nic ",
                      prov->fabric_attr->prov_name,
                      " pci ",
                      (int)pci.domain_id,
                      ":",
                      (int)pci.bus_id,
                      ":",
                      (int)pci.device_id,
                      ":",
                      (int)pci.function_id,
                      " with nic ",
                      others[i]->fabric_attr->prov_name,
                      " pci ",
                      (int)other_pci.domain_id,
                      ":",
                      (int)other_pci.bus_id,
                      ":",
                      (int)other_pci.device_id,
                      ":",
                      (int)other_pci.function_id);
            if (pci.domain_id == other_pci.domain_id && pci.bus_id == other_pci.bus_id &&
                pci.device_id == other_pci.device_id && pci.function_id == other_pci.function_id)
                return 1;
        }
        else {
            LOG_TRACE("compare nic ",
                      atl_ofi_get_nic_name(prov),
                      " with nic ",
                      atl_ofi_get_nic_name(others[i]));
            if (atl_ofi_get_short_nic_name(prov) == atl_ofi_get_short_nic_name(others[i]))
                return 1;
        }
    }
    return 0;
}

/* return true if the NIC is bound to the same socket as calling process */
int atl_ofi_is_nic_local(struct fi_info* info) {
    if (info->nic && info->nic->bus_attr->bus_type == FI_BUS_PCI) {
        struct fi_pci_attr pci = info->nic->bus_attr->attr.pci;
        return ccl::global_data::get().hwloc_wrapper->is_dev_close_by_pci(
            pci.domain_id, pci.bus_id, pci.device_id, pci.function_id);
    }
    return 0;
}

atl_status_t atl_ofi_parse_mnic_name(atl_ofi_ctx_t& ctx, std::string str_to_parse) {
    atl_status_t ret = ATL_STATUS_SUCCESS;

    std::string include_str;
    std::string exclude_str;

    auto pos = str_to_parse.find('^');

    if (pos == 0) {
        exclude_str = str_to_parse.substr(1);
    }
    else {
        if (pos != std::string::npos) {
            include_str = str_to_parse.substr(0, pos - 1);
            exclude_str = str_to_parse.substr(pos + 1);
        }
        else {
            include_str = str_to_parse.substr(0, pos);
        }
    }

    if (!include_str.empty()) {
        LOG_DEBUG("include names str: ", include_str);
    }

    if (!exclude_str.empty()) {
        LOG_DEBUG("exclude names str: ", exclude_str);
    }

    auto include_names = ccl::utils::tokenize<std::vector<std::string>>(include_str, ',');
    auto exclude_names = ccl::utils::tokenize<std::vector<std::string>>(exclude_str, ',');

    if (!include_names.empty() && !exclude_names.empty()) {
        auto include_set = std::set<std::string>(include_names.begin(), include_names.end());
        auto exclude_set = std::set<std::string>(exclude_names.begin(), exclude_names.end());

        std::set<std::string> intersect;
        std::set_intersection(include_set.begin(),
                              include_set.end(),
                              exclude_set.begin(),
                              exclude_set.end(),
                              std::inserter(intersect, intersect.begin()));
        if (!intersect.empty()) {
            LOG_ERROR("include and exclude sets can not intersect");
            ret = ATL_STATUS_FAILURE;
        }

        for (auto include_name : include_names) {
            for (auto exclude_name : exclude_names) {
                std::string& larger_name =
                    (include_name.size() > exclude_name.size()) ? include_name : exclude_name;
                std::string& smaller_name =
                    (include_name.size() > exclude_name.size()) ? exclude_name : include_name;
                if (larger_name.substr(0, smaller_name.size()) == smaller_name) {
                    LOG_ERROR("include name ",
                              include_name,
                              " and exclude name ",
                              exclude_name,
                              " have commom prefix");
                    ret = ATL_STATUS_FAILURE;
                    break;
                }
            }
        }
    }

    if (ret == ATL_STATUS_SUCCESS) {
        LOG_DEBUG("include names: ", ccl::utils::vec_to_string(include_names));
        LOG_DEBUG("exclude names: ", ccl::utils::vec_to_string(exclude_names));
        ctx.mnic_include_names = include_names;
        ctx.mnic_exclude_names = exclude_names;
    }

    return ret;
}

int atl_ofi_is_allowed_nic_name(atl_ofi_ctx_t& ctx, struct fi_info* info) {
    auto& include_names = ctx.mnic_include_names;
    auto& exclude_names = ctx.mnic_exclude_names;
    std::string nic_name = atl_ofi_get_short_nic_name(info);

    int should_include = 0;
    int should_exclude = 0;

    if (include_names.empty()) {
        should_include = 1;
    }

    for (auto name : include_names) {
        if (nic_name.substr(0, name.size()) == name) {
            should_include = 1;
            break;
        }
    }

    for (auto name : exclude_names) {
        if (nic_name.substr(0, name.size()) == name) {
            should_exclude = 1;
            break;
        }
    }

    return (should_include && !should_exclude);
}

bool atl_ofi_compare_nics(const struct fi_info* nic1, const struct fi_info* nic2) {
    if (nic1->nic && !nic2->nic) {
        return true;
    }
    else if (!nic1->nic && nic2->nic) {
        return false;
    }
    return (atl_ofi_get_short_nic_name(nic1) < atl_ofi_get_short_nic_name(nic2));
}

atl_status_t atl_ofi_open_nw_provs(atl_ofi_ctx_t& ctx,
                                   const atl_proc_coord_t& coord,
                                   struct fi_info* base_hints,
                                   atl_attr_t* attr,
                                   std::shared_ptr<ipmi> pmi,
                                   std::vector<ep_names_t>& ep_names,
                                   bool log_on_error) {
    atl_status_t ret = ATL_STATUS_SUCCESS;
    struct fi_info* prov_list = nullptr;
    struct fi_info* prov_iter = nullptr;
    size_t idx = 0, prov_idx = 0;
    char* prov_name = nullptr;
    atl_ofi_prov_t* prov = nullptr;
    std::vector<struct fi_info*> name_provs;
    std::vector<struct fi_info*> topo_provs;
    std::vector<struct fi_info*> final_provs;
    std::set<std::string> all_nic_names;
    int prov_offset = 0;

    ctx.nw_prov_count = 0;

    /* 1. get full list of providers */
    if (strlen(global_data.prov_env_copy) && !strstr(global_data.prov_env_copy, ","))
        prov_name = global_data.prov_env_copy;
    else
        prov_name = nullptr;
    ret = atl_ofi_get_prov_list(ctx, prov_name, base_hints, &prov_list);
    if (ret != ATL_STATUS_SUCCESS) {
        if (log_on_error) {
            LOG_ERROR(
                "atl_ofi_get_prov_list(ctx, prov_name, base_hints, &prov_list)\n fails with status: ",
                ret);
        }
        goto err;
    }

    /* 2. filter out by names */
    prov_iter = prov_list;
    while (prov_iter) {
        LOG_DEBUG("name filter: check nic ", atl_ofi_get_nic_name(prov_iter));
        if (atl_ofi_is_nic_down(prov_iter)) {
            LOG_DEBUG("nic ", atl_ofi_get_nic_name(prov_iter), " is in down state, skip");
        }
        else if (!atl_ofi_nic_already_used(prov_iter, name_provs)) {
            all_nic_names.insert(atl_ofi_get_short_nic_name(prov_iter));
            if (atl_ofi_is_allowed_nic_name(ctx, prov_iter)) {
                LOG_DEBUG("name filter: found suitable nic ", atl_ofi_get_nic_name(prov_iter));
                name_provs.push_back(fi_dupinfo(prov_iter));
            }
        }
        prov_iter = prov_iter->next;
    }

    /* sort by names */
    std::sort(name_provs.begin(), name_provs.end(), atl_ofi_compare_nics);

    if (name_provs.empty()) {
        LOG_ERROR("name filter: can not find network providers",
                  ", include names: ",
                  ccl::utils::vec_to_string(ctx.mnic_include_names),
                  ", exclude names: ",
                  ccl::utils::vec_to_string(ctx.mnic_exclude_names),
                  ", all names: ",
                  ccl::utils::vec_to_string(all_nic_names));
        goto err;
    }

    /* 3. filter out by topo */
    if (ctx.mnic_type == ATL_MNIC_NONE) {
        topo_provs.push_back(fi_dupinfo(name_provs[0]));
    }
    else {
        struct fid_nic* nic = nullptr;
        for (idx = 0; idx < name_provs.size(); idx++) {
            prov_iter = name_provs[idx];
            LOG_DEBUG("topo filter: check nic ", atl_ofi_get_nic_name(prov_iter));
            nic = prov_iter->nic;

            LOG_DEBUG("topo filter: check nic ",
                      atl_ofi_get_nic_name(prov_iter),
                      ", has nic_attr ",
                      (nic != nullptr));

            if (!atl_ofi_nic_already_used(prov_iter, topo_provs)) {
                int is_local = atl_ofi_is_nic_local(prov_iter);
                LOG_DEBUG(
                    "topo filter: nic ", atl_ofi_get_nic_name(prov_iter), ", is_local ", is_local);
                if (ctx.mnic_type == ATL_MNIC_GLOBAL ||
                    (ctx.mnic_type == ATL_MNIC_LOCAL && is_local)) {
                    LOG_DEBUG("topo filter: found suitable nic ", atl_ofi_get_nic_name(prov_iter));
                    topo_provs.push_back(fi_dupinfo(prov_iter));
                }
            }
            else {
                LOG_DEBUG("topo filter: nic ", atl_ofi_get_nic_name(prov_iter), " already used");
            }
        }
    }

    if (topo_provs.empty()) {
        LOG_ERROR("topo filter: can not find network providers, mnic_type ", ctx.mnic_type);
        goto err;
    }

    /* 4. reorder according to desired offset */
    if (ctx.mnic_offset == ATL_MNIC_OFFSET_LOCAL_PROC_IDX) {
        prov_offset = coord.local_idx % topo_provs.size();
    }
    LOG_DEBUG("rotate: prov_offset ", prov_offset, ", vec_size ", topo_provs.size());
    std::rotate(topo_provs.begin(), topo_provs.begin() + prov_offset, topo_provs.end());

    /* 5. filter out by count */
    for (idx = 0; idx < topo_provs.size(); idx++) {
        prov_iter = topo_provs[idx];
        LOG_DEBUG("count filter: check nic ", atl_ofi_get_nic_name(prov_iter));
        if (final_provs.size() < ctx.mnic_count) {
            LOG_DEBUG("count filter: found suitable nic ",
                      atl_ofi_get_nic_name(prov_iter),
                      ", nic idx ",
                      final_provs.size());
            final_provs.push_back(fi_dupinfo(prov_iter));
        }
        else {
            break;
        }
    }

    if (final_provs.empty()) {
        LOG_ERROR("count filter: can not find network providers, mnic_count ", ctx.mnic_count);
        goto err;
    }

    /* 6. create network providers */
    LOG_INFO("found ", final_provs.size(), " nic(s) according to all filters");
    ctx.nw_prov_count = final_provs.size();
    if (ep_names.size() < ctx.nw_prov_count + ctx.nw_prov_first_idx) {
        ep_names.resize(ctx.nw_prov_count + ctx.nw_prov_first_idx);
    }
    for (idx = 0; idx < ctx.nw_prov_count; idx++) {
        prov_idx = ctx.nw_prov_first_idx + idx;
        prov = &ctx.provs[prov_idx];
        prov->idx = prov_idx;
        prov->is_shm = 0;
        ATL_CALL(
            atl_ofi_prov_init(ctx, coord, final_provs[idx], prov, attr, pmi, ep_names[prov->idx]),
            goto err);
    }

exit:
    for (idx = 0; idx < final_provs.size(); idx++) {
        if (final_provs[idx])
            fi_freeinfo(final_provs[idx]);
    }

    for (idx = 0; idx < topo_provs.size(); idx++) {
        if (topo_provs[idx])
            fi_freeinfo(topo_provs[idx]);
    }

    for (idx = 0; idx < name_provs.size(); idx++) {
        if (name_provs[idx])
            fi_freeinfo(name_provs[idx]);
    }

    fi_freeinfo(prov_list);

    ctx.prov_count += ctx.nw_prov_count;

    return ret;

err:
    if (log_on_error) {
        LOG_ERROR("can not open network providers");
    }
    else {
        LOG_DEBUG("can not open network providers");
    }
    ret = ATL_STATUS_FAILURE;
    goto exit;
}

void atl_ofi_init_req(atl_req_t& req, atl_ofi_prov_ep_t* prov_ep, struct fid_ep* fi_ep) {
    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);
    ofi_req->prov_ep = prov_ep;
    ofi_req->fi_ep = fi_ep;
    ofi_req->comp_state = ATL_OFI_COMP_POSTED;
    req.is_completed = 0;
}
