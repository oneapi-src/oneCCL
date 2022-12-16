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
#ifdef CCL_ENABLE_MPI

#include "atl/atl_def.h"
#include "atl/mpi/atl_mpi.hpp"
#include "atl/util/pm/pm_rt.h"

atl_mpi::~atl_mpi() {
    if (!is_finalized) {
        LOG_WARN("unexpected atl_mpi object delete without finalize")
        finalize(0);
    }
}

atl_status_t atl_mpi::init(int* argc,
                           char*** argv,
                           atl_attr_t* attr,
                           const char* main_addr,
                           std::shared_ptr<ipmi> pmi) {
    CCL_THROW_IF_NOT(!inited, "atl_mpi reinit is not expected");
    inited = true;

    CCL_THROW_IF_NOT((sizeof(atl_mpi_req_t) <= sizeof(atl_req_t) - offsetof(atl_req_t, internal)),
                     "unexpected offset: atl_mpi_request size ",
                     sizeof(atl_mpi_req_t),
                     ", atl_request size ",
                     sizeof(atl_req_t),
                     ", expected offset ",
                     offsetof(atl_req_t, internal));

    int ret = MPI_SUCCESS;
    int is_tag_ub_set = 0;
    void* tag_ub_ptr = NULL;
    int required_thread_level = MPI_THREAD_MULTIPLE, provided_thread_level;

    if (atl_mpi_ctx::set_env(*attr)) {
        goto err_init;
    }

    MPI_Initialized(&ctx.is_external_init);
    if (!ctx.is_external_init) {
        ret = MPI_Init_thread(argc, argv, required_thread_level, &provided_thread_level);
        if (provided_thread_level < required_thread_level) {
            LOG_ERROR("unexpected MPI thread level: required ",
                      required_thread_level,
                      ", provided ",
                      provided_thread_level);
            goto err_init;
        }
    }
    else {
        LOG_DEBUG("MPI was initialized externaly");
        MPI_Query_thread(&provided_thread_level);
        if (provided_thread_level < required_thread_level) {
            LOG_WARN("MPI was initialized externaly but with unexpected thread level: "
                     "required ",
                     required_thread_level,
                     ", provided ",
                     provided_thread_level);
        }
    }

    if (ret)
        goto err_init;

    if (ctx.update_global_data(*attr) == ATL_STATUS_FAILURE) {
        goto err_init;
    }

    ctx.ep_count = attr->in.ep_count;

    char* progress_mode_env;
    progress_mode_env = getenv(ATL_PROGRESS_MODE_ENV);
    if (progress_mode_env) {
        ctx.progress_mode = (atl_progress_mode_t)atoi(progress_mode_env);
    }
    ctx.sync_coll = attr->in.enable_sync_coll;

    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &is_tag_ub_set);

    /* report actual attributes back to upper level */
    attr->out.enable_shm = 0;
    attr->out.enable_rma = 0;
    attr->out.enable_hmem = attr->in.enable_hmem & ctx.mpi_lib_attr.hmem;
    attr->out.mnic_type = ctx.mnic_type;
    attr->out.mnic_count = ctx.mnic_count;
    attr->out.tag_bits = 32;
    attr->out.max_tag = (is_tag_ub_set) ? *((int*)tag_ub_ptr) : 0;
    attr->out.max_order_waw_size = 0;

    return ATL_STATUS_SUCCESS;

err_init:
    return ATL_STATUS_FAILURE;
}

atl_status_t atl_mpi::send(atl_ep_t& ep,
                           const void* buf,
                           size_t len,
                           int dst_proc_idx,
                           uint64_t tag,
                           atl_req_t& req) {
    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    int ret = MPI_Isend(
        buf, len, MPI_CHAR, dst_proc_idx, (int)tag, mpi_ep->mpi_comm, &mpi_req->native_req);

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::recv(atl_ep_t& ep,
                           void* buf,
                           size_t len,
                           int src_proc_idx,
                           uint64_t tag,
                           atl_req_t& req) {
    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    int ret = MPI_Irecv(
        buf, len, MPI_CHAR, src_proc_idx, (int)tag, mpi_ep->mpi_comm, &mpi_req->native_req);

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::probe(atl_ep_t& ep,
                            int src_proc_idx,
                            uint64_t tag,
                            int* found,
                            size_t* recv_len) {
    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);

    int flag = 0, len = 0, ret;
    MPI_Status status;

    ret = MPI_Iprobe(src_proc_idx, tag, mpi_ep->mpi_comm, &flag, &status);
    if (flag) {
        MPI_Get_count(&status, MPI_BYTE, &len);
    }

    if (found)
        *found = flag;
    if (recv_len)
        *recv_len = len;

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::allgatherv(atl_ep_t& ep,
                                 const void* send_buf,
                                 size_t send_len,
                                 void* recv_buf,
                                 const int* recv_lens,
                                 const int* offsets,
                                 atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Allgatherv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             send_len,
                             MPI_CHAR,
                             recv_buf,
                             recv_lens,
                             offsets,
                             MPI_CHAR,
                             mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Iallgatherv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                              send_len,
                              MPI_CHAR,
                              recv_buf,
                              recv_lens,
                              offsets,
                              MPI_CHAR,
                              mpi_ep->mpi_comm,
                              &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::allreduce(atl_ep_t& ep,
                                const void* send_buf,
                                void* recv_buf,
                                size_t len,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Allreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            recv_buf,
                            len,
                            mpi_dtype,
                            mpi_op,
                            mpi_ep->mpi_comm);
    }
    else {
        //printf("atl_mpi: send_buf %p, recv_buf %p\n", send_buf, recv_buf);
        ret = MPI_Iallreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             recv_buf,
                             len,
                             mpi_dtype,
                             mpi_op,
                             mpi_ep->mpi_comm,
                             &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::alltoall(atl_ep_t& ep,
                               const void* send_buf,
                               void* recv_buf,
                               int len,
                               atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Alltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                           len,
                           MPI_CHAR,
                           recv_buf,
                           len,
                           MPI_CHAR,
                           mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Ialltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            len,
                            MPI_CHAR,
                            recv_buf,
                            len,
                            MPI_CHAR,
                            mpi_ep->mpi_comm,
                            &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::alltoallv(atl_ep_t& ep,
                                const void* send_buf,
                                const int* send_lens,
                                const int* send_offsets,
                                void* recv_buf,
                                const int* recv_lens,
                                const int* recv_offsets,
                                atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Alltoallv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            send_lens,
                            send_offsets,
                            MPI_CHAR,
                            recv_buf,
                            recv_lens,
                            recv_offsets,
                            MPI_CHAR,
                            mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Ialltoallv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             send_lens,
                             send_offsets,
                             MPI_CHAR,
                             recv_buf,
                             recv_lens,
                             recv_offsets,
                             MPI_CHAR,
                             mpi_ep->mpi_comm,
                             &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::barrier(atl_ep_t& ep, atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Barrier(mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Ibarrier(mpi_ep->mpi_comm, &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::bcast(atl_ep_t& ep, void* buf, size_t len, int root, atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Bcast(buf, len, MPI_CHAR, root, mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Ibcast(buf, len, MPI_CHAR, root, mpi_ep->mpi_comm, &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::reduce(atl_ep_t& ep,
                             const void* send_buf,
                             void* recv_buf,
                             size_t len,
                             int root,
                             atl_datatype_t dtype,
                             atl_reduction_t op,
                             atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    int my_proc_idx = ep.coord.global_idx;
    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    init_req(req);

    if (ctx.sync_coll) {
        ret = MPI_Reduce(
            (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            len,
            mpi_dtype,
            mpi_op,
            root,
            mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Ireduce(
            (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            len,
            mpi_dtype,
            mpi_op,
            root,
            mpi_ep->mpi_comm,
            &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::reduce_scatter(atl_ep_t& ep,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_len,
                                     atl_datatype_t dtype,
                                     atl_reduction_t op,
                                     atl_req_t& req) {
    int ret = MPI_SUCCESS;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    init_req(req);

    if (ctx.sync_coll) {
        ret =
            MPI_Reduce_scatter_block((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                                     recv_buf,
                                     recv_len,
                                     mpi_dtype,
                                     mpi_op,
                                     mpi_ep->mpi_comm);
    }
    else {
        ret = MPI_Ireduce_scatter_block(
            (send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            recv_len,
            mpi_dtype,
            mpi_op,
            mpi_ep->mpi_comm,
            &mpi_req->native_req);
    }

    check_ep(ep);

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::wait(atl_ep_t& ep, atl_req_t& req) {
    int ret;
    MPI_Status status;
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);
    ret = MPI_Wait(&mpi_req->native_req, &status);
    mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    req.is_completed = 1;
    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::wait_all(atl_ep_t& ep, std::vector<atl_req_t>& reqs, size_t count) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::cancel(atl_ep_t& ep, atl_req_t& req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::poll(atl_ep_t& ep) {
    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    if (ctx.progress_mode == ATL_PROGRESS_POLL) {
        return progress_ep(ep, &(mpi_ep->dummy_req));
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi::check(atl_ep_t& ep, atl_req_t& req) {
    atl_status_t status;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);

    CCL_THROW_IF_NOT(!req.is_completed, "request is already completed");
    CCL_THROW_IF_NOT(mpi_req->comp_state == ATL_MPI_COMP_POSTED, "request is already completed");

    if (mpi_req->native_req == MPI_REQUEST_NULL) {
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    }

    req.is_completed = (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED);
    if (req.is_completed) {
        return ATL_STATUS_SUCCESS;
    }

    status = progress_ep(ep, mpi_req);
    req.is_completed = (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED);

    return status;
}

atl_proc_coord_t atl_mpi::create_proc_coord(atl_ep_t& ep) {
    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    MPI_Comm comm = mpi_ep->mpi_comm;
    return create_proc_coord(comm);
}

atl_proc_coord_t atl_mpi::create_proc_coord(MPI_Comm comm) {
    atl_proc_coord_t res;

    MPI_Comm_rank(comm, (int*)&(res.global_idx));
    MPI_Comm_size(comm, (int*)&(res.global_count));

    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, res.global_count, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, (int*)&(res.local_idx));
    MPI_Comm_size(local_comm, (int*)&(res.local_count));
    MPI_Comm_free(&local_comm);

    char my_hostname[ATL_MAX_HOSTNAME_LEN] = { 0 };
    gethostname(my_hostname, ATL_MAX_HOSTNAME_LEN - 1);
    res.hostname_hash = std::hash<std::string>{}(my_hostname);

    res.validate();

    return res;
}

void atl_mpi::comms_free(std::vector<atl_ep_t>& eps) {
    for (size_t i = 0; i < eps.size(); i++) {
        atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)eps[i].internal);
        if (ctx.progress_mode == ATL_PROGRESS_POLL) {
            MPI_Cancel(&(mpi_ep->dummy_req.native_req));
            MPI_Comm_free(&mpi_ep->dummy_comm);
        }
        MPI_Comm_free(&mpi_ep->mpi_comm);
    }
}

atl_status_t atl_mpi::comm_split(const std::vector<atl_ep_t>& base_eps,
                                 std::vector<atl_ep_t>& eps,
                                 size_t color,
                                 int local_idx) {
    int ret = 0;
    for (size_t idx = 0; idx < ctx.ep_count; idx++) {
        atl_ep_t ep;

        atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
        atl_mpi_ep_t* base_mpi_ep = ((atl_mpi_ep_t*)base_eps[idx].internal);

        ssize_t mpi_ep_idx = get_ep_idx(idx);
        char mpi_ep_idx_str[MPI_MAX_INFO_VAL] = { 0 };

        size_t nic_idx = 0;
        char nic_idx_str[MPI_MAX_INFO_VAL] = { 0 };

        ret = MPI_Comm_split(base_mpi_ep->mpi_comm, color, 0, &mpi_ep->mpi_comm);
        if (ret) {
            LOG_ERROR("MPI_Comm_split error, ep_idx ", idx);
            break;
        }

        MPI_Info info;
        MPI_Info_create(&info);

        /* set EP index */
        snprintf(mpi_ep_idx_str, MPI_MAX_INFO_VAL, "%zd", mpi_ep_idx);
        MPI_Info_set(info, ctx.EP_IDX_KEY, mpi_ep_idx_str);

        /* pre-requisite for pref-nic hint */
        MPI_Info_set(info, "mpi_assert_no_any_source", "true");
        MPI_Info_set(info, "mpi_assert_no_any_tag", "true");

        if (ctx.mnic_type != ATL_MNIC_NONE) {
            /* set NIC index */
            nic_idx = idx;
            if (ctx.mnic_offset == ATL_MNIC_OFFSET_LOCAL_PROC_IDX) {
                nic_idx += local_idx;
            }
            nic_idx %= ctx.mnic_count;
            snprintf(nic_idx_str, MPI_MAX_INFO_VAL, "%zu", nic_idx);
            MPI_Info_set(info, ctx.NIC_IDX_KEY, nic_idx_str);

            LOG_DEBUG(
                "select nic: ep_idx ", idx, ", local_proc_idx ", local_idx, ", nic_idx ", nic_idx);
        }

        MPI_Comm_set_info(mpi_ep->mpi_comm, info);

        if (ctx.progress_mode == ATL_PROGRESS_POLL) {
            ret = MPI_Comm_split(base_mpi_ep->dummy_comm, color, 0, &mpi_ep->dummy_comm);
            if (ret) {
                LOG_ERROR("MPI_Comm_split error, ep_idx ", idx);
                break;
            }
            MPI_Comm_set_info(mpi_ep->dummy_comm, info);
            MPI_Irecv(NULL, 0, MPI_CHAR, 0, 0, mpi_ep->dummy_comm, &(mpi_ep->dummy_req.native_req));

            check_comm_ep_idx(mpi_ep->dummy_comm, mpi_ep_idx);
            if (ctx.mnic_type != ATL_MNIC_NONE) {
                check_comm_nic_idx(mpi_ep->dummy_comm, nic_idx);
            }
        }

        MPI_Info_free(&info);

        check_comm_ep_idx(mpi_ep->mpi_comm, mpi_ep_idx);
        if (ctx.mnic_type != ATL_MNIC_NONE) {
            check_comm_nic_idx(mpi_ep->mpi_comm, nic_idx);
        }

        LOG_DEBUG("atl-mpi-ep: ", idx, ", ep_idx ", mpi_ep_idx, ", nic_idx ", nic_idx);

        ep.idx = idx;
        eps.push_back(ep);
    }

    if (ret) {
        comms_free(eps);
        ctx.bf16_finalize();
        ctx.fp16_finalize();
        if (!ctx.is_external_init) {
            MPI_Finalize();
        }
    }

    return ATL_MPI_RET(ret);
}

std::string atl_mpi::to_string() {
    std::stringstream ss;
    ss << "atl-mpi:\n" << ctx.to_string();
    return ss.str();
}

atl_status_t atl_mpi::finalize(int global_idx) {
    CCL_THROW_IF_NOT(!is_finalized, "atl_mpi refinalize is not expected");
    is_finalized = true;
    inited = false;

    int ret = MPI_SUCCESS;

    if (global_idx == 0) {
        LOG_INFO("finalize atl-mpi");
    }

    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);
    if (!is_mpi_finalized) {
        ctx.bf16_finalize();
        ctx.fp16_finalize();
        if (!ctx.is_external_init) {
            ret = MPI_Finalize();
        }
        else {
            LOG_DEBUG("MPI_Init has been called externally, skip MPI_Finalize");
        }

        if (global_idx == 0) {
            LOG_INFO("finalized atl-mpi");
        }
    }
    else {
        if (global_idx == 0) {
            LOG_WARN("MPI_Finalize has been called before CCL finalization");
        }
    }

    return ATL_MPI_RET(ret);
}

atl_status_t atl_mpi::comm_create(int comm_size,
                                  const std::vector<int>& comm_ranks,
                                  std::shared_ptr<ipmi> pmi,
                                  MPI_Comm* new_comm) {
    MPI_Group world_group, new_group;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int my_mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);
    std::string my_mpi_rank_str = std::to_string(my_mpi_rank);

    int my_pmi_rank = pmi->get_rank();

    std::vector<int> mpi_ranks(comm_size);
    ATL_CHECK_STATUS(pmi->pmrt_kvs_put((char*)ATL_MPI_RANK_INFO_PM_KEY,
                                       my_pmi_rank,
                                       my_mpi_rank_str.c_str(),
                                       ATL_MPI_RANK_STR_SIZE),
                     "pmrt_kvs_put: error");

    char returned_mpi_rank[ATL_MPI_RANK_STR_SIZE];
    for (int i = 0; i < comm_size; ++i) {
        ATL_CHECK_STATUS(
            pmi->pmrt_kvs_get(
                (char*)ATL_MPI_RANK_INFO_PM_KEY, i, returned_mpi_rank, ATL_MPI_RANK_STR_SIZE),
            "pmrt_kvs_get: error");
        mpi_ranks[i] = std::atoi(returned_mpi_rank);
    }

    MPI_Group_incl(world_group, comm_size, mpi_ranks.data(), &new_group);
    int ret = MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, new_comm);
    if (ret) {
        LOG_ERROR("MPI_Comm_create_group error");
        return ATL_STATUS_FAILURE;
    }
    if (*new_comm == MPI_COMM_NULL) {
        LOG_ERROR("MPI_Comm_create_group error, new_comm == MPI_COMM_NULL");
        return ATL_STATUS_FAILURE;
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi::ep_init(std::vector<atl_ep_t>& eps, MPI_Comm global_comm, int local_idx) {
    atl_ep_t base_ep;
    base_ep.idx = 0;

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)base_ep.internal);
    mpi_ep->mpi_comm = global_comm;
    mpi_ep->dummy_comm = global_comm;

    std::vector<atl_ep_t> base_eps(ctx.ep_count, base_ep);
    return comm_split(base_eps, eps, 0, local_idx);
}

void atl_mpi::set_env(const atl_attr_t& attr) {
    atl_mpi_ctx::set_env(attr);
}

atl_mpi_env_info_t atl_mpi::get_env_info(const char* key) {
    atl_mpi_env_info_t res;
    snprintf(res.key, MPI_MAX_INFO_KEY, "%s", key);
    MPI_Info_get(MPI_INFO_ENV, key, MPI_MAX_INFO_VAL, res.value, &res.found);
    return res;
}

atl_mpi_comm_info_t atl_mpi::get_comm_info(MPI_Comm comm, const char* key) {
    MPI_Info info;
    atl_mpi_comm_info_t res;

    res.comm = comm;
    snprintf(res.key, MPI_MAX_INFO_KEY, "%s", key);

    MPI_Comm_get_info(res.comm, &info);
    MPI_Info_get(info, key, MPI_MAX_INFO_VAL, res.value, &res.found);
    MPI_Info_free(&info);

    return res;
}

MPI_Datatype atl_mpi::atl2mpi_dtype(atl_datatype_t dtype) {
    switch (dtype) {
        case ATL_DTYPE_INT8: return MPI_CHAR;
        case ATL_DTYPE_UINT8: return MPI_UNSIGNED_CHAR;
        case ATL_DTYPE_INT16: return MPI_INT16_T;
        case ATL_DTYPE_UINT16: return MPI_UINT16_T;
        case ATL_DTYPE_INT32: return MPI_INT;
        case ATL_DTYPE_UINT32: return MPI_UINT32_T;
        case ATL_DTYPE_INT64: return MPI_LONG_LONG;
        case ATL_DTYPE_UINT64: return MPI_UNSIGNED_LONG_LONG;
        case ATL_DTYPE_FLOAT16: return MPI_FLOAT16;
        case ATL_DTYPE_FLOAT32: return MPI_FLOAT;
        case ATL_DTYPE_FLOAT64: return MPI_DOUBLE;
        case ATL_DTYPE_BFLOAT16: return MPI_BFLOAT16;
        default: printf("unknown datatype: %d\n", dtype); exit(1);
    }
}

MPI_Op atl_mpi::atl2mpi_op(atl_reduction_t rtype, MPI_Datatype dtype) {
    if (dtype == ctx.bf16.dtype) {
        return ctx.atl2mpi_op_bf16(rtype);
    }

#ifdef ATL_MPI_FP16
    if (dtype == ctx.fp16.dtype) {
        return ctx.atl2mpi_op_fp16(rtype);
    }
#endif // ATL_MPI_FP16

    (void)dtype;
    switch (rtype) {
        case ATL_REDUCTION_SUM: return MPI_SUM;
        case ATL_REDUCTION_PROD: return MPI_PROD;
        case ATL_REDUCTION_MIN: return MPI_MIN;
        case ATL_REDUCTION_MAX: return MPI_MAX;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
}

void atl_mpi::init_req(atl_req_t& req) {
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req.internal);
    mpi_req->native_req = MPI_REQUEST_NULL;
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    req.is_completed = 0;
}

inline atl_status_t atl_mpi::progress_ep(atl_ep_t& ep, atl_mpi_req_t* req) {
    int flag = 0;
    int ret = MPI_Test(&req->native_req, &flag, MPI_STATUS_IGNORE);

    if (flag) {
        req->comp_state = ATL_MPI_COMP_COMPLETED;
    }

    return ATL_MPI_RET(ret);
}

void atl_mpi::check_comm_nic_idx(MPI_Comm comm, size_t expected_idx) {
    char expected_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(expected_idx_str, MPI_MAX_INFO_VAL, "%zu", expected_idx);
    check_comm_info(comm, ctx.NIC_IDX_KEY, expected_idx_str);
}

void atl_mpi::check_comm_ep_idx(MPI_Comm comm, size_t expected_idx) {
    if (ctx.mpi_lib_attr.type == ctx.ATL_MPI_LIB_NONE)
        return;

    char expected_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(expected_idx_str, MPI_MAX_INFO_VAL, "%zu", expected_idx);
    check_comm_info(comm, ctx.EP_IDX_KEY, expected_idx_str);
}

void atl_mpi::check_comm_info(MPI_Comm comm, const char* key, const char* expected_value) {
    atl_mpi_comm_info_t info = atl_mpi::get_comm_info(comm, key);

    CCL_THROW_IF_NOT(info.found, "MPI comm key ", key, " was not set");
    CCL_THROW_IF_NOT(!strcmp(info.value, expected_value),
                     "MPI comm key ",
                     key,
                     ": expected: ",
                     expected_value,
                     ", read: ",
                     info.value);
}

void atl_mpi::check_ep(atl_ep_t& ep) {
#ifdef ENABLE_DEBUG
    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)ep.internal);
    check_comm_ep_idx(mpi_ep->mpi_comm, get_ep_idx(ep.idx));
#endif // ENABLE_DEBUG
}

size_t atl_mpi::get_ep_idx(size_t ep_idx) {
    size_t mpi_ep_idx = ep_idx;
    if (ctx.extra_ep)
        mpi_ep_idx += ctx.extra_ep;
    return mpi_ep_idx;
}

#endif // CCL_ENABLE_MPI
