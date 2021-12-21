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

#include "atl_def.h"
#include "atl_mpi.hpp"

#define MPI_BFLOAT16 \
    ({ \
        CCL_THROW_IF_NOT(global_data.bf16.dtype != MPI_DATATYPE_NULL, \
                         "unsupported datatype: ATL_DTYPE_BF16"); \
        global_data.bf16.dtype; \
    })

#define MPI_FLOAT16 \
    ({ \
        CCL_THROW_IF_NOT(global_data.fp16.dtype != MPI_DATATYPE_NULL, \
                         "unsupported datatype: ATL_DTYPE_FP16"); \
        global_data.fp16.dtype; \
    })

#define RET2ATL(ret) (ret != MPI_SUCCESS) ? ATL_STATUS_FAILURE : ATL_STATUS_SUCCESS

atl_mpi_global_data atl_mpi::global_data{};

atl_status_t atl_mpi::init(int* argc,
                           char*** argv,
                           atl_attr_t* attr,
                           const char* main_addr,
                           std::shared_ptr<ipmi> pmi) {
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

    if (global_data.ctx_count == 0) {
        if (global_data.set_env(*attr)) {
            goto err_init;
        }

        MPI_Initialized(&global_data.is_external_init);

        if (!global_data.is_external_init) {
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

        if (global_data.update_global_data(attr) == ATL_STATUS_FAILURE) {
            goto err_init;
        }
    }
    global_data.ctx_count++;

    coord_update(MPI_COMM_WORLD, global_coord);

    ep_count = attr->in.ep_count;

    char* progress_mode_env;
    progress_mode_env = getenv(ATL_PROGRESS_MODE_ENV);
    if (progress_mode_env) {
        progress_mode = (atl_progress_mode_t)atoi(progress_mode_env);
    }
    else {
        progress_mode = ATL_PROGRESS_CHECK;
    }
    sync_coll = attr->in.enable_sync_coll;

    if (global_coord.global_idx == 0) {
        global_data.print_log_info();
        LOG_INFO("atl-mpi-ctx: ", (global_data.ctx_count - 1));
        LOG_INFO("  progress_mode: ", progress_mode);
        LOG_INFO("  sync_coll: ", sync_coll);
    }

    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &is_tag_ub_set);

    /* report actual attributes back to upper level */
    attr->out.enable_shm = 0;
    attr->out.enable_rma = 0;
    attr->out.enable_hmem = attr->in.enable_hmem & global_data.mpi_lib_attr.hmem;
    attr->out.mnic_type = global_data.mnic_type;
    attr->out.mnic_count = global_data.mnic_count;
    attr->out.tag_bits = 32;
    attr->out.max_tag = (is_tag_ub_set) ? *((int*)tag_ub_ptr) : 0;
    attr->out.max_order_waw_size = 0;

    return ATL_STATUS_SUCCESS;

err_init:
    return ATL_STATUS_FAILURE;
}

void atl_mpi::coord_update(MPI_Comm base_comm, atl_proc_coord_t& coord) {
    MPI_Comm_rank(base_comm, (int*)&(coord.global_idx));
    MPI_Comm_size(base_comm, (int*)&(coord.global_count));

    MPI_Comm local_comm;
    MPI_Comm_split_type(
        base_comm, MPI_COMM_TYPE_SHARED, coord.global_count, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, (int*)&(coord.local_idx));
    MPI_Comm_size(local_comm, (int*)&(coord.local_count));
    MPI_Comm_free(&local_comm);

    char my_hostname[ATL_MAX_HOSTNAME_LEN] = { 0 };
    gethostname(my_hostname, ATL_MAX_HOSTNAME_LEN - 1);
    coord.hostname_hash = std::hash<std::string>{}(my_hostname);
}

void atl_mpi::comms_free(std::vector<atl_mpi_ep_t>& eps) {
    for (size_t i = 0; i < eps.size(); i++) {
        atl_mpi_ep_t& mpi_ep = eps[i];

        if (progress_mode == ATL_PROGRESS_POLL) {
            MPI_Cancel(&(mpi_ep.dummy_req.native_req));
            MPI_Comm_free(&mpi_ep.dummy_comm);
        }
        MPI_Comm_free(&mpi_ep.mpi_comm);
    }
}

atl_status_t atl_mpi::finalize() {
    is_finalized = true;

    int ret = MPI_SUCCESS;

    global_data.ctx_count--;
    if (global_coord.global_idx == 0) {
        LOG_INFO("finalize atl-mpi ctx, remaining ctx_count ", global_data.ctx_count);
    }

    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);

    if (!is_mpi_finalized) {
        if (global_data.ctx_count == 0) {
            global_data.bf16_finalize();
            global_data.fp16_finalize();
            if (!global_data.is_external_init) {
                ret = MPI_Finalize();
            }
            else {
                LOG_DEBUG("MPI_Init has been called externally, skip MPI_Finalize");
            }

            if (global_coord.global_idx == 0) {
                LOG_INFO("finalized last atl-mpi ctx");
            }
        }
    }
    else {
        if ((global_data.ctx_count == 0) && (global_coord.global_idx == 0)) {
            LOG_WARN("MPI_Finalize has been called before CCL finalization");
        }
    }

    return RET2ATL(ret);
}

atl_status_t atl_mpi::update(std::shared_ptr<ipmi> pmi) {
    (void)pmi;
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::mr_reg(const void* buf, size_t len, atl_mr_t** mr) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::mr_dereg(atl_mr_t* mr) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::send(atl_mpi_ep_t& ep,
                           const void* buf,
                           size_t len,
                           int dst_proc_idx,
                           uint64_t tag,
                           atl_req_t* req) {
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    int ret =
        MPI_Isend(buf, len, MPI_CHAR, dst_proc_idx, (int)tag, ep.mpi_comm, &mpi_req->native_req);

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::recv(atl_mpi_ep_t& ep,
                           void* buf,
                           size_t len,
                           int src_proc_idx,
                           uint64_t tag,
                           atl_req_t* req) {
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    int ret =
        MPI_Irecv(buf, len, MPI_CHAR, src_proc_idx, (int)tag, ep.mpi_comm, &mpi_req->native_req);

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::probe(atl_mpi_ep_t& ep,
                            int src_proc_idx,
                            uint64_t tag,
                            int* found,
                            size_t* recv_len) {
    int flag = 0, len = 0, ret;
    MPI_Status status;

    ret = MPI_Iprobe(src_proc_idx, tag, ep.mpi_comm, &flag, &status);
    if (flag) {
        MPI_Get_count(&status, MPI_BYTE, &len);
    }

    if (found)
        *found = flag;
    if (recv_len)
        *recv_len = len;

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::allgatherv(atl_mpi_ep_t& ep,
                                 const void* send_buf,
                                 size_t send_len,
                                 void* recv_buf,
                                 const int* recv_lens,
                                 const int* offsets,
                                 atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Allgatherv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             send_len,
                             MPI_CHAR,
                             recv_buf,
                             recv_lens,
                             offsets,
                             MPI_CHAR,
                             ep.mpi_comm);
    }
    else {
        ret = MPI_Iallgatherv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                              send_len,
                              MPI_CHAR,
                              recv_buf,
                              recv_lens,
                              offsets,
                              MPI_CHAR,
                              ep.mpi_comm,
                              &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::allreduce(atl_mpi_ep_t& ep,
                                const void* send_buf,
                                void* recv_buf,
                                size_t len,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Allreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            recv_buf,
                            len,
                            mpi_dtype,
                            mpi_op,
                            ep.mpi_comm);
    }
    else {
        //printf("atl_mpi: send_buf %p, recv_buf %p\n", send_buf, recv_buf);
        ret = MPI_Iallreduce((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                             recv_buf,
                             len,
                             mpi_dtype,
                             mpi_op,
                             ep.mpi_comm,
                             &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::alltoall(atl_mpi_ep_t& ep,
                               const void* send_buf,
                               void* recv_buf,
                               int len,
                               atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Alltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                           len,
                           MPI_CHAR,
                           recv_buf,
                           len,
                           MPI_CHAR,
                           ep.mpi_comm);
    }
    else {
        ret = MPI_Ialltoall((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            len,
                            MPI_CHAR,
                            recv_buf,
                            len,
                            MPI_CHAR,
                            ep.mpi_comm,
                            &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::alltoallv(atl_mpi_ep_t& ep,
                                const void* send_buf,
                                const int* send_lens,
                                const int* send_offsets,
                                void* recv_buf,
                                const int* recv_lens,
                                const int* recv_offsets,
                                atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Alltoallv((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                            send_lens,
                            send_offsets,
                            MPI_CHAR,
                            recv_buf,
                            recv_lens,
                            recv_offsets,
                            MPI_CHAR,
                            ep.mpi_comm);
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
                             ep.mpi_comm,
                             &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::barrier(atl_mpi_ep_t& ep, atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Barrier(ep.mpi_comm);
    }
    else {
        ret = MPI_Ibarrier(ep.mpi_comm, &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::bcast(atl_mpi_ep_t& ep, void* buf, size_t len, int root, atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Bcast(buf, len, MPI_CHAR, root, ep.mpi_comm);
    }
    else {
        ret = MPI_Ibcast(buf, len, MPI_CHAR, root, ep.mpi_comm, &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::reduce(atl_mpi_ep_t& ep,
                             const void* send_buf,
                             void* recv_buf,
                             size_t len,
                             int root,
                             atl_datatype_t dtype,
                             atl_reduction_t op,
                             atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    int my_proc_idx = ep.coord->global_idx;
    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    init_req(req);

    if (sync_coll) {
        ret = MPI_Reduce(
            (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            len,
            mpi_dtype,
            mpi_op,
            root,
            ep.mpi_comm);
    }
    else {
        ret = MPI_Ireduce(
            (send_buf && (send_buf == recv_buf) && (root == my_proc_idx)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            len,
            mpi_dtype,
            mpi_op,
            root,
            ep.mpi_comm,
            &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::reduce_scatter(atl_mpi_ep_t& ep,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_len,
                                     atl_datatype_t dtype,
                                     atl_reduction_t op,
                                     atl_req_t* req) {
    int ret = MPI_SUCCESS;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    MPI_Datatype mpi_dtype = atl2mpi_dtype(dtype);
    MPI_Op mpi_op = atl2mpi_op(op, mpi_dtype);

    init_req(req);

    if (sync_coll) {
        ret =
            MPI_Reduce_scatter_block((send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
                                     recv_buf,
                                     recv_len,
                                     mpi_dtype,
                                     mpi_op,
                                     ep.mpi_comm);
    }
    else {
        ret = MPI_Ireduce_scatter_block(
            (send_buf && (send_buf == recv_buf)) ? MPI_IN_PLACE : send_buf,
            recv_buf,
            recv_len,
            mpi_dtype,
            mpi_op,
            ep.mpi_comm,
            &mpi_req->native_req);
    }

    check_ep(ep);

    return RET2ATL(ret);
}

atl_status_t atl_mpi::read(atl_mpi_ep_t& ep,
                           void* buf,
                           size_t len,
                           atl_mr_t* mr,
                           uint64_t addr,
                           uintptr_t remote_key,
                           int dst_proc_idx,
                           atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::write(atl_mpi_ep_t& ep,
                            const void* buf,
                            size_t len,
                            atl_mr_t* mr,
                            uint64_t addr,
                            uintptr_t remote_key,
                            int dst_proc_idx,
                            atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::wait(atl_mpi_ep_t& ep, atl_req_t* req) {
    int ret;
    MPI_Status status;
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    ret = MPI_Wait(&mpi_req->native_req, &status);
    mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    return RET2ATL(ret);
}

atl_status_t atl_mpi::wait_all(atl_mpi_ep_t& ep, atl_req_t* req, size_t count) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::cancel(atl_mpi_ep_t& ep, atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::poll(atl_mpi_ep_t& ep) {
    if (progress_mode == ATL_PROGRESS_POLL) {
        return ep_progress(ep, &(ep.dummy_req));
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi::check(atl_mpi_ep_t& ep, atl_req_t* req) {
    atl_status_t status;

    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);

    CCL_THROW_IF_NOT(!req->is_completed, "request is already completed");
    CCL_THROW_IF_NOT(mpi_req->comp_state == ATL_MPI_COMP_POSTED, "request is already completed");

    if (mpi_req->native_req == MPI_REQUEST_NULL) {
        mpi_req->comp_state = ATL_MPI_COMP_COMPLETED;
    }

    req->is_completed = (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED);
    if (req->is_completed) {
        return ATL_STATUS_SUCCESS;
    }

    status = ep_progress(ep, mpi_req);
    req->is_completed = (mpi_req->comp_state == ATL_MPI_COMP_COMPLETED);

    return status;
}

atl_mpi::~atl_mpi() {
    if (!is_finalized)
        finalize();
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

inline atl_status_t atl_mpi::ep_progress(atl_mpi_ep_t& ep, atl_mpi_req_t* req) {
    int flag = 0;
    int ret = MPI_Test(&req->native_req, &flag, MPI_STATUS_IGNORE);

    if (flag) {
        req->comp_state = ATL_MPI_COMP_COMPLETED;
    }

    return RET2ATL(ret);
}

void atl_mpi::init_req(atl_req_t* req) {
    atl_mpi_req_t* mpi_req = ((atl_mpi_req_t*)req->internal);
    mpi_req->native_req = MPI_REQUEST_NULL;
    mpi_req->comp_state = ATL_MPI_COMP_POSTED;
    req->is_completed = 0;
}

MPI_Op atl_mpi::atl2mpi_op(atl_reduction_t rtype, MPI_Datatype dtype) {
#ifdef ATL_MPI_BF16
    if (dtype == global_data.bf16.dtype)
        return global_data.atl2mpi_op_bf16(rtype);
#endif // ATL_MPI_BF16

#ifdef ATL_MPI_FP16
    if (dtype == global_data.fp16.dtype)
        return global_data.atl2mpi_op_fp16(rtype);
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

size_t atl_mpi::get_ep_idx(size_t ep_idx) {
    size_t mpi_ep_idx = ep_idx;
    if (global_data.extra_ep)
        mpi_ep_idx += global_data.extra_ep;
    return mpi_ep_idx;
}

atl_status_t atl_mpi::ep_init(std::vector<atl_mpi_ep_t>& eps) {
    atl_mpi_ep_t base_ep;
    base_ep.mpi_comm = MPI_COMM_WORLD;
    base_ep.dummy_comm = MPI_COMM_WORLD;
    base_ep.idx = 0;
    base_ep.coord = nullptr;
    std::vector<atl_mpi_ep_t> base_eps(ep_count, base_ep);
    return comm_split(base_eps, eps, 0);
}

#ifdef ENABLE_DEBUG
void atl_mpi::check_ep(atl_mpi_ep_t& ep) {
    check_comm_ep_idx(ep.mpi_comm, get_ep_idx(ep.idx));
}
#endif // ENABLE_DEBUG

void atl_mpi::check_comm_nic_idx(MPI_Comm comm, size_t expected_idx) {
    char expected_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(expected_idx_str, MPI_MAX_INFO_VAL, "%zu", expected_idx);
    check_comm_info(comm, global_data.NIC_IDX_KEY, expected_idx_str);
}

void atl_mpi::check_comm_ep_idx(MPI_Comm comm, size_t expected_idx) {
    if (global_data.mpi_lib_attr.type == global_data.ATL_MPI_LIB_NONE)
        return;

    char expected_idx_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(expected_idx_str, MPI_MAX_INFO_VAL, "%zu", expected_idx);
    check_comm_info(comm, global_data.EP_IDX_KEY, expected_idx_str);
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

void atl_mpi::set_env(const atl_attr_t& attr) {
    global_data.set_env(attr);
}

atl_status_t atl_mpi::comm_split(const std::vector<atl_mpi_ep_t>& base_eps,
                                 std::vector<atl_mpi_ep_t>& eps,
                                 size_t color) {
    int ret;
    atl_mpi_ep_t ep;
    for (size_t idx = 0; idx < ep_count; idx++) {
        ssize_t mpi_ep_idx = get_ep_idx(idx);
        char mpi_ep_idx_str[MPI_MAX_INFO_VAL] = { 0 };

        size_t nic_idx = 0;
        char nic_idx_str[MPI_MAX_INFO_VAL] = { 0 };

        ret = MPI_Comm_split(base_eps[idx].mpi_comm, color, 0, &ep.mpi_comm);
        if (ret) {
            LOG_ERROR("MPI_Comm_split error, ep_idx ", idx);
            break;
        }

        MPI_Info info;
        MPI_Info_create(&info);

        /* set EP index */
        snprintf(mpi_ep_idx_str, MPI_MAX_INFO_VAL, "%zu", mpi_ep_idx);
        MPI_Info_set(info, global_data.EP_IDX_KEY, mpi_ep_idx_str);

        if (global_data.mnic_type != ATL_MNIC_NONE) {
            /* set NIC index */
            nic_idx = idx;
            if (global_data.mnic_offset == ATL_MNIC_OFFSET_LOCAL_PROC_IDX) {
                nic_idx += global_coord.local_idx;
            }
            nic_idx %= global_data.mnic_count;
            snprintf(nic_idx_str, MPI_MAX_INFO_VAL, "%zu", nic_idx);
            MPI_Info_set(info, global_data.NIC_IDX_KEY, nic_idx_str);

            LOG_INFO("select nic: ep_idx ",
                     idx,
                     ", local_proc_idx ",
                     global_coord.local_idx,
                     ", nic_idx ",
                     nic_idx);
        }

        MPI_Comm_set_info(ep.mpi_comm, info);

        if (progress_mode == ATL_PROGRESS_POLL) {
            ret = MPI_Comm_split(base_eps[idx].dummy_comm, color, 0, &ep.dummy_comm);
            if (ret) {
                LOG_ERROR("MPI_Comm_split error, ep_idx ", idx);
                break;
            }
            MPI_Comm_set_info(ep.dummy_comm, info);
            MPI_Irecv(NULL, 0, MPI_CHAR, 0, 0, ep.dummy_comm, &(ep.dummy_req.native_req));

            check_comm_ep_idx(ep.dummy_comm, mpi_ep_idx);
            if (global_data.mnic_type != ATL_MNIC_NONE) {
                check_comm_nic_idx(ep.dummy_comm, nic_idx);
            }
        }

        MPI_Info_free(&info);

        check_comm_ep_idx(ep.mpi_comm, mpi_ep_idx);
        if (global_data.mnic_type != ATL_MNIC_NONE) {
            check_comm_nic_idx(ep.mpi_comm, nic_idx);
        }

        LOG_DEBUG("atl-mpi-ep: ", idx, ", ep_idx ", mpi_ep_idx, ", nic_idx ", nic_idx);

        ep.idx = idx;
        eps.push_back(ep);
    }

    if (ret) {
        comms_free(eps);
        global_data.ctx_count--;
        if (global_data.ctx_count == 0) {
            global_data.bf16_finalize();
            global_data.fp16_finalize();
            if (!global_data.is_external_init) {
                MPI_Finalize();
            }
        }
    }

    return RET2ATL(ret);
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

#endif // CCL_ENABLE_MPI
