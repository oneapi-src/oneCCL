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
#include "oneapi/ccl/type_traits.hpp"
#include "coll/algorithms/sparse_allreduce/sparse_handler.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_COALESCE_RESERVE_SIZE 16

#define CCL_BF16_ONE 0x3f80
#define CCL_BF16_MAX 0x7f7f
#define CCL_BF16_MIN 0xff7f

#define CCL_SPARSE_ALLREDUCE_SELECT_ALGO(itype, vtype, algo) \
    do { \
        switch (algo) { \
            case ccl_coll_sparse_allreduce_ring: \
                CCL_CALL((ccl_coll_build_sparse_allreduce_ring<itype, vtype>(sched, \
                                                                             send_ind_buf, \
                                                                             send_ind_count, \
                                                                             send_val_buf, \
                                                                             send_val_count, \
                                                                             recv_ind_buf, \
                                                                             recv_ind_count, \
                                                                             recv_val_buf, \
                                                                             recv_val_count, \
                                                                             index_dtype, \
                                                                             value_dtype, \
                                                                             reduction, \
                                                                             comm))); \
                break; \
            case ccl_coll_sparse_allreduce_mask: \
                CCL_CALL((ccl_coll_build_sparse_allreduce_mask<itype, vtype>(sched, \
                                                                             send_ind_buf, \
                                                                             send_ind_count, \
                                                                             send_val_buf, \
                                                                             send_val_count, \
                                                                             recv_ind_buf, \
                                                                             recv_ind_count, \
                                                                             recv_val_buf, \
                                                                             recv_val_count, \
                                                                             index_dtype, \
                                                                             value_dtype, \
                                                                             reduction, \
                                                                             comm))); \
                break; \
            case ccl_coll_sparse_allreduce_3_allgatherv: \
                CCL_CALL( \
                    (ccl_coll_build_sparse_allreduce_3_allgatherv<itype, vtype>(sched, \
                                                                                send_ind_buf, \
                                                                                send_ind_count, \
                                                                                send_val_buf, \
                                                                                send_val_count, \
                                                                                recv_ind_buf, \
                                                                                recv_ind_count, \
                                                                                recv_val_buf, \
                                                                                recv_val_count, \
                                                                                index_dtype, \
                                                                                value_dtype, \
                                                                                reduction, \
                                                                                comm))); \
                break; \
            default: \
                CCL_FATAL("unexpected sparse_allreduce_algo ", ccl_coll_algorithm_to_str(algo)); \
                return ccl::status::invalid_arguments; \
        } \
    } while (0)

#define CCL_SPARSE_ALLREDUCE_SELECT_V_DTYPE(itype, vtype, algo) \
    do { \
        switch (vtype.idx()) { \
            case ccl::datatype::float32: \
                CCL_SPARSE_ALLREDUCE_SELECT_ALGO(itype, float, algo); \
                break; \
            case ccl::datatype::bfloat16: \
                CCL_SPARSE_ALLREDUCE_SELECT_ALGO(itype, ccl::bfloat16, algo); \
                break; \
            default: \
                CCL_FATAL("value datatype ", \
                          ccl::global_data::get().dtypes->name(vtype), \
                          " is not supported yet"); \
                return ccl::status::invalid_arguments; \
        } \
    } while (0)

/* TODO: used for ring and mask, refactor to work with dst_ibuf, dst_vbuf */
#define CCL_SPARSE_ALLREDUCE_IF_SINGLE_RANK() \
    ({ \
        if (sa_handler->comm_size == 1) { \
            *sa_handler->recv_icount = iv_map_cnt; \
            *sa_handler->recv_vcount = iv_map_cnt * sa_handler->val_dim_cnt; \
            *sa_handler->recv_ibuf = sa_handler->dst_buf; \
            *sa_handler->recv_vbuf = \
                (char*)sa_handler->dst_buf + sa_handler->itype_size * iv_map_cnt; \
        } \
    })

#define CCL_SPARSE_ALLREDUCE_CREATE_HANDLER() \
    do { \
        /* create handler for sched function callbacks */ \
        sa_handler = static_cast<ccl_sparse_allreduce_handler*>( \
            sched->alloc_buffer(sizeof(ccl_sparse_allreduce_handler)).get_ptr()); \
\
        sa_handler->comm = comm; \
        sa_handler->comm_size = comm_size; \
        sa_handler->val_dim_cnt = val_dim_cnt; \
        sa_handler->itype_size = itype_size; \
        sa_handler->vtype_size = vtype_size; \
        sa_handler->index_dtype = index_dtype; \
        sa_handler->value_dtype = value_dtype; \
        sa_handler->op = op; \
        sa_handler->recv_ibuf = r_ind_buf; \
        sa_handler->recv_vbuf = r_val_buf; \
        sa_handler->recv_vcount = recv_val_count; \
        sa_handler->recv_icount = recv_ind_count; \
        sa_handler->sched = sched; \
\
        sa_handler->size_per_rank = \
            static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) * comm_size).get_ptr()); \
\
        for (int i = 0; i < comm_size; i++) \
            sa_handler->size_per_rank[i] = sizeof(size_t); \
\
        sa_handler->send_ibuf = send_ind_buf.get_ptr(); \
        sa_handler->send_vbuf = send_val_buf.get_ptr(); \
\
        sa_handler->send_count[0] = send_ind_count; \
        sa_handler->send_count[1] = send_val_count; \
\
        if (sa_handler->sched->coll_attr.sparse_coalesce_mode == \
                ccl::sparse_coalesce_mode::keep_precision && \
            sa_handler->value_dtype.idx() == ccl::datatype::bfloat16) { \
            sa_handler->tmp = \
                static_cast<float*>(sched->alloc_buffer(sizeof(float) * val_dim_cnt).get_ptr()); \
            sa_handler->acc = \
                static_cast<float*>(sched->alloc_buffer(sizeof(float) * val_dim_cnt).get_ptr()); \
        } \
        else { \
            sa_handler->tmp = nullptr; \
            sa_handler->acc = nullptr; \
        } \
    } while (0)

#define CCL_SPARSE_ALLREDUCE_ADD_NNZ_ENTRY() \
    do { \
        ccl_coll_entry_param param_nnz{}; \
        param_nnz.ctype = ccl_coll_allgatherv; \
        param_nnz.send_buf = ccl_buffer(sa_handler->send_count, sizeof(size_t)); \
        param_nnz.recv_buf = ccl_buffer(sa_handler->recv_counts, sizeof(size_t) * comm_size); \
        param_nnz.send_count = sizeof(size_t); \
        param_nnz.recv_counts = sa_handler->size_per_rank; \
        param_nnz.dtype = ccl_datatype_int8; \
        param_nnz.comm = comm; \
\
        entry_factory::make_entry<coll_entry>(sched, param_nnz); \
        sched->add_barrier(); \
    } while (0)

template <typename vtype>
typename std::enable_if<!std::is_same<vtype, ccl::bfloat16>::value, vtype>::type get_mask(
    ccl::reduction op) {
    switch (op) {
        case ccl::reduction::sum: return 0;
        case ccl::reduction::prod: return 1;
        case ccl::reduction::min: return std::numeric_limits<vtype>::max();
        case ccl::reduction::max: return std::numeric_limits<vtype>::min();
        case ccl::reduction::custom:
            CCL_FATAL("custom reduction is not supported for sparse_allreduce/mask algorithm");
            return ccl::status::invalid_arguments;
        default: return 0;
    }
}

template <typename vtype>
typename std::enable_if<std::is_same<vtype, ccl::bfloat16>::value, vtype>::type get_mask(
    ccl::reduction op) {
    switch (op) {
        case ccl::reduction::sum: return ccl::bfloat16(0);
        case ccl::reduction::prod: return ccl::bfloat16(CCL_BF16_ONE);
        case ccl::reduction::min: return ccl::bfloat16(CCL_BF16_MAX);
        case ccl::reduction::max: return ccl::bfloat16(CCL_BF16_MIN);
        case ccl::reduction::custom:
            CCL_FATAL("custom reduction is not supported for sparse_allreduce/mask algorithm");
            return ccl::status::invalid_arguments;
        default: return 0;
    }
}

template <typename i_type, typename v_type>
void sparse_coalesce(ccl_sparse_allreduce_handler* sah) {
    std::unique_ptr<idx_offset_map> iv_map(new idx_offset_map);
    i_type* src_i = (i_type*)sah->send_ibuf;
    v_type* src_v = (v_type*)sah->send_vbuf;

    /* fill in the <index:value_offset> map */
    for (size_t i = 0; i < sah->send_count[0]; i++) {
        auto it = iv_map->find(src_i[i]);
        if (it == iv_map->end()) {
            std::vector<size_t> tmp = { i * sah->val_dim_cnt };
            tmp.reserve(CCL_COALESCE_RESERVE_SIZE);
            iv_map->emplace(src_i[i], tmp);
        }
        else {
            it->second.push_back(i * sah->val_dim_cnt);
        }
    }

    /* create buffer w/o duplicates */
    size_t iv_map_cnt = iv_map->size();

    i_type* dst_i = nullptr;
    v_type* dst_v = nullptr;

    ccl_sched* sched = sah->sched;

    if (sah->comm->size() == 1 && sched->coll_attr.sparse_allreduce_alloc_fn) {
        /* the final buffers for comm_size == 1 are allocated here, so use alloc_fn */
        /* TODO: enable for ring/mask */
        sched->coll_attr.sparse_allreduce_alloc_fn(iv_map_cnt,
                                                   sah->index_dtype.idx(),
                                                   iv_map_cnt * sah->val_dim_cnt,
                                                   sah->value_dtype.idx(),
                                                   sched->coll_attr.sparse_allreduce_fn_ctx,
                                                   &sah->dst_ibuf,
                                                   &sah->dst_vbuf);
        dst_i = (i_type*)sah->dst_ibuf;
        dst_v = (v_type*)sah->dst_vbuf;
    }
    else {
        /* TODO: split dst_buf on ibuf and vbuf for ring/mask */
        sah->dst_buf =
            sched->alloc_buffer(iv_map_cnt * (sah->itype_size + sah->val_dim_cnt * sah->vtype_size))
                .get_ptr();

        sah->dst_ibuf = sah->dst_buf;
        sah->dst_vbuf = (char*)sah->dst_buf + sah->itype_size * iv_map_cnt;

        dst_i = (i_type*)sah->dst_ibuf;
        dst_v = (v_type*)sah->dst_vbuf;
    }

    CCL_THROW_IF_NOT(dst_i && dst_v);

    size_t idx_offset = 0;
    size_t val_offset = 0;

    for (auto& it : *iv_map) {
        dst_i[idx_offset] = it.first;
        val_offset = idx_offset * sah->val_dim_cnt;

        std::copy(
            src_v + it.second[0], src_v + it.second[0] + sah->val_dim_cnt, dst_v + val_offset);
        it.second[0] = val_offset;

        /* reduce values from duplicate indices */
        if (it.second.size() > 1) {
            ccl_comp_batch_reduce(src_v,
                                  it.second,
                                  sah->val_dim_cnt,
                                  dst_v + val_offset,
                                  nullptr,
                                  sah->value_dtype,
                                  sah->op,
                                  nullptr,
                                  nullptr,
                                  (sched->coll_attr.sparse_coalesce_mode ==
                                       ccl::sparse_coalesce_mode::keep_precision &&
                                   sah->value_dtype.idx() == ccl::datatype::bfloat16),
                                  sah->tmp,
                                  sah->acc);

            it.second.resize(1);
        }
        idx_offset++;
    }
    sah->iv_map = std::move(iv_map);
}

template <typename i_type, typename v_type>
ccl::status sparse_reduce_ring(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    /* Having received the msg we should prepare it for further send operation to the next neighbour. 
       sa_handler->recv_counts contains all the nnz count for all the ranks. And every iteration
       (sa_handler->iter) we need to take corresponding nnz count from recv_counts array, according to
       the scheme
                     rank id:      0        1        2        3
       send_buf_id in iter 0:    | 0 | -> | 1 | -> | 2 | -> | 3 |
                                   ↑__________________________|
                   
                     rank id:      0        1        2        3
       send_buf_id in iter 1:    | 3 | -> | 0 | -> | 1 | -> | 2 |
                                   ↑__________________________|

                     rank id:      0        1        2        3
       send_buf_id in iter 2:    | 2 | -> | 3 | -> | 0 | -> | 1 |
                                   ↑__________________________|

                     rank id:      0        1        2        3
       send_buf_id in iter 3:    | 1 | -> | 2 | -> | 3 | -> | 0 |
                                   ↑__________________________|
    */
    sa_handler->send_count[0] =
        sa_handler->recv_counts[(sa_handler->recv_from - sa_handler->iter + sa_handler->comm_size) %
                                sa_handler->comm_size];
    sa_handler->send_count[1] = sa_handler->send_count[0] * sa_handler->val_dim_cnt;

    i_type* snd_i = (i_type*)(sa_handler->dst_buf);
    v_type* snd_v =
        (v_type*)((char*)(sa_handler->dst_buf) + sa_handler->itype_size * sa_handler->dst_count[0]);

    /* copy data from recv_buf so that it would be easier to identify unique indices */
    size_t idx_size = sa_handler->itype_size * sa_handler->send_count[0];
    i_type* rcv_i = (i_type*)sa_handler->recv_buf;
    v_type* rcv_v = (v_type*)((char*)(sa_handler->recv_buf) + idx_size);
    std::vector<size_t> unique_indices_ids;

    /* look at received indices and the ones we already have. Check if there are equal
    ones, then the values could be reduced right away. The indices left will be copied
    along with correspoinding values*/
    for (size_t idx = 0; idx < sa_handler->send_count[0]; idx++) {
        auto it = sa_handler->iv_map->find(rcv_i[idx]);
        if (it != sa_handler->iv_map->end()) {
            ccl_comp_reduce(sa_handler->sched,
                            (void*)(rcv_v + idx * sa_handler->val_dim_cnt),
                            sa_handler->val_dim_cnt,
                            snd_v + it->second[0],
                            nullptr,
                            sa_handler->value_dtype,
                            sa_handler->op,
                            nullptr,
                            nullptr);
        }
        else {
            /* we'll run through these unique indices later */
            unique_indices_ids.push_back(idx);
        }
    }

    /* were there any unique indices? */
    if (unique_indices_ids.size() > 0) {
        /* prepare buf for combined data */
        size_t merge_idx_len = sa_handler->iv_map->size() + unique_indices_ids.size();

        std::vector<i_type> buf_i(merge_idx_len);
        std::vector<v_type> buf_v(merge_idx_len * sa_handler->val_dim_cnt);

        /* copy what we already have reduced*/
        ccl_comp_copy(snd_i,
                      buf_i.data(),
                      sa_handler->itype_size * sa_handler->dst_count[0],
                      ccl_datatype_int8);
        ccl_comp_copy(snd_v,
                      buf_v.data(),
                      sa_handler->vtype_size * sa_handler->dst_count[1],
                      ccl_datatype_int8);

        size_t idx_offset = 0;
        for (auto id : unique_indices_ids) {
            buf_i[sa_handler->dst_count[0] + idx_offset] = rcv_i[id];

            for (size_t k = 0; k < sa_handler->val_dim_cnt; k++) {
                buf_v[sa_handler->dst_count[1] + idx_offset * sa_handler->val_dim_cnt + k] =
                    rcv_v[id * sa_handler->val_dim_cnt + k];
            }

            /* upd the map */
            std::vector<size_t> tmp = { sa_handler->dst_count[1] +
                                        idx_offset * sa_handler->val_dim_cnt };
            tmp.reserve(CCL_COALESCE_RESERVE_SIZE);
            sa_handler->iv_map->emplace(rcv_i[id], tmp);
            idx_offset++;
        }

        /* we definitely have to increase the size of dst buffer because
        of the unique indices that came from our neighbour */
        size_t new_dst_size = merge_idx_len * (sa_handler->vtype_size * sa_handler->val_dim_cnt +
                                               sa_handler->itype_size);
        sa_handler->dst_buf =
            (sa_handler->sched->update_buffer(
                 ccl_buffer(sa_handler->dst_buf,
                            sa_handler->dst_count[0] * sa_handler->itype_size +
                                sa_handler->dst_count[1] * sa_handler->vtype_size),
                 new_dst_size))
                .get_ptr();

        ccl_comp_copy(buf_i.data(),
                      (i_type*)(sa_handler->dst_buf),
                      sa_handler->itype_size * merge_idx_len,
                      ccl_datatype_int8);

        ccl_comp_copy(
            buf_v.data(),
            (v_type*)((char*)(sa_handler->dst_buf) + sa_handler->itype_size * merge_idx_len),
            sa_handler->vtype_size * merge_idx_len * sa_handler->val_dim_cnt,
            ccl_datatype_int8);

        sa_handler->dst_count[0] = merge_idx_len;
        sa_handler->dst_count[1] = merge_idx_len * sa_handler->val_dim_cnt;

    } // if unique_indices > 0

    ccl_comp_copy(sa_handler->recv_buf,
                  sa_handler->send_tmp_buf,
                  idx_size + sa_handler->send_count[1] * sa_handler->vtype_size,
                  ccl_datatype_int8);

    sa_handler->iter++;

    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status sparse_prepare_result_ring(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    /* data should be returned as sorted in the result buffer */
    i_type* ibuf = (i_type*)(sa_handler->dst_buf);
    v_type* vbuf = (v_type*)((i_type*)(sa_handler->dst_buf) + sa_handler->iv_map->size());
    std::vector<v_type> tmp(vbuf, vbuf + sa_handler->iv_map->size() * sa_handler->val_dim_cnt);
    size_t idx_offset = 0;
    for (auto& it : *sa_handler->iv_map) {
        ibuf[idx_offset] = it.first;
        std::copy(tmp.begin() + it.second[0],
                  tmp.begin() + it.second[0] + sa_handler->val_dim_cnt,
                  vbuf + idx_offset * sa_handler->val_dim_cnt);
        idx_offset++;
    }

    *sa_handler->recv_icount = sa_handler->iv_map->size();
    *sa_handler->recv_vcount = *sa_handler->recv_icount * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = sa_handler->dst_buf;
    *sa_handler->recv_vbuf =
        ((char*)sa_handler->dst_buf + sa_handler->itype_size * (*sa_handler->recv_icount));

    sa_handler->iv_map->clear();

    return ccl::status::success;
}

ccl::status sparse_get_send_count_ring(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t* cnt_ptr = (size_t*)field_ptr;
    *cnt_ptr = sa_handler->send_count[0] *
               (sa_handler->itype_size + sa_handler->val_dim_cnt * sa_handler->vtype_size);
    return ccl::status::success;
}

ccl::status sparse_get_send_buf_ring(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->send_tmp_buf);
    return ccl::status::success;
}

ccl::status sparse_get_recv_count_ring(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    size_t* cnt_ptr = (size_t*)field_ptr;
    size_t nnz =
        sa_handler->recv_counts[(sa_handler->recv_from - sa_handler->iter + sa_handler->comm_size) %
                                sa_handler->comm_size];

    *cnt_ptr = nnz * (sa_handler->itype_size + sa_handler->val_dim_cnt * sa_handler->vtype_size);
    return ccl::status::success;
}

ccl::status sparse_get_recv_buf_ring(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->recv_buf);
    return ccl::status::success;
}

ccl::status sparse_set_max_buf_size_ring(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t max_nnz = sa_handler->recv_counts[0];

    for (int i = 1; i < sa_handler->comm_size; i++) {
        if (max_nnz < sa_handler->recv_counts[i]) {
            max_nnz = sa_handler->recv_counts[i];
        }
    }

    size_t common_size_part =
        sa_handler->itype_size + sa_handler->vtype_size * sa_handler->val_dim_cnt;
    size_t max_size = max_nnz * common_size_part;

    sa_handler->send_tmp_buf = sa_handler->sched->alloc_buffer(max_size).get_ptr();
    CCL_MEMCPY(
        sa_handler->send_tmp_buf, sa_handler->dst_buf, sa_handler->dst_count[0] * common_size_part);
    sa_handler->recv_buf = sa_handler->sched->alloc_buffer(max_size).get_ptr();

    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status sparse_coalesce_ring(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    sparse_coalesce<i_type, v_type>(sa_handler);

    size_t iv_map_cnt = sa_handler->iv_map->size();

    sa_handler->send_count[0] = iv_map_cnt; /* index count */
    sa_handler->send_count[1] = iv_map_cnt * sa_handler->val_dim_cnt; /* value count */
    CCL_MEMCPY(&sa_handler->dst_count, &sa_handler->send_count, sizeof(size_t) * 2);

    CCL_SPARSE_ALLREDUCE_IF_SINGLE_RANK();
    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status ccl_coll_build_sparse_allreduce_ring(ccl_sched* sched,
                                                 ccl_buffer send_ind_buf,
                                                 size_t send_ind_count,
                                                 ccl_buffer send_val_buf,
                                                 size_t send_val_count,
                                                 void** recv_ind_buf,
                                                 size_t* recv_ind_count,
                                                 void** recv_val_buf,
                                                 size_t* recv_val_count,
                                                 const ccl_datatype& index_dtype,
                                                 const ccl_datatype& value_dtype,
                                                 ccl::reduction op,
                                                 ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    int comm_size = comm->size();
    int rank = comm->rank();

    /* get data type sizes */
    size_t vtype_size = sizeof(v_type);
    size_t itype_size = sizeof(i_type);

    /* get value dimension */
    size_t val_dim_cnt = send_val_count / send_ind_count;

    CCL_ASSERT(recv_ind_buf && recv_val_buf, "recv buffers are null");
    CCL_ASSERT(recv_ind_count && recv_val_count, "recv counts are null");

    void** r_ind_buf = recv_ind_buf;
    void** r_val_buf = recv_val_buf;

    ccl_sparse_allreduce_handler* sa_handler;
    CCL_SPARSE_ALLREDUCE_CREATE_HANDLER();

    /* send from left to right (ring)*/
    /* receive from the left neighbour */
    int recv_from = (rank - 1 + comm_size) % comm_size;

    /* send to the right neighbour */
    int send_to = (rank + 1) % comm_size;

    sa_handler->recv_from = recv_from;
    sa_handler->iter = 0;

    sa_handler->recv_counts =
        static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) * comm_size).get_ptr());

    entry_factory::make_entry<function_entry>(
        sched, sparse_coalesce_ring<i_type, v_type>, sa_handler);
    sched->add_barrier();

    if (comm_size > 1) {
        CCL_SPARSE_ALLREDUCE_ADD_NNZ_ENTRY();

        entry_factory::make_entry<function_entry>(sched, sparse_set_max_buf_size_ring, sa_handler);
        sched->add_barrier();

        for (int i = 0; i < comm_size - 1; i++) {
            /* send local data to the right neighbour */
            send_entry* se = entry_factory::make_entry<send_entry>(
                sched, ccl_buffer(), 0, ccl_datatype_int8, send_to, comm);
            se->set_field_fn<ccl_sched_entry_field_buf>(sparse_get_send_buf_ring, sa_handler);
            se->set_field_fn<ccl_sched_entry_field_cnt>(sparse_get_send_count_ring, sa_handler);

            /* receive data from the left neighbour */
            recv_entry* re = entry_factory::make_entry<recv_entry>(
                sched, ccl_buffer(), 0, ccl_datatype_int8, recv_from, comm);
            re->set_field_fn<ccl_sched_entry_field_buf>(sparse_get_recv_buf_ring, sa_handler);
            re->set_field_fn<ccl_sched_entry_field_cnt>(sparse_get_recv_count_ring, sa_handler);
            sched->add_barrier();

            /* reduce data */
            entry_factory::make_entry<function_entry>(
                sched, sparse_reduce_ring<i_type, v_type>, sa_handler);
            sched->add_barrier();
        }

        /* copy all reduced data to recv_buf */
        entry_factory::make_entry<function_entry>(
            sched, sparse_prepare_result_ring<i_type, v_type>, sa_handler);
        sched->add_barrier();
    }

    return status;
}

template <typename i_type, typename v_type>
ccl::status sparse_create_matrix_mask(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    LOG_TRACE("sa_handler: ",
              sa_handler,
              ", sa_handler->recv_buf_count: ",
              sa_handler->recv_buf_count,
              ", sa_handler->recv_buf: ",
              sa_handler->recv_buf);

    /* get rid of the duplicates in allgathered indices list */
    std::set<i_type> idx_set(
        static_cast<i_type*>(sa_handler->recv_buf),
        static_cast<i_type*>(sa_handler->recv_buf) + sa_handler->recv_buf_count);

    /* create a matrix expanded with zeros for indices that are not
       present in the unique indices list specified for this very process */
    size_t value_line_size = sa_handler->vtype_size * sa_handler->val_dim_cnt;
    size_t idx_cnt = idx_set.size();
    size_t matrix_size = idx_cnt * value_line_size;
    v_type* matrix = static_cast<v_type*>(CCL_MALLOC(matrix_size, "matrix"));
    v_type* values =
        (v_type*)((char*)(sa_handler->dst_buf) + sa_handler->itype_size * sa_handler->dst_count[0]);

    v_type mask_value = get_mask<v_type>(sa_handler->op);
    size_t idx_offset = 0;
    for (typename std::set<i_type>::iterator it = idx_set.begin(); it != idx_set.end(); ++it) {
        auto elem = sa_handler->iv_map->find(*it);
        if (elem != sa_handler->iv_map->end()) {
            /* copy values from dst_buf to matrix */
            CCL_MEMCPY(matrix + idx_offset * sa_handler->val_dim_cnt,
                       values + elem->second[0],
                       value_line_size);
        }
        else {
            /* no index was found locally, fill the line with mask */
            std::fill(matrix + idx_offset * sa_handler->val_dim_cnt,
                      matrix + (idx_offset + 1) * sa_handler->val_dim_cnt,
                      mask_value);
        }
        idx_offset++;
    }

    sa_handler->dst_buf =
        sa_handler->sched
            ->find_and_realloc_buffer(sa_handler->dst_buf,
                                      idx_cnt * sa_handler->itype_size + matrix_size,
                                      sa_handler->itype_size * sa_handler->dst_count[0] +
                                          sa_handler->vtype_size * sa_handler->dst_count[1])
            .get_ptr();

    ccl_comp_copy(matrix,
                  (char*)sa_handler->dst_buf + idx_cnt * sa_handler->itype_size,
                  matrix_size,
                  ccl_datatype_int8);

    CCL_FREE(matrix);
    sa_handler->iv_map->clear();
    std::copy(idx_set.begin(), idx_set.end(), (i_type*)(sa_handler->dst_buf));

    *sa_handler->recv_icount = idx_cnt;
    *sa_handler->recv_vcount = idx_cnt * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = sa_handler->dst_buf;
    *sa_handler->recv_vbuf = ((char*)sa_handler->dst_buf + sa_handler->itype_size * idx_cnt);

    return ccl::status::success;
}

ccl::status sparse_get_allreduce_buf_mask(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(*sa_handler->recv_vbuf);
    return ccl::status::success;
}

ccl::status sparse_get_allreduce_count_mask(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t* cnt_ptr = (size_t*)field_ptr;
    *cnt_ptr = *sa_handler->recv_vcount;
    return ccl::status::success;
}

ccl::status sparse_nnz_per_rank_mask(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    sa_handler->recv_buf_count = 0;
    for (int i = 0; i < sa_handler->comm_size; i++) {
        sa_handler->recv_buf_count += sa_handler->recv_counts[i];
    }

    sa_handler->recv_buf =
        sa_handler->sched->alloc_buffer(sa_handler->itype_size * sa_handler->recv_buf_count)
            .get_ptr();

    return ccl::status::success;
}

ccl::status sparse_get_allgatherv_buf_mask(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->recv_buf);
    return ccl::status::success;
}

ccl::status sparse_get_send_buf_mask(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->dst_buf);
    return ccl::status::success;
}

ccl::status sparse_get_send_count_mask(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t* count = (size_t*)field_ptr;
    *count = sa_handler->dst_count[0];
    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status sparse_coalesce_mask(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    sparse_coalesce<i_type, v_type>(sa_handler);

    size_t iv_map_cnt = sa_handler->iv_map->size();

    sa_handler->dst_count[0] = iv_map_cnt;
    sa_handler->dst_count[1] = iv_map_cnt * sa_handler->val_dim_cnt;

    CCL_SPARSE_ALLREDUCE_IF_SINGLE_RANK();
    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status ccl_coll_build_sparse_allreduce_mask(ccl_sched* sched,
                                                 ccl_buffer send_ind_buf,
                                                 size_t send_ind_count,
                                                 ccl_buffer send_val_buf,
                                                 size_t send_val_count,
                                                 void** recv_ind_buf,
                                                 size_t* recv_ind_count,
                                                 void** recv_val_buf,
                                                 size_t* recv_val_count,
                                                 const ccl_datatype& index_dtype,
                                                 const ccl_datatype& value_dtype,
                                                 ccl::reduction op,
                                                 ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    int comm_size = comm->size();

    /* get data type sizes */
    size_t itype_size = sizeof(i_type);
    size_t vtype_size = sizeof(v_type);

    /* get value dimension */
    size_t val_dim_cnt = send_val_count / send_ind_count;

    CCL_ASSERT(recv_ind_buf && recv_val_buf, "recv buffers are null");
    CCL_ASSERT(recv_ind_count && recv_val_count, "recv counts are null");

    void** r_ind_buf = recv_ind_buf;
    void** r_val_buf = recv_val_buf;

    ccl_sparse_allreduce_handler* sa_handler;
    CCL_SPARSE_ALLREDUCE_CREATE_HANDLER();

    sa_handler->recv_counts =
        static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) * comm_size).get_ptr());

    entry_factory::make_entry<function_entry>(
        sched, sparse_coalesce_mask<i_type, v_type>, sa_handler);
    sched->add_barrier();

    if (comm_size > 1) {
        CCL_SPARSE_ALLREDUCE_ADD_NNZ_ENTRY();

        entry_factory::make_entry<function_entry>(sched, sparse_nnz_per_rank_mask, sa_handler);
        sched->add_barrier();

        ccl_coll_entry_param param_allgatherv{};
        param_allgatherv.ctype = ccl_coll_allgatherv;
        param_allgatherv.send_buf = ccl_buffer();
        param_allgatherv.recv_buf = ccl_buffer();
        param_allgatherv.send_count = 0;
        param_allgatherv.recv_counts = sa_handler->recv_counts;
        param_allgatherv.dtype = index_dtype;
        param_allgatherv.comm = comm;

        /* gather indices from all the processes */
        coll_entry* e = entry_factory::make_entry<coll_entry>(sched, param_allgatherv);
        e->set_field_fn<ccl_sched_entry_field_send_buf>(sparse_get_send_buf_mask, sa_handler);
        e->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_allgatherv_buf_mask, sa_handler);
        e->set_field_fn<ccl_sched_entry_field_send_count>(sparse_get_send_count_mask, sa_handler);
        sched->add_barrier();

        entry_factory::make_entry<function_entry>(
            sched, sparse_create_matrix_mask<i_type, v_type>, sa_handler);
        sched->add_barrier();

        ccl_coll_entry_param param_allreduce{};
        param_allreduce.ctype = ccl_coll_allreduce;
        param_allreduce.send_buf = ccl_buffer();
        param_allreduce.recv_buf = ccl_buffer();
        param_allreduce.count = 0;
        param_allreduce.dtype = value_dtype;
        param_allreduce.reduction = op;
        param_allreduce.comm = comm;

        /* coll allreduce on matrix data */
        coll_entry* ce = entry_factory::make_entry<coll_entry>(sched, param_allreduce);
        ce->set_field_fn<ccl_sched_entry_field_send_buf>(sparse_get_allreduce_buf_mask, sa_handler);
        ce->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_allreduce_buf_mask, sa_handler);
        ce->set_field_fn<ccl_sched_entry_field_cnt>(sparse_get_allreduce_count_mask, sa_handler);
        sched->add_barrier();
    }

    return status;
}

ccl::status sparse_alloc_result_buf_allgatherv(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    sa_handler->recv_buf_count = 0;
    for (int i = 0; i < sa_handler->comm_size; i++) {
        sa_handler->recv_buf_count += sa_handler->recv_counts[i];
    }

    LOG_TRACE("sa_handle: ",
              sa_handler,
              ",allocate all buffers - indices size: ",
              sa_handler->recv_buf_count * sa_handler->itype_size,
              ", values size: ",
              sa_handler->recv_buf_count * sa_handler->vtype_size * sa_handler->val_dim_cnt,
              ", sa_handler->recv_counts: ",
              sa_handler->recv_counts);

    ccl_sched* sched = sa_handler->sched;

    if (sched->coll_attr.sparse_coalesce_mode == ccl::sparse_coalesce_mode::disable &&
        sched->coll_attr.sparse_allreduce_alloc_fn) {
        /* with coalesce_disable the final buffers are allocated here, so use alloc_fn */
        sched->coll_attr.sparse_allreduce_alloc_fn(
            sa_handler->recv_buf_count,
            sa_handler->index_dtype.idx(),
            sa_handler->recv_buf_count * sa_handler->val_dim_cnt,
            sa_handler->value_dtype.idx(),
            sched->coll_attr.sparse_allreduce_fn_ctx,
            &sa_handler->all_idx_buf,
            &sa_handler->all_val_buf);
    }
    else {
        sa_handler->all_idx_buf =
            sched->alloc_buffer(sa_handler->recv_buf_count * sa_handler->itype_size).get_ptr();
        sa_handler->all_val_buf =
            sched
                ->alloc_buffer(sa_handler->recv_buf_count * sa_handler->vtype_size *
                               sa_handler->val_dim_cnt)
                .get_ptr();
    }

    CCL_THROW_IF_NOT(sa_handler->all_idx_buf && sa_handler->all_val_buf);

    return ccl::status::success;
}

template <size_t stride_per_comm>
ccl::status sparse_set_v_counts_allgatherv(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t stride = stride_per_comm * sa_handler->comm_size;
    for (int i = 0; i < sa_handler->comm_size; i++) {
        sa_handler->recv_counts[i + stride] = sa_handler->recv_counts[i] * sa_handler->val_dim_cnt;
    }

    return ccl::status::success;
}

ccl::status sparse_return_gathered_allgatherv(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    *sa_handler->recv_icount = sa_handler->recv_buf_count;
    *sa_handler->recv_vcount = sa_handler->recv_buf_count * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = sa_handler->all_idx_buf;
    *sa_handler->recv_vbuf = sa_handler->all_val_buf;

    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status sparse_reduce_gathered_allgatherv(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    i_type* indices = static_cast<i_type*>(sa_handler->all_idx_buf);
    v_type* values = static_cast<v_type*>(sa_handler->all_val_buf);

    std::unique_ptr<idx_offset_map> iv_map(new idx_offset_map);
    for (size_t i = 0; i < sa_handler->recv_buf_count; i++) {
        auto it = iv_map->find(indices[i]);
        if (it == iv_map->end()) {
            std::vector<size_t> tmp = { i * sa_handler->val_dim_cnt };
            tmp.reserve(CCL_COALESCE_RESERVE_SIZE);
            iv_map->emplace(indices[i], tmp);
        }
        else {
            it->second.push_back(i * sa_handler->val_dim_cnt);
        }
    }

    size_t idx_cnt = iv_map->size();
    size_t i_new_size = sa_handler->itype_size * idx_cnt;
    size_t v_new_size = sa_handler->vtype_size * idx_cnt * sa_handler->val_dim_cnt;

    i_type* i_recv = nullptr;
    v_type* v_recv = nullptr;

    ccl_sched* sched = sa_handler->sched;

    if (sched->coll_attr.sparse_allreduce_alloc_fn) {
        sched->coll_attr.sparse_allreduce_alloc_fn(idx_cnt,
                                                   sa_handler->index_dtype.idx(),
                                                   idx_cnt * sa_handler->val_dim_cnt,
                                                   sa_handler->value_dtype.idx(),
                                                   sched->coll_attr.sparse_allreduce_fn_ctx,
                                                   &sa_handler->dst_ibuf,
                                                   &sa_handler->dst_vbuf);

        i_recv = (i_type*)sa_handler->dst_ibuf;
        v_recv = (v_type*)sa_handler->dst_vbuf;
    }
    else {
        sa_handler->dst_ibuf = sched->alloc_buffer(i_new_size).get_ptr();
        sa_handler->dst_vbuf = sched->alloc_buffer(v_new_size).get_ptr();

        i_recv = (i_type*)sa_handler->dst_ibuf;
        v_recv = (v_type*)sa_handler->dst_vbuf;
    }

    CCL_THROW_IF_NOT(i_recv && v_recv);

    size_t idx_offset = 0;
    size_t val_offset = 0;

    for (auto& it : *iv_map) {
        i_recv[idx_offset] = it.first;
        val_offset = idx_offset * sa_handler->val_dim_cnt;
        std::copy(values + it.second[0],
                  values + it.second[0] + sa_handler->val_dim_cnt,
                  v_recv + val_offset);
        it.second[0] = val_offset;

        /* reduce values from duplicate indices */
        if (it.second.size() > 1) {
            ccl_comp_batch_reduce(values,
                                  it.second,
                                  sa_handler->val_dim_cnt,
                                  v_recv + val_offset,
                                  nullptr,
                                  sa_handler->value_dtype,
                                  sa_handler->op,
                                  nullptr,
                                  nullptr,
                                  sched->coll_attr.sparse_coalesce_mode ==
                                          ccl::sparse_coalesce_mode::keep_precision &&
                                      sa_handler->value_dtype.idx() == ccl::datatype::bfloat16,
                                  sa_handler->tmp,
                                  sa_handler->acc);
        }
        idx_offset++;
    }

    iv_map->clear();

    *sa_handler->recv_icount = idx_cnt;
    *sa_handler->recv_vcount = idx_cnt * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = i_recv;
    *sa_handler->recv_vbuf = v_recv;

    return ccl::status::success;
}

ccl::status sparse_get_i_recv_allgatherv(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->all_idx_buf);
    return ccl::status::success;
}

ccl::status sparse_get_i_send_allgatherv(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->dst_ibuf);
    return ccl::status::success;
}

template <int send_count_src_index>
ccl::status sparse_get_send_count_allgatherv(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t* send_buf_count = (size_t*)field_ptr;
    *send_buf_count = sa_handler->send_count[send_count_src_index];
    return ccl::status::success;
}

ccl::status sparse_get_v_recv_allgatherv(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->all_val_buf);
    return ccl::status::success;
}

ccl::status sparse_get_v_send_allgatherv(const void* ctx, void* field_ptr) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    if (sa_handler->sched->coll_attr.sparse_coalesce_mode == ccl::sparse_coalesce_mode::disable) {
        buf_ptr->set(sa_handler->send_vbuf);
    }
    else {
        buf_ptr->set(sa_handler->dst_vbuf);
    }

    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status sparse_coalesce_allgatherv(const void* ctx) {
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    sparse_coalesce<i_type, v_type>(sa_handler);

    size_t iv_map_cnt = sa_handler->iv_map->size();
    sa_handler->iv_map->clear();
    sa_handler->send_count[0] = iv_map_cnt;
    sa_handler->send_count[1] = iv_map_cnt * sa_handler->val_dim_cnt;

    if (sa_handler->comm_size == 1) {
        *sa_handler->recv_icount = iv_map_cnt;
        *sa_handler->recv_vcount = iv_map_cnt * sa_handler->val_dim_cnt;
        *sa_handler->recv_ibuf = sa_handler->dst_ibuf;
        *sa_handler->recv_vbuf = sa_handler->dst_vbuf;
    }

    return ccl::status::success;
}

template <typename i_type, typename v_type>
ccl::status ccl_coll_build_sparse_allreduce_3_allgatherv(ccl_sched* sched,
                                                         ccl_buffer send_ind_buf,
                                                         size_t send_ind_count,
                                                         ccl_buffer send_val_buf,
                                                         size_t send_val_count,
                                                         void** recv_ind_buf,
                                                         size_t* recv_ind_count,
                                                         void** recv_val_buf,
                                                         size_t* recv_val_count,
                                                         const ccl_datatype& index_dtype,
                                                         const ccl_datatype& value_dtype,
                                                         ccl::reduction op,
                                                         ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    int comm_size = comm->size();

    /* get data type sizes */
    size_t vtype_size = sizeof(v_type);
    size_t itype_size = sizeof(i_type);

    /* get value dimension */
    size_t val_dim_cnt = send_val_count / send_ind_count;

    CCL_ASSERT(recv_ind_buf && recv_val_buf, "recv buffers are null");
    CCL_ASSERT(recv_ind_count && recv_val_count, "recv counts are null");

    void** r_ind_buf = recv_ind_buf;
    void** r_val_buf = recv_val_buf;

    ccl_sparse_allreduce_handler* sa_handler;
    CCL_SPARSE_ALLREDUCE_CREATE_HANDLER();

    constexpr size_t parallel_requests_count = 2; //indices + values
    sa_handler->recv_counts = static_cast<size_t*>(
        sched->alloc_buffer(sizeof(size_t) * comm_size * parallel_requests_count).get_ptr());

    LOG_DEBUG("sa_handler: ",
              sa_handler,
              ", sa_handler->recv_ibuf: ",
              sa_handler->recv_ibuf,
              ", sa_handler->recv_vbuf: ",
              sa_handler->recv_vbuf,
              ", sa_handler->val_dim_cnt: ",
              sa_handler->val_dim_cnt,
              ", sa_handler->recv_counts: ",
              sa_handler->recv_counts);

    if (sched->coll_attr.sparse_coalesce_mode != ccl::sparse_coalesce_mode::disable) {
        entry_factory::make_entry<function_entry>(
            sched, sparse_coalesce_allgatherv<i_type, v_type>, sa_handler);
        sched->add_barrier();

        if (comm_size == 1)
            return status;
    }
    else {
        sa_handler->dst_ibuf = sa_handler->send_ibuf;
        sa_handler->dst_vbuf = sa_handler->send_vbuf;
    }

    CCL_SPARSE_ALLREDUCE_ADD_NNZ_ENTRY();

    entry_factory::make_entry<function_entry>(
        sched, sparse_alloc_result_buf_allgatherv, sa_handler);
    sched->add_barrier();

    // allgather indices
    size_t parallel_request_index = 0;
    ccl_coll_entry_param param_i{};
    param_i.ctype = ccl_coll_allgatherv;
    param_i.send_buf = ccl_buffer();
    param_i.recv_buf = ccl_buffer();
    param_i.send_count = 0;
    param_i.recv_counts = sa_handler->recv_counts;
    param_i.dtype = index_dtype;
    param_i.comm = comm;

    coll_entry* ce = entry_factory::make_entry<coll_entry>(sched, param_i, parallel_request_index);
    ce->set_field_fn<ccl_sched_entry_field_send_buf>(sparse_get_i_send_allgatherv, sa_handler);
    ce->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_i_recv_allgatherv, sa_handler);
    ce->set_field_fn<ccl_sched_entry_field_send_count>(sparse_get_send_count_allgatherv<0>,
                                                       sa_handler);
    entry_factory::make_entry<function_entry>(sched, sparse_set_v_counts_allgatherv<1>, sa_handler);

    // allgather values
    parallel_request_index++;
    ccl_coll_entry_param param_v{};
    param_v.ctype = ccl_coll_allgatherv;
    param_v.send_buf = ccl_buffer();
    param_v.recv_buf = ccl_buffer();
    param_v.send_count = 0;
    param_v.recv_counts = &sa_handler->recv_counts[comm_size];
    param_v.dtype = value_dtype;
    param_v.comm = comm;

    ce = entry_factory::make_entry<coll_entry>(sched, param_v, parallel_request_index);
    ce->set_field_fn<ccl_sched_entry_field_send_buf>(sparse_get_v_send_allgatherv, sa_handler);
    ce->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_v_recv_allgatherv, sa_handler);
    ce->set_field_fn<ccl_sched_entry_field_send_count>(sparse_get_send_count_allgatherv<1>,
                                                       sa_handler);
    sched->add_barrier();

    if (sched->coll_attr.sparse_coalesce_mode == ccl::sparse_coalesce_mode::disable) {
        entry_factory::make_entry<function_entry>(
            sched, sparse_return_gathered_allgatherv, sa_handler);
    }
    else {
        entry_factory::make_entry<function_entry>(
            sched, sparse_reduce_gathered_allgatherv<i_type, v_type>, sa_handler);
    }
    sched->add_barrier();

    return status;
}
