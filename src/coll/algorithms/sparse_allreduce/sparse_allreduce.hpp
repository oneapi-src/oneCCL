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
#include "ccl_type_traits.hpp"
#include "coll/algorithms/sparse_allreduce/sparse_handler.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_BFP16_ONE 0x3f80
#define CCL_BFP16_MAX 0x7f7f
#define CCL_BFP16_MIN 0xff7f

#define CCL_DEFINE_ALGO(itype, vtype)                                                     \
    do {                                                                                  \
         switch (algo)                                                                    \
         {                                                                                \
              case ccl_coll_sparse_allreduce_ring:                                        \
                  CCL_CALL((ccl_coll_build_sparse_allreduce_ring <itype,vtype> (sched,    \
                                                           send_ind_buf, send_ind_count,  \
                                                           send_val_buf, send_val_count,  \
                                                           recv_ind_buf, recv_ind_count,  \
                                                           recv_val_buf, recv_val_count,  \
                                                           index_dtype,  value_dtype,     \
                                                           reduction, comm)));            \
                  break;                                                                  \
              case ccl_coll_sparse_allreduce_mask:                                        \
                  CCL_CALL((ccl_coll_build_sparse_allreduce_mask <itype,vtype> (sched,    \
                                                           send_ind_buf, send_ind_count,  \
                                                           send_val_buf, send_val_count,  \
                                                           recv_ind_buf, recv_ind_count,  \
                                                           recv_val_buf, recv_val_count,  \
                                                           index_dtype,  value_dtype,     \
                                                           reduction, comm)));            \
                  break;                                                                  \
              case ccl_coll_sparse_allreduce_3_allgatherv:                                \
                  CCL_CALL((ccl_coll_build_sparse_allreduce_3_allgatherv <itype,vtype> (sched,\
                                                           send_ind_buf, send_ind_count,  \
                                                           send_val_buf, send_val_count,  \
                                                           recv_ind_buf, recv_ind_count,  \
                                                           recv_val_buf, recv_val_count,  \
                                                           index_dtype,  value_dtype,     \
                                                           reduction, comm)));            \
                  break;                                                                  \
              default:                                                                    \
                  CCL_FATAL("unexpected sparse_allreduce_algo ",                          \
                             ccl_coll_algorithm_to_str(algo));                            \
                  return ccl_status_invalid_arguments;                                    \
         }                                                                                \
    } while (0)


#define CCL_DEFINE_VALUE(itype)                                         \
    do {                                                                \
         switch (value_dtype.idx())                                     \
         {                                                              \
              case ccl_dtype_float:                                     \
                  CCL_DEFINE_ALGO(itype, float);                        \
                  break;                                                \
              case ccl_dtype_double:                                    \
                  CCL_DEFINE_ALGO(itype, double);                       \
                  break;                                                \
              case ccl_dtype_char:                                      \
                  CCL_DEFINE_ALGO(itype, char);                         \
                  break;                                                \
              case ccl_dtype_int:                                       \
                  CCL_DEFINE_ALGO(itype, int);                          \
                  break;                                                \
              case ccl_dtype_int64:                                     \
                  CCL_DEFINE_ALGO(itype, int64_t);                      \
                  break;                                                \
              case ccl_dtype_uint64:                                    \
                  CCL_DEFINE_ALGO(itype, uint64_t);                     \
                  break;                                                \
              case ccl_dtype_bfp16:                                     \
                  CCL_DEFINE_ALGO(itype, ccl::bfp16);                   \
                  break;                                                \
              default:                                                  \
                  CCL_FATAL("value datatype ",                          \
                            global_data.dtypes->name(value_dtype),      \
                            " is not supported yet");                   \
                  return ccl_status_invalid_arguments;                  \
         }                                                              \
    } while (0)

#define REMOVE_DUPS()                                                             \
    do {                                                                          \
        /* fill in the <index:value_offset> map */                                \
        for (size_t i = 0; i < send_ind_count; i++)                               \
        {                                                                         \
            auto it = iv_map->find(src_i[i]);                                     \
            if (it == iv_map->end())                                              \
            {                                                                     \
                /* save index and starting addr of values */                      \
                /* set according to that index */                                 \
                iv_map->emplace(src_i[i], i * val_dim_cnt);                       \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                /* reduce values from duplicate indices */                        \
                ccl_comp_reduce((void*)(src_v + i * val_dim_cnt), val_dim_cnt,    \
                                (void*)(src_v + it->second), nullptr, value_dtype,\
                                op, nullptr, nullptr);                            \
            }                                                                     \
        }                                                                         \
                                                                                  \
        /* create buffer w/o duplicates */                                        \
        iv_map_cnt = iv_map->size();                                              \
        no_dup_size = iv_map_cnt * (itype_size + val_dim_cnt * vtype_size);       \
        dst = sched->alloc_buffer(no_dup_size).get_ptr();                         \
                                                                                  \
        dst_i = (i_type*)dst;                                                     \
        dst_v = (v_type*)((char*)dst + itype_size * iv_map_cnt);                  \
                                                                                  \
        size_t idx_offset = 0;                                                    \
        size_t val_offset = 0;                                                    \
                                                                                  \
        /* update value offsets in the map, because */                            \
        /* we copy data to dst buffer from source buffer */                       \
        for (auto& it : *iv_map)                                                  \
        {                                                                         \
            dst_i[idx_offset] = it.first;                                         \
            val_offset = idx_offset * val_dim_cnt;                                \
            CCL_MEMCPY(dst_v + val_offset,                                        \
                       src_v + it.second,                                         \
                       vtype_size * val_dim_cnt);                                 \
            it.second = val_offset;                                               \
            idx_offset++;                                                         \
        }                                                                         \
    } while (0)


#define IF_COMM_SIZE_IS_ONE()                          \
    do {                                               \
        if (comm_size == 1)                            \
        {                                              \
            *recv_ind_count = iv_map_cnt;              \
            *recv_val_count = iv_map_cnt * val_dim_cnt;\
            *r_ind_buf = (void*)dst_i;                 \
            *r_val_buf = (void*)dst_v;                 \
            return status;                             \
        }                                              \
    } while (0)

#define SET_SPARSE_HANDLER_COMMON_FIELDS()                                                  \
    do {                                                                                    \
        sa_handler =                                                                        \
            static_cast<ccl_sparse_allreduce_handler*>(                                     \
            sched->alloc_buffer(sizeof(ccl_sparse_allreduce_handler)).get_ptr());           \
                                                                                            \
        sa_handler->comm = comm;                                                            \
        sa_handler->comm_size = comm_size;                                                  \
        sa_handler->val_dim_cnt = val_dim_cnt;                                              \
        sa_handler->itype_size = itype_size;                                                \
        sa_handler->vtype_size = vtype_size;                                                \
        sa_handler->value_dtype = value_dtype;                                              \
        sa_handler->op = op;                                                                \
        sa_handler->recv_ibuf = r_ind_buf;                                                  \
        sa_handler->recv_vbuf = r_val_buf;                                                  \
        sa_handler->recv_vcount = recv_val_count;                                           \
        sa_handler->recv_icount = recv_ind_count;                                           \
        sa_handler->sched = sched;                                                          \
                                                                                            \
        sa_handler->size_per_rank =                                                         \
            static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) * comm_size).get_ptr());\
                                                                                            \
        for (size_t i = 0; i < comm_size; i++)                                              \
            sa_handler->size_per_rank[i] = sizeof(size_t);                                  \
    } while (0)

#define GET_NNZ()                                                   \
    do {                                                            \
        ccl_coll_entry_param param_nnz{};                           \
        param_nnz.ctype = ccl_coll_allgatherv;                      \
        param_nnz.send_buf = ccl_buffer(&sa_handler->send_count,    \
                                        sizeof(size_t));            \
        param_nnz.recv_buf = ccl_buffer(sa_handler->recv_counts,    \
                                        sizeof(size_t) * comm_size);\
        param_nnz.send_count = sizeof(size_t);                      \
        param_nnz.recv_counts = sa_handler->size_per_rank;          \
        param_nnz.dtype = ccl_datatype_char;                        \
        param_nnz.comm = comm;                                      \
                                                                    \
        entry_factory::make_entry<coll_entry>(sched, param_nnz);    \
        sched->add_barrier();                                       \
    } while (0)
    
template<typename vtype>
typename std::enable_if<!std::is_same<vtype, ccl::bfp16>::value, vtype>::type
get_mask(ccl_reduction_t op)
{
    switch (op)
    {
        case ccl_reduction_sum:
            return 0;
        case ccl_reduction_prod:
            return 1;
        case ccl_reduction_min:
            return std::numeric_limits<vtype>::max();
        case ccl_reduction_max:
            return std::numeric_limits<vtype>::min(); 
        case ccl_reduction_custom:
            CCL_FATAL("custom reduction is not supported for sparse_allreduce/mask algorithm");
            return ccl_status_invalid_arguments;
        default:
            return 0;
    }
}

template <typename vtype>
typename std::enable_if<std::is_same<vtype, ccl::bfp16>::value, vtype>::type
get_mask(ccl_reduction_t op)
{
    switch (op)
    {
        case ccl_reduction_sum:
            return 0;
        case ccl_reduction_prod:
            return CCL_BFP16_ONE;
        case ccl_reduction_min:
            return CCL_BFP16_MAX;
        case ccl_reduction_max:
            return CCL_BFP16_MIN; 
        case ccl_reduction_custom:
            CCL_FATAL("custom reduction is not supported for sparse_allreduce/mask algorithm");
            return ccl_status_invalid_arguments;
        default:
            return 0;
    }
}

template<typename i_type, typename v_type>
ccl_status_t sparse_reduce(const void* ctx)
{
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
    sa_handler->send_count[0] = sa_handler->recv_counts[(sa_handler->recv_from - 
                                                         sa_handler->iter + 
                                                         sa_handler->comm_size) % sa_handler->comm_size];
    sa_handler->send_count[1] = sa_handler->send_count[0] * sa_handler->val_dim_cnt;

    i_type* snd_i = (i_type*)(sa_handler->dst_buf);
    v_type* snd_v = (v_type*)((char*)(sa_handler->dst_buf) + sa_handler->itype_size * sa_handler->dst_count[0]);

    /* copy data from recv_buf so that it would be easier to identify unique indices */
    size_t idx_size = sa_handler->itype_size * sa_handler->send_count[0];
    i_type* rcv_i = (i_type*)sa_handler->recv_buf;
    v_type* rcv_v = (v_type*)((char*)(sa_handler->recv_buf) + idx_size);
    std::vector<size_t> unique_indices_ids;

    /* look at received indices and the ones we already have. Check if there are equal
    ones, then the values could be reduced right away. The indices left will be copied
    along with correspoinding values*/
    for (size_t idx = 0; idx < sa_handler->send_count[0]; idx++)
    {
        auto it = sa_handler->iv_map->find(rcv_i[idx]);
        if (it != sa_handler->iv_map->end())
        {
            ccl_comp_reduce((void*)(rcv_v + idx * sa_handler->val_dim_cnt), sa_handler->val_dim_cnt,
                            (void*)(snd_v + it->second), nullptr,
                            sa_handler->value_dtype, sa_handler->op, nullptr, nullptr);
        }
        else
        {
            /* we'll run through these unique indices later */
            unique_indices_ids.push_back(idx);
        }
    }

    /* were there any unique indices? */
    if (unique_indices_ids.size() > 0)
    {
        /* prepare buf for combined data */
        size_t merge_idx_len = sa_handler->iv_map->size() + unique_indices_ids.size();

        std::vector<i_type> buf_i(merge_idx_len);
        std::vector<v_type> buf_v(merge_idx_len * sa_handler->val_dim_cnt);

        /* copy what we already have reduced*/
        ccl_comp_copy(snd_i, buf_i.data(), sa_handler->itype_size * sa_handler->dst_count[0], ccl_datatype_char);
        ccl_comp_copy(snd_v, buf_v.data(), sa_handler->vtype_size * sa_handler->dst_count[1], ccl_datatype_char);

        size_t idx_offset = 0;
        for (auto id : unique_indices_ids)
        {
            buf_i[sa_handler->dst_count[0] + idx_offset] = rcv_i[id];

            for (size_t k = 0; k < sa_handler->val_dim_cnt; k++)
            {
                buf_v[sa_handler->dst_count[1] + idx_offset * sa_handler->val_dim_cnt + k] =
                    rcv_v[id * sa_handler->val_dim_cnt + k];
            }

            /* upd the map */
            sa_handler->iv_map->emplace(rcv_i[id], sa_handler->dst_count[1] +
                                                    idx_offset * sa_handler->val_dim_cnt);
            idx_offset++;
        }

        /* we definitely have to increase the size of dst buffer because
        of the unique indices that came from our neighbour */
        size_t new_dst_size = merge_idx_len * (sa_handler->vtype_size * sa_handler->val_dim_cnt +
                                               sa_handler->itype_size);
        sa_handler->dst_buf =
            (sa_handler->sched->update_buffer(ccl_buffer(sa_handler->dst_buf,
                                                         sa_handler->dst_count[0] * sa_handler->itype_size +
                                                         sa_handler->dst_count[1] * sa_handler->vtype_size),
                                                         new_dst_size)).get_ptr();

        ccl_comp_copy(buf_i.data(),
                      (i_type*)(sa_handler->dst_buf),
                      sa_handler->itype_size * merge_idx_len, ccl_datatype_char);

        ccl_comp_copy(buf_v.data(),
                      (v_type*)((char*)(sa_handler->dst_buf) + sa_handler->itype_size * merge_idx_len),
                      sa_handler->vtype_size * merge_idx_len * sa_handler->val_dim_cnt, ccl_datatype_char);

        sa_handler->dst_count[0] = merge_idx_len;
        sa_handler->dst_count[1] = merge_idx_len * sa_handler->val_dim_cnt;

    } // if unique_indices > 0

    ccl_comp_copy(sa_handler->recv_buf,
                  sa_handler->send_tmp_buf,
                  idx_size + sa_handler->send_count[1] * sa_handler->vtype_size,
                  ccl_datatype_char);

    sa_handler->iter++;

    return ccl_status_success;
}

ccl_status_t sparse_prepare_result(const void* ctx)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    *sa_handler->recv_icount = sa_handler->iv_map->size();
    *sa_handler->recv_vcount = *sa_handler->recv_icount * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = sa_handler->dst_buf;
    *sa_handler->recv_vbuf = ((char*)sa_handler->dst_buf + sa_handler->itype_size * (*sa_handler->recv_icount));

    sa_handler->iv_map->clear();

    return ccl_status_success;
}

ccl_status_t sparse_get_send_count(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t* cnt_ptr = (size_t*)field_ptr;
    *cnt_ptr = sa_handler->send_count[0] * (sa_handler->itype_size + sa_handler->val_dim_cnt * sa_handler->vtype_size);
    return ccl_status_success;
}

ccl_status_t sparse_get_send_buf(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->send_tmp_buf);
    return ccl_status_success;
}

ccl_status_t sparse_get_recv_count(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    size_t* cnt_ptr = (size_t*)field_ptr;
    size_t nnz =
        sa_handler->recv_counts[(sa_handler->recv_from - sa_handler->iter + sa_handler->comm_size) %
                                sa_handler->comm_size];

    *cnt_ptr = nnz * (sa_handler->itype_size + sa_handler->val_dim_cnt * sa_handler->vtype_size);
    return ccl_status_success;
}

ccl_status_t sparse_get_recv_buf(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->recv_buf);
    return ccl_status_success;
}

ccl_status_t sparse_set_max_buf_size(const void* ctx)
{
    ccl_sparse_allreduce_handler *sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t max_nnz = sa_handler->recv_counts[0];

    for (size_t i = 1; i < sa_handler->comm_size; i++)
    {
        if (max_nnz < sa_handler->recv_counts[i])
        {
            max_nnz = sa_handler->recv_counts[i];
        }
    }

    size_t common_size_part = sa_handler->itype_size + sa_handler->vtype_size * sa_handler->val_dim_cnt;
    size_t max_size = max_nnz * common_size_part;

    sa_handler->send_tmp_buf = sa_handler->sched->alloc_buffer(max_size).get_ptr();
    CCL_MEMCPY(sa_handler->send_tmp_buf, sa_handler->dst_buf, sa_handler->dst_count[0] * common_size_part);
    sa_handler->recv_buf = sa_handler->sched->alloc_buffer(max_size).get_ptr();

    return ccl_status_success;
}

template<typename i_type, typename v_type>
ccl_status_t ccl_coll_build_sparse_allreduce_ring(ccl_sched* sched,
                                                  ccl_buffer send_ind_buf, size_t send_ind_count,
                                                  ccl_buffer send_val_buf, size_t send_val_count,
                                                  void** recv_ind_buf, size_t* recv_ind_count,
                                                  void** recv_val_buf, size_t* recv_val_count,
                                                  const ccl_datatype& index_dtype,
                                                  const ccl_datatype& value_dtype,
                                                  ccl_reduction_t op,
                                                  ccl_comm* comm)
{
    ccl_status_t status = ccl_status_success;

    size_t comm_size = comm->size();
    size_t rank = comm->rank();

    /* get data type sizes */
    size_t vtype_size = sizeof(v_type);
    size_t itype_size = sizeof(i_type);

    /* get value dimension */
    size_t val_dim_cnt = send_val_count / send_ind_count;

    /* buffers for in_data */
    i_type* src_i = (i_type*)send_ind_buf.get_ptr();
    v_type* src_v = (v_type*)send_val_buf.get_ptr();

    CCL_ASSERT(recv_ind_buf && recv_ind_buf, "recv buffers are null");
    CCL_ASSERT(recv_ind_count && recv_val_count, "recv counts are null");

    void** r_ind_buf = recv_ind_buf;
    void** r_val_buf = recv_val_buf;

    std::unique_ptr<idx_offset_map> iv_map(new idx_offset_map);
    size_t iv_map_cnt, no_dup_size;
    
    /* the accumulated result will be kept here */
    void* dst = nullptr;
    i_type* dst_i;
    v_type* dst_v;

    REMOVE_DUPS();

    IF_COMM_SIZE_IS_ONE();

    /* send from left to right (ring)*/

    /* receive from the left neighbour */
    size_t recv_from = (rank - 1 + comm_size) % comm_size;

    /* send to the right neighbour */
    size_t send_to = (rank + 1) % comm_size;

    /* create handler for sched function callbacks */
    ccl_sparse_allreduce_handler *sa_handler;

    SET_SPARSE_HANDLER_COMMON_FIELDS();

    /* _count variables needed for sending/receiving */
    sa_handler->send_count[0] = iv_map_cnt; /* index count */
    sa_handler->send_count[1] = iv_map_cnt * val_dim_cnt; /* value count */
    CCL_MEMCPY(&sa_handler->dst_count, &sa_handler->send_count, sizeof(size_t) * 2);
    sa_handler->dst_buf = dst;
    sa_handler->iv_map = std::move(iv_map);
    sa_handler->recv_from = recv_from;
    sa_handler->iter = 0;

    sa_handler->recv_counts =
        static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) * comm_size).get_ptr());

    GET_NNZ();

    entry_factory::make_entry<function_entry>(sched, sparse_set_max_buf_size, sa_handler);
    sched->add_barrier();

    for (size_t i = 0; i < comm_size - 1; i++)
    {
        /* send local data to the right neighbour */
        send_entry* se =
            entry_factory::make_entry<send_entry>(sched, ccl_buffer(), 0,
                                                  ccl_datatype_char, send_to, comm);
        se->set_field_fn<ccl_sched_entry_field_buf>(sparse_get_send_buf, sa_handler);
        se->set_field_fn<ccl_sched_entry_field_cnt>(sparse_get_send_count, sa_handler);

        /* receive data from the left neighbour */
        recv_entry* re =
            entry_factory::make_entry<recv_entry>(sched, ccl_buffer(), 0,
                                                  ccl_datatype_char, recv_from, comm);
        re->set_field_fn<ccl_sched_entry_field_buf>(sparse_get_recv_buf, sa_handler);
        re->set_field_fn<ccl_sched_entry_field_cnt>(sparse_get_recv_count, sa_handler);
        sched->add_barrier();

        /* reduce data */
        entry_factory::make_entry<function_entry>(sched, sparse_reduce<i_type, v_type>, sa_handler);
        sched->add_barrier();
    }

    /* copy all reduced data to recv_buf */
    entry_factory::make_entry<function_entry>(sched, sparse_prepare_result, sa_handler);
    sched->add_barrier();

    return status;
}

template<typename i_type, typename v_type>
ccl_status_t sparse_create_matrix(const void* ctx)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    LOG_TRACE("sa_handler: ", sa_handler,
              ", sa_handler->recv_buf_count: ", sa_handler->recv_buf_count,
              ", sa_handler->recv_buf: ", sa_handler->recv_buf);

    /* get rid of the duplicates in allgathered indices list */
    std::set<i_type> idx_set(static_cast<i_type*>(sa_handler->recv_buf), 
                             static_cast<i_type*>(sa_handler->recv_buf) + 
                                                  sa_handler->recv_buf_count);

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
    for (typename std::set<i_type>::iterator it = idx_set.begin(); it != idx_set.end(); ++it)
    {
        auto elem = sa_handler->iv_map->find(*it);
        if (elem != sa_handler->iv_map->end())
        {
            /* copy values from dst_buf to matrix */
            CCL_MEMCPY(matrix + idx_offset * sa_handler->val_dim_cnt, values + elem->second, value_line_size);
        }
        else
        {
            /* no index was found locally, fill the line with mask */
            std::fill(matrix + idx_offset * sa_handler->val_dim_cnt, 
                      matrix + idx_offset * sa_handler->val_dim_cnt + sa_handler->val_dim_cnt,
                      mask_value);
        }
        idx_offset++;
    }

    sa_handler->dst_buf =
        sa_handler->sched->find_and_realloc_buffer(
          sa_handler->dst_buf,
          idx_cnt * sa_handler->itype_size + matrix_size,
          sa_handler->itype_size * sa_handler->dst_count[0] +
          sa_handler->vtype_size * sa_handler->dst_count[1]).get_ptr();

    ccl_comp_copy(matrix,
                  (char*)sa_handler->dst_buf + idx_cnt * sa_handler->itype_size,
                  matrix_size, ccl_datatype_char);

    CCL_FREE(matrix);
    sa_handler->iv_map->clear();
    std::copy(idx_set.begin(), idx_set.end(), (i_type*)(sa_handler->dst_buf));
    
    *sa_handler->recv_icount = idx_cnt;
    *sa_handler->recv_vcount = idx_cnt * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = sa_handler->dst_buf;
    *sa_handler->recv_vbuf = ((char*)sa_handler->dst_buf + sa_handler->itype_size * idx_cnt);

    return ccl_status_success;
}

ccl_status_t sparse_get_allreduce_buf(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(*sa_handler->recv_vbuf);
    return ccl_status_success;
}

ccl_status_t sparse_get_allreduce_count(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t* cnt_ptr = (size_t*)field_ptr;
    *cnt_ptr = *sa_handler->recv_vcount;
    return ccl_status_success;
}

ccl_status_t sparse_nnz_per_rank(const void* ctx)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    sa_handler->recv_buf_count = 0;
    for (size_t i = 0; i < sa_handler->comm_size; i++)
    {
        sa_handler->recv_buf_count += sa_handler->recv_counts[i];
    }

    sa_handler->recv_buf =
        sa_handler->sched->alloc_buffer(sa_handler->itype_size * sa_handler->recv_buf_count).get_ptr();

    return ccl_status_success;
}

ccl_status_t sparse_get_allgatherv_buf(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->recv_buf);
    return ccl_status_success;
}

template<typename i_type, typename v_type>
ccl_status_t ccl_coll_build_sparse_allreduce_mask(ccl_sched* sched,
                                                  ccl_buffer send_ind_buf, size_t send_ind_count,
                                                  ccl_buffer send_val_buf, size_t send_val_count,
                                                  void** recv_ind_buf, size_t* recv_ind_count,
                                                  void** recv_val_buf, size_t* recv_val_count,
                                                  const ccl_datatype& index_dtype,
                                                  const ccl_datatype& value_dtype,
                                                  ccl_reduction_t op,
                                                  ccl_comm* comm)
{
    ccl_status_t status = ccl_status_success;

    size_t comm_size = comm->size();

    /* get data type sizes */
    size_t vtype_size = sizeof(v_type);
    size_t itype_size = sizeof(i_type);

    /* get value dimension */
    size_t val_dim_cnt = send_val_count / send_ind_count;

    /* buffers for in_data */
    i_type* src_i = (i_type*)send_ind_buf.get_ptr();
    v_type* src_v = (v_type*)send_val_buf.get_ptr();

    CCL_ASSERT(recv_ind_buf && recv_ind_buf, "recv buffers are null");
    CCL_ASSERT(recv_ind_count && recv_val_count, "recv counts are null");

    void** r_ind_buf = recv_ind_buf;
    void** r_val_buf = recv_val_buf;

    std::unique_ptr<idx_offset_map> iv_map(new idx_offset_map);
    size_t iv_map_cnt, no_dup_size;
   
    /* the accumulated result will be kept here */
    void* dst = nullptr;
    i_type* dst_i;
    v_type* dst_v;

    REMOVE_DUPS();

    IF_COMM_SIZE_IS_ONE();

    /* create handler for sched function callbacks */
    ccl_sparse_allreduce_handler* sa_handler;

    SET_SPARSE_HANDLER_COMMON_FIELDS();

    sa_handler->iv_map = std::move(iv_map);
    sa_handler->dst_buf = dst;
    sa_handler->dst_count[0] = iv_map_cnt;
    sa_handler->dst_count[1] = iv_map_cnt * val_dim_cnt;
    sa_handler->send_count[0] = send_ind_count;
    sa_handler->send_count[1] = send_val_count;

    sa_handler->recv_counts = 
        static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) * comm_size).get_ptr());
        
    GET_NNZ();

    entry_factory::make_entry<function_entry>(sched, sparse_nnz_per_rank, sa_handler);
    sched->add_barrier();   

    ccl_coll_entry_param param_allgatherv{};
    param_allgatherv.ctype = ccl_coll_allgatherv;
    param_allgatherv.send_buf = ccl_buffer(src_i, send_ind_count * itype_size);
    param_allgatherv.recv_buf = ccl_buffer();
    param_allgatherv.send_count = send_ind_count;
    param_allgatherv.recv_counts = sa_handler->recv_counts;
    param_allgatherv.dtype = index_dtype;
    param_allgatherv.comm = comm;

    /* gather indices from all the processes */
    coll_entry* e = entry_factory::make_entry<coll_entry>(sched, param_allgatherv);
    e->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_allgatherv_buf, sa_handler);
    sched->add_barrier();

    entry_factory::make_entry<function_entry>(sched, sparse_create_matrix<i_type, v_type>, sa_handler);
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
    ce->set_field_fn<ccl_sched_entry_field_send_buf>(sparse_get_allreduce_buf, sa_handler);
    ce->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_allreduce_buf, sa_handler);
    ce->set_field_fn<ccl_sched_entry_field_cnt>(sparse_get_allreduce_count, sa_handler);
    sched->add_barrier();

    return status;
}

ccl_status_t sparse_alloc_result_buf(const void* ctx)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;

    sa_handler->recv_buf_count = 0;
    for (size_t i = 0; i < sa_handler->comm_size; i++)
    {
        sa_handler->recv_buf_count += sa_handler->recv_counts[i];
    }

    LOG_TRACE("sa_handle: ", sa_handler,
      ",allocate all buffers - indices size: ", sa_handler->recv_buf_count * sa_handler->itype_size,
      ", values size: ", sa_handler->recv_buf_count * sa_handler->vtype_size * sa_handler->val_dim_cnt,
      ", sa_handler->recv_counts: ", sa_handler->recv_counts);

    sa_handler->all_idx_buf =
        sa_handler->sched->alloc_buffer(sa_handler->recv_buf_count *
                                        sa_handler->itype_size).get_ptr();
    sa_handler->all_val_buf =
        sa_handler->sched->alloc_buffer(sa_handler->recv_buf_count *
                                        sa_handler->vtype_size *
                                        sa_handler->val_dim_cnt).get_ptr();

    return ccl_status_success;
}

template<size_t stride_per_comm>
ccl_status_t sparse_set_v_counts(const void* ctx)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    size_t stride = stride_per_comm * sa_handler->comm_size;
    for (size_t i = 0; i < sa_handler->comm_size; i++)
    {
        sa_handler->recv_counts[i + stride] = sa_handler->recv_counts[i] * sa_handler->val_dim_cnt;
    }

    return ccl_status_success;
}

template <typename i_type, typename v_type>
ccl_status_t sparse_reduce_gathered(const void* ctx)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    i_type* indices = static_cast<i_type*>(sa_handler->all_idx_buf);
    v_type* values = static_cast<v_type*>(sa_handler->all_val_buf);

    std::unique_ptr<idx_offset_map> iv_map(new idx_offset_map);
    for (size_t i = 0; i < sa_handler->recv_buf_count; i++)
    {
        auto it = iv_map->find(indices[i]);
        if (it == iv_map->end())
        {
            iv_map->emplace(indices[i], i * sa_handler->val_dim_cnt);
        }
        else
        {
            ccl_comp_reduce((void*)(values + i * sa_handler->val_dim_cnt), sa_handler->val_dim_cnt, 
                            (void*)(values + it->second), nullptr, sa_handler->value_dtype,
                            sa_handler->op, nullptr, nullptr);
        }
    }

    size_t idx_cnt = iv_map->size();
    size_t i_new_size = sa_handler->itype_size * idx_cnt;
    size_t v_new_size = sa_handler->vtype_size * idx_cnt * sa_handler->val_dim_cnt;
    
    sa_handler->dst_buf = sa_handler->sched->alloc_buffer(i_new_size + v_new_size).get_ptr();

    i_type* i_recv = (i_type*)(sa_handler->dst_buf);
    v_type* v_recv = (v_type*)((char*)sa_handler->dst_buf + i_new_size);

    size_t idx_offset = 0;
    size_t val_offset = 0;
    for (auto& it : *iv_map)
    {
        i_recv[idx_offset] = it.first;
        val_offset = idx_offset * sa_handler->val_dim_cnt;
        CCL_MEMCPY(v_recv + val_offset,
                   values + it.second,
                   sa_handler->vtype_size * sa_handler->val_dim_cnt);
        it.second = val_offset;
        idx_offset++;
    }

    iv_map->clear();

    *sa_handler->recv_icount = idx_cnt;
    *sa_handler->recv_vcount = idx_cnt * sa_handler->val_dim_cnt;

    *sa_handler->recv_ibuf = sa_handler->dst_buf;
    *sa_handler->recv_vbuf = ((char*)sa_handler->dst_buf + i_new_size);

    return ccl_status_success;
}

ccl_status_t sparse_get_i_recv(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->all_idx_buf);
    return ccl_status_success;
}

ccl_status_t sparse_get_v_recv(const void* ctx, void* field_ptr)
{
    ccl_sparse_allreduce_handler* sa_handler = (ccl_sparse_allreduce_handler*)ctx;
    ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
    buf_ptr->set(sa_handler->all_val_buf);
    return ccl_status_success;
}

template<typename i_type, typename v_type> ccl_status_t
ccl_coll_build_sparse_allreduce_3_allgatherv(ccl_sched *sched,
                                             ccl_buffer send_ind_buf, size_t send_ind_count,
                                             ccl_buffer send_val_buf, size_t send_val_count,
                                             void** recv_ind_buf, size_t* recv_ind_count,
                                             void** recv_val_buf, size_t* recv_val_count,
                                             const ccl_datatype& index_dtype,
                                             const ccl_datatype& value_dtype,
                                             ccl_reduction_t op,
                                             ccl_comm* comm)
{
    ccl_status_t status = ccl_status_success;

    size_t comm_size = comm->size();

    /* get data type sizes */
    size_t vtype_size = sizeof(v_type);
    size_t itype_size = sizeof(i_type);

    /* get value dimension */
    size_t val_dim_cnt = send_val_count / send_ind_count;

    /* buffers for in_data */
    i_type* src_i = (i_type*)send_ind_buf.get_ptr();
    v_type* src_v = (v_type*)send_val_buf.get_ptr();

    CCL_ASSERT(recv_ind_buf && recv_ind_buf, "recv buffers are null");
    CCL_ASSERT(recv_ind_count && recv_val_count, "recv counts are null");

    void** r_ind_buf = recv_ind_buf;
    void** r_val_buf = recv_val_buf;

    std::unique_ptr<idx_offset_map> iv_map(new idx_offset_map);
    size_t iv_map_cnt, no_dup_size;

    /* the accumulated result will be kept here */
    void* dst = nullptr;
    i_type* dst_i;
    v_type* dst_v;

    REMOVE_DUPS();
    
    iv_map->clear();

    IF_COMM_SIZE_IS_ONE();

    /* create handler for sched function callbacks */
    ccl_sparse_allreduce_handler* sa_handler;

    SET_SPARSE_HANDLER_COMMON_FIELDS();

    /* _count variables needed for sending/receiving */
    sa_handler->send_count[0] = iv_map_cnt; /* index count */
    sa_handler->send_count[1] = iv_map_cnt * val_dim_cnt; /* value count */

    constexpr size_t parallel_requests_count = 2; //indices + values
    sa_handler->recv_counts = 
        static_cast<size_t*>(sched->alloc_buffer(sizeof(size_t) *
                                                 comm_size *
                                                 parallel_requests_count).get_ptr());

    LOG_TRACE("sa_handler: ", sa_handler,
              ", sa_handler->recv_ibuf: ", sa_handler->recv_ibuf,
              ", sa_handler->recv_vbuf: ", sa_handler->recv_vbuf,
              ", sa_handler->val_dim_cnt: ", sa_handler->val_dim_cnt,
              ", sa_handler->recv_counts: ", sa_handler->recv_counts);
        
    GET_NNZ();

    entry_factory::make_entry<function_entry>(sched, sparse_alloc_result_buf, sa_handler);
    sched->add_barrier();

    // allgather indices
    size_t parallel_request_index = 0;
    ccl_coll_entry_param param_i{};
    param_i.ctype = ccl_coll_allgatherv;
    param_i.send_buf = ccl_buffer(dst_i, sa_handler->send_count[parallel_request_index] * itype_size);
    param_i.recv_buf = ccl_buffer();
    param_i.send_count = sa_handler->send_count[parallel_request_index];
    param_i.recv_counts = &sa_handler->recv_counts[parallel_request_index * comm_size];
    param_i.dtype = index_dtype;
    param_i.comm = comm;

    coll_entry* ce = entry_factory::make_entry<coll_entry>(sched, param_i, parallel_request_index);
    ce->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_i_recv, sa_handler);
    entry_factory::make_entry<function_entry>(sched, sparse_set_v_counts<1>, sa_handler);
    
    // allgather values
    parallel_request_index ++;
    ccl_coll_entry_param param_v{};
    param_v.ctype = ccl_coll_allgatherv;
    param_v.send_buf = ccl_buffer(dst_v, sa_handler->send_count[parallel_request_index] * vtype_size);
    param_v.recv_buf = ccl_buffer();
    param_v.send_count = sa_handler->send_count[parallel_request_index];
    param_v.recv_counts = &sa_handler->recv_counts[parallel_request_index * comm_size];
    param_v.dtype = value_dtype;
    param_v.comm = comm;

    ce = entry_factory::make_entry<coll_entry>(sched, param_v, parallel_request_index);
    ce->set_field_fn<ccl_sched_entry_field_recv_buf>(sparse_get_v_recv, sa_handler);
    sched->add_barrier();

    entry_factory::make_entry<function_entry>(sched, sparse_reduce_gathered<i_type, v_type>, sa_handler);
    sched->add_barrier();
    return status;
}
