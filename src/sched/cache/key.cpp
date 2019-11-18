/*
 Copyright 2016-2019 Intel Corporation
 
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

#include "common/env/env.hpp"
#include "sched/cache/key.hpp"

#include <cstring>

ccl_sched_key::ccl_sched_key(const ccl_coll_param& param,
                             const ccl_coll_attr& attr)
{
    set(param, attr);
}

void ccl_sched_key::set(const ccl_coll_param& param,
                        const ccl_coll_attr& attr)
{
    prologue_fn = attr.prologue_fn;
    epilogue_fn = attr.epilogue_fn;
    reduction_fn = attr.reduction_fn;
    match_id = attr.match_id;

    ctype = param.ctype;
    dtype = param.dtype->type;
    comm = param.comm;

    switch (ctype)
    {
        case ccl_coll_allgatherv:
            buf = (void*)param.recv_counts;
            count1 = param.send_count;
            break;
        case ccl_coll_allreduce:
            count1 = param.count;
            reduction = param.reduction;
            break;
        case ccl_coll_alltoall:
            count1 = param.count;
            break;
        case ccl_coll_barrier:
            break;
        case ccl_coll_bcast:
            count1 = param.count;
            root = param.root;
            break;
        case ccl_coll_reduce:
            count1 = param.count;
            reduction = param.reduction;
            root = param.root;
            break;
        case ccl_coll_sparse_allreduce:
            count1 = param.sparse_param.send_ind_count;
            count2 = param.sparse_param.send_val_count;
            count3 = param.sparse_param.recv_ind_count;
            count4 = param.sparse_param.recv_val_count;
            itype = param.sparse_param.itype->type;
            reduction = param.reduction;
            break;
        default:
            CCL_THROW("unexpected coll_type ", ctype);
    }
}

bool ccl_sched_key::check(const ccl_coll_param& param, const ccl_coll_attr& attr)
{
    bool result = true;

    result &= (attr.prologue_fn == prologue_fn ||
               attr.epilogue_fn == epilogue_fn ||
               attr.reduction_fn == reduction_fn ||
               param.ctype == ctype ||
               param.dtype->type == dtype ||
               param.comm == comm);

    switch (ctype)
    {
        case ccl_coll_allgatherv:
            result &= (param.recv_counts == buf &&
                       param.send_count == count1);
            break;
        case ccl_coll_allreduce:
            result &= (param.count == count1 &&
                       param.reduction == reduction);
            break;
        case ccl_coll_alltoall:
            result &= (param.count == count1);
            break;
        case ccl_coll_barrier:
            break;
        case ccl_coll_bcast:
            result &= (param.count == count1 &&
                       param.root == root);
            break;
        case ccl_coll_reduce:
            result &= (param.count == count1 &&
                       param.reduction == reduction &&
                       param.root == root);
            break;
        case ccl_coll_sparse_allreduce:
            result &= (param.sparse_param.send_ind_count == count1 &&
                       param.sparse_param.send_val_count == count2 &&
                       param.sparse_param.recv_ind_count == count3 &&
                       param.sparse_param.recv_val_count == count4 &&
                       param.sparse_param.itype->type == itype &&
                       param.reduction == reduction);
            break;
        default:
            CCL_THROW("unexpected coll_type ", ctype);
    }

    return result;
}

const char* ccl_cache_key_type_to_str(ccl_cache_key_type type)
{
    switch (type)
    {
        case ccl_cache_key_full:
            return "full";
        case ccl_cache_key_match_id:
            return "match_id";
        default:
            CCL_FATAL("unknown cache_key_type ", type);
    }
}

bool ccl_sched_key::operator== (const ccl_sched_key& k) const
{
    char* first_field1 = (char*)&ctype;
    char* last_field1 = (char*)&match_id;
    void* first_field2 = (char*)&k.ctype;
    size_t bytes_to_compare = last_field1 - first_field1;
    bool is_fields_equal = (env_data.cache_key_type == ccl_cache_key_full) ?
        !memcmp(first_field1, first_field2, bytes_to_compare) : 1;

    bool is_equal = is_fields_equal && !match_id.compare(k.match_id);
    LOG_DEBUG("is_equal ", is_equal);
    print();
    k.print();
    return is_equal;
}

size_t ccl_sched_key_hasher::operator()(const ccl_sched_key& k) const
{
    if (k.has_hasher_result)
        return k.get_hasher_result();

    size_t hash_value = string_hasher(k.match_id);
    if (env_data.cache_key_type == ccl_cache_key_full)
    {
        hash_value += k.ctype + k.dtype + k.itype + k.reduction +
            k.count1 + k.count2 + k.root + (size_t)k.buf +
            (size_t)k.count3 + (size_t)k.count4 + (size_t)k.comm +
            (size_t)k.prologue_fn + (size_t)k.epilogue_fn + (size_t)k.reduction_fn;
    }

    const_cast<ccl_sched_key&>(k).set_hasher_result(hash_value);

    LOG_DEBUG("hash_value ", hash_value);
    k.print();

    return hash_value;
}
