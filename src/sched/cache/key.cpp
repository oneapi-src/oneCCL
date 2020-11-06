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
#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "sched/cache/key.hpp"
#include "common/utils/enums.hpp"

#include <cstring>

std::map<ccl_cache_key_type, std::string> ccl_sched_key::key_type_names = {
    std::make_pair(ccl_cache_key_full, "full"),
    std::make_pair(ccl_cache_key_match_id, "match_id")
};

ccl_sched_key::ccl_sched_key(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    set(param, attr);
}

void ccl_sched_key::set(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    if (ccl::global_data::env().cache_key_type == ccl_cache_key_full) {
        /* to zerioize holes in memory layout */
        memset((void*)&f, 0, sizeof(ccl_sched_key_inner_fields));
    }

    f.prologue_fn = attr.prologue_fn;
    f.epilogue_fn = attr.epilogue_fn;
    f.reduction_fn = attr.reduction_fn;
    match_id = attr.match_id;

    f.ctype = param.ctype;
    f.dtype = param.dtype.idx();
    f.comm = param.comm;

    switch (f.ctype) {
        case ccl_coll_allgatherv: f.count1 = param.send_count; break;
        case ccl_coll_allreduce:
            f.count1 = param.count;
            f.reduction = param.reduction;
            break;
        case ccl_coll_alltoall: f.count1 = param.count; break;
        case ccl_coll_alltoallv:
            f.buf1 = (void*)param.send_counts;
            f.buf2 = (void*)param.recv_counts;
            break;
        case ccl_coll_barrier: break;
        case ccl_coll_bcast:
            f.count1 = param.count;
            f.root = param.root;
            break;
        case ccl_coll_reduce:
            f.count1 = param.count;
            f.reduction = param.reduction;
            f.root = param.root;
            break;
        case ccl_coll_reduce_scatter:
            f.count1 = param.count;
            f.reduction = param.reduction;
            break;
        case ccl_coll_sparse_allreduce:
            f.count1 = param.sparse_param.send_ind_count;
            f.count2 = param.sparse_param.send_val_count;
            f.count3 = param.sparse_param.recv_ind_count;
            f.count4 = param.sparse_param.recv_val_count;
            f.itype = param.sparse_param.itype.idx();
            f.reduction = param.reduction;
            break;
        default: CCL_THROW("unexpected coll_type ", f.ctype);
    }
}

bool ccl_sched_key::check(const ccl_coll_param& param, const ccl_coll_attr& attr) const {
    bool result = true;

    result &= (attr.prologue_fn == f.prologue_fn || attr.epilogue_fn == f.epilogue_fn ||
               attr.reduction_fn == f.reduction_fn || param.ctype == f.ctype ||
               param.dtype.idx() == f.dtype || param.comm == f.comm);

    switch (f.ctype) {
        case ccl_coll_allgatherv: result &= (param.send_count == f.count1); break;
        case ccl_coll_allreduce:
            result &= (param.count == f.count1 && param.reduction == f.reduction);
            break;
        case ccl_coll_alltoall: result &= (param.count == f.count1); break;
        case ccl_coll_alltoallv:
            result &= (param.send_counts == f.buf1 && param.recv_counts == f.buf2);
            break;
        case ccl_coll_barrier: break;
        case ccl_coll_bcast: result &= (param.count == f.count1 && param.root == f.root); break;
        case ccl_coll_reduce:
            result &=
                (param.count == f.count1 && param.reduction == f.reduction && param.root == f.root);
            break;
        case ccl_coll_reduce_scatter:
            result &= (param.count == f.count1 && param.reduction == f.reduction);
            break;
        case ccl_coll_sparse_allreduce:
            result &= (param.sparse_param.send_ind_count == f.count1 &&
                       param.sparse_param.send_val_count == f.count2 &&
                       param.sparse_param.recv_ind_count == f.count3 &&
                       param.sparse_param.recv_val_count == f.count4 &&
                       param.sparse_param.itype.idx() == f.itype &&
                       param.reduction == f.reduction);
            break;
        default: CCL_THROW("unexpected coll_type ", f.ctype);
    }

    return result;
}

bool ccl_sched_key::operator==(const ccl_sched_key& k) const {
    bool are_fields_equal = (ccl::global_data::env().cache_key_type == ccl_cache_key_full)
                                ? !memcmp(&f, &(k.f), sizeof(ccl_sched_key_inner_fields))
                                : 1;
    bool are_keys_equal = are_fields_equal && !match_id.compare(k.match_id);

    LOG_DEBUG("are_keys_equal ", are_keys_equal);

    print();
    k.print();
    return are_keys_equal;
}

void ccl_sched_key::print() const {
    LOG_DEBUG("ctype ",
              ccl_coll_type_to_str(f.ctype),
              ", dtype ",
              ccl::global_data::get().dtypes->name(f.dtype),
              ", itype ",
              ccl::global_data::get().dtypes->name(f.itype),
              ", reduction ",
              ccl_reduction_to_str(f.reduction),
              ", buf1 ",
              f.buf1,
              ", buf2 ",
              f.buf2,
              ", count1 ",
              f.count1,
              ", count2 ",
              f.count2,
              ", count3 ",
              f.count3,
              ", count4 ",
              f.count4,
              ", root ",
              f.root,
              ", comm ",
              f.comm,
              ", prologue_fn ",
              (void*)f.prologue_fn,
              ", epilogue_fn ",
              (void*)f.epilogue_fn,
              ", reduction_fn ",
              (void*)f.reduction_fn,
              ", match_id ",
              match_id);
}

size_t ccl_sched_key_hasher::operator()(const ccl_sched_key& k) const {
    if (k.has_hasher_result)
        return k.get_hasher_result();

    size_t hash_value = string_hasher(k.match_id);
    if (ccl::global_data::env().cache_key_type == ccl_cache_key_full) {
        hash_value += k.f.ctype + utils::enum_to_underlying(k.f.dtype) + 
                      utils::enum_to_underlying(k.f.itype) + utils::enum_to_underlying(k.f.reduction) +
                      k.f.count1 + k.f.count2 + k.f.root + (size_t)k.f.buf1 + (size_t)k.f.buf2 +
                      (size_t)k.f.count3 + (size_t)k.f.count4 + (size_t)k.f.comm +
                      (size_t)k.f.prologue_fn + (size_t)k.f.epilogue_fn + (size_t)k.f.reduction_fn;
    }

    const_cast<ccl_sched_key&>(k).set_hasher_result(hash_value);

    LOG_DEBUG("hash_value ", hash_value);
    k.print();

    return hash_value;
}
