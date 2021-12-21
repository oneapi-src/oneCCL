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
#include <numeric>
#include <sstream>

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/log/log.hpp"

const char* ccl_coll_type_to_str(ccl_coll_type type) {
    switch (type) {
        case ccl_coll_allgatherv: return "allgatherv";
        case ccl_coll_allreduce: return "allreduce";
        case ccl_coll_alltoall: return "alltoall";
        case ccl_coll_alltoallv: return "alltoallv";
        case ccl_coll_barrier: return "barrier";
        case ccl_coll_bcast: return "bcast";
        case ccl_coll_reduce: return "reduce";
        case ccl_coll_reduce_scatter: return "reduce_scatter";
        case ccl_coll_sparse_allreduce: return "sparse_allreduce";
        case ccl_coll_partial: return "partial";
        case ccl_coll_undefined: return "undefined";
        default: return "unknown";
    }
    return "unknown";
}

void ccl_get_segment_sizes(size_t dtype_size,
                           size_t elem_count,
                           size_t requested_seg_size,
                           std::vector<size_t>& seg_sizes) {
    seg_sizes.clear();

    if (dtype_size * elem_count == 0) {
        return;
    }
    else if (dtype_size >= requested_seg_size) {
        seg_sizes.resize(elem_count, 1);
    }
    else {
        size_t seg_size = (requested_seg_size + dtype_size - 1) / dtype_size;
        size_t total_seg_count = std::max((elem_count + seg_size - 1) / seg_size, 1UL);
        size_t regular_seg_size = elem_count / total_seg_count;
        size_t large_seg_size = regular_seg_size + ((elem_count % total_seg_count) != 0);
        size_t regular_seg_count = total_seg_count * large_seg_size - elem_count;

        seg_sizes.resize(total_seg_count, regular_seg_size);
        std::fill(seg_sizes.begin() + regular_seg_count, seg_sizes.end(), large_seg_size);

        size_t sum = std::accumulate(seg_sizes.begin(), seg_sizes.end(), 0);
        if (sum != elem_count) {
            std::stringstream ss;
            for (size_t idx = 0; idx < seg_sizes.size(); idx++) {
                ss << seg_sizes[idx] << " ";
            }
            CCL_THROW_IF_NOT(false,
                             "unexpected sum of seg_sizes ",
                             sum,
                             ", expected ",
                             elem_count,
                             ", total_seg_count ",
                             total_seg_count,
                             ", regular_seg_count ",
                             regular_seg_count,
                             ", regular_seg_size ",
                             regular_seg_size,
                             ", large_seg_size ",
                             large_seg_size,
                             ", all seg_sizes: ",
                             ss.str());
        }
    }
}
