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
#pragma once
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_kernels.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl_impl.hpp"

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event reduce_scatter_small_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     ccl_comm* comm,
                                     ccl_stream* global_stream,
                                     const ccl::vector_class<ccl::event>& deps) {
    constexpr int N = NE * NP;
    sycl::queue q = global_stream->get_native_stream();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    const size_t count = recv_count;
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    // as data size increases, increase the vector size and subgroup size
    // to reduce the overhead of atomic increment in local sync phase
    const size_t sec_1 = 32800;
    const size_t sec_2 = 131100;

    auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;

    // VS : vec_size, SGS : sub_group_size, LB : use_local_barrier, GB : use_global_barrier
    auto reduce_sum_invoke = [=, &q]<int VS, int SGS, int LB, int GB>(std::vector<sycl::event> l_dep_events) {
        constexpr int use_block = 1;
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = count / vec_size + count % vec_size;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

        std::array<void*, MAX_NODE_RANKS> remote_ptrs_offset;
        // TODO: is it better to only pass remote_ptrs to the kernel and do this calculation there
        for (int i = 0; i < comm_size; i++) {
            remote_ptrs_offset[i] = (char*)remote_ptrs[i] + comm_rank * count * dsize;
        }

        size_t* sync_ptr = get_sync_ptr(true);
        ccl_barrier_data barrier_data = node_comm->barrier_inc();

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(l_dep_events);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sg_size)]] {
                    reduce_sum<T, N, VS, use_block, LB, GB, 1, N>(
                        send_buf, recv_buf, local_tmp_buf, remote_ptrs_offset, sync_ptr, barrier_data, count, it);
                });
        });
        return local_event;
    };

    if (count == 0) {
        kernel_event = submit_wait_on_events(q, dep_events);
    }
    else if (ccl::global_data::env().sycl_ccl_barrier) {
        // TODO: use 64 bit instead of 32 bit for larger data types
        constexpr int vec_size = 4 / (sizeof(T) / sizeof(char));
        sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count * comm_size);
        });
        // invoke cpu barrier instead of kernel comm_barrier
        kernel_event = invoke_barrier(node_comm, q, { memcpy_event }, true);
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 32, 0, 0>({ kernel_event });
    }
    // for 16 bit types with odd count or non 32 bit aligned data, use a vector size of 32 bits
    else if (!can_use_full_vector(send_buf, recv_buf, count * dsize)) {
        constexpr int vec_size = 4 / (sizeof(T) / sizeof(char));
        sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count * comm_size);
        });
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 32, 0, 1>({ memcpy_event });
    }
    else if (count * dsize < sec_1) {
        constexpr int vec_size = 8 / (sizeof(T) / sizeof(char));
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    else if (count * dsize < sec_2) {
        constexpr int vec_size = 16 / (sizeof(T) / sizeof(char));
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 32, 1, 1>(dep_events);
    }
    // overhead of local sync is too high and therefore split into two multiple kernels
    else {
        constexpr int vec_size = 16 / (sizeof(T) / sizeof(char));
        sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count * comm_size);
        });
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 32, 0, 1>({ memcpy_event });
    }

    return ccl::event::create_from_native(kernel_event);
}
