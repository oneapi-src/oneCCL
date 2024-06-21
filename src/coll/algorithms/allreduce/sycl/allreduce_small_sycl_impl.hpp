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
#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl_rw_kernel.hpp"

template <typename T, int N, int read_all>
inline void reduce_kernel(void *recv, std::array<void *, MAX_NODE_RANKS> in, size_t idx) {
    T tmp_arr[N];
    // copy from remote to local array
    T sum = ((T *)in[0])[idx];
#pragma unroll
    for (int i = 1; i < N; i++) {
        tmp_arr[i] = ((T *)in[i])[idx];
    }

    // reduce from local array
    for (int i = 1; i < N; i++) {
        sum += tmp_arr[i];
    }

    // write to local recv buffer
    if (read_all) {
        ((T *)recv)[idx] = sum;
    }
    // write back to remote tmp buffers
    else {
#pragma unroll
        for (int i = 0; i < N; i++) {
            ((T *)in[i])[idx] = sum;
        }
    }
}

template <typename T,
          int N,
          int vec_size,
          int use_block,
          int use_local_barrier,
          int use_global_barrier,
          int read_all = 1,
          int M = 1>
void inline reduce_sum(const void *send,
                       void *recv,
                       void *tmp,
                       std::array<void *, MAX_NODE_RANKS> in,
                       size_t *sync_ptr,
                       const ccl_barrier_data barrier_data,
                       const size_t count,
                       const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    const size_t packed_count = count / vec_size;

    if (use_local_barrier) {
        // copy data from send buffer to tmp buffer
        if (use_block && idx < packed_count) {
            using MAT = sycl::marray<AT, M>;
            ((MAT *)tmp)[idx] = ((MAT *)send)[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count) {
                using MT = sycl::marray<T, M>;
                ((MT *)tmp)[new_idx] = ((MT *)send)[new_idx];
            }
        }

        // local barrier within gpu
        kernel_barrier(sync_ptr, it);
    }

    if (use_global_barrier) {
        // global communication barrier across ranks
        comm_barrier(barrier_data, it);
    }

    // reset local barrier counter
    if (use_local_barrier && idx == 0) {
        *sync_ptr = 0;
    }

    if (use_block && idx < packed_count) {
        reduce_kernel<AT, N, read_all>(recv, in, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            reduce_kernel<T, N, read_all>(recv, in, new_idx);
        }
    }
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event allreduce_small_impl(const void *send_buf,
                                void *recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm *comm,
                                ccl_stream *global_stream,
                                const ccl::vector_class<ccl::event> &deps) {
    constexpr int N = NE * NP;
    sycl::queue q = global_stream->get_native_stream();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    const size_t sec_1 = 65536; // 4 16

    auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;

    // VS : vec_size, SGS : sub_group_size, LB : use_local_barrier, GB: use_global_barrier
    auto reduce_sum_invoke =
        [=, &q]<int VS, int SGS, int LB, int GB>(std::vector<sycl::event> l_dep_events) {
            constexpr int use_block = 1;
            constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
            const size_t kernel_threads = count / vec_size + count % vec_size;
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

            size_t *sync_ptr = get_sync_ptr(true);
            ccl_barrier_data barrier_data = node_comm->barrier_inc();

            sycl::event local_event = q.submit([=](sycl::handler &h) {
                h.depends_on(l_dep_events);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, wg_size),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sg_size)]] {
                        reduce_sum<T, N, VS, use_block, LB, GB>(send_buf,
                                                                recv_buf,
                                                                local_tmp_buf,
                                                                remote_ptrs,
                                                                sync_ptr,
                                                                barrier_data,
                                                                count,
                                                                it);
                    });
            });
            return local_event;
        };

    constexpr int vec_size = 8 / (sizeof(T) / sizeof(char));

    if (ccl::global_data::env().sycl_ccl_barrier) {
        // TODO: use ccl_barrier option with read_write kernel
        sycl::event memcpy_event = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count);
        });
        // invoke cpu barrier instead of kernel comm_barrier
        kernel_event = invoke_barrier(node_comm, q, { memcpy_event }, true);
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 0, 0>({ kernel_event });
    }
    else if (count * dsize < sec_1) {
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    else {
        // overhead of local sync is too high and therefore split into two multiple kernels
        // use read_all kernel for 2 ranks since read_all with 2 ranks does not increase data
        if (comm_size <= 2) {
            sycl::event memcpy_event = q.submit([=](sycl::handler &h) {
                h.depends_on(dep_events);
                h.memcpy(local_tmp_buf, send_buf, dsize * count);
            });
            // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
            kernel_event =
                reduce_sum_invoke.template operator()<vec_size, 16, 0, 1>({ memcpy_event });
        }
        // use read_write kernel for more than 2 ranks, since read_all will increase data too much (* vs +)
        // e.g. read_all with 12 ranks has 12*12 data reads whereas read_write has 12 read + 12 write
        else {
            const int rank_align_count = ccl::global_data::env().kernel_mem_align / dsize;
            const size_t align_count = comm_size * rank_align_count;
            const size_t count_rem = count % align_count;
            const size_t count_per_rank_tmp = (count - count_rem) / comm_size;
            const size_t offset = count_per_rank_tmp * comm_rank;
            // if count is not divisible by comm_size,
            // then last rank has the remaining data
            const size_t count_per_rank =
                count_per_rank_tmp + (comm_rank == comm_size - 1 ? count_rem : 0);

            constexpr int vec_size_cp = vec_size * N;
            const int wg_size = 32, sg_size = 32;
            const size_t kernel_threads_cp = count / vec_size_cp + count % vec_size_cp;
            const size_t kernel_threads_red = count_per_rank / vec_size + count_per_rank % vec_size;
            const size_t kernel_threads = std::max(kernel_threads_cp, kernel_threads_red);
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

            std::array<void *, MAX_NODE_RANKS> remote_ptrs_rank;
            for (int i = 0; i < comm_size; i++) {
                remote_ptrs_rank[i] = (char *)(remote_ptrs[i]) + offset * dsize;
            }

            size_t *sync_ptr = get_sync_ptr(true);
            size_t *sync_ptr_next = get_sync_ptr(true);
            ccl_barrier_data barrier_data = node_comm->barrier_inc();
            ccl_barrier_data barrier_data_next = node_comm->barrier_inc();

            kernel_event = q.submit([=](sycl::handler &h) {
                h.depends_on(dep_events);
                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, wg_size),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sg_size)]] {
                        // <vec_size, use_block, use_local_barrier, use_global_barrier, read_all, multiplier>
                        reduce_sum_general<T, N, vec_size, 1, 1, 1, 0, N>(send_buf,
                                                                          recv_buf,
                                                                          local_tmp_buf,
                                                                          remote_ptrs_rank,
                                                                          sync_ptr,
                                                                          barrier_data,
                                                                          count,
                                                                          count_per_rank,
                                                                          it);

                        // local barrier within gpu
                        kernel_barrier(sync_ptr_next, it);
                        // global communication barrier across ranks
                        comm_barrier(barrier_data_next, it);
                        // reset local barrier counter
                        if (it.get_global_linear_id() == 0) {
                            *sync_ptr_next = 0;
                        }

                        // <vec_size, multiplier>
                        copy_data<T, vec_size, N>(recv_buf, local_tmp_buf, count, it);
                    });
            });
        }
    }

    return ccl::event::create_from_native(kernel_event);
}
