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

template <typename T, int N>
inline void gather_kernel(std::array<void*, MAX_NODE_RANKS> out,
                          std::array<void*, MAX_NODE_RANKS> in,
                          size_t idx) {
    T tmp_arr[N];

#pragma unroll
    // copy from remote to local array
    for (int i = 0; i < N; i++) {
        tmp_arr[i] = ((T*)in[i])[idx];
    }

#pragma unroll
    // copy from local array to output
    for (int i = 0; i < N; i++) {
        ((T*)out[i])[idx] = tmp_arr[i];
    }
}

template <typename T, int N, int vec_size, int use_block, int use_local_barrier, int use_global_barrier>
void inline gather(const void* send,
                   void* tmp,
                   std::array<void*, MAX_NODE_RANKS> out,
                   std::array<void*, MAX_NODE_RANKS> in,
                   size_t* sync_ptr,
                   const ccl_barrier_data barrier_data,
                   const size_t count,
                   const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    const size_t packed_count = count / vec_size;

    if (use_local_barrier) {
        // copy data from send buffer to tmp buffer
        if (use_block && idx < packed_count) {
            ((AT*)tmp)[idx] = ((AT*)send)[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count) {
                ((T*)tmp)[new_idx] = ((T*)send)[new_idx];
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
        gather_kernel<AT, N>(out, in, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            gather_kernel<T, N>(out, in, new_idx);
        }
    }
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event allgatherv_small_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const ccl::vector_class<size_t>& recv_counts,
                                 ccl::datatype dtype,
                                 ccl_comm* comm,
                                 ccl_stream* global_stream,
                                 const ccl::vector_class<ccl::event>& deps) {
    constexpr int N = NE * NP;
    sycl::queue q = global_stream->get_native_stream();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    const size_t count = send_count;

    // TODO: which communicator to use for scaleout
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;

    // VS : vec_size, SGS : sub_group_size, LB : use_local_barrier, GB : use_global_barrier
    auto gather_invoke = [=, &q]<int VS, int SGS, int LB, int GB>(std::vector<sycl::event> sycl_deps) {
        constexpr int use_block = 1;
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = count / vec_size + count % vec_size;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

        std::array<void*, MAX_NODE_RANKS> out_ptrs;
        // TODO: is it better to only pass recv_buf to the kernel and do this calculation there
        for (int i = 0; i < comm_size; i++) {
            out_ptrs[i] = (char*)recv_buf + i * count * dsize;
        }

        size_t* sync_ptr = get_sync_ptr(true);
        ccl_barrier_data barrier_data = node_comm->barrier_inc();

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sg_size)]] {
                    gather<T, N, VS, use_block, LB, GB>(
                        send_buf, local_tmp_buf, out_ptrs, remote_ptrs, sync_ptr, barrier_data, count, it);
                });
        });
        return local_event;
    };

    if (ccl::global_data::env().sycl_ccl_barrier) {
        // TODO: use 64 bit instead of 32 bit for larger data types
        constexpr int vec_size = 4 / (sizeof(T) / sizeof(char));
        sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count);
        });
        // invoke cpu barrier instead of kernel comm_barrier
        kernel_event = invoke_barrier(node_comm, q, { memcpy_event }, true);

        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = gather_invoke.template operator()<vec_size, 32, 0, 0>(dep_events);
    }
    else if (count % 2 == 0 || dsize >= sizeof(int)) {
        constexpr int vec_size = 8 / (sizeof(T) / sizeof(char));
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = gather_invoke.template operator()<vec_size, 32, 1, 1>(dep_events);
    }
    // for 16 bit types with odd count, use a vector size of 32 bits
    else {
        constexpr int vec_size = 4 / (sizeof(T) / sizeof(char));

        // beyond 65536, there are too many threads in local barrier which causes deadlock
        if (count * dsize <= 65536) {
            // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
            kernel_event = gather_invoke.template operator()<vec_size, 32, 1, 1>(dep_events);
        }
        else {
            sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(local_tmp_buf, send_buf, dsize * count);
            });

            // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
            kernel_event = gather_invoke.template operator()<vec_size, 32, 0, 1>(dep_events);
        }
    }
    return ccl::event::create_from_native(kernel_event);
}
