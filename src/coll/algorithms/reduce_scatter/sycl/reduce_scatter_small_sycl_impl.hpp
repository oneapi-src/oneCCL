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

    size_t hw_threads = get_total_threads(q);

    // create sections as data size increases and increase the vector size
    // used in each thread so as to reduce the number of threads and this
    // reduces the overhead of atomic increment in local sync
    // and help to contain number of sycl thread <= hw threads.
    // for data sizes not contained within the sections, divide single kernel
    // into multiple kernels to avoid deadlocks when sycl thread > hw threads
    // currently a maximum of 2048 threads are used in each sections.
    const size_t sec_0 = std::min(2048ul, hw_threads);
    const size_t sec_1 = sec_0 * 8; // 8 byte vectors
    const size_t sec_2 = sec_1 * 2; // 16 byte vectors
    const size_t sec_3 = sec_2 * 2; // 32 byte vectors

    // use full vector (>= 8 bytes) if buffers and data size are 4 byte aligned
    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count * dsize);
    const bool use_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    const bool use_kernel_sync = ccl::global_data::env().sycl_kernel_sync;

    auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;

    // VS : vec_size, SGS : sub_group_size, LB : use_local_barrier, GB : use_global_barrier
    auto reduce_sum_invoke = [=, &q]<int VS, int SGS, int LB, int GB>(std::vector<sycl::event> l_dep_events) {
        constexpr int use_block = 1;
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = count / vec_size + count % vec_size;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

        // total number of hw threads is a multiple of sub_group
        CCL_THROW_IF_NOT(hw_threads % SGS == 0);
        // if synchronization is there, make sure sycl threads can be contained within hw threads
        if (LB == 1 || GB == 1) {
            if (kernel_size > hw_threads) {
                CCL_THROW("sycl threads : ",
                          kernel_size,
                          " > hw threads : ",
                          hw_threads,
                          " is not allowed in reduce_scatter small for count :",
                          count);
            }
        }

        std::array<void*, MAX_NODE_RANKS> remote_ptrs_offset;
        // TODO: is it better to only pass remote_ptrs to the kernel and do this calculation there
        for (int i = 0; i < comm_size; i++) {
            remote_ptrs_offset[i] = (char*)remote_ptrs[i] + comm_rank * count * dsize;
        }

        ccl_kernel_barrier_data kernel_barrier_data = get_kernel_barrier_data().inc_slot();
        // if global barrier is not used, then do not increment the barrier counter
        ccl_comm_barrier_data comm_barrier_data = GB ? node_comm->barrier_inc() : node_comm->barrier_data();

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(l_dep_events);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sg_size)]] {
                    reduce_sum<T, N, VS, use_block, LB, GB, 1, N>(send_buf,
                                                                  recv_buf,
                                                                  local_tmp_buf,
                                                                  remote_ptrs_offset,
                                                                  remote_ptrs_offset,
                                                                  kernel_barrier_data,
                                                                  comm_barrier_data,
                                                                  count,
                                                                  it);
                });
        });
        return local_event;
    };

    // three phases of allgather: local copy to tmp, comm_barrier, reduce
    // VS : vec_size
    auto memcpy_reduce_sum = [=, &q]<int VS>() {
        sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count * comm_size);
        });

        sycl::event barrier_event = invoke_barrier(node_comm, q, { memcpy_event }, use_cpu_barrier);

        sycl::event local_event = reduce_sum_invoke.template operator()<VS, 32, 0, 0>({ barrier_event });

        return local_event;
    };

    // run the three phases of collective as separate kernels.
    // when cpu barrier is enabled we cannot use single gpu kernel
    // since control has to go to cpu and perform the barrier.
    // also when user asks to remove the synchronization within kernel
    // run them as separate kernels since single kernel algorithm
    // will require the threads to synchronize between phases
    if (use_cpu_barrier || !use_kernel_sync) {
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8>();
            kernel_event = memcpy_reduce_sum.template operator()<vec_size>();
        }
        else {
            // for unaligned data use vector size of 1
            kernel_event = memcpy_reduce_sum.template operator()<1>();
        }
    }
    // for unaligned data use vector size of 1
    else if (!use_full_vector) {
        // if sycl threads are not more than hw threads we can
        // use single kernel that executes barrier internally
        if (count <= sec_0) {
            // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
            kernel_event = reduce_sum_invoke.template operator()<1, 32, 1, 1>(dep_events);
        }
        else {
            kernel_event = memcpy_reduce_sum.template operator()<1>();
        }
    }
    else if (count * dsize <= sec_0) {
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<1, 16, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_1) {
        constexpr int vec_size = get_num_elements<T, 8>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_2) {
        constexpr int vec_size = get_num_elements<T, 16>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 32, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_3) {
        constexpr int vec_size = get_num_elements<T, 32>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    // overhead of local sync is too high and therefore split into two multiple kernels
    // also sycl threads more than hw threads can cause deadlock
    else {
        constexpr int vec_size = get_num_elements<T, 8>();
        kernel_event = memcpy_reduce_sum.template operator()<vec_size>();
    }

    return ccl::event::create_from_native(kernel_event);
}
