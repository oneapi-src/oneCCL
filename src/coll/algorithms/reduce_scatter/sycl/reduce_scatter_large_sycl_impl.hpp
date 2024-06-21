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
#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl_impl.hpp"

template <typename T, int N>
void inline read_reduce_write_kernel(std::array<void *, MAX_GPUS> pair_ptrs,
                                     std::array<void *, MAX_GPUS> local_ptrs,
                                     std::array<void *, MAX_GPUS> even_ptrs,
                                     const bool is_multi_tile,
                                     size_t idx) {
    if (is_multi_tile) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            const T pair_val = ((T *)pair_ptrs[i])[idx];
            const T local_val = ((T *)local_ptrs[i])[idx];
            const T red_val = pair_val + local_val;
            ((T *)even_ptrs[i])[idx] = red_val;
        }
    }
    else {
#pragma unroll
        for (int i = 0; i < N; i++) {
            ((T *)even_ptrs[i])[idx] = ((T *)local_ptrs[i])[idx];
        }
    }
}

template <typename T, int N, int vec_size>
void inline read_reduce_write(std::array<void *, MAX_GPUS> pair_ptrs,
                              std::array<void *, MAX_GPUS> local_ptrs,
                              std::array<void *, MAX_GPUS> even_ptrs,
                              const bool is_multi_tile,
                              const size_t count,
                              const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();

    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        read_reduce_write_kernel<AT, N>(pair_ptrs, local_ptrs, even_ptrs, is_multi_tile, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            read_reduce_write_kernel<T, N>(pair_ptrs, local_ptrs, even_ptrs, is_multi_tile, new_idx);
        }
    }
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP, bool is_odd>
ccl::event reduce_scatter_large_impl(const void *send_buf,
                                     void *recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     ccl_comm *comm,
                                     ccl_stream *global_stream,
                                     const ccl::vector_class<ccl::event> &deps) {
    constexpr int N = NE;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    sycl::queue q_use = q;

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();

    constexpr int pipeline_size = 2;

    const bool is_multi_tile = pair_comm_size > 1;
    const bool is_aligned = (recv_count * dsize) % ccl::global_data::env().kernel_mem_align == 0;
    const bool is_use_tmp = ccl::global_data::env().sycl_reduce_scatter_tmp_buf || is_odd ||
                            (!is_aligned && ccl::global_data::env().sycl_auto_use_tmp_buf);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::array<void *, MAX_GPUS> l_mdfi_send_ptrs, l_xelink_work_ptrs, l_send_ptrs, l_send_ptrs_orig,
        l_send_ptrs_use;
    std::array<void *, MAX_GPUS> l_cp_src_ptrs, l_cp_dst_ptrs, l_cp_src_ptrs_next, l_cp_dst_ptrs_next;
    std::array<void *, MAX_NODE_RANKS> l_work_ptrs, l_work_ptrs_prev;
    void *l_recv_ptr = recv_buf, *l_recv_ptr_prev = recv_buf;

    const size_t recv_bytes = recv_count * dsize;
    const size_t chunk_size = get_tmp_buf_size_per_rank();
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);

    // 0 index is used for tmp work buffer and
    // 1 index is used to copy input data
    void *work_buf = get_tmp_buf(0);
    std::array<void *, MAX_GPUS> work_bufs[pipeline_size];
    std::array<void *, MAX_GPUS> xelink_work_bufs[pipeline_size];
    xelink_work_bufs[0] = get_remote_even_tmp_buf(0);
    const size_t pipeline_offset = chunk_size * even_comm_size;
    for (int i = 0; i < even_comm_size; i++) {
        work_bufs[0][i] = (char *)work_buf + chunk_size * i;
        work_bufs[1][i] = (char *)(work_bufs[0][i]) + pipeline_offset;
        xelink_work_bufs[0][i] = (char *)(xelink_work_bufs[0][i]) + chunk_size * even_comm->rank();
        xelink_work_bufs[1][i] = (char *)(xelink_work_bufs[0][i]) + pipeline_offset;
    }

    void *tmp_bufs[pipeline_size];
    tmp_bufs[0] = get_tmp_buf(1);
    tmp_bufs[1] = (char *)(tmp_bufs[0]) + pipeline_offset;

    void *tmp_send_buf = get_tmp_buf(2);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    for (size_t nc = 0; nc < num_chunks; nc++) {
        const int pipeline_index = nc % pipeline_size;
        const int pipeline_index_next = (nc + 1) % pipeline_size;

        const size_t chunk_offset = nc * chunk_size;
        const size_t data_count = ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t data_count_next = ((nc + 1 < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t data_count_prev = chunk_size / dsize;

        for (int i = 0; i < even_comm_size; i++) {
            int global_rank = even_comm->get_node_rank(i);
            // TODO: is there a better way to find the pair_neighbor global rank
            int global_rank_neighbor = (global_rank / pair_comm_size) * pair_comm_size;
            if (global_rank % pair_comm_size == 0) {
                global_rank_neighbor = global_rank_neighbor + 1;
            }

            // offset is direct offset within send_buf and
            // offset_tmp is offset within tmp_buf where data is copied from send_buf
            const size_t offset = global_rank * recv_bytes + chunk_offset;
            const size_t offset_neigh = global_rank_neighbor * recv_bytes + chunk_offset;
            const size_t offset_tmp = i * chunk_size;
            const size_t mdfi_offset_tmp = pipeline_index * pipeline_offset + offset_tmp;

            l_cp_src_ptrs[i] = (char *)send_buf + offset_neigh;
            l_cp_dst_ptrs[i] = (char *)tmp_bufs[pipeline_index] + offset_tmp;

            l_cp_src_ptrs_next[i] = (char *)l_cp_src_ptrs[i] + chunk_size;
            l_cp_dst_ptrs_next[i] = (char *)tmp_bufs[pipeline_index_next] + offset_tmp;

            l_mdfi_send_ptrs[i] = (char *)mdfi_ptr_rd + (is_use_tmp ? mdfi_offset_tmp : offset);
            l_send_ptrs_orig[i] = (char *)send_buf + offset;
            l_send_ptrs_use[i] = (char *)tmp_send_buf + offset_tmp;
            l_send_ptrs[i] = is_odd ? l_send_ptrs_use[i] : l_send_ptrs_orig[i];
            l_xelink_work_ptrs[i] = (char *)xelink_work_bufs[pipeline_index][i];

            l_work_ptrs[i] = (char *)work_bufs[pipeline_index][i];
        }
        l_recv_ptr = (char *)recv_buf + chunk_offset;

        constexpr bool subgroup_api = false;
        // for 16 bit types with odd count, use 4 byte vectors instead of 8 bytes
        constexpr int vec_size_tmp = is_odd ? 4 : 8;
        constexpr int vec_size = vec_size_tmp / (sizeof(T) / sizeof(char));
        constexpr int vec_size_cp = is_odd ? 1 : vec_size;
        const size_t work_group_size = 16;
        const size_t work_group_size_cp = is_odd ? 32 : work_group_size;
        const size_t sub_group_size = 16;
        const size_t sub_group_size_cp = is_odd ? 32 : sub_group_size;

        // pipeline prologue
        // this data copy can also be done from the main kernel as first step with a guard of nc == 0
        bool is_deps_added = false;
        if (is_use_tmp && nc == 0 && is_multi_tile) {
            is_deps_added = true;
            work_event = q_use.submit([=](sycl::handler &h) {
                const size_t kernel_threads = data_count / vec_size_cp + data_count % vec_size_cp;
                const size_t kernel_size =
                    ((kernel_threads + work_group_size_cp - 1) / work_group_size_cp) * work_group_size_cp;
                h.depends_on(dep_events);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size_cp),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sub_group_size_cp)]] {
                        copy_data<T, N, vec_size_cp, subgroup_api>(l_cp_dst_ptrs, l_cp_src_ptrs, data_count, it);
                    });
            });
        }

        if (is_cpu_barrier) {
            std::vector<sycl::event> barrier_deps;
            if (nc == 0 && !is_deps_added) {
                is_deps_added = true;
                barrier_deps = dep_events;
            }
            else {
                barrier_deps.push_back(work_event);
            }
            work_event = invoke_barrier(node_comm, q_use, barrier_deps, is_cpu_barrier);
        }

        // for 16 bit types with odd count, copy data to a tmp buffer with
        // good alignment so that read_reduce_write kernel can use the aligned data
        if (is_odd) {
            work_event = q_use.submit([=](sycl::handler &h) {
                const size_t kernel_threads = data_count / vec_size_cp + data_count % vec_size_cp;
                const size_t kernel_size =
                    ((kernel_threads + work_group_size_cp - 1) / work_group_size_cp) * work_group_size_cp;
                h.depends_on(work_event);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size_cp),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sub_group_size_cp)]] {
                        copy_data<T, N, vec_size_cp, subgroup_api>(
                            l_send_ptrs_use, l_send_ptrs_orig, data_count, it);
                    });
            });
        }

        work_event = q_use.submit([=](sycl::handler &h) {
            const size_t kernel_threads_curr = data_count / vec_size + data_count % vec_size;
            const size_t kernel_threads_prev =
                nc > 0 ? data_count_prev / vec_size + data_count_prev % vec_size : 0;
            const size_t kernel_threads_next = is_use_tmp && nc < num_chunks - 1 && is_multi_tile && !is_odd
                                                   ? data_count_next / vec_size + data_count_next % vec_size
                                                   : 0;
            const size_t kernel_threads =
                std::max({ kernel_threads_curr, kernel_threads_prev, kernel_threads_next });
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            if (nc == 0 && !is_deps_added) {
                h.depends_on(dep_events);
            }
            else {
                h.depends_on(work_event);
            }

            ccl_barrier_data barrier_data = node_comm->barrier_inc(!is_cpu_barrier);

            h.parallel_for(
                sycl::nd_range<1>(kernel_size, work_group_size),
                [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sub_group_size)]] {
                    comm_barrier(barrier_data, it, !is_cpu_barrier);

                    read_reduce_write<T, N, vec_size>(
                        l_mdfi_send_ptrs, l_send_ptrs, l_xelink_work_ptrs, is_multi_tile, data_count, it);

                    if (nc > 0) {
                        reduce_sum<T, N, vec_size, 1, 0, 0>(nullptr,
                                                            l_recv_ptr_prev,
                                                            nullptr,
                                                            l_work_ptrs_prev,
                                                            nullptr,
                                                            barrier_data,
                                                            data_count_prev,
                                                            it);
                    }

                    if (is_use_tmp && nc < num_chunks - 1 && is_multi_tile && !is_odd) {
                        copy_data<T, N, vec_size, subgroup_api>(
                            l_cp_dst_ptrs_next, l_cp_src_ptrs_next, data_count_next, it);
                    }
                });
        });

        if (is_use_tmp && nc < num_chunks - 1 && is_multi_tile && is_odd) {
            work_event = q_use.submit([=](sycl::handler &h) {
                const size_t kernel_threads = data_count_next / vec_size_cp + data_count_next % vec_size_cp;
                const size_t kernel_size =
                    ((kernel_threads + work_group_size_cp - 1) / work_group_size_cp) * work_group_size_cp;
                h.depends_on(work_event);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size_cp),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sub_group_size_cp)]] {
                        copy_data<T, N, vec_size_cp, subgroup_api>(
                            l_cp_dst_ptrs_next, l_cp_src_ptrs_next, data_count_next, it);
                    });
            });
        }

        // save prev pointers to be used in next iteration
        for (int i = 0; i < even_comm_size; i++) {
            l_work_ptrs_prev[i] = l_work_ptrs[i];
        }
        l_recv_ptr_prev = l_recv_ptr;

        // pipeline epilogue
        // this reduction can also be done from the main kernel as last step with a guard of nc == num_chunks - 1
        if (nc == num_chunks - 1) {
            work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);

            const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
            ccl_barrier_data barrier_data = node_comm->barrier_data();

            work_event = q_use.submit([=](sycl::handler &h) {
                h.depends_on(work_event);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sub_group_size)]] {
                        reduce_sum<T, N, vec_size, 1, 0, 0>(
                            nullptr, l_recv_ptr, nullptr, l_work_ptrs, nullptr, barrier_data, data_count, it);
                    });
            });
        }
    }

    return ccl::event::create_from_native(work_event);
}
