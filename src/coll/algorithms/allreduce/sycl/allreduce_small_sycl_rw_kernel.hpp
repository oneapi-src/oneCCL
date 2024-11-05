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
template <typename T, int N, int read_all>
inline void reduce_kernel(void *recv,
                          std::array<void *, MAX_NODE_RANKS> in,
                          std::array<void *, MAX_NODE_RANKS> out,
                          size_t idx);

// copy data from src to dst
template <typename T, int vec_size, int M>
void inline copy_data(void *dst, const void *src, const size_t count, const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    constexpr int vec_size_cp = vec_size * M;
    const size_t packed_count = count / vec_size_cp;

    if (idx < packed_count) {
        using MAT = sycl::marray<AT, M>;
        ((MAT *)dst)[idx] = ((MAT *)src)[idx];
    }
    else {
        const size_t new_idx = idx + (vec_size_cp - 1) * packed_count;
        if (new_idx < count) {
            ((T *)dst)[new_idx] = ((T *)src)[new_idx];
        }
    }
}

template <typename T,
          int N,
          int vec_size,
          int use_block,
          int use_local_barrier,
          int use_global_barrier,
          int read_all,
          int M>
void inline reduce_sum_general(const void *send,
                               void *recv,
                               void *tmp,
                               std::array<void *, MAX_NODE_RANKS> in,
                               std::array<void *, MAX_NODE_RANKS> out,
                               ccl_kernel_barrier_data kernel_barrier_data,
                               const ccl_comm_barrier_data comm_barrier_data,
                               const size_t count_cp,
                               const size_t count_red,
                               const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    if (use_local_barrier) {
        // copy data from send buffer to local temp buffer
        copy_data<T, vec_size, M>(tmp, send, count_cp, it);

        // local barrier within gpu
        kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);
    }

    if (use_global_barrier) {
        // global communication barrier across ranks
        comm_barrier(comm_barrier_data, it);
    }

    // reset local barrier counter
    if (use_local_barrier && idx == 0) {
        kernel_barrier_data.reset_sync_data();
    }

    const size_t packed_count = count_red / vec_size;

    // reduce data from all ranks
    if (idx < packed_count) {
        reduce_kernel<AT, N, read_all>(recv, in, out, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count_red) {
            reduce_kernel<T, N, read_all>(recv, in, out, new_idx);
        }
    }
}
