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

#ifdef CCL_ENABLE_NUMA
#include <numa.h>
#endif // CCL_ENABLE_NUMA

#include "coll.hpp"

/* cpu-specific base implementation */
template <class Dtype, class strategy>
struct cpu_base_coll : base_coll, protected strategy {
    using coll_strategy = strategy;

    template <class... Args>
    cpu_base_coll(bench_init_attr init_attr, Args&&... args)
            : base_coll(init_attr),
              coll_strategy() {
        int result = 0;

        size_t send_multiplier = coll_strategy::get_send_multiplier();
        size_t recv_multiplier = coll_strategy::get_recv_multiplier();

        for (size_t rank_idx = 0; rank_idx < base_coll::get_ranks_per_proc(); rank_idx++) {
            for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                send_bufs[idx][rank_idx] =
                    alloc_buffer(base_coll::get_max_elem_count() * sizeof(Dtype) * send_multiplier);
                if (base_coll::get_inplace()) {
                    recv_bufs[idx][rank_idx] = send_bufs[idx][rank_idx];
                }
                else {
                    recv_bufs[idx][rank_idx] = alloc_buffer(base_coll::get_max_elem_count() *
                                                            sizeof(Dtype) * recv_multiplier);
                }
            }
        }

        ASSERT(result == 0, "failed to allocate buffers");
    }

    cpu_base_coll(bench_init_attr init_attr) : cpu_base_coll(init_attr, 1, 1) {}

    virtual ~cpu_base_coll() {
        size_t send_multiplier = coll_strategy::get_send_multiplier();
        size_t recv_multiplier = coll_strategy::get_recv_multiplier();
        for (size_t rank_idx = 0; rank_idx < base_coll::get_ranks_per_proc(); rank_idx++) {
            for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                free_buffer(send_bufs[idx][rank_idx],
                            base_coll::get_max_elem_count() * sizeof(Dtype) * send_multiplier);
                if (!base_coll::get_inplace()) {
                    free_buffer(recv_bufs[idx][rank_idx],
                                base_coll::get_max_elem_count() * sizeof(Dtype) * recv_multiplier);
                }
            }
        }
    }

    const char* name() const noexcept override {
        return coll_strategy::class_name();
    }

    virtual void start(size_t count,
                       size_t buf_idx,
                       const bench_exec_attr& attr,
                       req_list_t& reqs) override {
        auto& transport = transport_data::instance();
        auto& comms = transport.get_comms();
        size_t ranks_per_proc = base_coll::get_ranks_per_proc();

        for (size_t rank_idx = 0; rank_idx < ranks_per_proc; rank_idx++) {
            coll_strategy::start_internal(comms[rank_idx],
                                          count,
                                          static_cast<Dtype*>(send_bufs[buf_idx][rank_idx]),
                                          static_cast<Dtype*>(recv_bufs[buf_idx][rank_idx]),
                                          attr,
                                          reqs,
                                          coll_strategy::get_op_attr(attr));
        }
    }

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        int local_rank = comm.rank();

        size_t send_count = coll_strategy::get_send_multiplier() * elem_count;
        size_t recv_count = coll_strategy::get_recv_multiplier() * elem_count;

        size_t send_bytes = send_count * base_coll::get_dtype_size();
        size_t recv_bytes = recv_count * base_coll::get_dtype_size();

        std::vector<Dtype> fill_vector(send_count);
        std::fill(fill_vector.begin(), fill_vector.end(), local_rank);

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            memcpy(send_bufs[b_idx][rank_idx], fill_vector.data(), send_bytes);
            if (!base_coll::get_inplace()) {
                memset(recv_bufs[b_idx][rank_idx], 0, recv_bytes);
            }
        }
    }

    void* alloc_buffer(size_t bytes) {
        void* ptr = nullptr;
#ifdef CCL_ENABLE_NUMA
        int numa_node = base_coll::get_numa_node();
        if (numa_node != DEFAULT_NUMA_NODE) {
            ASSERT(numa_available() >= 0, "libnuma is not available");
            ASSERT(numa_node <= numa_max_node(),
                   "requsted NUMA node %d is larger than max NUMA node %d",
                   numa_node,
                   numa_max_node());

            long long free_bytes = 0;
            numa_node_size64(numa_node, &free_bytes);
            ASSERT(bytes <= (size_t)free_bytes,
                   "no enough free memory on NUMA node %d, requested %zu, free %lld",
                   numa_node,
                   bytes,
                   free_bytes);

            ptr = numa_alloc_onnode(bytes, numa_node);
            ASSERT(
                ptr, "failed to allocate buffer with size %zu on NUMA node %d", bytes, numa_node);
        }
        else
#endif // CCL_ENABLE_NUMA
        {
            size_t alignment = REG_MSG_ALIGNMENT;
            if (bytes >= LARGE_MSG_THRESHOLD)
                alignment = LARGE_MSG_ALIGNMENT;

            int result = posix_memalign(&ptr, alignment, bytes);
            ASSERT((result == 0) && ptr, "failed to allocate buffer with size %zu", bytes);
        }

        return ptr;
    }

    void free_buffer(void* ptr, size_t bytes) {
#ifdef CCL_ENABLE_NUMA
        int numa_node = base_coll::get_numa_node();
        if (numa_node != DEFAULT_NUMA_NODE) {
            ASSERT(numa_available() >= 0, "libnuma is not available");
            numa_free(ptr, bytes);
        }
        else
#endif // CCL_ENABLE_NUMA
        {
            free(ptr);
        }
    }

    ccl::datatype get_dtype() const override final {
        return get_ccl_dtype<Dtype>();
    }
};
