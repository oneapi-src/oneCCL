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

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <sycl/sycl.hpp>
#include <ext/intel/esimd.hpp>

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "comm/comm.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "ccl_api_functions_generators.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"
#include "coll/algorithms/utils/sycl_kernels.hpp"

// TODO: timers can re used, but place in more general place
class timer {
public:
    virtual double get_us(uint32_t i) const = 0;
    virtual int size() const = 0;
};

template <uint32_t steps_per_instance = 1>
class gpu_timer : timer {
    std::array<sycl::event, steps_per_instance> v_events;

public:
    inline void record(uint32_t i, sycl::event e) {
        v_events[i] = e;
    }
    double get_us(uint32_t i) const {
        auto start =
            v_events[i].template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end =
            v_events[i].template get_profiling_info<sycl::info::event_profiling::command_end>();
        return (end - start) / 1000.0;
    }
    double get_start_us(uint32_t i) const {
        auto start =
            v_events[i].template get_profiling_info<sycl::info::event_profiling::command_start>();
        return start / 1000.0;
    }
    double get_end_us(uint32_t i) const {
        auto end =
            v_events[i].template get_profiling_info<sycl::info::event_profiling::command_end>();
        return end / 1000.0;
    }
    int size() const {
        return steps_per_instance;
    }
};

template <uint32_t steps_per_instance = 1>
class cpu_timer : timer {
    std::array<std::chrono::time_point<std::chrono::steady_clock>, steps_per_instance> v_start,
        v_end;

public:
    inline void start(uint32_t i) {
        v_start[i] = std::chrono::steady_clock::now();
    }
    inline void stop(uint32_t i) {
        v_end[i] = std::chrono::steady_clock::now();
    }
    double get_us(uint32_t i) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(v_end[i] - v_start[i]).count();
    }
    int size() const {
        return steps_per_instance;
    }
};

inline void gpu_kernel_copy(char *d, const char *s, size_t n) {
    while (n >= 8) {
        *(int64_t *)d = *(int64_t *)s;
        d += 8;
        s += 8;
        n -= 8;
    }
    if (n & 4) {
        *(int *)d = *(int *)s;
        d += 4;
        s += 4;
        n -= 4;
    }
    if (n & 2) {
        *(int16_t *)d = *(int16_t *)s;
        d += 2;
        s += 2;
        n -= 2;
    }
    if (n == 1) {
        *(char *)d = *(char *)s;
    }
}

template <typename data_type>
struct sycl_coll_base {
public:
    sycl_coll_base() {
        initialized = false;
    }

    inline int inited() {
        return initialized;
    }

protected:
    void exchange_peer_ipc_mem(sycl::queue &queue,
                               ccl_comm *comm,
                               ccl_stream *stream,
                               void *send_ptr,
                               void *recv_ptr,
                               int rank,
                               int world,
                               int data_size_per_buffer,
                               void **send_buffers,
                               void **sync_buffer,
                               size_t *offsets,
                               ze_ipc_mem_handle_t *ipc_handle,
                               void **recv_buffers,
                               void **mmap_buffers = NULL,
                               bool to_cache = true) {
        // use infrastructure
        // 10us to create a sched
        // 6-14us to create exchange_entry
        // 80-128us  to call update
        // 10us to fill buffers
        // 20-30us to free
        ccl_comm *node_comm = comm->get_node_comm().get();
        std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

        in_buffers = {
            { send_ptr, ccl::ze::ipc_mem_type::memory },
        };
        if (recv_ptr) {
            in_buffers.push_back({ recv_ptr, ccl::ze::ipc_mem_type::memory });
        }

        ccl_coll_param param{};
        param.comm = comm;
        param.stream = stream;
        ccl_coll_attr attr{};
        ccl_sched *sched = ccl_sched::create(param, attr);

        ccl::utils::pt2pt_handle_exchange_info info = {};
        int skip_rank = ccl_comm::invalid_rank;
        ze_handle_exchange_entry *exchange_entry =
            new ze_handle_exchange_entry(sched, node_comm, in_buffers, skip_rank, info);
        // start the entry
        exchange_entry->start();
        while (!exchange_entry->is_completed()) {
            exchange_entry->update(); //    128us
        }

        size_t send_buf_idx = 0;
        std::vector<ccl_buffer> peer_send_bufs(world - 1);
        for (int i = 0; i < world - 1; i++) {
            int peer_rank = (rank + i + 1) % world;
            sched->get_memory().handle_manager.get(peer_rank,
                                                   send_buf_idx,
                                                   peer_send_bufs[i],
                                                   node_comm,
                                                   false /*pt2pt_op*/,
                                                   to_cache);
            send_buffers[peer_rank] = peer_send_bufs[i].get_ptr();
            CCL_THROW_IF_NOT(send_buffers[peer_rank], "null IPC buffer is received");
        }
        send_buffers[rank] = send_ptr;
        if (sync_buffer) {
            for (int i = 0; i < world; i++) {
                sync_buffer[i] = (char *)send_buffers[i] + data_size_per_buffer;
            }
        }
        if (recv_ptr) {
            size_t recv_buf_idx = 1;
            std::vector<ccl_buffer> peer_recv_bufs(world - 1);
            for (int i = 0; i < world - 1; i++) {
                int peer_rank = (rank + i + 1) % world;
                sched->get_memory().handle_manager.get(peer_rank,
                                                       recv_buf_idx,
                                                       peer_recv_bufs[i],
                                                       node_comm,
                                                       false /*pt2pt_op*/,
                                                       to_cache);
                recv_buffers[peer_rank] = peer_recv_bufs[i].get_ptr();
                CCL_THROW_IF_NOT(recv_buffers[peer_rank], "null IPC buffer is received");
            }
            recv_buffers[rank] = recv_ptr;
        }
        delete exchange_entry;
        delete sched;
    }

    bool initialized;
};

void pipe_prep(size_t min_msg_count, size_t max_msg_count, int dsize, size_t &nchunks);

sycl::event pipe_sendrecv(sycl::queue &q,
                          const void *send_buf,
                          size_t send_count,
                          int dest,
                          int sendtag,
                          void *recv_buf,
                          size_t recv_count,
                          int src,
                          int recvtag,
                          ccl::datatype dtype,
                          size_t nchunks,
                          ccl_comm *comm,
                          const ccl::vector_class<sycl::event> &deps,
                          bool use_rdma = false);

class ccl_kernel_barrier_data {
public:
    static constexpr int slots = 3;

    ccl_kernel_barrier_data() = default;
    ccl_kernel_barrier_data(const ccl_kernel_barrier_data &) = default;
    ccl_kernel_barrier_data &operator=(const ccl_kernel_barrier_data &) = default;

    void set_sync_ptrs(size_t *ptrs) {
        m_sync_ptrs = ptrs;
    }

    // advance to the next slot
    ccl_kernel_barrier_data inc_slot(int count = 1) {
        m_count += count;
        return *this;
    }

    // get current slot pointer
    size_t *get_sync_ptr() const {
        size_t curr_slot = m_count % slots;
        return m_sync_ptrs + curr_slot;
    }

    // reset data in the farthest slot
    void reset_sync_data() {
        size_t reset_slot = (m_count + (slots - 1)) % slots;
        m_sync_ptrs[reset_slot] = 0;
    }

private:
    size_t *m_sync_ptrs;
    size_t m_count = 0;
};

std::pair<ccl_sched *, ze_handle_exchange_entry *> do_ipc_exchange(ccl_comm *comm,
                                                                   ccl_stream *stream,
                                                                   std::vector<void *> ptrs);

void coll_init(ccl_comm *comm, ccl_stream *stream);

ccl_kernel_barrier_data &get_kernel_barrier_data();

void *get_tmp_buf(int index);

std::array<void *, MAX_NODE_RANKS> get_remote_node_tmp_buf(int index);

std::array<void *, MAX_GPUS> get_remote_even_tmp_buf(int index);

std::array<void *, MAX_TILES> get_remote_pair_tmp_buf(int index);

size_t get_tmp_buf_size_per_rank();

std::vector<sycl::event> get_sycl_events(const ccl::vector_class<ccl::event> &deps);

sycl::event invoke_barrier(const std::shared_ptr<ccl_comm> comm,
                           sycl::queue q,
                           const std::vector<sycl::event> &dep_events,
                           bool use_cpu);

int get_num_lce();

sycl::queue get_mce_queue(sycl::queue q);

sycl::queue get_lce_queue(sycl::queue q, int index);

sycl::queue &get_default_queue();

void copy_data(const int dsize,
               const int N,
               std::array<void *, MAX_GPUS> dst,
               std::array<void *, MAX_GPUS> src,
               const size_t count,
               sycl::queue q,
               std::vector<sycl::event> deps,
               std::vector<sycl::event> &out);

template <typename T, int N>
std::array<T *, N> get_ipc_ptrs(std::shared_ptr<ccl_comm> comm,
                                const int handle_index,
                                void *local_ptr,
                                ccl_sched *sched,
                                sycl::queue q,
                                bool dummy_copy = false) {
    std::array<T *, N> remote_ptrs;

    const int rank = comm->rank();
    const int size = comm->size();
    remote_ptrs[rank] = (T *)local_ptr;

    for (int i = 1; i < size; i++) {
        int peer_rank = (rank + i) % size;
        ccl_buffer tmp_ccl_buf;
        sched->get_memory().handle_manager.get(peer_rank, handle_index, tmp_ccl_buf, comm.get());
        CCL_THROW_IF_NOT(tmp_ccl_buf.get_ptr(), "null IPC buffer is received");
        remote_ptrs[peer_rank] = (T *)tmp_ccl_buf.get_ptr();
        if (dummy_copy) {
            q.memcpy(remote_ptrs[rank], remote_ptrs[peer_rank], 1);
        }
    }
    return remote_ptrs;
}

template <typename T, int N>
std::array<T *, N> get_ipc_ptrs(std::shared_ptr<ccl_comm> comm,
                                const int handle_index,
                                void *local_ptr,
                                ccl_sched *sched,
                                bool dummy_copy = false) {
    auto q = get_default_queue();

    return get_ipc_ptrs<T, N>(std::move(comm), handle_index, local_ptr, sched, q, dummy_copy);
}

template <int NE, int NP, typename L>
ccl::event invoke_collective_type(L lambda, ccl::datatype dtype) {
    ccl::event e;
    switch (dtype) {
        case ccl::datatype::int16: e = lambda.template operator()<short, NE, NP>(); break;
        case ccl::datatype::float16:
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
            e = lambda.template operator()<sycl::half, NE, NP>();
#else
            CCL_THROW(
                "The Sycl compilers do not support Sycl::vec kernels with float16, please switch to ESIMD kernels, or build oneCCL with the latest version of cmake and oneAPI compiler");
#endif
            break;
        case ccl::datatype::bfloat16:
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
            e = lambda.template operator()<sycl::ext::oneapi::bfloat16, NE, NP>();
#else
            CCL_THROW(
                "The Sycl compilers do not support Sycl::vec kernels with bfloat16, please switch to ESIMD kernels, or build oneCCL with oneAPI compiler that is newer than 2024.2.0");
#endif
            break;
        case ccl::datatype::float32: e = lambda.template operator()<float, NE, NP>(); break;
        case ccl::datatype::int32: e = lambda.template operator()<int, NE, NP>(); break;
        default: CCL_THROW("unsupported datatype ", dtype); break;
    }
    return e;
}

template <int NP, typename L>
ccl::event invoke_collective_size(L lambda, int even_comm_size, ccl::datatype dtype) {
    ccl::event e;
    switch (even_comm_size) {
        case 1: e = invoke_collective_type<1, NP>(lambda, dtype); break;
        case 2: e = invoke_collective_type<2, NP>(lambda, dtype); break;
        case 3: e = invoke_collective_type<3, NP>(lambda, dtype); break;
        case 4: e = invoke_collective_type<4, NP>(lambda, dtype); break;
        case 5: e = invoke_collective_type<5, NP>(lambda, dtype); break;
        case 6: e = invoke_collective_type<6, NP>(lambda, dtype); break;
        case 7: e = invoke_collective_type<7, NP>(lambda, dtype); break;
        case 8: e = invoke_collective_type<8, NP>(lambda, dtype); break;
        default: CCL_THROW("unsupported even_comm size ", even_comm_size); break;
    }
    return e;
}

template <typename L>
ccl::event invoke_collective(L lambda, ccl_comm *global_comm, ccl::datatype dtype) {
    std::shared_ptr<ccl_comm> pair_comm = global_comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = global_comm->get_even_comm();

    ccl::event e;
    switch (pair_comm->size()) {
        case 1: e = invoke_collective_size<1>(lambda, even_comm->size(), dtype); break;
        case 2: e = invoke_collective_size<2>(lambda, even_comm->size(), dtype); break;
        default: CCL_THROW("unsupported pair_comm size ", pair_comm->size()); break;
    }
    return e;
}

template <typename L>
sycl::event invoke_scaleout(L lambda, ccl::datatype dtype) {
    sycl::event e;
    switch (dtype) {
        case ccl::datatype::int16: e = lambda.template operator()<short>(); break;
        case ccl::datatype::float16:
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
            e = lambda.template operator()<sycl::half>();
#else
            CCL_THROW(
                "The Sycl compilers do not support Sycl::vec kernels with float16, please switch to ESIMD kernels, or build oneCCL with the latest version of cmake and oneAPI compiler");
#endif
            break;
        case ccl::datatype::bfloat16:
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
            e = lambda.template operator()<sycl::ext::oneapi::bfloat16>();
#else
            CCL_THROW(
                "The Sycl compilers do not support Sycl::vec kernels with bfloat16, please switch to ESIMD kernels, or build oneCCL with oneAPI compiler that is newer than 2024.2.0");
#endif
            break;
        case ccl::datatype::float32: e = lambda.template operator()<float>(); break;
        case ccl::datatype::int32: e = lambda.template operator()<int>(); break;
        default: CCL_THROW("unsupported datatype ", dtype); break;
    }
    return e;
}

template <typename T>
T *ptr_offset(T *ptr, size_t offset) {
    return static_cast<char *>(ptr) + offset;
}

template <typename T>
const T *ptr_offset(const T *ptr, size_t offset) {
    return static_cast<const char *>(ptr) + offset;
}

inline bool is_aligned(const void *buf, const size_t size, const int alignment) {
    return (size_t)buf % alignment == 0 && size % alignment == 0;
}

inline bool is_aligned(const void *send_buf,
                       const void *recv_buf,
                       const size_t size,
                       const int alignment) {
    return (size_t)send_buf % alignment == 0 && (size_t)recv_buf % alignment == 0 &&
           size % alignment == 0;
}

inline bool all_aligned(std::vector<void *> ptrs, size_t size, size_t alignment) {
    if (size % alignment) {
        return false;
    }
    for (const void *ptr : ptrs) {
        if ((size_t)ptr % alignment) {
            return false;
        }
    }
    return true;
}

inline bool all_aligned(void **ptrs, int count, size_t size, size_t alignment) {
    if (size % alignment) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        if ((size_t)ptrs[i] % alignment) {
            return false;
        }
    }
    return true;
}

inline bool can_use_full_vector(const void *send_buf,
                                const void *recv_buf,
                                const size_t size,
                                const int alignment = 4) {
    return is_aligned(send_buf, recv_buf, size, alignment) &&
           ccl::global_data::env().sycl_full_vector;
}

inline sycl::event submit_wait_on_events(sycl::queue q, const std::vector<sycl::event> &deps) {
    return q.submit([=](sycl::handler &h) {
        h.depends_on(deps);
        h.single_task([]() {});
    });
}

// call by ESIMD kernel
template <size_t align, typename L>
auto invoke_esimd_function(L lambda, int world) {
    switch (world) {
        case 2: lambda.template operator()<2, align>(); break;
        case 4: lambda.template operator()<4, align>(); break;
        case 6: lambda.template operator()<6, align>(); break;
        case 8: lambda.template operator()<8, align>(); break;
        case 10: lambda.template operator()<10, align>(); break;
        case 12: lambda.template operator()<12, align>(); break;
        case 14: lambda.template operator()<14, align>(); break;
        case 16: lambda.template operator()<16, align>(); break;
        default: break;
    }
}

// helper function to check NAN or INF numbers in the input buffer
template <typename data_type>
sycl::event sycl_check_nan(sycl::queue &queue,
                           int rank,
                           const data_type *in_buffer,
                           size_t count,
                           const char *str,
                           const std::vector<sycl::event> &deps) {
    int wg_size = 16;
    int nthreads = (count + wg_size - 1) / wg_size * wg_size;
    auto e = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::nd_range<1>(nthreads, wg_size), [=](sycl::nd_item<1> idx2) {
            uint32_t idx = idx2.get_global_id();
            if (idx >= count)
                return;
            data_type *in = (data_type *)in_buffer;
            bool has_inf = false;
            if (std::numeric_limits<data_type>::has_infinity) {
                has_inf = in[idx] == std::numeric_limits<data_type>::infinity() ||
                          -in[idx] == std::numeric_limits<data_type>::infinity();
            }
            if (in[idx] != in[idx] || has_inf) {
                sycl::_V1::ext::oneapi::experimental::printf(
                    "[%d] NAN error %s: idx:%d count: %d val: %f \n",
                    rank,
                    str,
                    idx,
                    count,
                    in[idx]);
            }
        });
    });
    return e;
}

inline uint32_t get_total_threads(sycl::queue q) {
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ssize_t dev_id{ ccl::utils::invalid_device_id };
    if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
        CCL_THROW("unable to get global id for device\n");
    }
    return ccl::global_data::get().ze_data->devices[dev_id].total_threads;
}

template <typename T, size_t bytes, bool use_full_vector = true>
constexpr inline size_t get_num_elements() {
    // TODO: should use_full_vector be int instead of bool
    // to enable various levels of vectorization
    if (use_full_vector == true) {
        CCL_ASSERT(bytes >= 8);
        return bytes / sizeof(T);
    }
    else {
        // use 4 bytes when full vector cannot be used
        return 4 / sizeof(T);
    }
}
