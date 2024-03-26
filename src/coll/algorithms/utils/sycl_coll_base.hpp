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
        using namespace std::chrono;
        return duration_cast<microseconds>(v_end[i] - v_start[i]).count();
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
        sched = NULL;
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
        if (recv_ptr && send_ptr != recv_ptr) {
            in_buffers.push_back({ recv_ptr, ccl::ze::ipc_mem_type::memory });
        }

        if (!sched) {
            ccl_coll_param param{};
            param.comm = comm;
            param.stream = stream;
            ccl_coll_attr attr{};
            sched = ccl_sched::create(param, attr);
        }

        ccl::utils::pt2pt_handle_exchange_info info = {};
        int skip_rank = ccl_comm::invalid_rank;
        ze_handle_exchange_entry *exchange_entry =
            new ze_handle_exchange_entry(sched, node_comm, in_buffers, skip_rank, info);
        exchange_entry->update(); //    128us

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
            if (send_ptr != recv_ptr) {
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
            else {
                for (int i = 0; i < world; i++) {
                    recv_buffers[i] = send_buffers[i];
                }
            }
        }
        delete exchange_entry;
        sched->clear_memory();
    }

    bool initialized;
    ccl_sched *sched;
};
