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

#include "coll/algorithms/utils/sycl_coll_base.hpp"

#define SIMD_MAX       256
#define SIMD           (SIMD_MAX / sizeof(data_type))
#define SIMD_ATOMIC    16
#define MAX_RANK       16
#define UNROLL_SIZE    1
#define TRIPLE_BUFFER  3
#define SYNC_BYTE      (SIMD_ATOMIC * sizeof(int) * 2)
#define ALIGNMENT_BYTE 256
#define EU_COUNT       512
#define THREADS_PER_EU 8
#define MAX_THREAD     (EU_COUNT * THREADS_PER_EU)
#define MAX_COUNT      (SIMD * UNROLL_SIZE * MAX_THREAD)
#define INIT_SIZE      64
#define INIT_COUNT     1
#define SIMD_INIT      (INIT_SIZE * INIT_COUNT)

template <uint32_t TEMP_WORLD, typename data_type>
ESIMD_INLINE void gather_write(int offset,
                               const void *send_buf,
                               int myoffset,
                               void *recv_buf,
                               uint32_t send_count,
                               void *temp_buffer[],
                               uint32_t temp_rank,
                               int size_per_buffer_kernel,
                               int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    simd<data_type, SIMD * UNROLL_SIZE * TEMP_WORLD> buffer;
    for (uint32_t r = 0; r < TEMP_WORLD; r++) {
        if (r == temp_rank && send_buf == recv_buf)
            continue;
        data_type *peer_ptr = (data_type *)temp_buffer[r] + (buffer_index_kernel * size_per_buffer_kernel);
#pragma unroll
        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
            buffer.template select<SIMD, 1>(r * UNROLL_SIZE * SIMD + unroll_i * SIMD) =
                lsc_block_load<data_type,
                               SIMD,
                               lsc_data_size::default_size,
                               cache_hint::cached,
                               cache_hint::cached>(peer_ptr + offset + unroll_i * SIMD);
        }
    }

    for (uint32_t r = 0; r < TEMP_WORLD; r++) {
        if (r == temp_rank && send_buf == recv_buf)
            continue;
        uint32_t r_offset = send_count * r + offset;
#pragma unroll
        for (int i = 0; i < UNROLL_SIZE; i++) {
            lsc_block_store<data_type,
                            SIMD,
                            lsc_data_size::default_size,
                            cache_hint::uncached,
                            cache_hint::uncached>(
                (data_type *)recv_buf + r_offset + i * SIMD,
                buffer.template select<SIMD, 1>(r * UNROLL_SIZE * SIMD + i * SIMD));
        }
    }
}

template <typename dtype>
class Allgatherv_small_kernel_esimd;
template <typename dtype, int wg_size>
class Allgatherv_small_kernel_scalar;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_allgatherv_small : public sycl_coll_base<data_type> {
public:
    sycl_allgatherv_small() : sycl_coll_base<data_type>() {
        buffer_index = 0;
        size_per_buffer = 0;
    }

    void init(sycl::queue &queue, ccl_comm *comm, ccl_stream *stream, uint32_t rank_in, uint32_t world_in) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        data_size_per_buffer = ((MAX_COUNT + SIMD * UNROLL_SIZE - 1) / (SIMD * UNROLL_SIZE)) * SIMD * UNROLL_SIZE;
        data_size_per_buffer = ((data_size_per_buffer * sizeof(data_type) + ALIGNMENT_BYTE - 1) / ALIGNMENT_BYTE) *
                               ALIGNMENT_BYTE / sizeof(data_type); //aligned size
        size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;

        void *local_triple_buffer = sycl::malloc_device(size_per_buffer * TRIPLE_BUFFER, queue);
        auto e = queue.memset(local_triple_buffer, 0, size_per_buffer * TRIPLE_BUFFER);
        e.wait();
        this->exchange_peer_ipc_mem(queue,
                                    comm,
                                    stream,
                                    local_triple_buffer,
                                    NULL,
                                    rank,
                                    world,
                                    data_size_per_buffer * sizeof(data_type),
                                    (void **)buffers,
                                    (void **)sync_buffer,
                                    offsets,
                                    ipc_handle,
                                    NULL,
                                    NULL /* mmap_buffers */,
                                    false /* to_cache */);
        this->initialized = true;

        int wg_size = 1;
        //dummy kernel to avoid hang. The hang happens when there is no dummy kernel and allreduce() is called right after init().
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>({ 1 }, wg_size), [=](sycl::item<1> idx) SYCL_ESIMD_KERNEL {

            });
        });
        e.wait();
    }

    ccl::event allgatherv(sycl::queue &queue,
                          const void *send_buf,
                          size_t send_count,
                          void *recv_buf,
                          const ccl::vector_class<size_t> &recv_counts,
                          bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        assert(this->initialized == true);

        for (uint32_t i = 0; i < recv_counts.size(); i++) {
            if (recv_counts[i] != send_count) {
                fprintf(stderr, "not all recv_counts are the same as send_count\n");
                abort();
            }
        }

        done = true;
        if (send_count * world <= MAX_THREAD) {
            if (send_count * world < 3000)
                e = allgatherv_scalar<8>(queue, send_buf, send_count, recv_buf, recv_counts);
            else
                e = allgatherv_scalar<16>(queue, send_buf, send_count, recv_buf, recv_counts);
        }
        else {
            if ((send_count * sizeof(data_type)) % 4 == 0) {
                e = allgatherv_esimd(queue, send_buf, send_count, recv_buf, recv_counts);
            }
            else {
                done = false;
            }
        }

        return ccl::event::create_from_native(e);
    }

    sycl::event allgatherv_esimd(sycl::queue &queue,
                                 const void *send_buf,
                                 size_t send_count,
                                 void *recv_buf,
                                 const ccl::vector_class<size_t> &recv_counts) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;

        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        uint32_t myoffset = 0;
        if (send_buf == recv_buf)
            myoffset = send_count * temp_rank;

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel = size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        int buffer_index_kernel = buffer_index;
        buffer_index++;
        buffer_index %= TRIPLE_BUFFER;

        uint32_t total_threads_needed = (send_count + SIMD * UNROLL_SIZE - 1) / (SIMD * UNROLL_SIZE); //ceiling
        int wg_size = 8;
        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<Allgatherv_small_kernel_esimd<data_type>>(
                sycl::nd_range<1>({ total_threads_dispatched }, wg_size),
                [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL {
                    uint32_t idx = idx2.get_global_id();

                    //ESIMD kernel
                    uint32_t offset = idx * SIMD * UNROLL_SIZE;
                    simd<data_type, SIMD * UNROLL_SIZE> buffer_small;
                    simd<ushort, SIMD_ATOMIC> ramp;
                    simd_mask<SIMD_ATOMIC> pred;
                    simd<int, SIMD_ATOMIC> status0;
                    int *local_sync_ptr;

#pragma unroll
                    for (uint32_t i = 0; i < SIMD_ATOMIC; i++) {
                        ramp[i] = i * sizeof(int);
                    }

                    //use the temp buffer for the current rank to copy the data to.
                    data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                    local_temp_ptr +=
                        buffer_index_kernel *
                        size_per_buffer_kernel; //point to the correct buffer inside the triple buffer

                    //process the input only if the thread is useful
                    if (idx < total_threads_needed) {
                    //do copy from input buffer to temp buffer.
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer_small.template select<SIMD, 1>(unroll_i * SIMD) =
                                lsc_block_load<data_type,
                                               SIMD,
                                               lsc_data_size::default_size,
                                               cache_hint::cached,
                                               cache_hint::cached>((data_type *)send_buf + myoffset + offset +
                                                                   unroll_i * SIMD);
                        }

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type,
                                            SIMD,
                                            lsc_data_size::default_size,
                                            cache_hint::uncached,
                                            cache_hint::uncached>(
                                local_temp_ptr + offset + unroll_i * SIMD,
                                buffer_small.template select<SIMD, 1>(unroll_i * SIMD));
                        }
                        //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                        //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    }

                    //sync locally within local GPU first.
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_for_sync_kernel);

                    //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                    if (total_threads_needed > 1) {
                        //do local sync in two steps. First using TG barrier. Then global L3 atomics.
                        uint32_t local_tid = idx2.get_local_linear_id();

                        pred = false;
                        pred[0] = true;
                        if (local_tid == 0) {
                            status0 = lsc_atomic_update<atomic_op::inc,
                                                        int,
                                                        SIMD_ATOMIC,
                                                        lsc_data_size::default_size,
                                                        cache_hint::none,
                                                        cache_hint::none>(local_sync_ptr, ramp, pred);
                            //wait for all the local TG to sync. Then sync the other remote GPUs
                            while (status0[0] != total_wg_count) {
                                status0 = lsc_atomic_update<atomic_op::load,
                                                            int,
                                                            SIMD_ATOMIC,
                                                            lsc_data_size::default_size,
                                                            cache_hint::none,
                                                            cache_hint::none>(local_sync_ptr, ramp, pred);
                            }
                        }
                        barrier();
                    }

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    pred = false;
                    pred[1] = true; //use different lane for the remote gpu sync

                    if (total_threads_dispatched >= temp_world) {
                        if (idx < temp_world) {
                            status0 = total_threads_needed;
                            int *sync_ptr = (int *)temp_sync_buffer[idx];
                            sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                            lsc_atomic_update<atomic_op::add,
                                              int,
                                              SIMD_ATOMIC,
                                              lsc_data_size::default_size,
                                              cache_hint::none,
                                              cache_hint::none>(sync_ptr, ramp, status0, pred);
                        }
                    }
                    else if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        status0 = total_threads_needed;
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;
                            sync_ptr = (int *)temp_sync_buffer
                                [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                            lsc_atomic_update<atomic_op::add,
                                              int,
                                              SIMD_ATOMIC,
                                              lsc_data_size::default_size,
                                              cache_hint::none,
                                              cache_hint::none>(sync_ptr, ramp, status0, pred);
                        }
                    }

                    //once the local sync is done, retire useless threads
                    if (idx >= total_threads_needed)
                        return;

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    status0 = lsc_atomic_update<atomic_op::load,
                                                int,
                                                SIMD_ATOMIC,
                                                lsc_data_size::default_size,
                                                cache_hint::none,
                                                cache_hint::none>(local_sync_ptr, ramp, pred);
                    while (status0[1] != total_threads_needed * temp_world) {
                        status0 = lsc_atomic_update<atomic_op::load,
                                                    int,
                                                    SIMD_ATOMIC,
                                                    lsc_data_size::default_size,
                                                    cache_hint::none,
                                                    cache_hint::none>(local_sync_ptr, ramp, pred);
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset = (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                        status0 = 0;
                        pred = true;
                        local_sync_ptr = (int *)temp_sync_buffer
                            [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_for_sync_kernel);
                        lsc_atomic_update<atomic_op::store,
                                          int,
                                          SIMD_ATOMIC,
                                          lsc_data_size::default_size,
                                          cache_hint::none,
                                          cache_hint::none>(
                            local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //for (uint32_t r = 0; r < temp_world; r++)  {
                    if (offset + SIMD * UNROLL_SIZE <= send_count) {
                        switch (temp_world) {
                            case 2:
                                gather_write<2, data_type>(offset,
                                                           send_buf,
                                                           myoffset,
                                                           recv_buf,
                                                           send_count,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel);
                                break;
                            case 4:
                                gather_write<4, data_type>(offset,
                                                           send_buf,
                                                           myoffset,
                                                           recv_buf,
                                                           send_count,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel);
                                break;
                            case 6:
                                gather_write<6, data_type>(offset,
                                                           send_buf,
                                                           myoffset,
                                                           recv_buf,
                                                           send_count,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel);
                                break;
                            case 8:
                                gather_write<8, data_type>(offset,
                                                           send_buf,
                                                           myoffset,
                                                           recv_buf,
                                                           send_count,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel);
                                break;
                            case 10:
                                gather_write<10, data_type>(offset,
                                                            send_buf,
                                                            myoffset,
                                                            recv_buf,
                                                            send_count,
                                                            (void **)temp_buffer,
                                                            temp_rank,
                                                            size_per_buffer_kernel,
                                                            buffer_index_kernel);
                                break;
                            case 12:
                                gather_write<12, data_type>(offset,
                                                            send_buf,
                                                            myoffset,
                                                            recv_buf,
                                                            send_count,
                                                            (void **)temp_buffer,
                                                            temp_rank,
                                                            size_per_buffer_kernel,
                                                            buffer_index_kernel);
                                break;
                            case 14:
                                gather_write<14, data_type>(offset,
                                                            send_buf,
                                                            myoffset,
                                                            recv_buf,
                                                            send_count,
                                                            (void **)temp_buffer,
                                                            temp_rank,
                                                            size_per_buffer_kernel,
                                                            buffer_index_kernel);
                                break;
                            case 16:
                                gather_write<16, data_type>(offset,
                                                            send_buf,
                                                            myoffset,
                                                            recv_buf,
                                                            send_count,
                                                            (void **)temp_buffer,
                                                            temp_rank,
                                                            size_per_buffer_kernel,
                                                            buffer_index_kernel);
                                break;
                            default: break;
                        }
                    }
                    else {
                        for (uint32_t r = 0; r < temp_world; r++) {
                            data_type *src_ptr = (data_type *)temp_buffer[r] +
                                                 buffer_index_kernel * size_per_buffer_kernel + offset;
                            data_type *dest_ptr = (data_type *)recv_buf + send_count * r + offset;
                            for (size_t i = offset; i < send_count; i++) {
                                *dest_ptr = *src_ptr;
                                src_ptr++;
                                dest_ptr++;
                            }
                        }
                    }
                });
        });
        //e.wait();
        return e;
    }

    template <int wg_size>
    sycl::event allgatherv_scalar(sycl::queue &queue,
                                  const void *send_buf,
                                  size_t send_count,
                                  void *recv_buf,
                                  const ccl::vector_class<size_t> &recv_counts) {
        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        uint32_t myoffset = 0;
        if (send_buf == recv_buf)
            myoffset = send_count * temp_rank;

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel = size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        //uint32_t total_threads_needed = send_count;
        assert(wg_size >= 8);
        uint32_t total_threads_needed = send_count * temp_world;
        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;
        uint32_t total_threads_needed_copy = send_count;

        int buffer_index_kernel = buffer_index;
        buffer_index++;
        buffer_index %= TRIPLE_BUFFER;

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<Allgatherv_small_kernel_scalar<
                data_type,
                wg_size>>(sycl::nd_range<1>({ total_threads_dispatched }, wg_size), [=](sycl::nd_item<1> idx2) {
                uint32_t idx = idx2.get_global_id();

                int *local_sync_ptr;

                //use the temp buffer for the current rank to copy the data to.
                data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                local_temp_ptr += buffer_index_kernel *
                                  size_per_buffer_kernel; //point to the correct buffer inside the triple buffer

                //process the input only if the thread is useful
                //if (idx < total_threads_needed)
                if (idx < total_threads_needed_copy) {
                    local_temp_ptr[idx] = *((data_type *)send_buf + myoffset + idx);
                }

                //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank

                //sync locally within local GPU first.
                local_sync_ptr = (int *)temp_sync_buffer
                    [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                local_sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;

                //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                if (total_threads_needed > 1) {
                    //do local sync in two steps. First using TG barrier. Then global L3 atomics.
                    uint32_t local_tid = idx2.get_local_linear_id();

                    if (local_tid == 0) {
                        sycl::atomic_ref<int,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_p(local_sync_ptr[0]);
                        atomic_p += 1;

                        //wait for all the local TG to sync. Then sync the other remote GPUs
                        uint32_t val = atomic_p.load();
                        while (val != total_wg_count) {
                            val = atomic_p.load();
                        }
                    }
                    idx2.barrier();
                }

                //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                if (total_threads_dispatched >= temp_world) {
                    if (idx < temp_world) {
                        uint32_t status0 = total_threads_needed;
                        int *sync_ptr = (int *)temp_sync_buffer
                            [idx]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                        sycl::atomic_ref<int,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_p(sync_ptr[1]);
                        atomic_p += status0;
                    }
                }
                else if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                {
                    uint32_t status0 = total_threads_needed;
                    for (uint32_t i = 0; i < temp_world; i++) {
                        int *sync_ptr;
                        sync_ptr = (int *)temp_sync_buffer
                            [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                        sycl::atomic_ref<int,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_p(sync_ptr[1]);
                        atomic_p += status0;
                    }
                }

                //once the local sync is done, retire useless threads
                if (idx >= total_threads_needed)
                    return;

                //once all the local TGs are sync, do fence so that other GPU can see.
                //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                //wait for completion of the atomic sync
                sycl::atomic_ref<int,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_p(local_sync_ptr[1]);
                uint32_t val = atomic_p.load();
                while (val != total_threads_needed * temp_world) {
                    val = atomic_p.load();
                }

                //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                {
                    int buffer_index_to_reset = (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += buffer_index_to_reset * size_per_buffer_for_sync_kernel;
                    local_sync_ptr[0] = local_sync_ptr[1] = 0;
                }

                //at this point, all the threads are done copying data from input buffer to temp buffer.
                uint32_t r = idx / send_count;
                data_type *peer_ptr = (data_type *)temp_buffer[r] + buffer_index_kernel * size_per_buffer_kernel;
                int ii = idx % send_count;
                *((data_type *)recv_buf + idx) = peer_ptr[ii];
            });
        });
        return e;
    }

private:
    void *buffers[max_rank]{};
    void *sync_buffer[max_rank]{};
    size_t offsets[max_rank]{};
    ze_ipc_mem_handle_t ipc_handle[max_rank]{};
    int rank{ ccl::utils::invalid_rank }, world{ ccl::utils::invalid_err_code };
    int buffer_index{ ccl::utils::invalid_err_code };
    int size_per_buffer{ ccl::utils::invalid_bytes_value };
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
};
