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
#include "common/api_wrapper/mpi_api_wrapper.hpp"

namespace ccl::v1 {
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};
}; // namespace ccl::v1

static constexpr int MAX_NODE_RANKS = 16;
static constexpr int MAX_GPUS = 8;

void* tmp_buf = nullptr;
std::array<size_t*, MAX_NODE_RANKS> sync_remote_ptrs;
std::array<void*, MAX_GPUS> xelink_ptrs;
void* mdfi_ptr = nullptr;

constexpr bool use_sycl_kernel_block = true;
constexpr int vec_size = 1;

template <typename T,
          sycl::access::address_space Space = sycl::access::address_space::global_space,
          sycl::access::decorated IsDecorated = sycl::access::decorated::yes>
sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes> get_multi_ptr(T* ptr) {
    return sycl::address_space_cast<Space, IsDecorated>(ptr);
}

template <typename T, int N>
void inline read_write(std::array<void*, MAX_GPUS> peer_even_ptrs,
                       std::array<void*, MAX_GPUS> local_ptrs,
                       std::array<void*, MAX_GPUS> peer_pair_ptrs,
                       const size_t count,
                       const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    sycl::sub_group sg = it.get_sub_group();
    const size_t sgSize = sg.get_local_range()[0];

    int base = (idx / sgSize) * sgSize * vec_size;
    const long rem_elem_count = count - base;

    if (idx < count) {
        if (use_sycl_kernel_block && rem_elem_count > 0 && (size_t)rem_elem_count >= sgSize) {
#pragma unroll
            for (int i = 0; i < N; i++) {
                auto val = sg.load<vec_size>(get_multi_ptr(&(((T*)peer_even_ptrs[i])[base])));
                sg.store<vec_size>(get_multi_ptr(&(((T*)peer_pair_ptrs[i])[base])), val);
                sg.store<vec_size>(get_multi_ptr(&(((T*)local_ptrs[i])[base])), val);
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < N; i++) {
                const T val = ((T*)peer_even_ptrs[i])[idx];
                ((T*)peer_pair_ptrs[i])[idx] = val;
                ((T*)local_ptrs[i])[idx] = val;
            }
        }
    }
}

template <typename T, int N>
void inline copy_data(std::array<void*, MAX_GPUS> dst,
                      std::array<void*, MAX_GPUS> src,
                      const size_t count,
                      const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    sycl::sub_group sg = it.get_sub_group();
    const size_t sgSize = sg.get_local_range()[0];

    int base = (idx / sgSize) * sgSize * vec_size;

    if (idx < count) {
        if (use_sycl_kernel_block) {
#pragma unroll
            for (int i = 0; i < N; i++) {
                const sycl::vec<T, vec_size> val = sg.load<vec_size>(get_multi_ptr(&(((T*)src[i])[base])));
                sg.store<vec_size>(get_multi_ptr(&(((T*)dst[i])[base])), val);
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < N; i++) {
                ((T*)dst[i])[idx] = ((T*)src[i])[idx];
            }
        }
    }
}

sycl::event sycl_barrier(std::shared_ptr<ccl_comm> comm,
                         sycl::queue q,
                         std::vector<sycl::event> dep_events,
                         std::array<size_t*, MAX_NODE_RANKS> local_sync_remote_ptrs) {
    static size_t barrier_count_val = 0;
    barrier_count_val++;

    const size_t counter = barrier_count_val;
    const int comm_rank = comm->rank();
    const int comm_size = comm->size();
    auto evt = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);

        //h.parallel_for(1, [=](sycl::item<1> idx) {
        h.parallel_for(comm_size, [=](sycl::item<1> idx) {
            //TODO: find out the best consitency parameters to be used for atomic variables

            int i = idx;
            //for(int i=0; i<comm_size; i++)
            {
                sycl::atomic_ref<size_t,
                                 sycl::memory_order::seq_cst,
                                 sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_p(local_sync_remote_ptrs[i][comm_rank]);
                atomic_p += 1;
            }

            //for(int i=0; i<comm_size; i++)
            {
                sycl::atomic_ref<size_t,
                                 sycl::memory_order::seq_cst,
                                 sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>
                    atomic_p(local_sync_remote_ptrs[comm_rank][i]);

                size_t val = atomic_p.load();
                while (val < counter) {
                    val = atomic_p.load();
                }
            }
        });
    });
    return evt;
}

sycl::event invoke_barrier(std::shared_ptr<ccl_comm> comm,
                           sycl::queue q,
                           std::vector<sycl::event> dep_events,
                           std::array<size_t*, MAX_NODE_RANKS> local_sync_remote_ptrs,
                           const size_t num_chunks) {
    sycl::event ret_event;
    const bool use_sycl_barrier = !ccl::global_data::env().use_ccl_barrier;
    if (ccl::global_data::env().use_sycl_barrier || use_sycl_barrier) {
        ret_event = sycl_barrier(comm, q, dep_events, local_sync_remote_ptrs);
    }
    else {
        const bool use_sync_barrier = false;
        if (use_sync_barrier) {
            for (auto dep_event : dep_events) {
                dep_event.wait();
            }
            //MPI_Barrier(MPI_COMM_WORLD);
            ccl::impl_dispatch disp;
            comm->barrier(disp(ccl::default_stream), ccl::default_barrier_attr).wait();
        }
        else {
            ret_event = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.host_task([=]() {
                    //MPI_Barrier(MPI_COMM_WORLD);
                    ccl::impl_dispatch disp;
                    comm->barrier(disp(ccl::default_stream), ccl::default_barrier_attr).wait();
                });
            });
        }
    }
    return ret_event;
}

sycl::queue create_main_ce_queue(sycl::queue q) {
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    // Create Command Queue
    ze_command_queue_desc_t Qdescriptor = {};
    Qdescriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    Qdescriptor.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    Qdescriptor.ordinal = 1;
    Qdescriptor.index = 0;
    Qdescriptor.flags = 0;
    Qdescriptor.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    //ze_command_queue_handle_t ze_cmd_queue = nullptr;
    //ze_result_t result = zeCommandQueueCreate(ze_ctx, ze_dev, &Qdescriptor, &ze_cmd_queue);

    ze_command_list_handle_t ze_imm_cmd_list = nullptr;
    ze_result_t result = zeCommandListCreateImmediate(ze_ctx, ze_dev, &Qdescriptor, &ze_imm_cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandQueueCreate failed\n";
        return q;
    }

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device> InteropDeviceInput{ ze_dev };
    sycl::device InteropDevice = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(InteropDeviceInput);

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{
        ze_ctx, std::vector<sycl::device>(1, InteropDevice), sycl::ext::oneapi::level_zero::ownership::keep
    };
    sycl::context InteropContext = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);

    //sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCQ{
    //  ze_cmd_queue, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep};

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInputCL{
        ze_imm_cmd_list, InteropDevice, sycl::ext::oneapi::level_zero::ownership::keep
    };

    //sycl::queue q_mce = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCQ, InteropContext);
    sycl::queue q_mce =
        sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(InteropQueueInputCL, InteropContext);

    return q_mce;
}

sycl::queue get_main_ce_queue(sycl::queue q) {
    static sycl::queue q_mce = create_main_ce_queue(q);
    return q_mce;
}

template <typename T, int N>
ccl::event allgatherv_large_impl_ipc(const void* send_buf,
                                     size_t send_count,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::communicator& comm,
                                     const ccl::stream& op_stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl::impl_dispatch disp;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = op_stream.get_native();
    assert(q.is_in_order());

    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    std::shared_ptr<ccl_comm> even_comm = global_comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();

    std::array<void*, MAX_GPUS> local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs;
    for (int i = 0; i < even_comm->size(); i++) {
        // offsets for read_write kernel
        const int global_rank = even_comm->get_global_rank(i);
        const size_t offset_bytes = send_count * global_rank * dsize;
        local_peer_even_ptrs[i] = (char*)xelink_ptrs[i] + offset_bytes;
        local_local_ptrs[i] = (char*)recv_buf + offset_bytes;
        local_peer_pair_ptrs[i] = (char*)mdfi_ptr + offset_bytes;
    }

    invoke_barrier(node_comm, q, {}, sync_remote_ptrs, 1);

    assert(vec_size == 1);
    const size_t work_group_size = 32;
    const size_t kernel_size = ((send_count + work_group_size - 1) / work_group_size) * work_group_size;

    q.submit([=](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range(sycl::range{ kernel_size }, sycl::range{ work_group_size }), [=](sycl::nd_item<1> it) {
                read_write<T, N / 2>(local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs, send_count, it);
            });
    });

    sycl::event barrier_event = invoke_barrier(node_comm, q, {}, sync_remote_ptrs, 1);
    return ccl::event::create_from_native(barrier_event);
}

template <typename T, int N>
ccl::event allgatherv_large_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const ccl::vector_class<size_t>& recv_counts,
                                 ccl::datatype dtype,
                                 const ccl::communicator& comm,
                                 const ccl::stream& op_stream,
                                 const ccl::allgatherv_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    if (!ccl::global_data::env().allgatherv_use_tmp_buf) {
        ccl::event e = allgatherv_large_impl_ipc<T, N>(
            send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
        return e;
    }

    ccl::impl_dispatch disp;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = op_stream.get_native();
    assert(q.is_in_order());

    static sycl::queue q_worker(q.get_device());
    static sycl::queue q_copy = get_main_ce_queue(q);

    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    std::shared_ptr<ccl_comm> pair_comm = global_comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = global_comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();

    const int even_comm_size = even_comm->size();

    const int node_comm_size = node_comm->size();

    const int comm_rank = comm.rank();

    constexpr int pipeline_size = 2;
    const size_t tmp_chunk_size = ccl::global_data::env().allgatherv_chunk_size * node_comm_size;
    void* tmp_bufs[pipeline_size];
    tmp_bufs[0] = tmp_buf;
    tmp_bufs[1] = ((char*)tmp_buf) + tmp_chunk_size;

    std::array<void*, MAX_GPUS> local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs;
    std::array<void*, MAX_GPUS> recv_buf_dst_ptrs, tmp_buf_src_ptrs;
    std::array<void*, MAX_GPUS> recv_buf_dst_ptrs_prev, tmp_buf_src_ptrs_prev;
    std::array<void*, MAX_GPUS> tmp_send_buf_next, my_send_buf_next;

    const size_t chunk_size = ccl::global_data::env().allgatherv_chunk_size;
    const size_t chunk_count = chunk_size / dsize;
    const size_t num_chunks = send_count / chunk_count + (send_count % chunk_count != 0);

    std::vector<sycl::event> work_events;
    sycl::event output_event;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        // setup pointers

        // alternate between tmp buffers since we use a pipeline of size 2
        // i.e. copy previous output from one tmp_buffer when allgatherv
        // is operating on the second tmp_buffer
        const int tmp_chunk_id = nc % pipeline_size;
        void* tmp_buf_use = tmp_bufs[tmp_chunk_id];
        void* tmp_buf_other = tmp_bufs[!tmp_chunk_id];

        // TODO: which rank to be used for scaleout, node_comm or global_comm

        // offset on send buffer
        const size_t my_offset_count_send = chunk_count * nc;
        // offset on recv buffer
        const size_t my_offset_count = send_count * comm_rank + my_offset_count_send;
        // offset on tmp buffer
        const size_t my_offset_count_tmp = chunk_count * comm_rank;

        void* my_send_buf = (char*)send_buf + my_offset_count_send * dsize;
        if (send_buf == recv_buf) {
            my_send_buf = (char*)recv_buf + my_offset_count * dsize;
        }
        void* tmp_send_buf = (char*)tmp_buf_use + my_offset_count_tmp * dsize;

        my_send_buf_next[0] = (char*)my_send_buf + chunk_count * dsize;
        tmp_send_buf_next[0] = (char*)tmp_buf_other + my_offset_count_tmp * dsize;

        for (int i = 0; i < even_comm_size; i++) {
            // offsets for read_write kernel
            int global_rank = even_comm->get_global_rank(i);
            const size_t offset_bytes = (send_count * global_rank + chunk_count * nc) * dsize;
            const size_t offset_bytes_tmp = chunk_count * global_rank * dsize;

            // xelink and mdfi ptrs are the tmp buffers in the other ranks
            const size_t tmp_chunk_offset = tmp_chunk_id * tmp_chunk_size;
            local_peer_even_ptrs[i] = (char*)xelink_ptrs[i] + offset_bytes_tmp + tmp_chunk_offset;
            local_local_ptrs[i] = (char*)recv_buf + offset_bytes;
            local_peer_pair_ptrs[i] = (char*)mdfi_ptr + offset_bytes_tmp + tmp_chunk_offset;

            // offsets for copy kernel
            // TODO: is there a better way to find the pair_neighbor global rank
            int global_rank_neighbor = (global_rank / pair_comm_size) * pair_comm_size;
            if (global_rank % pair_comm_size == 0) {
                global_rank_neighbor = global_rank_neighbor + 1;
            }
            const size_t offset_bytes_c = (send_count * global_rank_neighbor + chunk_count * nc) * dsize;
            const size_t offset_bytes_c_tmp = chunk_count * global_rank_neighbor * dsize;
            recv_buf_dst_ptrs[i] = (char*)recv_buf + offset_bytes_c;
            tmp_buf_src_ptrs[i] = (char*)tmp_buf_use + offset_bytes_c_tmp;

            recv_buf_dst_ptrs_prev[i] = (char*)recv_buf_dst_ptrs[i] - chunk_count * dsize;
            // offset of prev tmp buffer is same but use tmp_buf_other instead of tmp_buf_use
            tmp_buf_src_ptrs_prev[i] = (char*)tmp_buf_other + offset_bytes_c_tmp;
        }

        // start the collective

        // if send_count is not a multiple of chunk_count, then last chunk will contain only remainder data
        const size_t data_count = (nc < send_count / chunk_count) ? chunk_count : send_count % chunk_count;
        const size_t data_count_next =
            (nc < (send_count / chunk_count) - 1) ? chunk_count : send_count % chunk_count;

        // TODO: move this outside the looop
        // pipeline prologue - copy first chunk from send_buf to tmp_buf using in-order queue
        if (nc == 0) {
            sycl::event e = q.submit([=](sycl::handler& h) {
                h.memcpy(tmp_send_buf, my_send_buf, dsize * data_count);
            });
            work_events.push_back(e);
        }

        sycl::queue q_use = q_worker;

        sycl::event barrier_event1 = invoke_barrier(node_comm, q_use, work_events, sync_remote_ptrs, num_chunks);
        work_events.clear();

        // use kernel for local pipeline copies of next and prev buffers,
        // by default we using main copy engine using memcpy
        constexpr bool use_kernel_copy = false;

        const size_t work_group_size = 32;

        const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
        const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

        sycl::event kernel_event = q_use.submit([=](sycl::handler& h) {
            h.depends_on(barrier_event1);
            h.parallel_for(sycl::nd_range(sycl::range{ kernel_size }, sycl::range{ work_group_size }),
                           [=](sycl::nd_item<1> it) {
                               read_write<T, N / 2>(
                                   local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs, data_count, it);
                               // copy next input chunk
                               if (use_kernel_copy && nc < num_chunks - 1) {
                                   copy_data<T, 1>(tmp_send_buf_next, my_send_buf_next, data_count_next, it);
                               }
                               // copy prev output chunk
                               if (use_kernel_copy && nc > 0) {
                                   copy_data<T, N / 2>(
                                       recv_buf_dst_ptrs_prev, tmp_buf_src_ptrs_prev, data_count, it);
                               }
                           });
        });
        work_events.push_back(kernel_event);

        std::vector<sycl::event> copy_events;
        if (!use_kernel_copy) {
            // copy next input chunk
            if (nc < num_chunks - 1) {
                sycl::event e = q_copy.submit([=](sycl::handler& h) {
                    h.depends_on(barrier_event1);
                    const size_t data_count_next =
                        (nc < (send_count / chunk_count) - 1) ? chunk_count : send_count % chunk_count;
                    h.memcpy(tmp_send_buf_next[0], my_send_buf_next[0], dsize * data_count_next);
                });
                copy_events.push_back(e);
            }

            // copy prev output chunk
            if (nc > 0) {
                // for last iteration, if read_write kernel is small, then use
                // compute engine for copying since it is faster than copy engine
                // and there is very less overlap with read_write since it is small
                const size_t small_size_threshold = ccl::global_data::env().allgatherv_small_size_threshold;

                //TODO: should we use single kernel copy when q_use is used
                sycl::queue q_copy_use =
                    (nc == num_chunks - 1 && data_count < small_size_threshold) ? q_use : q_copy;
                for (int i = 0; i < even_comm_size; i++) {
                    sycl::event e = q_copy_use.submit([=](sycl::handler& h) {
                        h.depends_on(barrier_event1);
                        const size_t data_count_prev = chunk_count;
                        h.memcpy(recv_buf_dst_ptrs_prev[i], tmp_buf_src_ptrs_prev[i], dsize * data_count_prev);
                    });
                    copy_events.push_back(e);
                }
            }
        }
        else if (use_kernel_copy && chunk_count > data_count && nc > 0) {
            // in case the last chunk is small than chunk_count,
            // we still need to copy the rest of prev output chunk
            assert(nc == num_chunks - 1);
            const size_t copy_count = chunk_count - data_count;
            for (int i = 0; i < even_comm_size; i++) {
                sycl::event e = q_copy.submit([=](sycl::handler& h) {
                    h.depends_on(barrier_event1);
                    void* src = (char*)(tmp_buf_src_ptrs_prev[i]) + data_count * dsize;
                    void* dst = (char*)(recv_buf_dst_ptrs_prev[i]) + data_count * dsize;
                    h.memcpy(dst, src, dsize * copy_count);
                });
                copy_events.push_back(e);
            }
        }

        // WA: directly connecting the output event of q_copy to gpu kernels
        // cause failure when MPI binding is used - I_MPI_PIN_PROCESSOR_LIST
        if (!copy_events.empty()) {
            sycl::event e = q_use.submit([=](sycl::handler& h) {
                h.depends_on(copy_events);
                h.host_task([]() {});
            });
            work_events.push_back(e);
        }

        // TODO: move this outside of the looop
        // pipeline epilogue - copy the final output chunk from tmp_buffer to recv_buffer
        if (nc == num_chunks - 1) {
            sycl::event barrier_event2;
            barrier_event2 = invoke_barrier(
                node_comm, q_use, work_events, /*ccl::global_data::get().*/ sync_remote_ptrs, num_chunks);
            work_events.clear();

            // TODO: find when to use single kernel copy vs memcpys
            constexpr bool use_single_kernel_copy = true;
            // use a single kernel to copy from tmp_buffer to recv_buffer
            if (use_single_kernel_copy) {
                output_event = q.submit([=](sycl::handler& h) {
                    h.depends_on(barrier_event2);
                    constexpr int mult = 4;
                    const size_t packed_size = data_count / mult;
                    const size_t rem_size = data_count % mult;
                    const size_t kernel_size = packed_size + rem_size;
                    using AT = sycl::vec<T, mult>;
                    h.parallel_for(kernel_size, [=](sycl::item<1> idx) {
                        if (idx < packed_size) {
#pragma unroll
                            for (int i = 0; i < N / 2; i++) {
                                ((AT*)recv_buf_dst_ptrs[i])[idx] = ((AT*)tmp_buf_src_ptrs[i])[idx];
                            }
                        }
                        else {
#pragma unroll
                            for (int i = 0; i < N / 2; i++) {
                                const size_t new_idx = idx + (mult - 1) * packed_size;
                                ((T*)recv_buf_dst_ptrs[i])[new_idx] = ((T*)tmp_buf_src_ptrs[i])[new_idx];
                            }
                        }
                    });
                });
            }
            // use memcpys to copy from tmp_buffer to recv_buffer
            else {
                for (int i = 0; i < even_comm_size; i++) {
                    sycl::event e = q_use.submit([=](sycl::handler& h) {
                        h.depends_on(barrier_event2);
                        h.memcpy(recv_buf_dst_ptrs[i], tmp_buf_src_ptrs[i], dsize * data_count);
                    });
                    work_events.push_back(e);
                }
                output_event = q.ext_oneapi_submit_barrier(work_events);
            }
        }
    } // nc

    return ccl::event::create_from_native(output_event);
}
