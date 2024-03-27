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
#include "coll/algorithms/allgatherv/sycl/allgatherv_large_sycl_impl.hpp"
#include "sched/entry/factory/entry_factory.hpp"

std::pair<ccl_sched*, ze_handle_exchange_entry*> do_ipc_exchange(ccl_comm* comm,
                                                                 ccl_stream* stream,
                                                                 std::vector<void*> ptrs) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

    for (auto ptr : ptrs) {
        in_buffers.emplace_back(ptr, ccl::ze::ipc_mem_type::memory);
    }

    ccl_coll_param param{};
    param.comm = comm;
    param.stream = stream;
    ccl_coll_attr attr{};
    static ccl_sched* sched = ccl_sched::create(param, attr);
    ccl::utils::pt2pt_handle_exchange_info info = {};
    int skip_rank = ccl_comm::invalid_rank;

    ze_handle_exchange_entry* exchange_entry =
        new ze_handle_exchange_entry(sched, node_comm, in_buffers, skip_rank, info);
    exchange_entry->update();
    return { sched, exchange_entry };
}

template <int N>
ccl::event allgatherv_large_type(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const ccl::vector_class<size_t>& recv_counts,
                                 ccl::datatype dtype,
                                 const ccl::communicator& comm,
                                 const ccl::stream& op_stream,
                                 const ccl::allgatherv_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    ccl::event e;
    switch (dtype) {
        case ccl::datatype::float16:
            e = allgatherv_large_impl<sycl::half, N>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case ccl::datatype::bfloat16:
            e = allgatherv_large_impl<short, N>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case ccl::datatype::int32:
            e = allgatherv_large_impl<int, N>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case ccl::datatype::float32:
            e = allgatherv_large_impl<float, N>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        default: assert(false); break;
    }
    return e;
}

ccl::event allgatherv_large(const void* send_buf,
                            size_t send_count,
                            void* recv_buf,
                            const ccl::vector_class<size_t>& recv_counts,
                            ccl::datatype dtype,
                            const ccl::communicator& comm,
                            const ccl::stream& op_stream,
                            const ccl::allgatherv_attr& attr,
                            const ccl::vector_class<ccl::event>& deps) {
    ccl::impl_dispatch disp;
    static size_t allgatherv_count = 0;

    sycl::queue q = op_stream.get_native();
    assert(q.is_in_order());

    const int comm_rank = comm.rank();
    const int comm_size = comm.size();

    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    std::shared_ptr<ccl_comm> pair_comm = global_comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = global_comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();
    const int node_comm_size = node_comm->size();

    allgatherv_count++;
    const bool is_use_tmp = ccl::global_data::env().allgatherv_use_tmp_buf;

    if (allgatherv_count == 1) {
        LOG_INFO("invoking allgatherv large kernel first time");
        const size_t tmp_buffer_size = ccl::global_data::env().allgatherv_chunk_size * node_comm_size * 2;
        tmp_buf = sycl::aligned_alloc_device<char>(4096, tmp_buffer_size, q);

        sync_remote_ptrs[node_comm->rank()] = sycl::malloc_device<size_t>(MAX_NODE_RANKS, q);
        q.memset(sync_remote_ptrs[node_comm->rank()], 0, MAX_NODE_RANKS * sizeof(size_t)).wait();
    }
    {
        // ipc exchange needs to be done everytime if tmp buffer is not used and
        // when tmp buffer is used, it needs to be done only for the first time
        if (!is_use_tmp || allgatherv_count == 1) {
            void* data_buf_send = is_use_tmp ? tmp_buf : (void*)send_buf;
            void* data_buf_recv = is_use_tmp ? tmp_buf : recv_buf;
            bool to_cache = !is_use_tmp;

            std::vector<void*> ptrs{ data_buf_send, data_buf_recv }; // index 0 and 1
            // need to get sync remote pointers only onece
            if (allgatherv_count == 1) {
                ptrs.push_back(sync_remote_ptrs[comm_rank]); // index 2
            }

            auto [sched, exchange_entry] = do_ipc_exchange(global_comm, disp(op_stream).get(), ptrs);

            // kernels are unable to read/write data from the ipc pointers and
            // as a workaround touching them once with memcpy seems to fix the issue
            std::vector<sycl::event> dummy_copy_events;
            static sycl::queue q_worker(q.get_device());
            // need to get sync remote pointers only onece
            if (allgatherv_count == 1) {
                for (int i = 1; i < node_comm_size; i++) {
                    int peer_rank = (node_comm->rank() + i) % node_comm_size;
                    ccl_buffer tmp_ccl_buf;
                    sched->get_memory().handle_manager.get(
                        peer_rank, 2, tmp_ccl_buf, node_comm.get(), false /*pt2pt_op*/, false);
                    CCL_THROW_IF_NOT(tmp_ccl_buf.get_ptr(), "null IPC buffer is received");
                    sync_remote_ptrs[peer_rank] = (size_t*)tmp_ccl_buf.get_ptr();
                    dummy_copy_events.push_back(q_worker.memcpy(tmp_buf, sync_remote_ptrs[peer_rank], 1));
                }
            }

            // get xelink remote pointers
            auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
            const int dsize = ccl_dtype.size();
            const int global_rank = even_comm->get_global_rank(even_comm->rank());
            const size_t adjust_offset_count = (is_use_tmp || send_buf == recv_buf) ? 0 : send_count * global_rank;

            // for inplace, send_buf is at an offset of send_count*global_rank in the recv_buf
            // but for non-inplace send_buf is separate and there is no send_count*global_rank offset
            // and therefore for non-inplace, subtract send_count*global_rank to the ptr, so that
            // the algorithm can always access send buffer at an index of send_count*global_rank
            xelink_ptrs[even_comm->rank()] = (char*)data_buf_send - adjust_offset_count * dsize;

            for (int i = 1; i < even_comm_size; i++) {
                int peer_rank = (even_comm->rank() + i) % even_comm_size;
                ccl_buffer tmp_ccl_buf;
                sched->get_memory().handle_manager.get(
                    peer_rank, 0, tmp_ccl_buf, even_comm.get(), false /*pt2pt_op*/, to_cache);
                CCL_THROW_IF_NOT(tmp_ccl_buf.get_ptr(), "null IPC buffer is received");
                const int global_rank_pr = even_comm->get_global_rank(peer_rank);
                const size_t adjust_offset_count_pr =
                    (is_use_tmp || send_buf == recv_buf) ? 0 : send_count * global_rank_pr;
                xelink_ptrs[peer_rank] = (char*)tmp_ccl_buf.get_ptr() - adjust_offset_count_pr * dsize;
                if (allgatherv_count == 1)
                    dummy_copy_events.push_back(q_worker.memcpy(tmp_buf, tmp_ccl_buf.get_ptr(), 1));
            }

            // get mdfi remote pointers
            if (pair_comm_size > 1) {
                int peer_rank = (pair_comm->rank() + 1) % pair_comm_size;
                ccl_buffer tmp_ccl_buf;
                sched->get_memory().handle_manager.get(
                    peer_rank, 1, tmp_ccl_buf, pair_comm.get(), false /*pt2pt_op*/, to_cache);
                CCL_THROW_IF_NOT(tmp_ccl_buf.get_ptr(), "null IPC buffer is received");
                mdfi_ptr = tmp_ccl_buf.get_ptr();
                if (allgatherv_count == 1)
                    dummy_copy_events.push_back(q_worker.memcpy(tmp_buf, mdfi_ptr, 1));
            }

            if (allgatherv_count == 1) {
                // combine the dummy copy events into the inorder queue
                q.ext_oneapi_submit_barrier(dummy_copy_events);
            }

            // sycl_barrier is working only if a barrier is there
            // between ipc exchange and first invocation of sycl_barrier
            if (allgatherv_count == 1) {
                ccl_comm* node_comm_ptr = node_comm.get();
                q.submit([=](sycl::handler& h) {
                    h.host_task([=]() {
                        ccl::impl_dispatch disp;
                        node_comm_ptr->barrier(disp(ccl::default_stream), ccl::default_barrier_attr).wait();
                    });
                });
            }

            delete exchange_entry;
            sched->clear_memory();
        }
    }

    LOG_DEBUG("|CCL_SYCL| allgatherv selects large kernel without ccl scheduler autotune");

    // TODO: which is better as outer switch - comm_size of dtype?
    ccl::event e;
    //TODO: for multi-node we need to use node_comm_size
    switch (comm_size) {
        case 2:
            e = allgatherv_large_type<2>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 4:
            e = allgatherv_large_type<4>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 6:
            e = allgatherv_large_type<6>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 8:
            e = allgatherv_large_type<8>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 10:
            e = allgatherv_large_type<10>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 12:
            e = allgatherv_large_type<12>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 14:
            e = allgatherv_large_type<14>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        case 16:
            e = allgatherv_large_type<16>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            break;
        default: assert(false); break;
    }
    return e;
}
