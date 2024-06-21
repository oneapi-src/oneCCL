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

ccl::event allgatherv_large(const void* send_buf,
                            size_t send_count,
                            void* recv_buf,
                            const ccl::vector_class<size_t>& recv_counts,
                            ccl::datatype dtype,
                            ccl_comm* comm,
                            ccl_stream* global_stream,
                            const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("invoking allgatherv_large");

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();

    const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();
    const bool is_odd = send_count % 2 != 0 && dsize < sizeof(int);
    const bool is_aligned = (send_count * dsize) % ccl::global_data::env().kernel_mem_align == 0;
    // do not use tmp_buf when copy engines are used
    // use tmp buf for 16 bit types with odd count with 32 bit vectors
    // use tmp buf when the count is not aligned
    const bool is_use_tmp = !ccl::global_data::env().sycl_copy_engine &&
                            (ccl::global_data::env().sycl_allgatherv_tmp_buf || is_odd ||
                             (!is_aligned && ccl::global_data::env().sycl_auto_use_tmp_buf));

    // if tmp buffer is not used, perform ipc exchange on send and recv buffer
    if (!is_use_tmp) {
        std::vector<void*> ptrs{ (void*)send_buf, recv_buf }; // index 0 and 1
        auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ptrs);

        xelink_ptrs_rd = get_ipc_ptrs<void, MAX_GPUS>(even_comm, 0, (void*)send_buf, sched);
        xelink_ptrs_wr = get_ipc_ptrs<void, MAX_GPUS>(even_comm, 1, recv_buf, sched);

        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            mdfi_ptr_rd = get_ipc_ptrs<void, MAX_TILES>(pair_comm, 0, (void*)send_buf, sched)[peer_pair_rank];
            mdfi_ptr_wr = get_ipc_ptrs<void, MAX_TILES>(pair_comm, 1, recv_buf, sched)[peer_pair_rank];
        }
        delete exchange_entry;
        delete sched;

        coll_init(comm, global_stream);
    }
    else {
        coll_init(comm, global_stream);
        xelink_ptrs_rd = get_remote_even_tmp_buf(0);
        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            mdfi_ptr_wr = get_remote_pair_tmp_buf(0)[peer_pair_rank];
        }
    }

    auto lambda = [&]<typename T, int NE, int NP>() {
        if (is_odd) {
            return allgatherv_large_impl<T, NE, NP, true>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
        }
        else {
            return allgatherv_large_impl<T, NE, NP, false>(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
        }
    };

    return invoke_collective(lambda, comm, dtype);
}
