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

#include "bcast_strategy.hpp"

#ifdef CCL_ENABLE_SYCL
#include "sycl_coll.hpp"

template <class Dtype>
struct sycl_bcast_coll : sycl_base_coll<Dtype, bcast_strategy_impl> {
    using coll_base = sycl_base_coll<Dtype, bcast_strategy_impl>;
    using coll_base::recv_bufs;
    using coll_base::host_recv_buf;

    sycl_bcast_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        int comm_rank = comm.rank();

        size_t count = elem_count;
        size_t bytes = count * base_coll::get_dtype_size();

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            std::fill(host_recv_buf.begin(), host_recv_buf.end(), b_idx);

            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                if (comm_rank == COLL_ROOT)
                    stream.get_native()
                        .memcpy(recv_bufs[b_idx][rank_idx], host_recv_buf.data(), bytes)
                        .wait();
                else
                    stream.get_native().memset(recv_bufs[b_idx][rank_idx], 0, bytes).wait();
            }
            else {
                stream.get_native()
                    .submit([&](handler& h) {
                        auto recv_buf =
                            (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));
                        auto recv_buf_acc = recv_buf->template get_access<mode::write>(h);
                        h.parallel_for(range<1>{ elem_count }, [=](item<1> e_idx) {
                            if (comm_rank == COLL_ROOT)
                                recv_buf_acc[e_idx] = e_idx.get_id(0);
                            else
                                recv_buf_acc[e_idx] = 0;
                        });
                    })
                    .wait();
            }
        }
    }

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        size_t bytes = elem_count * base_coll::get_dtype_size();

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                stream.get_native()
                    .memcpy(host_recv_buf.data(), recv_bufs[b_idx][rank_idx], bytes)
                    .wait();
            }
            else {
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));
                auto recv_buf_acc = recv_buf->template get_access<mode::read>();

                stream.get_native()
                    .memcpy(host_recv_buf.data(), recv_buf_acc.get_pointer(), bytes)
                    .wait();
            }

            Dtype value;

            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                value = host_recv_buf[e_idx];
                if (value != b_idx) {
                    std::cout << this->name() << " recv_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected " << (Dtype)b_idx
                              << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};
#endif /* CCL_ENABLE_SYCL */
