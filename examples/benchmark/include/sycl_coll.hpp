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

#include <iostream>
#include <map>
#include <set>
#include <string>

#include "coll.hpp"
#include "sycl_base.hpp" /* from examples/include */

#ifdef CCL_ENABLE_SYCL

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::access;

/* sycl-specific base implementation */
template <class Dtype, class strategy>
struct sycl_base_coll : base_coll, private strategy {
    using coll_strategy = strategy;

    template <class... Args>
    sycl_base_coll(bench_init_attr init_attr, Args&&... args)
            : base_coll(init_attr),
              coll_strategy() {
        auto& transport = transport_data::instance();
        auto streams = transport.get_bench_streams();

        size_t send_multiplier = coll_strategy::get_send_multiplier();
        size_t recv_multiplier = coll_strategy::get_recv_multiplier();

        for (size_t rank_idx = 0; rank_idx < base_coll::get_ranks_per_proc(); rank_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                allocators.push_back(buf_allocator<Dtype>(streams[rank_idx].get_native()));

                auto& allocator = allocators[rank_idx];

                sycl::usm::alloc usm_alloc_type;
                auto bench_alloc_type = base_coll::get_sycl_usm_type();
                if (bench_alloc_type == SYCL_USM_SHARED)
                    usm_alloc_type = usm::alloc::shared;
                else if (bench_alloc_type == SYCL_USM_DEVICE)
                    usm_alloc_type = usm::alloc::device;
                else
                    ASSERT(0, "unexpected bench_alloc_type %d", bench_alloc_type);

                for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                    send_bufs[idx][rank_idx] = allocator.allocate(
                        base_coll::get_max_elem_count() * send_multiplier, usm_alloc_type);

                    if (base_coll::get_inplace()) {
                        recv_bufs[idx][rank_idx] = send_bufs[idx][rank_idx];
                    }
                    else {
                        recv_bufs[idx][rank_idx] = allocator.allocate(
                            base_coll::get_max_elem_count() * recv_multiplier, usm_alloc_type);
                    }
                }
            }
            else {
                for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                    send_bufs[idx][rank_idx] = new cl::sycl::buffer<Dtype, 1>(
                        base_coll::get_max_elem_count() * send_multiplier);
                    recv_bufs[idx][rank_idx] = new cl::sycl::buffer<Dtype, 1>(
                        base_coll::get_max_elem_count() * recv_multiplier);
                }
            }
        }

        host_send_buf.resize(base_coll::get_max_elem_count() * send_multiplier);
        host_recv_buf.resize(base_coll::get_max_elem_count() * recv_multiplier);
    }

    sycl_base_coll(bench_init_attr init_attr) : sycl_base_coll(init_attr, 1, 1) {}

    virtual ~sycl_base_coll() {
        for (size_t rank_idx = 0; rank_idx < base_coll::get_ranks_per_proc(); rank_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_BUF) {
                for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
                    delete static_cast<sycl_buffer_t<Dtype>*>(send_bufs[idx][rank_idx]);
                    if (!base_coll::get_inplace()) {
                        delete static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[idx][rank_idx]);
                    }
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
        auto streams = transport.get_streams();
        size_t ranks_per_proc = base_coll::get_ranks_per_proc();

        for (size_t rank_idx = 0; rank_idx < ranks_per_proc; rank_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                coll_strategy::start_internal(comms[rank_idx],
                                              count,
                                              static_cast<Dtype*>(send_bufs[buf_idx][rank_idx]),
                                              static_cast<Dtype*>(recv_bufs[buf_idx][rank_idx]),
                                              attr,
                                              reqs,
                                              streams[rank_idx],
                                              coll_strategy::get_op_attr(attr));
            }
            else {
                /*sycl_buffer_t<Dtype>& send_buf =
                    *(static_cast<sycl_buffer_t<Dtype>*>(send_bufs[buf_idx][rank_idx]));
                sycl_buffer_t<Dtype>& recv_buf =
                    *(static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[buf_idx][rank_idx]));
                coll_strategy::template start_internal<sycl_buffer_t<Dtype>&>(
                    comms[rank_idx],
                    count,
                    send_buf,
                    recv_buf,
                    attr,
                    reqs,
                    streams[rank_idx],
                    coll_strategy::get_op_attr(attr));*/

                throw std::runtime_error(std::string(__FUNCTION__) +
                                         " - only USM buffers are supported\n");
            }
        }
    }

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        int comm_rank = comm.rank();

        size_t send_count = coll_strategy::get_send_multiplier() * elem_count;
        size_t recv_count = coll_strategy::get_recv_multiplier() * elem_count;

        size_t send_bytes = send_count * base_coll::get_dtype_size();
        size_t recv_bytes = recv_count * base_coll::get_dtype_size();

        std::fill(host_send_buf.begin(), host_send_buf.end(), comm_rank);

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (base_coll::get_sycl_mem_type() == SYCL_MEM_USM) {
                stream.get_native()
                    .memcpy(send_bufs[b_idx][rank_idx], host_send_buf.data(), send_bytes)
                    .wait();

                if (!base_coll::get_inplace()) {
                    stream.get_native().memset(recv_bufs[b_idx][rank_idx], 0, recv_bytes).wait();
                }
            }
            else {
                stream.get_native()
                    .submit([&](handler& h) {
                        auto send_buf =
                            (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx][rank_idx]));
                        auto send_buf_acc =
                            send_buf->template get_access<mode::write>(h, send_count);
                        h.fill(send_buf_acc, static_cast<Dtype>(comm_rank));
                    })
                    .wait();

                stream.get_native()
                    .submit([&](handler& h) {
                        auto recv_buf =
                            (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx][rank_idx]));
                        auto recv_buf_acc =
                            recv_buf->template get_access<mode::write>(h, recv_count);
                        h.fill(recv_buf_acc, static_cast<Dtype>(0));
                    })
                    .wait();
            }
        }
    }

    ccl::datatype get_dtype() const override final {
        return get_ccl_dtype<Dtype>();
    }

    /* used on fill/check phases */
    std::vector<Dtype> host_send_buf;
    std::vector<Dtype> host_recv_buf;

private:
    std::vector<buf_allocator<Dtype>> allocators;
};

#endif /* CCL_ENABLE_SYCL */
