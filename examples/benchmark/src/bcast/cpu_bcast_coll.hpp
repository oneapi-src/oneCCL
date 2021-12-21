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

#include "cpu_coll.hpp"
#include "bcast_strategy.hpp"

template <class Dtype>
struct cpu_bcast_coll : cpu_base_coll<Dtype, bcast_strategy_impl> {
    using coll_base = cpu_base_coll<Dtype, bcast_strategy_impl>;
    using coll_base::recv_bufs;

    cpu_bcast_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                if (comm.rank() == COLL_ROOT) {
                    ((Dtype*)recv_bufs[b_idx][rank_idx])[e_idx] =
                        get_val<Dtype>(static_cast<float>(b_idx));
                }
                else {
                    ((Dtype*)recv_bufs[b_idx][rank_idx])[e_idx] = 0;
                }
            }
        }
    }

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        Dtype value;
        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                value = ((Dtype*)recv_bufs[b_idx][rank_idx])[e_idx];
                Dtype expected = get_val<Dtype>(static_cast<float>(b_idx));
                if (value != expected) {
                    std::cout << this->name() << " recv_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected " << expected
                              << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};
