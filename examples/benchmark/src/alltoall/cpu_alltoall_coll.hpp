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
#ifndef CPU_ALLTOALL_COLL
#define CPU_ALLTOALL_COLL

#include "cpu_coll.hpp"
#include "alltoall_strategy.hpp"

template<class Dtype>
struct cpu_alltoall_coll : cpu_base_coll<Dtype, alltoall_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, alltoall_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::stream;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::comm;

    cpu_alltoall_coll() : coll_base(base_coll::comm->size(), base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t idx = 0; idx < comm->size(); idx++)
            {
                for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
                {
                    ((Dtype*)send_bufs[b_idx])[idx * elem_count + e_idx] = comm->rank();
                    ((Dtype*)recv_bufs[b_idx])[idx * elem_count + e_idx] = 0;
                }
            }
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        Dtype sbuf_expected = comm->rank();
        Dtype rbuf_expected;
        Dtype value;
        size_t comm_size = comm->size();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count * comm_size; e_idx++)
            {
                value = ((Dtype*)send_bufs[b_idx])[e_idx];
                rbuf_expected = e_idx / elem_count;
                if (value != sbuf_expected)
                {
                    std::cout << this->name() << " send_bufs: buf_idx "
                              << b_idx << ", elem_idx " << e_idx << ", expected "
                              << sbuf_expected << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }

                value = ((Dtype*)recv_bufs[b_idx])[e_idx];
                if (value != rbuf_expected)
                {
                    std::cout << this->name() << " recv_bufs: buf_idx "
                              << b_idx << ", elem_idx " << e_idx << ", expected "
                              << rbuf_expected << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

#endif /* CPU_ALLTOALL_COLL */
