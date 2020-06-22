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
#ifndef CPU_COLL_HPP
#define CPU_COLL_HPP

#include "coll.hpp"

/* cpu-specific base implementation */

template<class Dtype, class strategy>
struct cpu_base_coll : virtual base_coll, protected strategy
{
    using coll_strategy = strategy;

    template<class ...Args>
    cpu_base_coll(size_t sbuf_multiplier, size_t rbuf_multiplier, Args&& ...args):
        coll_strategy(std::forward<Args>(args)...)
    {
        int result = 0;
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            result = posix_memalign((void**)&send_bufs[idx], ALIGNMENT,
                                    ELEM_COUNT * sizeof(Dtype) * sbuf_multiplier);
            result = posix_memalign((void**)&recv_bufs[idx], ALIGNMENT,
                                    ELEM_COUNT * sizeof(Dtype) * rbuf_multiplier);
        }
        result = posix_memalign((void**)&single_send_buf, ALIGNMENT,
                                SINGLE_ELEM_COUNT * sizeof(Dtype) * sbuf_multiplier);
        result = posix_memalign((void**)&single_recv_buf, ALIGNMENT,
                                SINGLE_ELEM_COUNT * sizeof(Dtype) * rbuf_multiplier);
        (void)result;
    }

    cpu_base_coll() : cpu_base_coll(1, 1)
    {
    }

    virtual ~cpu_base_coll()
    {
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            free(send_bufs[idx]);
            free(recv_bufs[idx]);
        }
        free(single_send_buf);
        free(single_recv_buf);
    }

    const char* name() const noexcept override
    {
        return coll_strategy::class_name();
    }

    virtual void start(size_t count, size_t buf_idx,
                       const ccl::coll_attr& attr,
                       req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm, count,
                                      static_cast<Dtype*>(send_bufs[buf_idx]),
                                      static_cast<Dtype*>(recv_bufs[buf_idx]),
                                      attr, stream, reqs);
    }

    virtual void start_single(size_t count,
                              const ccl::coll_attr& attr,
                              req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm, count,
                                      static_cast<Dtype*>(single_send_buf),
                                      static_cast<Dtype*>(single_recv_buf),
                                      attr, stream, reqs);
    }

    ccl::datatype get_dtype() const override final
    {
        return ccl::native_type_info<typename std::remove_pointer<Dtype>::type>::ccl_datatype_value;
    }
};

#endif /* CPU_COLL_HPP */
