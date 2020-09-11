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
#ifndef SYCL_COLL_HPP
#define SYCL_COLL_HPP

#include "coll.hpp"

#ifdef CCL_ENABLE_SYCL
/* sycl-specific base implementation */
template <class Dtype, class strategy>
struct sycl_base_coll : base_coll, private strategy, device_specific_data {
    using coll_strategy = strategy;

    template <class... Args>
    sycl_base_coll(bench_coll_init_attr init_attr,
                   size_t sbuf_multiplier,
                   size_t rbuf_multiplier,
                   Args&&... args)
            : base_coll(init_attr),
              coll_strategy(std::forward<Args>(args)...) {
        for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
            send_bufs[idx] =
                new cl::sycl::buffer<Dtype, 1>(base_coll::get_max_elem_count() * sbuf_multiplier);
            recv_bufs[idx] =
                new cl::sycl::buffer<Dtype, 1>(base_coll::get_max_elem_count() * rbuf_multiplier);
        }

        single_send_buf = new cl::sycl::buffer<Dtype, 1>(
            base_coll::get_single_buf_max_elem_count() * sbuf_multiplier);

        single_recv_buf = new cl::sycl::buffer<Dtype, 1>(
            base_coll::get_single_buf_max_elem_count() * rbuf_multiplier);
    }

    sycl_base_coll(bench_coll_init_attr init_attr) : sycl_base_coll(init_attr, 1, 1) {}

    virtual ~sycl_base_coll() {
        for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
            delete static_cast<sycl_buffer_t<Dtype>*>(send_bufs[idx]);
            delete static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[idx]);
        }
        delete static_cast<sycl_buffer_t<Dtype>*>(single_send_buf);
        delete static_cast<sycl_buffer_t<Dtype>*>(single_recv_buf);
    }

    const char* name() const noexcept override {
        return coll_strategy::class_name();
    }

    virtual void start(size_t count,
                       size_t buf_idx,
                       const bench_coll_exec_attr& attr,
                       req_list_t& reqs) override {
        sycl_buffer_t<Dtype>& send_buf = *(static_cast<sycl_buffer_t<Dtype>*>(send_bufs[buf_idx]));
        sycl_buffer_t<Dtype>& recv_buf = *(static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[buf_idx]));
        coll_strategy::template start_internal<sycl_buffer_t<Dtype>&>(
            comm(),
            count,
            send_buf,
            recv_buf,
            attr,
            reqs,
            stream(),
            coll_strategy::get_op_attr(attr));
    }

    virtual void start_single(size_t count,
                              const bench_coll_exec_attr& attr,
                              req_list_t& reqs) override {
        sycl_buffer_t<Dtype>& send_buf = *(static_cast<sycl_buffer_t<Dtype>*>(single_send_buf));
        sycl_buffer_t<Dtype>& recv_buf = *(static_cast<sycl_buffer_t<Dtype>*>(single_recv_buf));
        coll_strategy::template start_internal<sycl_buffer_t<Dtype>&>(
            comm(),
            count,
            send_buf,
            recv_buf,
            attr,
            reqs,
            stream(),
            coll_strategy::get_op_attr(attr));
    }

    ccl::datatype get_dtype() const override final {
        return ccl::native_type_info<typename std::remove_pointer<Dtype>::type>::ccl_datatype_value;
    }

    /* global communicator & stream for all cpu collectives */
    static ccl::device_communicator& comm() {
        if (!device_specific_data::comm_ptr) {
        }
        return *device_specific_data::comm_ptr;
    }

    static ccl::stream& stream() {
        if (!device_specific_data::stream_ptr) {
        }
        return *device_specific_data::stream_ptr;
    }
};
#endif /* CCL_ENABLE_SYCL */

#endif /* SYCL_COLL_HPP */
