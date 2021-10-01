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
#if 0

#pragma once

#ifdef CCL_ENABLE_SYCL

#include "sycl_coll.hpp"

template <class VType,
          class IType,
          template <class> class IndicesDistributorType =
              sparse_detail::incremental_indices_distributor>
struct sycl_sparse_allreduce_coll : base_sparse_allreduce_coll<cl::sycl::buffer<VType, 1>,
                                                               cl::sycl::buffer<IType, 1>,
                                                               IndicesDistributorType> {
    using sycl_indices_t = cl::sycl::buffer<IType, 1>;
    using sycl_values_t = cl::sycl::buffer<VType, 1>;
    using coll_base =
        base_sparse_allreduce_coll<sycl_values_t, sycl_indices_t, IndicesDistributorType>;
    using coll_strategy = typename coll_base::coll_strategy;

    using coll_base::send_ibufs;
    using coll_base::send_vbufs;
    using coll_base::recv_ibufs;
    using coll_base::recv_vbufs;
    using coll_base::recv_icount;
    using coll_base::recv_vcount;
    using coll_base::fn_ctxs;

    sycl_sparse_allreduce_coll(bench_init_attr init_attr,
                               size_t sbuf_size_modifier = 1,
                               size_t rbuf_size_modifier = 1)
            : coll_base(init_attr, transport_data::get_comm_size()) {
        // size_t max_elem_count = base_coll::get_max_elem_count();

        // int comm_size = transport_data::get_comm_size();

        // for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
        //     send_ibufs[idx] = new sycl_indices_t(max_elem_count * sbuf_size_modifier);
        //     send_vbufs[idx] = new sycl_values_t(max_elem_count * sbuf_size_modifier);

        //     recv_ibufs[idx] =
        //         new sycl_indices_t(max_elem_count * rbuf_size_modifier * comm_size);
        //     recv_vbufs[idx] =
        //         new sycl_values_t(max_elem_count * rbuf_size_modifier * comm_size);

        //     stream.get_native().submit([&](handler& h) {
        //         auto send_ibuf = (static_cast<sycl_indices_t*>(send_ibufs[idx]));
        //         auto send_vbuf = (static_cast<sycl_values_t*>(send_vbufs[idx]));

        //         auto recv_ibuf = (static_cast<sycl_indices_t*>(recv_ibufs[idx]));
        //         auto recv_vbuf = (static_cast<sycl_values_t*>(recv_vbufs[idx]));

        //         auto send_ibuf_acc = send_ibuf->template get_access<mode::write>(h);
        //         auto send_vbuf_acc = send_vbuf->template get_access<mode::write>(h);
        //         auto recv_ibuf_acc = recv_ibuf->template get_access<mode::write>(h);
        //         auto recv_vbuf_acc = recv_vbuf->template get_access<mode::write>(h);

        //         h.parallel_for(range<1>{max_elem_count*comm_size}, [=](item<1> e_idx)
        //         {
        //             if (e_idx.get_linear_id() < max_elem_count) {
        //                 send_ibuf_acc[e_idx] = 0;
        //                 send_vbuf_acc[e_idx] = 0;
        //             }
        //             recv_ibuf_acc[e_idx] = 0;
        //             recv_vbuf_acc[e_idx] = 0;
        //         });
        //     }).wait();
        // }

        // for (size_t idx = 0; idx < base_coll::get_buf_count(); idx++) {
        //     fn_ctxs[idx].recv_ibuf = (void**)(&(recv_ibufs[idx]));
        //     fn_ctxs[idx].recv_vbuf = (void**)(&(recv_vbufs[idx]));
        // }
    }

    virtual void prepare(size_t elem_count) override {
        // TODO not implemented yet
    }

    virtual void finalize(size_t elem_count) override {
        // TODO not implemented yet
    }

    virtual void prepare_internal(size_t elem_count,
                         ccl::communicator& comm,
                         ccl::stream& stream,
                         size_t rank_idx) override {

        // TODO not implemented yet
    }

    virtual void finalize_internal(size_t elem_count,
                          ccl::communicator& comm,
                          ccl::stream& stream,
                          size_t rank_idx) override {

        // TODO not implemented yet
    }
    virtual void start(size_t count,
                       size_t buf_idx,
                       const bench_exec_attr& attr,
                       req_list_t& reqs) override {
        /*coll_strategy::start_internal(
            comm(),
            *static_cast<const cl::sycl::buffer<IType>*>(send_ibufs[buf_idx]),
            count,
            *reinterpret_cast<const cl::sycl::buffer<VType>*>(send_vbufs[buf_idx]),
            count,
            *static_cast<cl::sycl::buffer<IType>*>(recv_ibufs[buf_idx]),
            recv_icount[buf_idx],
            *reinterpret_cast<cl::sycl::buffer<VType>*>(recv_vbufs[buf_idx]),
            recv_vcount[buf_idx],
            attr,
            reqs,
            fn_ctxs[buf_idx],
            stream(),
            coll_strategy::get_op_attr(attr));*/
    }
};

#endif // CCL_ENABLE_SYCL

#endif
