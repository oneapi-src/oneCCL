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

#include "sparse_allreduce_strategy.hpp"

template <class VType, class IType, template <class> class IndicesDistributorType>
struct base_sparse_allreduce_coll
        : base_coll,
          protected sparse_allreduce_strategy_impl<IType, IndicesDistributorType> {
    using ITypeNonMod = typename std::remove_pointer<IType>::type;
    using VTypeNonMod = typename std::remove_pointer<VType>::type;

    using coll_base = base_coll;
    using coll_strategy = sparse_allreduce_strategy_impl<IType, IndicesDistributorType>;

    std::vector<std::vector<ITypeNonMod*>> send_ibufs;
    std::vector<std::vector<VTypeNonMod*>> send_vbufs;

    /* buffers from these arrays will be reallocated inside completion callback */
    std::vector<std::vector<ITypeNonMod*>> recv_ibufs;
    std::vector<std::vector<VTypeNonMod*>> recv_vbufs;

    size_t* recv_icount = nullptr;
    size_t* recv_vcount = nullptr;
    std::vector<std::vector<sparse_allreduce_fn_ctx_t>> fn_ctxs;

    base_sparse_allreduce_coll(bench_init_attr init_attr, size_t size)
            : base_coll(init_attr),
              coll_strategy(init_attr.v2i_ratio, size) {
        int result = 0;

        result =
            posix_memalign((void**)&recv_icount, ALIGNMENT, init_attr.buf_count * sizeof(size_t));
        result =
            posix_memalign((void**)&recv_vcount, ALIGNMENT, init_attr.buf_count * sizeof(size_t));

        std::memset(recv_icount, 0, init_attr.buf_count * sizeof(size_t));
        std::memset(recv_vcount, 0, init_attr.buf_count * sizeof(size_t));
        (void)result;

        fn_ctxs.resize(init_attr.buf_count);
        send_ibufs.resize(init_attr.buf_count);
        send_vbufs.resize(init_attr.buf_count);
        recv_ibufs.resize(init_attr.buf_count);
        recv_vbufs.resize(init_attr.buf_count);

        for (size_t idx = 0; idx < init_attr.buf_count; idx++) {
            fn_ctxs[idx].resize(init_attr.ranks_per_proc);
            send_ibufs[idx].resize(init_attr.ranks_per_proc);
            send_vbufs[idx].resize(init_attr.ranks_per_proc);
            recv_ibufs[idx].resize(init_attr.ranks_per_proc);
            recv_vbufs[idx].resize(init_attr.ranks_per_proc);
        }
    }

    virtual ~base_sparse_allreduce_coll() {
        free(recv_icount);
        free(recv_vcount);
        fn_ctxs.clear();
    }

    const char* name() const noexcept override {
        return coll_strategy::class_name();
    }

    ccl::datatype get_dtype() const override final {
        return ccl::native_type_info<typename std::remove_pointer<VType>::type>::dtype;
    }

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        ASSERT(0, "unexpected");
    }

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        ASSERT(0, "unexpected");
    }
};
