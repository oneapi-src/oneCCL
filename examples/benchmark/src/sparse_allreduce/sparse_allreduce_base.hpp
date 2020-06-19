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
#ifndef SPARSE_ALLREDUCE_BASE_HPP
#define SPARSE_ALLREDUCE_BASE_HPP

#include "sparse_allreduce_strategy.hpp"

template<class VType, class IType, template<class> class IndicesDistributorType>
struct base_sparse_allreduce_coll :
        virtual base_coll,
        protected sparse_allreduce_strategy_impl<IType,
                                                 IndicesDistributorType>
{
    using ITypeNonMod = typename std::remove_pointer<IType>::type;
    using VTypeNonMod = typename std::remove_pointer<VType>::type;

    using coll_base = base_coll;
    using coll_strategy = sparse_allreduce_strategy_impl<IType,
                                                         IndicesDistributorType>;

    using coll_base::stream;
    using coll_base::comm;

    using coll_strategy::value_to_indices_ratio;
    using coll_strategy::vdim_size;
    using coll_strategy::minimal_indices_count;

    ITypeNonMod* send_ibufs[BUF_COUNT] = { nullptr };
    VTypeNonMod* send_vbufs[BUF_COUNT] = { nullptr };

    /* buffers from these arrays will be reallocated inside completion callback */
    ITypeNonMod* recv_ibufs[BUF_COUNT] = { nullptr };
    VTypeNonMod* recv_vbufs[BUF_COUNT] = { nullptr };

    size_t* recv_icount = nullptr;
    size_t* recv_vcount = nullptr;
    sparse_allreduce_user_ctx_t user_ctxs[BUF_COUNT];

    ITypeNonMod* single_send_ibuf = nullptr;
    VTypeNonMod* single_send_vbuf = nullptr;
    ITypeNonMod* single_recv_ibuf = nullptr;
    VTypeNonMod* single_recv_vbuf = nullptr;
    size_t single_recv_icount {};
    size_t single_recv_vcount {};
    sparse_allreduce_user_ctx_t single_user_ctx;

    base_sparse_allreduce_coll(const std::string& args) :
     coll_strategy(args, base_coll::comm->size())
    {
        int result = 0;
        result = posix_memalign((void**)&recv_icount, ALIGNMENT,
                                BUF_COUNT * sizeof(size_t));
        result = posix_memalign((void**)&recv_vcount, ALIGNMENT,
                                BUF_COUNT * sizeof(size_t));

        std::memset(recv_icount, 0, BUF_COUNT * sizeof(size_t));
        std::memset(recv_vcount, 0, BUF_COUNT * sizeof(size_t));
        (void)result;
    }

    virtual ~base_sparse_allreduce_coll()
    {
        free(recv_icount);
        free(recv_vcount);
    }

    const char* name() const noexcept override
    {
        return coll_strategy::class_name();
    }

    ccl::datatype get_dtype() const override final
    {
        return ccl::native_type_info<typename std::remove_pointer<VType>::type>::ccl_datatype_value;
    }
};

#endif /* SPARSE_ALLREDUCE_BASE_HPP */
