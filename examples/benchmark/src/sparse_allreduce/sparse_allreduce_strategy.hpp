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

template <class type>
struct type_printer {
    static constexpr const char* sparse_class_name() {
        return "sparse_allreduce";
    }
};

template <>
struct type_printer<ccl::bf16> {
    static constexpr const char* sparse_class_name() {
        return "sparse_allreduce_bf16";
    }
};

typedef struct {
    void** recv_ibuf;
    void** recv_vbuf;
    size_t recv_ibuf_count;
    size_t recv_vbuf_count;
} sparse_allreduce_fn_ctx_t;

void sparse_allreduce_completion_fn(const void* ind_buf,
                                    size_t ind_count,
                                    ccl::datatype ind_dtype,
                                    const void* val_buf,
                                    size_t val_count,
                                    ccl::datatype val_dtype,
                                    const void* fn_ctx) {
    // printf("callback: ibuf %p, icnt %zu, idt %d, vbuf %p, cvnt %zu, vdt %d\n",
    //     ind_buf, ind_count, ind_dtype, val_buf, val_count, val_dtype);

    size_t ind_bytes = ind_count * ccl::get_datatype_size(ind_dtype);
    size_t val_bytes = val_count * ccl::get_datatype_size(val_dtype);

    ASSERT(fn_ctx, "fn_ctx is null");

    sparse_allreduce_fn_ctx_t* ctx = (sparse_allreduce_fn_ctx_t*)(fn_ctx);

    ASSERT(ctx->recv_ibuf && *ctx->recv_ibuf, "fn_ctx->recv_ibuf is null");
    ASSERT(ctx->recv_vbuf && *ctx->recv_vbuf, "fn_ctx->recv_vbuf is null");

    void* recv_ibuf = *ctx->recv_ibuf;
    void* recv_vbuf = *ctx->recv_vbuf;

    recv_ibuf = realloc(recv_ibuf, ind_bytes);
    recv_vbuf = realloc(recv_vbuf, val_bytes);

    ASSERT(recv_ibuf, "recv_ibuf is null after realloc");
    ASSERT(recv_vbuf, "recv_vbuf is null after realloc");

    memcpy(recv_ibuf, ind_buf, ind_bytes);
    memcpy(recv_vbuf, val_buf, val_bytes);

    *ctx->recv_ibuf = recv_ibuf;
    *ctx->recv_vbuf = recv_vbuf;
    ctx->recv_ibuf_count = ind_count;
    ctx->recv_vbuf_count = val_count;
}

void sparse_allreduce_alloc_fn(size_t ind_count,
                               ccl::datatype ind_dtype,
                               size_t val_count,
                               ccl::datatype val_dtype,
                               const void* fn_ctx,
                               void** out_ind_buf,
                               void** out_val_buf) {
    // printf("callback: icnt %zu, idt %d, cvnt %zu, vdt %d\n",
    //     ind_count, ind_dtype, val_count, val_dtype);

    size_t ind_bytes = ind_count * ccl::get_datatype_size(ind_dtype);
    size_t val_bytes = val_count * ccl::get_datatype_size(val_dtype);

    ASSERT(fn_ctx, "fn_ctx is null");

    sparse_allreduce_fn_ctx_t* ctx = (sparse_allreduce_fn_ctx_t*)(fn_ctx);

    ASSERT(ctx->recv_ibuf && *ctx->recv_ibuf, "fn_ctx->recv_ibuf is null");
    ASSERT(ctx->recv_vbuf && *ctx->recv_vbuf, "fn_ctx->recv_vbuf is null");

    void* recv_ibuf = *ctx->recv_ibuf;
    void* recv_vbuf = *ctx->recv_vbuf;

    recv_ibuf = realloc(recv_ibuf, ind_bytes);
    recv_vbuf = realloc(recv_vbuf, val_bytes);

    ASSERT(recv_ibuf, "recv_ibuf is null after realloc");
    ASSERT(recv_vbuf, "recv_vbuf is null after realloc");

    *ctx->recv_ibuf = recv_ibuf;
    *ctx->recv_vbuf = recv_vbuf;
    ctx->recv_ibuf_count = ind_count;
    ctx->recv_vbuf_count = val_count;

    *out_ind_buf = recv_ibuf;
    *out_val_buf = recv_vbuf;
}

template <class IType, template <class> class IndicesDistributorType>
struct sparse_allreduce_strategy_impl {
    static constexpr const char* class_name() {
        return type_printer<IType>::sparse_class_name();
    }

    static const ccl::sparse_allreduce_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::sparse_allreduce_attr>();
    }

    template <class T>
    using remove_ptr_t = typename std::remove_pointer<T>::type;
    template <class T>
    using remove_all_t = typename std::remove_const<remove_ptr_t<T>>::type;

    using IndicesDistributor = IndicesDistributorType<remove_all_t<IType>>;

    size_t v2i_ratio;
    size_t comm_size;
    const size_t minimal_indices_count = 1;

    void init_distributor(const std::pair<size_t, size_t>& elem_range) {
        size_t indices_count = std::get<0>(get_expected_recv_counts(elem_range.second));
        indices_distributor_impl.reset(new IndicesDistributor(elem_range.first, indices_count));
    }

    sparse_allreduce_strategy_impl(size_t v2i_ratio, size_t comm_size)
            : v2i_ratio(v2i_ratio),
              comm_size(comm_size) {}

    sparse_allreduce_strategy_impl(const allgatherv_strategy_impl&) = delete;
    sparse_allreduce_strategy_impl& operator=(const allgatherv_strategy_impl&) = delete;
    ~sparse_allreduce_strategy_impl() = default;

    std::tuple<size_t, size_t> get_expected_recv_counts(size_t elem_count) const {
        size_t indices_count = std::max(elem_count / v2i_ratio, minimal_indices_count);
        size_t vdim_count = (elem_count / indices_count);

        return std::tuple<size_t, size_t>(indices_count, indices_count * vdim_count);
    }

    template <class VType, class comm_t, class... Args>
    void start_internal(comm_t& comm,
                        const IType send_ibuf,
                        size_t send_icount,
                        const VType send_vbuf,
                        size_t send_vcount,
                        IType recv_ibuf,
                        size_t recv_icount,
                        VType recv_vbuf,
                        size_t recv_vcount,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        sparse_allreduce_fn_ctx_t& fn_ctx,
                        Args&&... args) {
        auto expected = get_expected_recv_counts(send_vcount);
        recv_icount = std::get<0>(expected);
        recv_vcount = std::get<1>(expected);

        auto& sparse_attr = const_cast<ccl::sparse_allreduce_attr&>(
            bench_attr.get_attr<ccl::sparse_allreduce_attr>());
        /* use completion_fn because it is supported by all algorithms */
        sparse_attr.set<ccl::sparse_allreduce_attr_id::completion_fn>(
            &sparse_allreduce_completion_fn);
        //sparse_attr.sparse_allreduce_alloc_fn = sparse_allreduce_alloc_fn;

        sparse_attr.set<ccl::sparse_allreduce_attr_id::fn_ctx>(static_cast<const void*>(&fn_ctx));
        sparse_attr.set<ccl::sparse_allreduce_attr_id::coalesce_mode>(
            ccl::sparse_coalesce_mode::keep_precision);

        reqs.push_back(ccl::preview::sparse_allreduce(send_ibuf,
                                                      std::get<0>(expected),
                                                      send_vbuf,
                                                      send_vcount,
                                                      recv_ibuf,
                                                      recv_icount,
                                                      recv_vbuf,
                                                      recv_vcount,
                                                      bench_attr.reduction,
                                                      comm,
                                                      std::forward<Args>(args)...));
    }

    std::unique_ptr<IndicesDistributor> indices_distributor_impl;
};
