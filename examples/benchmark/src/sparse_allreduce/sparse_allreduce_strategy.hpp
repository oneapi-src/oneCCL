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
#ifndef SPARSE_ALLREDUCE_STRATEGY_HPP
#define SPARSE_ALLREDUCE_STRATEGY_HPP

/* specific benchmark const expressions */
constexpr size_t default_value_to_indices_ratio = 3;
constexpr size_t default_vdim_size = ELEM_COUNT / 3;

template<class type>
struct type_printer
{
    static constexpr const char* sparse_class_name() { return "sparse_allreduce"; }
};

template<>
struct type_printer<ccl::bfp16>
{
    static constexpr const char* sparse_class_name() { return "sparse_allreduce_bfp16"; }
};

typedef struct
{
    void** recv_ibuf;
    void** recv_vbuf;
    size_t recv_ibuf_count;
    size_t recv_vbuf_count;
} sparse_allreduce_user_ctx_t;

ccl_status_t sparse_allreduce_completion_fn(
    const void* ind_buf, size_t ind_count, ccl_datatype_t ind_dtype,
    const void* val_buf, size_t val_count, ccl_datatype_t val_dtype,
    const ccl_fn_context_t* fn_ctx, const void* user_ctx)
{
    // printf("callback: ibuf %p, icnt %zu, idt %d, vbuf %p, cvnt %zu, vdt %d\n",
    //     ind_buf, ind_count, ind_dtype, val_buf, val_count, val_dtype);

    size_t ind_bytes = ind_count * ccl::datatype_get_size((ccl::datatype)(ind_dtype));
    size_t val_bytes = val_count * ccl::datatype_get_size((ccl::datatype)(val_dtype));

    ASSERT(user_ctx, "user_ctx is null");

    sparse_allreduce_user_ctx_t* ctx = (sparse_allreduce_user_ctx_t*)(user_ctx);

    ASSERT(ctx->recv_ibuf && *ctx->recv_ibuf, "user_ctx->recv_ibuf is null");
    ASSERT(ctx->recv_vbuf && *ctx->recv_vbuf, "user_ctx->recv_vbuf is null");

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

    return ccl_status_success;
}

template<class IType, template<class> class IndicesDistributorType>
struct sparse_allreduce_strategy_impl
{
    static constexpr const char* class_name()
    {
        return type_printer<IType>::sparse_class_name();
    }

    template<class T>
    using remove_ptr_t = typename std::remove_pointer<T>::type;
    template<class T>
    using remove_all_t = typename std::remove_const<remove_ptr_t<T>>::type;

    using IndicesDistributor = IndicesDistributorType<remove_all_t<IType>>;

    size_t value_to_indices_ratio;
    size_t vdim_size;
    size_t comm_size;
    const size_t minimal_indices_count = 1;

    void init_distributor(const std::pair<size_t, size_t>& elem_range)
    {
        size_t indices_count = std::get<0>(get_expected_recv_counts(elem_range.second));
        indices_distributor_impl.reset(new IndicesDistributor(elem_range.first,
                                                              indices_count));
    }

    sparse_allreduce_strategy_impl(const std::string& args, size_t size) :
        value_to_indices_ratio(),
        vdim_size(),
        comm_size(size)
    {
        std::vector<size_t> default_params { default_value_to_indices_ratio, default_vdim_size};
        if (!args.empty())
        {
            constexpr const char* masks = "[](){}";
            constexpr const char delim = ':';
            std::string arg_copy;
            arg_copy.reserve(args.size());
            std::remove_copy_if(args.begin(), args.end(),
                                std::back_inserter(arg_copy), [](char sym)
                                {
                                    return std::strchr(masks, sym);
                                });
            auto sparse_params = tokenize(arg_copy, delim);
            default_params.resize(std::max(sparse_params.size(), default_params.size()));
            std::transform(sparse_params.begin(), sparse_params.end(), default_params.begin(),
                           [](const std::string& val)
            {
                return std::stoull(val);
            });
        }

        value_to_indices_ratio = default_params[0];
        vdim_size = default_params[1];
    }

    sparse_allreduce_strategy_impl(const allgatherv_strategy_impl&) = delete;
    sparse_allreduce_strategy_impl& operator=(const allgatherv_strategy_impl&) = delete;
    ~sparse_allreduce_strategy_impl() = default;

    std::tuple<size_t, size_t> get_expected_recv_counts(size_t elem_count) const
    {
        size_t indices_count = std::max(elem_count / value_to_indices_ratio,
                                        minimal_indices_count);
        size_t vdim_count = (elem_count / indices_count);

        return std::tuple<size_t, size_t>(indices_count, indices_count * vdim_count);
    }

    template<class VType>
    void start_internal(ccl::communicator& comm,
                        const IType send_ibuf, size_t send_icount,
                        const VType send_vbuf, size_t send_vcount,
                        IType recv_ibuf, size_t recv_icount,
                        VType recv_vbuf, size_t recv_vcount,
                        const ccl::coll_attr& attr, ccl::stream_t& stream,
                        req_list_t& reqs,
                        sparse_allreduce_user_ctx_t& user_ctx)
    {
        auto expected = get_expected_recv_counts(send_icount);
        recv_icount = std::get<0>(expected);
        recv_vcount = std::get<1>(expected);

        auto& sparse_attr = const_cast<ccl_coll_attr_t&>(attr);
        sparse_attr.sparse_allreduce_completion_fn = sparse_allreduce_completion_fn;
        sparse_attr.sparse_allreduce_completion_ctx = &user_ctx;

        reqs.push_back(comm.sparse_allreduce(send_ibuf, std::get<0>(expected),
                                             send_vbuf, send_vcount,
                                             recv_ibuf, recv_icount,
                                             recv_vbuf, recv_vcount,
                                             ccl::reduction::sum,
                                             &sparse_attr, stream));
    }

    std::unique_ptr<IndicesDistributor> indices_distributor_impl;
};

#endif /* SPARSE_ALLREDUCE_STRATEGY_HPP */
