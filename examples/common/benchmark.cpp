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
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <math.h>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "base.hpp"
#include "bfp16.h"

#define BUF_COUNT         (16)
#define ELEM_COUNT        (128)
#define SINGLE_ELEM_COUNT (BUF_COUNT * ELEM_COUNT)
#define ALIGNMENT         (2 * 1024 * 1024)
#define DTYPE             float
#define MATCH_ID_SIZE     (256)

constexpr size_t default_value_to_indices_ratio = 3;
constexpr size_t default_vdim_size = ELEM_COUNT / 3;

/* different collectives with duplications */
#define DEFAULT_COLL_LIST "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce," \
                          "sparse_allreduce,sparse_allreduce_bfp16," \
                          "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce," \
                          "sparse_allreduce,sparse_allreduce_bfp16"

typedef enum
{
    LOOP_REGULAR,
    LOOP_UNORDERED
} loop_type_t;

#define DEFAULT_LOOP "regular"

class base_coll;

using coll_list_t = std::vector<std::unique_ptr<base_coll>>;
using req_list_t = std::vector<std::shared_ptr<ccl::request>>;

#ifdef CCL_ENABLE_SYCL
template<typename Dtype>
using sycl_buffer_t = cl::sycl::buffer<Dtype, 1>;

cl::sycl::queue sycl_queue;
#endif /* CCL_ENABLE_SYCL */

constexpr const char* help_message = "\nplease specify backend, loop type and comma-separated list of collective names\n\n"
                                     "example:\n\tcpu regular allgatherv,allreduce,sparse_allreduce,sparse_allreduce_bfp16\n"
                                     "example:\n\tsycl unordered bcast,reduce\n"
                                     "\n\n\tThe collectives \"sparse_*\" support additional configuration parameters:\n"
                                     "\n\t\t\"indices_to_value_ratio\" - to produce indices count not more than 'elem_count/indices_to_value_ratio\n"
                                     "\t\t\t(default value is 3)\n"
                                     "\n\t\t\"vdim_count\" - maximum value counts for index\n"
                                     "\t\t\t(default values determines all elapsed elements after \"indices_to_value_ratio\" recalculation application)\n"
                                     "\n\tUser can set this additional parameters to sparse collective in the way:\n"
                                     "\n\t\tsparse_allreduce[4:99]\n"
                                     "\t\t\t - to set \"indices_to_value_ratio\" in 4 and \"vdim_cout\" in 99\n"
                                     "\t\tsparse_allreduce[6]\n"
                                     "\t\t\t - to set \"indices_to_value_ratio\" in 6 and \"vdim_cout\" in default\n"
                                     "\n\tPlease use default configuration in most cases! You do not need to change it in general benchmark case\n";

std::list<std::string> tokenize(const std::string& input, char delimeter)
{
    std::stringstream ss(input);
    std::list<std::string> ret;
    std::string value;
    while (std::getline(ss, value, delimeter))
    {
        ret.push_back(value);
    }
    return ret;
}

// base polymorph collective wrapper class
struct base_coll
{
    virtual ~base_coll() = default;

    virtual const char* name() const noexcept { return nullptr; };

    virtual void prepare(size_t count) {};
    virtual void finalize(size_t count) {};

    virtual void start(size_t count, size_t buf_idx,
                       const ccl_coll_attr_t& coll_attr,
                       req_list_t& reqs) = 0;

    virtual void start_single(size_t count,
                              const ccl_coll_attr_t& coll_attr,
                              req_list_t& reqs) = 0;

    void* send_bufs[BUF_COUNT] = { nullptr };
    void* recv_bufs[BUF_COUNT] = { nullptr };
    void* single_send_buf = nullptr;
    void* single_recv_buf = nullptr;

    bool check_values = false;

    //the global communicator & stream for all collectives
    static ccl::communicator_t comm;
    static ccl::stream_t stream;
};

ccl::communicator_t base_coll::comm;
ccl::stream_t base_coll::stream;

// cpu-specific base implementation
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
                       const ccl_coll_attr_t& coll_attr,
                       req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm, count,
                                      static_cast<Dtype*>(send_bufs[buf_idx]),
                                      static_cast<Dtype*>(recv_bufs[buf_idx]),
                                      coll_attr, stream, reqs);
    }

    virtual void start_single(size_t count,
                              const ccl_coll_attr_t& coll_attr,
                              req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm, count,
                                      static_cast<Dtype*>(single_send_buf),
                                      static_cast<Dtype*>(single_recv_buf),
                                      coll_attr, stream, reqs);
    }
};

#ifdef CCL_ENABLE_SYCL
// sycl-specific base implementation
template<class Dtype, class strategy>
struct sycl_base_coll : virtual base_coll, private strategy
{
    using coll_strategy = strategy;

    template<class ...Args>
    sycl_base_coll(size_t sbuf_multiplier, size_t rbuf_multiplier, Args&& ...args) :
        coll_strategy(std::forward<Args>(args)...)
    {
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            send_bufs[idx] = new cl::sycl::buffer<Dtype, 1>(ELEM_COUNT * sbuf_multiplier);
            recv_bufs[idx] = new cl::sycl::buffer<Dtype, 1>(ELEM_COUNT * rbuf_multiplier);
        }
        single_send_buf = new cl::sycl::buffer<Dtype, 1>(SINGLE_ELEM_COUNT * sbuf_multiplier);
        single_recv_buf = new cl::sycl::buffer<Dtype, 1>(SINGLE_ELEM_COUNT * rbuf_multiplier);
    }

    sycl_base_coll() : sycl_base_coll(1, 1)
    {
    }

    virtual ~sycl_base_coll()
    {
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            delete static_cast<sycl_buffer_t<Dtype>*>(send_bufs[idx]);
            delete static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[idx]);
        }
        delete static_cast<sycl_buffer_t<Dtype>*>(single_send_buf);
        delete static_cast<sycl_buffer_t<Dtype>*>(single_recv_buf);
    }

    const char* name() const noexcept override
    {
        return coll_strategy::class_name();
    }

    virtual void start(size_t count, size_t buf_idx,
                       const ccl_coll_attr_t& coll_attr,
                       req_list_t& reqs) override
    {
        sycl_buffer_t<Dtype> &send_buf = *(static_cast<sycl_buffer_t<Dtype>*>(send_bufs[buf_idx]));
        sycl_buffer_t<Dtype> &recv_buf = *(static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[buf_idx]));
        coll_strategy::template start_internal<sycl_buffer_t<Dtype> &>(*comm, count,
                                      send_buf,
                                      recv_buf,
                                      coll_attr, stream, reqs);
    }

    virtual void start_single(size_t count,
                              const ccl_coll_attr_t& coll_attr,
                              req_list_t& reqs) override
    {
        sycl_buffer_t<Dtype> &send_buf = *(static_cast<sycl_buffer_t<Dtype>*>(single_send_buf));
        sycl_buffer_t<Dtype> &recv_buf = *(static_cast<sycl_buffer_t<Dtype>*>(single_recv_buf));
        coll_strategy::template start_internal<sycl_buffer_t<Dtype> &>(*comm, count,
                                      send_buf,
                                      recv_buf,
                                      coll_attr, stream, reqs);
    }
};
#endif /* CCL_ENABLE_SYCL */

// collectives strategy implementations
struct allgatherv_strategy_impl
{
    size_t comm_size = 0;
    size_t* recv_counts = nullptr;
    allgatherv_strategy_impl(size_t size) : comm_size(size)
    {
        int result = posix_memalign((void**)&recv_counts, ALIGNMENT, comm_size * sizeof(size_t));
        (void)result;
    }

    allgatherv_strategy_impl(const allgatherv_strategy_impl&) = delete;
    allgatherv_strategy_impl& operator=(const allgatherv_strategy_impl&) = delete;

    ~allgatherv_strategy_impl()
    {
        free(recv_counts);
    }

    static constexpr const char* class_name() { return "allgatherv"; }

    template<class Dtype>
    void start_internal(ccl::communicator& comm, size_t count, const Dtype send_buf, Dtype recv_buf,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        for (size_t idx = 0; idx < comm_size; idx++)
        {
            recv_counts[idx] = count;
        }
        reqs.push_back(comm.allgatherv(send_buf, count,
                                       recv_buf, recv_counts,
                                       &coll_attr, stream));
    }
};

struct allreduce_strategy_impl
{
    static constexpr const char* class_name() { return "allreduce"; }

    template<class Dtype>
    void start_internal(ccl::communicator &comm, size_t count, const Dtype send_buf, Dtype recv_buf,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        reqs.push_back(comm.allreduce(send_buf, recv_buf, count, ccl::reduction::sum,
                                      &coll_attr, stream));
    }
};

struct alltoall_strategy_impl
{
    static constexpr const char* class_name() { return "alltoall"; }

    template<class Dtype>
    void start_internal(ccl::communicator &comm, size_t count, const Dtype send_buf, Dtype recv_buf,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        reqs.push_back(comm.alltoall(send_buf, recv_buf, count, &coll_attr, stream));
    }
};

struct alltoallv_strategy_impl
{
    size_t comm_size = 0;
    size_t* send_counts = nullptr;
    size_t* recv_counts = nullptr;

    alltoallv_strategy_impl(size_t size) : comm_size(size)
    {
        int result = posix_memalign((void**)&send_counts, ALIGNMENT, comm_size * sizeof(size_t));
        result = posix_memalign((void**)&recv_counts, ALIGNMENT, comm_size * sizeof(size_t));
        (void)result;
    }

    alltoallv_strategy_impl(const alltoallv_strategy_impl&) = delete;
    alltoallv_strategy_impl& operator=(const alltoallv_strategy_impl&) = delete;

    ~alltoallv_strategy_impl()
    {
        free(send_counts);
        free(recv_counts);
    }

    static constexpr const char* class_name() { return "alltoallv"; }

    template<class Dtype>
    void start_internal(ccl::communicator& comm, size_t count, const Dtype send_buf, Dtype recv_buf,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        for (size_t idx = 0; idx < comm_size; idx++)
        {
            send_counts[idx] = count;
            recv_counts[idx] = count;
        }
        reqs.push_back(comm.alltoallv(send_buf, send_counts,
                                      recv_buf, recv_counts,
                                      &coll_attr, stream));
    }
};

struct bcast_strategy_impl
{
    static constexpr const char* class_name() { return "bcast"; }

    template<class Dtype>
    void start_internal(ccl::communicator &comm, size_t count, Dtype send_buf, Dtype recv_buf,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        (void)send_buf;
        reqs.push_back(comm.bcast(recv_buf, count, COLL_ROOT, &coll_attr, stream));
    }
};

struct reduce_strategy_impl
{
    static constexpr const char* class_name() { return "reduce"; }

    template<class Dtype>
    void start_internal(ccl::communicator &comm, size_t count, const Dtype send_buf, Dtype recv_buf,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        reqs.push_back(comm.reduce(send_buf, recv_buf,count, ccl::reduction::sum,
                                   COLL_ROOT, &coll_attr, stream));
    }
};

namespace sparse_detail
{
template <class IType>
struct incremental_indices_distributor
{
    incremental_indices_distributor(IType from_range, IType to_range, IType step = 1)
    {
        from = std::min(from_range, to_range);
        to = std::max(from_range, to_range);

        size_t unique_indices_count = (to - from) / step;
        if (unique_indices_count == 0)
        {
            throw std::runtime_error(std::string("Invalid range! from: ") + std::to_string(from) +
                                     ", to: " + std::to_string(to) +
                                     ", step: " + std::to_string(step));
        }

        generated_indices.resize(unique_indices_count);
        generated_indices[0] = from;
        for(size_t i = 1; i < unique_indices_count; i++)
        {
            generated_indices[i] = generated_indices[i - 1] + step;
        }

        cur_index = 0;
    }

    IType operator() ()
    {
        return generated_indices.at(cur_index ++ % generated_indices.size());
    }

private:
    std::vector<IType> generated_indices;
    IType from;
    IType to;
    std::atomic<size_t> cur_index;
};

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
    const size_t minimal_indices_cout = 1;

    void init_distributor(const std::pair<size_t, size_t>& elem_range)
    {
        size_t indices_count = std::get<0>(get_expected_recv_counts(elem_range.second));

        indices_distributor_impl.reset( new IndicesDistributor(elem_range.first, 
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
                                std::back_inserter(arg_copy), [masks](char sym)
                                {
                                    return std::strchr(masks, sym);
                                });
            auto sparse_params = tokenize(arg_copy, delim);
            default_params.resize(std::max(sparse_params.size(), default_params.size()));
            std::transform(sparse_params.begin(), sparse_params.end(), default_params.begin(), [](const std::string& val)
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
                                        minimal_indices_cout);
        size_t vdim_count = (elem_count / indices_count);

        return std::tuple<size_t, size_t>( indices_count, indices_count * vdim_count );
    }

    template<class Dtype>
    void start_internal(ccl::communicator& comm, const IType send_ibuf, size_t send_icount,
                        const Dtype send_vbuf, size_t send_vcount,
                        remove_all_t<IType>** recv_ibuf, size_t* recv_icount,
                        remove_all_t<Dtype>** recv_vbuf, size_t* recv_vcount,
                        const ccl_coll_attr_t& coll_attr, ccl::stream_t& stream,
                        req_list_t& reqs)
    {
        //TODO sparse do not works with cache actuall yet
        auto& mod_attr = const_cast<ccl_coll_attr_t&>(coll_attr);
        bool old_to_cache = mod_attr.to_cache;
        mod_attr.to_cache = 0;

        auto expected = get_expected_recv_counts(send_icount);
        *recv_icount = std::get<0>(expected);
        *recv_vcount = std::get<1>(expected);

        reqs.push_back(comm.sparse_allreduce(send_ibuf, std::get<0>(expected),
                                             send_vbuf, send_vcount,
                                             recv_ibuf, recv_icount,
                                             recv_vbuf, recv_vcount,
                                             ccl::reduction::sum,
                                             &coll_attr, stream));
        //TODO sparse do not works with cache actuall yet
        mod_attr.to_cache = old_to_cache;
    }

    std::unique_ptr<IndicesDistributor> indices_distributor_impl;
};

template<class ValueType, class IndexType, class IndicesDistributorType>
void fill_sparse_data(const std::tuple<size_t, size_t>& expected_recv_counts,
                      IndicesDistributorType& generator,
                      size_t elem_count, IndexType* send_ibuf, ValueType* send_buf,
                      ValueType* recv_buf, size_t& recv_icount, size_t& recv_vcount,
                      size_t rank)
{

    recv_icount = std::get<0>(expected_recv_counts);
    recv_vcount = std::get<1>(expected_recv_counts);
    size_t vdim_count = recv_vcount / recv_icount;
    for (size_t i_idx = 0; i_idx < recv_icount; i_idx++)
    {
        send_ibuf[i_idx] = generator();
        for (size_t e_idx = 0; e_idx < vdim_count; e_idx++)
        {
            if (i_idx * vdim_count + e_idx < elem_count)
            {
                send_buf[i_idx * vdim_count + e_idx] = (rank + 1 + e_idx);
            }
        }
    }

    std::fill(recv_buf, recv_buf + elem_count, ValueType{0});
}

// override for ccl::bfp16
template<class IndexType, class IndicesDistributorType>
void fill_sparse_data(const std::tuple<size_t, size_t>& expected_recv_counts,
                      IndicesDistributorType& generator,
                      size_t elem_count, IndexType* send_ibuf, ccl::bfp16* send_buf,
                      ccl::bfp16* recv_buf, size_t& recv_icount, size_t& recv_vcount,
                      size_t rank)
{

    recv_icount = std::get<0>(expected_recv_counts);
    recv_vcount = std::get<1>(expected_recv_counts);
    size_t vdim_count = recv_vcount / recv_icount;
    
    using from_type = float;
    std::vector<from_type> send_buf_from(elem_count);

    
    for (size_t i_idx = 0; i_idx < recv_icount; i_idx++)
    {
        send_ibuf[i_idx] = generator();
        for (size_t e_idx = 0; e_idx < vdim_count; e_idx++)
        {
            if (i_idx * vdim_count + e_idx < elem_count)
            {
                send_buf_from[i_idx * vdim_count + e_idx] = 1.0f / (rank + 1 + e_idx);
            }
        }
    }

    std::fill(recv_buf, recv_buf + elem_count, ccl::bfp16{0});
    // convert send_buf from float to send_buf ib bfp16
    convert_fp32_to_bfp16_arrays(send_buf_from.data(), send_buf, elem_count);
}

template<class ValueType, class IndexType>
void check_sparse_result(const std::tuple<size_t, size_t>& expected_recv_counts,
                         size_t elem_count,
                         const IndexType* send_ibuf, const ValueType* send_buf,
                         const IndexType* recv_ibuf, const ValueType* recv_buf,
                         size_t recv_icount, size_t recv_vcount,
                         size_t comm_size, size_t comm_rank)
{
    size_t indices_count, vdim_count;
    std::tie(indices_count, vdim_count) = expected_recv_counts;
    vdim_count = vdim_count / indices_count;

    std::vector<IndexType> aggregated_indices;
    aggregated_indices.reserve(indices_count * comm_size);

    std::vector<ValueType> aggregated_values;
    aggregated_values.reserve(indices_count * vdim_count * comm_size);

    std::vector<ValueType> base_send_data(send_buf, send_buf + indices_count * vdim_count);
    std::transform(base_send_data.begin(), base_send_data.end(),
                   base_send_data.begin(),
                   std::bind(std::minus<ValueType>(),
                             std::placeholders::_1, comm_rank));

    for(size_t rank_index = 0; rank_index < comm_size; rank_index++)
    {
        std::copy(send_ibuf, send_ibuf + indices_count,
                  std::back_inserter(aggregated_indices));

        std::transform(base_send_data.begin(), base_send_data.end(),
                       std::back_inserter(aggregated_values),
                       std::bind(std::plus<ValueType>(),
                                 std::placeholders::_1, rank_index));
    }

    /*calculate expected values*/
    using values_array = std::vector<ValueType>;
    using values_array_for_indices = std::unordered_map<IndexType, values_array>;

    values_array_for_indices expected;
    for (size_t index_pos = 0; index_pos < aggregated_indices.size(); index_pos++)
    {
        IndexType index = aggregated_indices[index_pos];

        typename values_array::iterator from = aggregated_values.begin();
        typename values_array::iterator to = aggregated_values.begin();
        std::advance(from, index_pos * vdim_count);
        std::advance(to, (index_pos + 1) * vdim_count);

        auto candidate_it = expected.find(index);
        if (candidate_it == expected.end())
        {
            expected.emplace(std::piecewise_construct,
                             std::forward_as_tuple(index),
                             std::forward_as_tuple(from, to));
        }
        else
        {
            std::transform(candidate_it->second.begin(), candidate_it->second.end(),
                           from, candidate_it->second.begin(), 
                           std::plus<ValueType>());
        }
    }

    // check received values
    for (size_t index_pos = 0; index_pos < recv_icount; index_pos++)
    {
        IndexType recv_index_value = recv_ibuf[index_pos];
        auto expected_it = expected.find(recv_index_value);
        if (expected_it == expected.end())
        {
            throw std::runtime_error(std::string(__FUNCTION__) + " - incorrect index received: " +
                                     std::to_string(recv_index_value));
        }

        const values_array& expected_values = expected_it->second;

        const ValueType* from = recv_buf + index_pos * vdim_count;
        const ValueType* to = from + vdim_count;

        values_array got_values(from, to);
        if (got_values != expected_values)
        {
            std::stringstream ss;
            const char* val_ptr = getenv("CCL_ATL_TRANSPORT");
            if (!val_ptr or !strcmp(val_ptr, "mpi"))
            {
                ss << "Environment value CCL_ATL_TRANSPORT=" << (val_ptr ? val_ptr : "<default>")
                   << ". MPI has a limited support of sparse allreduce. " 
                   << "Make sure, that the following conditions are satisfied: \n"
                   << "\tCCL_WORKER_OFFLOAD=0\n"
                   << "\tYou have only one non-repeated sparse collective in colls list.\n\n"
                   << "Failed scenario settings:\n";
            }
            ss << "elem_count: " << elem_count
               << ", indices_count: " << indices_count
               << ", vdim_count:" <<  vdim_count
               << ", recv_icount: " << recv_icount
               << ", recv_vcount: " << recv_vcount;

            ss << "\nValues got:\n";
            std::copy(got_values.begin(), got_values.end(),
                      std::ostream_iterator<ValueType>(ss, ","));
            ss << "\nValues expected:\n";
            std::copy(expected_values.begin(), expected_values.end(),
                      std::ostream_iterator<ValueType>(ss, ","));

            ss << "\nRank: " << comm_rank << ", send_bufs:\n";
            std::copy(send_buf, send_buf + indices_count * vdim_count,
                      std::ostream_iterator<ValueType>(ss, ","));
            ss << "\nRank: " << comm_rank << ", send_ibufs:\n";
            std::copy(send_ibuf, send_ibuf + indices_count,
                      std::ostream_iterator<IndexType>(ss, ","));

            ss << "\nRank: " << comm_rank << ", recv_bufs:\n";
            std::copy(recv_buf, recv_buf + indices_count * vdim_count,
                      std::ostream_iterator<ValueType>(ss, ","));
            ss << "\nRank: " << comm_rank << ", recv_ibufs:\n";
            std::copy(recv_ibuf, recv_ibuf + indices_count,
                      std::ostream_iterator<IndexType>(ss, ","));

            ss << "\nAggregated indices:\n";
            std::copy(aggregated_indices.begin(), aggregated_indices.end(),
                      std::ostream_iterator<IndexType>(ss, ","));
            ss << "\nAggregated values:\n";
            std::copy(aggregated_values.begin(), aggregated_values.end(),
                      std::ostream_iterator<ValueType>(ss, ","));

            throw std::runtime_error(std::string(__FUNCTION__) + " - incorrect values received!\n" + ss.str());
        }
    }
}

// override for ccl::bfp16
template<class IndexType>
void check_sparse_result(const std::tuple<size_t, size_t>& expected_recv_counts,
                         size_t elem_count,
                         const IndexType* send_ibuf, const ccl::bfp16* send_buf,
                         const IndexType* recv_ibuf, const ccl::bfp16* recv_buf,
                         size_t recv_icount, size_t recv_vcount,
                         size_t comm_size, size_t comm_rank)
{
    size_t indices_count, vdim_count;
    std::tie(indices_count, vdim_count) = expected_recv_counts;
    vdim_count = vdim_count / indices_count;

    std::vector<IndexType> aggregated_indices;
    aggregated_indices.reserve(indices_count * comm_size);

    std::vector<float> aggregated_values;
    aggregated_values.reserve(indices_count * vdim_count * comm_size);

    for(size_t rank_index = 0; rank_index < comm_size; rank_index++)
    {
        std::copy(send_ibuf, send_ibuf + indices_count,
                  std::back_inserter(aggregated_indices));

        for (size_t i_idx = 0; i_idx < indices_count; i_idx++)
        {
            for (size_t e_idx = 0; e_idx < vdim_count; e_idx++)
            {
                if (i_idx * vdim_count + e_idx < elem_count)
                {
                    std::back_inserter(aggregated_values) = 1.0f / (rank_index + 1 + e_idx);
                }
            }
        }
    }

    /*calculate expected values*/
    using values_array = std::vector<float>;
    using values_array_for_indices = std::unordered_map<IndexType, values_array>;

    values_array_for_indices expected;
    for (size_t index_pos = 0; index_pos < aggregated_indices.size(); index_pos++)
    {
        IndexType index = aggregated_indices[index_pos];

        typename values_array::iterator from = aggregated_values.begin();
        typename values_array::iterator to = aggregated_values.begin();
        std::advance(from, index_pos * vdim_count);
        std::advance(to, (index_pos + 1) * vdim_count);

        auto candidate_it = expected.find(index);
        if (candidate_it == expected.end())
        {
            expected.emplace(std::piecewise_construct,
                             std::forward_as_tuple(index),
                             std::forward_as_tuple(from, to));
        }
        else
        {
            std::transform(candidate_it->second.begin(), candidate_it->second.end(),
                           from, candidate_it->second.begin(), std::plus<float>());
        }
    }

    // check received values
    std::vector<float> recv_buf_float(recv_vcount, float{0});
    convert_bfp16_to_fp32_arrays(reinterpret_cast<void*>(const_cast<ccl::bfp16*>(recv_buf)), recv_buf_float.data(), recv_vcount);
    
    /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */
    /* added conversion error float->bfp16 for comm_size == 1*/
    constexpr double max_error = 0;
    double log_base2 = log(comm_size != 1 ? comm_size : 2 ) / log(2);
    double g = (log_base2 * BFP16_PRECISION)/(1 - (log_base2 * BFP16_PRECISION));
  
    for (size_t index_pos = 0; index_pos < recv_icount; index_pos++)
    {
        IndexType recv_index_value = recv_ibuf[index_pos];
        auto expected_it = expected.find(recv_index_value);
        if (expected_it == expected.end())
        {
            throw std::runtime_error(std::string(__FUNCTION__) + "_bfp16 - incorrect index received: " +
                                     std::to_string(recv_index_value));
        }

        const float* from = recv_buf_float.data() + index_pos * vdim_count;
        const float* to = from + vdim_count;
        const values_array& expected_values = expected_it->second;
        if (std::distance(from, to) != expected_values.size())
        {
            throw std::runtime_error(std::string(__FUNCTION__) + "_bfp16 - incorrect recv_buf count, got: " + 
                                     std::to_string(std::distance(from, to)) + ", expected: " +
                                     std::to_string(expected_values.size()));
        }
        
        for(size_t i = 0; i < expected_values.size(); i++)
        {
            double compare_max_error = g * expected_values[i] * 2;
            if (fabs(compare_max_error) < fabs(expected_values[i] - from[i]))
            {
                std::stringstream ss;
                const char* val_ptr = getenv("CCL_ATL_TRANSPORT");
                if (!val_ptr or !strcmp(val_ptr, "mpi"))
                {
                    ss << "Environment value CCL_ATL_TRANSPORT=" << (val_ptr ? val_ptr : "<default>")
                       << ". MPI has a limited support of sparse allreduce. "
                       << "Make sure, that the following conditions are satisfied: \n\n"
                       << "\tCCL_WORKER_OFFLOAD=0\n"
                       << "\tYou have only one non-repeated sparse collective in colls list.\n\n"
                       << "Failed scenario settings:\n";
                }
                
                ss << "elem_count: " << elem_count
                   << ", indices_count: " << indices_count
                   << ", vdim_count:" <<  vdim_count
                   << ", recv_icount: " << recv_icount
                   << ", recv_vcount: " << recv_vcount
                   << ", absolute_max_error: " << compare_max_error;

                ss << "\nValues got:\n";
                std::copy(from, to,
                          std::ostream_iterator<float>(ss, ","));
                ss << "\nValues expected:\n";
                std::copy(expected_values.begin(), expected_values.end(),
                          std::ostream_iterator<float>(ss, ","));

                ss << "\nRank: " << comm_rank << ", send_ibufs:\n";
                std::copy(send_ibuf, send_ibuf + indices_count,
                          std::ostream_iterator<IndexType>(ss, ","));

                ss << "\nRank: " << comm_rank << ", recv_bufs:\n";
                std::copy(recv_buf_float.begin(), recv_buf_float.end(),
                          std::ostream_iterator<float>(ss, ","));
                ss << "\nRank: " << comm_rank << ", recv_ibufs:\n";
                std::copy(recv_ibuf, recv_ibuf + indices_count,
                          std::ostream_iterator<IndexType>(ss, ","));

                ss << "\nAggregated indices:\n";
                std::copy(aggregated_indices.begin(), aggregated_indices.end(),
                          std::ostream_iterator<IndexType>(ss, ","));
                ss << "\nAggregated values:\n";
                std::copy(aggregated_values.begin(), aggregated_values.end(),
                          std::ostream_iterator<float>(ss, ","));

                throw std::runtime_error(std::string(__FUNCTION__) + "_bfp16 - incorrect values received!\n" + ss.str());
            }
        }
    }
}
}
// collective wrappers final imlementation
template<class Dtype>
struct cpu_allgatherv_coll : cpu_base_coll<Dtype, allgatherv_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, allgatherv_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    cpu_allgatherv_coll() : coll_base(1, base_coll::comm->size(),  base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                ((Dtype*)send_bufs[b_idx])[e_idx] = comm->rank();
            }

            for (size_t idx = 0; idx < comm->size(); idx++)
            {
                for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
                {
                    ((Dtype*)recv_bufs[b_idx])[idx * elem_count + e_idx] = 0;
                }
            }
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        Dtype sbuf_expected = comm->rank();
        Dtype value;
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                value = ((Dtype*)send_bufs[b_idx])[e_idx];
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx,
                           static_cast<float>(sbuf_expected),
                           static_cast<float>(value));
                    ASSERT(0, "unexpected value");
                }
            }

            for (size_t idx = 0; idx < comm->size(); idx++)
            {
                Dtype rbuf_expected = idx;
                for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
                {
                    value = ((Dtype*)recv_bufs[b_idx])[idx * elem_count + e_idx];
                    if (value != rbuf_expected)
                    {
                        printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                               this->name(), b_idx, e_idx,
                               static_cast<float>(rbuf_expected),
                               static_cast<float>(value));
                        ASSERT(0, "unexpected value");
                    }
                }
            }
        }
    }
};

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
struct sycl_allgatherv_coll : sycl_base_coll<Dtype, allgatherv_strategy_impl>
{
    using coll_base = sycl_base_coll<Dtype, allgatherv_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    sycl_allgatherv_coll() : coll_base(1, base_coll::comm->size(), base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        size_t local_rank = comm->rank();
        size_t local_size = comm->size();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class allatherv_buf_fill>(range<1>{elem_count}, [=](item<1> e_idx)
                {
                    send_buf_acc[e_idx] = local_rank;
                    for (size_t idx = 0; idx < local_size; idx++)
                    {
                        recv_buf_acc[idx * elem_count + e_idx.get_id(0)] = 0;
                    }
                });
            });
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        bool unexpected_device_value = false;
        size_t local_size = comm->size();
        Dtype sbuf_expected = comm->rank();

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class allgatherv_buf_check>(range<1>{elem_count}, [=](item<1> e_idx) mutable
                {
                    Dtype value = send_buf_acc[e_idx];
                    if (value != sbuf_expected)
                        unexpected_device_value = true;

                    for (size_t idx = 0; idx < local_size; idx++)
                    {
                        Dtype rbuf_expected = idx;
                        value = recv_buf_acc[idx * elem_count + e_idx.get_id(0)];
                        if (value != rbuf_expected)
                            unexpected_device_value = true;
                    }
                });
            });
        }

        Dtype value;
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
            auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
            auto send_buf_acc = send_buf->template get_access<mode::write>();
            auto recv_buf_acc = recv_buf->template get_access<mode::write>();

            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                value = send_buf_acc[e_idx];
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }

            for (size_t idx = 0; idx < comm->size(); idx++)
            {
                Dtype rbuf_expected = idx;
                for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
                {
                    value = recv_buf_acc[idx * elem_count + e_idx];
                    if (value != rbuf_expected)
                    {
                        printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                               this->name(), b_idx, e_idx, rbuf_expected, value);
                        ASSERT(0, "unexpected value");
                    }
                }
            }
        }

        if (unexpected_device_value)
            ASSERT(0, "unexpected value on device");
    }
};
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
struct cpu_allreduce_coll : cpu_base_coll<Dtype, allreduce_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, allreduce_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::stream;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                ((Dtype*)send_bufs[b_idx])[e_idx] = comm->rank();
                ((Dtype*)recv_bufs[b_idx])[e_idx] = 0;
            }
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        Dtype sbuf_expected = comm->rank();
        Dtype rbuf_expected = (comm->size() - 1) * ((float)comm->size() / 2);
        Dtype value;
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                value = ((Dtype*)send_bufs[b_idx])[e_idx];
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx,
                           static_cast<float>(sbuf_expected),
                           static_cast<float>(value));
                    ASSERT(0, "unexpected value");
                }

                value = ((Dtype*)recv_bufs[b_idx])[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx,
                           static_cast<float>(rbuf_expected),
                           static_cast<float>(value));
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
struct sycl_allreduce_coll : sycl_base_coll<Dtype, allreduce_strategy_impl>
{
    using coll_base = sycl_base_coll<Dtype, allreduce_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        size_t local_rank = comm->rank();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class allreduce_buf_fill>(range<1>{elem_count}, [=](item<1> e_idx)
                {
                    send_buf_acc[e_idx] = local_rank;
                    recv_buf_acc[e_idx] = 0;
                });
            });
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        bool unexpected_device_value = false;
        Dtype sbuf_expected = comm->rank();
        Dtype rbuf_expected = (comm->size() - 1) * ((float)comm->size() / 2);

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class allreduce_buf_check>(range<1>{elem_count}, [=](item<1> e_idx) mutable
                {
                    Dtype value = send_buf_acc[e_idx];
                    if (value != sbuf_expected)
                        unexpected_device_value = true;

                    value = recv_buf_acc[e_idx];
                    if (value != rbuf_expected)
                        unexpected_device_value = true;
                });
            });
        }

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
            auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
            auto send_buf_acc = send_buf->template get_access<mode::read>();
            auto recv_buf_acc = recv_buf->template get_access<mode::read>();

            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                Dtype value = send_buf_acc[e_idx];
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }

                value = recv_buf_acc[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, rbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }

        if (unexpected_device_value)
            ASSERT(0, "unexpected value on device");
    }
};
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
struct cpu_alltoall_coll : cpu_base_coll<Dtype, alltoall_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, alltoall_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::stream;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    cpu_alltoall_coll() : coll_base(base_coll::comm->size(), base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

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
        if (!check_values)
            return;

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
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }

                value = ((Dtype*)recv_bufs[b_idx])[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, rbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
struct sycl_alltoall_coll : sycl_base_coll<Dtype, alltoall_strategy_impl>
{
    using coll_base = sycl_base_coll<Dtype, alltoall_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    sycl_alltoall_coll() : coll_base(base_coll::comm->size(), base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        size_t local_rank = comm->rank();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class alltoall_buf_fill>(range<1>{elem_count*comm->size()}, [=](item<1> e_idx)
                {
                    send_buf_acc[e_idx] = local_rank;
                    recv_buf_acc[e_idx] = 0;
                });
            });
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        bool unexpected_device_value = false;
        Dtype sbuf_expected = comm->rank();
        size_t comm_size = comm->size();

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class alltoall_buf_check>(range<1>{elem_count * comm_size}, [=](item<1> e_idx) mutable
                {
                    Dtype value = send_buf_acc[e_idx];
                    Dtype rbuf_expected = static_cast<Dtype>(e_idx.get_id(0) / elem_count);
                    if (value != sbuf_expected)
                        unexpected_device_value = true;

                    value = recv_buf_acc[e_idx];
                    if (value != rbuf_expected)
                        unexpected_device_value = true;
                });
            });
        }

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
            auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
            auto send_buf_acc = send_buf->template get_access<mode::read>();
            auto recv_buf_acc = recv_buf->template get_access<mode::read>();

            for (size_t e_idx = 0; e_idx < elem_count * comm_size; e_idx++)
            {
                Dtype value = send_buf_acc[e_idx];
                Dtype rbuf_expected = e_idx / elem_count;
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }

                value = recv_buf_acc[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, rbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }

        if (unexpected_device_value)
            ASSERT(0, "unexpected value on device");
    }
};
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
struct cpu_alltoallv_coll : cpu_base_coll<Dtype, alltoallv_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, alltoallv_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    cpu_alltoallv_coll() : coll_base(base_coll::comm->size(), base_coll::comm->size(), base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

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
        if (!check_values)
            return;

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
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }

                value = ((Dtype*)recv_bufs[b_idx])[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, rbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
struct sycl_alltoallv_coll : sycl_base_coll<Dtype, alltoallv_strategy_impl>
{
    using coll_base = sycl_base_coll<Dtype, alltoallv_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    sycl_alltoallv_coll() : coll_base(base_coll::comm->size(), base_coll::comm->size(), base_coll::comm->size()) {}

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        size_t local_rank = comm->rank();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class alltoallv_buf_fill>(range<1>{elem_count*comm->size()}, [=](item<1> e_idx)
                {
                    send_buf_acc[e_idx] = local_rank;
                    recv_buf_acc[e_idx] = 0;
                });
            });
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        bool unexpected_device_value = false;
        Dtype sbuf_expected = comm->rank();
        size_t comm_size = comm->size();

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class alltoallv_buf_check>(range<1>{elem_count * comm_size}, [=](item<1> e_idx) mutable
                {
                    Dtype value = send_buf_acc[e_idx];
                    Dtype rbuf_expected = static_cast<Dtype>(e_idx.get_id(0) / elem_count);
                    if (value != sbuf_expected)
                        unexpected_device_value = true;

                    value = recv_buf_acc[e_idx];
                    if (value != rbuf_expected)
                        unexpected_device_value = true;
                });
            });
        }

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
            auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
            auto send_buf_acc = send_buf->template get_access<mode::read>();
            auto recv_buf_acc = recv_buf->template get_access<mode::read>();

            for (size_t e_idx = 0; e_idx < elem_count * comm_size; e_idx++)
            {
                Dtype value = send_buf_acc[e_idx];
                Dtype rbuf_expected = e_idx / elem_count;
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }

                value = recv_buf_acc[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, rbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }

        if (unexpected_device_value)
            ASSERT(0, "unexpected value on device");
    }
};
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
struct cpu_bcast_coll : cpu_base_coll<Dtype, bcast_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, bcast_strategy_impl>;
    using coll_base::recv_bufs;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                if (comm->rank() == COLL_ROOT)
                    ((Dtype*)recv_bufs[b_idx])[e_idx] = e_idx;
                else
                    ((Dtype*)recv_bufs[b_idx])[e_idx] = 0;
            }
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        Dtype value;
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                value = ((Dtype*)recv_bufs[b_idx])[e_idx];
                if (value != e_idx)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx,
                           static_cast<float>(e_idx),
                           static_cast<float>(value));
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
struct sycl_bcast_coll : sycl_base_coll<Dtype, bcast_strategy_impl>
{
    using coll_base = sycl_base_coll<Dtype, bcast_strategy_impl>;
    using coll_base::recv_bufs;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        size_t local_rank = comm->rank();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class bcast_buf_fill>(range<1>{elem_count}, [=](item<1> e_idx)
                {
                    if (local_rank == COLL_ROOT)
                        recv_buf_acc[e_idx] = e_idx.get_id(0);
                    else
                        recv_buf_acc[e_idx] = 0;
                });
            });
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        bool unexpected_device_value = false;

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class bcast_buf_check>(range<1>{elem_count}, [=](item<1> e_idx) mutable
                {
                    if (recv_buf_acc[e_idx] != e_idx.get_id(0))
                        unexpected_device_value = true;
                });
            });
        }

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
            auto recv_buf_acc = recv_buf->template get_access<mode::read>();

            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                Dtype value = recv_buf_acc[e_idx];
                if (value != e_idx)
                {
                    printf("%s: rend_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, (Dtype)e_idx, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }

        if (unexpected_device_value)
            ASSERT(0, "unexpected value on device");
    }
};
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
struct cpu_reduce_coll : cpu_base_coll<Dtype, reduce_strategy_impl>
{
    using coll_base = cpu_base_coll<Dtype, reduce_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                ((Dtype*)send_bufs[b_idx])[e_idx] = comm->rank();
                ((Dtype*)recv_bufs[b_idx])[e_idx] = 0;
            }
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        Dtype sbuf_expected = comm->rank();
        Dtype rbuf_expected = (comm->size() - 1) * ((float)comm->size() / 2);
        Dtype value;
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                value = ((Dtype*)send_bufs[b_idx])[e_idx];
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx,
                           static_cast<float>(sbuf_expected),
                           static_cast<float>(value));
                    ASSERT(0, "unexpected value");
                }

                if (comm->rank() != COLL_ROOT)
                    continue;

                value = ((Dtype*)recv_bufs[b_idx])[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx,
                           static_cast<float>(rbuf_expected),
                           static_cast<float>(value));
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
struct sycl_reduce_coll : sycl_base_coll<Dtype, reduce_strategy_impl>
{
    using coll_base = sycl_base_coll<Dtype, reduce_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;

    virtual void prepare(size_t elem_count) override
    {
        if (!check_values)
            return;

        size_t local_rank = comm->rank();
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class reduce_buf_fill>(range<1>{elem_count}, [=](item<1> e_idx)
                {
                    send_buf_acc[e_idx] = local_rank;
                    recv_buf_acc[e_idx] = 0;
                });
            });
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        bool unexpected_device_value = false;
        Dtype sbuf_expected = comm->rank();
        Dtype rbuf_expected = (comm->size() - 1) * ((float)comm->size() / 2);
        size_t local_rank = comm->rank();

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
                auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class reduce_buf_check>(range<1>{elem_count}, [=](item<1> e_idx) mutable
                {
                    Dtype value = send_buf_acc[e_idx];
                    if (value != sbuf_expected)
                        unexpected_device_value = true;

                    if (local_rank == COLL_ROOT)
                    {
                        value = recv_buf_acc[e_idx];
                        if (value != rbuf_expected)
                            unexpected_device_value = true;
                    }
                });
            });
        }

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            auto send_buf = (static_cast<sycl_buffer_t<Dtype>*>(send_bufs[b_idx]));
            auto recv_buf = (static_cast<sycl_buffer_t<Dtype>*>(recv_bufs[b_idx]));
            auto send_buf_acc = send_buf->template get_access<mode::read>();
            auto recv_buf_acc = recv_buf->template get_access<mode::read>();

            for (size_t e_idx = 0; e_idx < elem_count; e_idx++)
            {
                Dtype value = send_buf_acc[e_idx];
                if (value != sbuf_expected)
                {
                    printf("%s: send_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, sbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }

                if (local_rank != COLL_ROOT)
                    continue;

                value = recv_buf_acc[e_idx];
                if (value != rbuf_expected)
                {
                    printf("%s: recv_bufs: buf_idx %zu, elem_idx %zu, expected %f, got %f\n",
                           this->name(), b_idx, e_idx, rbuf_expected, value);
                    ASSERT(0, "unexpected value");
                }
            }
        }

        if (unexpected_device_value)
            ASSERT(0, "unexpected value on device");
    }
};
#endif /* CCL_ENABLE_SYCL */


template<class Dtype, class IType, template<class> class IndicesDistributorType>
struct base_sparse_allreduce_coll :
        virtual base_coll,
        protected sparse_detail::sparse_allreduce_strategy_impl<IType,
                                                                IndicesDistributorType>
{
    using ITypeNonMod = typename std::remove_pointer<IType>::type;

    using coll_base = base_coll;
    using coll_strategy = sparse_detail::sparse_allreduce_strategy_impl<IType,
                                                                        IndicesDistributorType>;

    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::stream;
    using coll_base::single_send_buf;
    using coll_base::single_recv_buf;
    using coll_base::check_values;
    using coll_base::comm;


    using coll_strategy::value_to_indices_ratio;
    using coll_strategy::vdim_size;
    using coll_strategy::minimal_indices_cout;

    ITypeNonMod* recv_ibufs[BUF_COUNT] = { nullptr };
    ITypeNonMod* send_ibufs[BUF_COUNT] = { nullptr };
    ITypeNonMod* single_send_ibuf = nullptr;
    ITypeNonMod* single_recv_ibuf = nullptr;

    size_t* recv_icount = nullptr;
    size_t* recv_vcount = nullptr;
    size_t single_recv_icount {};
    size_t single_recv_vcount {};

    base_sparse_allreduce_coll(const std::string& args) :
     coll_strategy(args, base_coll::comm->size())
    {
        int result = 0;
        result = posix_memalign((void**)&recv_icount, ALIGNMENT,
                                BUF_COUNT * sizeof(size_t) * base_coll::comm->size());
        result = posix_memalign((void**)&recv_vcount, ALIGNMENT,
                                BUF_COUNT * sizeof(size_t) * base_coll::comm->size());

        memset(recv_icount, 1, BUF_COUNT * sizeof(size_t) * base_coll::comm->size());
        memset(recv_vcount, 1, BUF_COUNT * sizeof(size_t) * base_coll::comm->size());
    }

    virtual ~base_sparse_allreduce_coll()
    {
        free (recv_icount);
        free (recv_vcount);
    }

    const char* name() const noexcept override
    {
        return coll_strategy::class_name();
    }
};

template<class Dtype, class IType,
         template<class> class IndicesDistributorType = sparse_detail::incremental_indices_distributor>
struct cpu_sparse_allreduce_coll : 
        base_sparse_allreduce_coll<Dtype *, IType *, IndicesDistributorType>
{
    using coll_base = base_sparse_allreduce_coll<Dtype *, IType *, IndicesDistributorType>;
    using coll_strategy = typename coll_base::coll_strategy;

    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::stream;
    using coll_base::single_send_buf;
    using coll_base::single_send_ibuf;
    using coll_base::single_recv_buf;
    using coll_base::single_recv_ibuf;
    using coll_base::single_recv_icount;
    using coll_base::single_recv_vcount;
    using coll_base::comm;
    using coll_base::check_values;

    using coll_base::recv_ibufs;
    using coll_base::send_ibufs;
    using coll_base::recv_icount;
    using coll_base::recv_vcount;

    cpu_sparse_allreduce_coll(const std::string& args, 
                              size_t sbuf_size_modifier = 1,
                              size_t rbuf_size_modifier = 1) : coll_base(args)
    {
        int result = 0;
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            result = posix_memalign((void**)&send_bufs[idx], ALIGNMENT,
                                    ELEM_COUNT * sizeof(Dtype) * sbuf_size_modifier);
            result = posix_memalign((void**)&recv_bufs[idx], ALIGNMENT,
                                    ELEM_COUNT * sizeof(Dtype) * rbuf_size_modifier *
                                    base_coll::comm->size());
            if (result != 0)
            {
                std::cerr << __FUNCTION__ << " - posix_memalign(values), error: "
                          << strerror(errno)
                          << ", on buffer idx: " <<  idx << std::endl;
            }

            result = posix_memalign((void**)&recv_ibufs[idx], ALIGNMENT,
                                    ELEM_COUNT * sizeof(IType) * rbuf_size_modifier *
                                    base_coll::comm->size());
            result = posix_memalign((void**)&send_ibufs[idx], ALIGNMENT,
                                    ELEM_COUNT * sizeof(IType) * sbuf_size_modifier);
            if (result != 0)
            {
                std::cerr << __FUNCTION__ << " - posix_memalign(indices), error: "
                          << strerror(errno)
                          << ", on buffer idx: " <<  idx << std::endl;
            }
        }

        result = posix_memalign((void**)&single_send_buf, ALIGNMENT,
                                SINGLE_ELEM_COUNT * sizeof(Dtype) * sbuf_size_modifier);
        result = posix_memalign((void**)&single_recv_buf, ALIGNMENT,
                                SINGLE_ELEM_COUNT * sizeof(Dtype) * rbuf_size_modifier * 
                                base_coll::comm->size());

        result = posix_memalign((void**)&single_send_ibuf, ALIGNMENT,
                                SINGLE_ELEM_COUNT * sizeof(IType) * sbuf_size_modifier);
        result = posix_memalign((void**)&single_recv_ibuf, ALIGNMENT,
                                SINGLE_ELEM_COUNT * sizeof(IType) * rbuf_size_modifier *
                                base_coll::comm->size());

        for( size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            memset(send_bufs[idx], 0, ELEM_COUNT * sizeof(Dtype) * sbuf_size_modifier);
            memset(recv_bufs[idx], 0, ELEM_COUNT * sizeof(Dtype) * rbuf_size_modifier * 
                                      base_coll::comm->size());
            memset(recv_ibufs[idx], 0, ELEM_COUNT * sizeof(IType) * base_coll::comm->size());
            memset(send_ibufs[idx], 0, ELEM_COUNT * sizeof(IType));

        }

        memset(single_send_buf, 0, SINGLE_ELEM_COUNT * sizeof(Dtype) * sbuf_size_modifier);
        memset(single_recv_buf, 0, SINGLE_ELEM_COUNT * sizeof(Dtype) * rbuf_size_modifier * 
                                   base_coll::comm->size());
        memset(single_send_ibuf, 0, SINGLE_ELEM_COUNT * sizeof(IType) * sbuf_size_modifier);
        memset(single_recv_ibuf, 0, SINGLE_ELEM_COUNT * sizeof(IType) * rbuf_size_modifier *
                                    base_coll::comm->size());
    }

    ~cpu_sparse_allreduce_coll()
    {
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            free(send_bufs[idx]);
            free(recv_bufs[idx]);
            free(recv_ibufs[idx]);
            free(send_ibufs[idx]);
        }

        free(single_send_buf);
        free(single_recv_buf);
        free(single_send_ibuf);
        free(single_recv_ibuf);
    }

    virtual void prepare(size_t elem_count) override
    {
        this->init_distributor({0, elem_count});
        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sparse_detail::fill_sparse_data(this->get_expected_recv_counts(elem_count),
                                            *this->indices_distributor_impl,
                                            elem_count, send_ibufs[b_idx],
                                            reinterpret_cast<Dtype*>(send_bufs[b_idx]),
                                            reinterpret_cast<Dtype*>(recv_bufs[b_idx]),
                                            recv_icount[b_idx],
                                            recv_vcount[b_idx],
                                            comm->rank());
        }
    }

    virtual void finalize(size_t elem_count) override
    {
        if (!check_values)
            return;

        for (size_t b_idx = 0; b_idx < BUF_COUNT; b_idx++)
        {
            sparse_detail::check_sparse_result(this->get_expected_recv_counts(elem_count),
                                               elem_count,
                                               send_ibufs[b_idx],
                                               static_cast<const Dtype* >(send_bufs[b_idx]),
                                               recv_ibufs[b_idx],
                                               static_cast<const Dtype* >(recv_bufs[b_idx]),
                                               recv_icount[b_idx],
                                               recv_vcount[b_idx],
                                               comm->size(),
                                               comm->rank());

        }
    }

    virtual void start(size_t count, size_t buf_idx,
                       const ccl_coll_attr_t& coll_attr,
                       req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm,
                                      send_ibufs[buf_idx],
                                      count,
                                      reinterpret_cast<const Dtype *>(send_bufs[buf_idx]),
                                      count,
                                      static_cast<IType**>(&recv_ibufs[buf_idx]),
                                      &recv_icount[buf_idx],
                                      reinterpret_cast<Dtype**>(&recv_bufs[buf_idx]),
                                      &recv_vcount[buf_idx],
                                      coll_attr, stream, reqs);
    }

    virtual void start_single(size_t count,
                              const ccl_coll_attr_t& coll_attr,
                              req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm,
                                      single_send_ibuf,
                                      count,
                                      reinterpret_cast<const Dtype *>(single_send_buf),
                                      count,
                                      static_cast<IType**>(&single_recv_ibuf),
                                      &single_recv_icount,
                                      reinterpret_cast<Dtype**>(&single_recv_buf),
                                      &single_recv_vcount,
                                      coll_attr, stream, reqs);
    }
};

#ifdef CCL_ENABLE_SYCL
template<class kernel_value_type, class kernel_index_type>
struct sparse_allreduce_kernel_name_bufs {};
template<class kernel_value_type, class kernel_index_type>
struct sparse_allreduce_kernel_name_single_bufs {};
    
template<class Dtype, class IType,
          template<class> class IndicesDistributorType = sparse_detail::incremental_indices_distributor>
struct sycl_sparse_allreduce_coll :
        base_sparse_allreduce_coll<cl::sycl::buffer<Dtype, 1>, 
                                   cl::sycl::buffer<IType, 1>,
                                   IndicesDistributorType>
{
    using sycl_indices_t = cl::sycl::buffer<IType, 1>;
    using sycl_values_t = cl::sycl::buffer<Dtype, 1>;
    using coll_base = base_sparse_allreduce_coll<sycl_values_t, 
                                                 sycl_indices_t,
                                                 IndicesDistributorType>;
    using coll_strategy = typename coll_base::coll_strategy;

    using coll_base::send_bufs;
    using coll_base::recv_bufs;
    using coll_base::stream;
    using coll_base::single_send_buf;
    using coll_base::single_send_ibuf;
    using coll_base::single_recv_buf;
    using coll_base::single_recv_ibuf;
    using coll_base::single_recv_icount;
    using coll_base::single_recv_vcount;
    using coll_base::comm;

    using coll_base::recv_ibufs;
    using coll_base::send_ibufs;
    using coll_base::recv_icount;
    using coll_base::recv_vcount;

    sycl_sparse_allreduce_coll(const std::string& args, 
                               size_t sbuf_size_modifier = 1,
                               size_t rbuf_size_modifier = 1) : coll_base(args)
    {
        int result = 0;
        for (size_t idx = 0; idx < BUF_COUNT; idx++)
        {
            send_bufs[idx] = new sycl_values_t(ELEM_COUNT * sbuf_size_modifier);
            send_ibufs[idx] = new sycl_indices_t(ELEM_COUNT * sbuf_size_modifier);
            recv_bufs[idx] = new sycl_values_t(ELEM_COUNT * rbuf_size_modifier *
                                               base_coll::comm->size());
            recv_ibufs[idx] = new sycl_indices_t(ELEM_COUNT * rbuf_size_modifier * 
                                                 base_coll::comm->size());

            sycl_queue.submit([&](handler& cgh)
            {
                auto send_buf = (static_cast<sycl_values_t*>(send_bufs[idx]));
                auto send_ibuf = (static_cast<sycl_indices_t*>(send_ibufs[idx]));

                auto recv_buf = (static_cast<sycl_values_t*>(recv_bufs[idx]));
                auto recv_ibuf = (static_cast<sycl_indices_t*>(recv_ibufs[idx]));

                auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
                auto send_ibuf_acc = send_ibuf->template get_access<mode::write>(cgh);
                auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
                auto recv_ibuf_acc = recv_buf->template get_access<mode::write>(cgh);
                cgh.parallel_for<class sparse_allreduce_kernel_name_bufs<Dtype,IType>>
                        (range<1>{ELEM_COUNT*base_coll::comm->size()}, [=](item<1> e_idx)
                {
                    if(e_idx.get_linear_id() < ELEM_COUNT)
                    {
                        send_buf_acc[e_idx] = 0;
                        send_ibuf_acc[e_idx] = 0;
                    }
                    recv_buf_acc[e_idx] = 0;
                    recv_ibuf_acc[e_idx] = 0;
                });
            });
        }

        single_send_buf = new sycl_values_t(SINGLE_ELEM_COUNT * sbuf_size_modifier);
        single_recv_buf = new sycl_values_t(SINGLE_ELEM_COUNT * rbuf_size_modifier *
                                            base_coll::comm->size());

        single_send_ibuf = new sycl_indices_t(SINGLE_ELEM_COUNT * sbuf_size_modifier);
        single_recv_ibuf = new sycl_indices_t(SINGLE_ELEM_COUNT * rbuf_size_modifier *
                                              base_coll::comm->size());
        sycl_queue.submit([&](handler& cgh)
        {
            auto send_buf = (static_cast<sycl_values_t*>(single_send_buf));
            auto send_ibuf = (static_cast<sycl_indices_t*>(single_send_ibuf));

            auto recv_buf = (static_cast<sycl_values_t*>(single_recv_buf));
            auto recv_ibuf = (static_cast<sycl_indices_t*>(single_recv_ibuf));

            auto send_buf_acc = send_buf->template get_access<mode::write>(cgh);
            auto send_ibuf_acc = send_ibuf->template get_access<mode::write>(cgh);

            auto recv_buf_acc = recv_buf->template get_access<mode::write>(cgh);
            auto recv_ibuf_acc = recv_buf->template get_access<mode::write>(cgh);
            cgh.parallel_for<class sparse_allreduce_kernel_name_single_bufs<Dtype, IType>>
                    (range<1>{SINGLE_ELEM_COUNT*base_coll::comm->size()}, [=](item<1> e_idx)
            {
                if(e_idx.get_linear_id() < SINGLE_ELEM_COUNT)
                {
                    send_buf_acc[e_idx] = 0;
                    send_ibuf_acc[e_idx] = 0;
                }
                recv_buf_acc[e_idx] = 0;
                recv_ibuf_acc[e_idx] = 0;
            });
        });
    }

    virtual void prepare(size_t elem_count) override
    {
        //TODO not implemented yet
    }

    virtual void finalize(size_t elem_count) override
    {
        //TODO not implemented yet
    }
    virtual void start(size_t count, size_t buf_idx,
                       const ccl_coll_attr_t& coll_attr,
                       req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm,
                                      *static_cast<const cl::sycl::buffer<IType> *>(send_ibufs[buf_idx]),
                                      count,
                                      *reinterpret_cast<const cl::sycl::buffer<Dtype> *>(send_bufs[buf_idx]),
                                      count,
                                      static_cast<cl::sycl::buffer<IType>**>(&recv_ibufs[buf_idx]),
                                      &recv_icount[buf_idx],
                                      reinterpret_cast<cl::sycl::buffer<Dtype>**>(&recv_bufs[buf_idx]),
                                      &recv_vcount[buf_idx],
                                      coll_attr, stream, reqs);
    }

    virtual void start_single(size_t count,
                              const ccl_coll_attr_t& coll_attr,
                              req_list_t& reqs) override
    {
        coll_strategy::start_internal(*comm,
                                      *static_cast<const cl::sycl::buffer<IType> *>(single_send_ibuf),
                                      count,
                                      *reinterpret_cast<const cl::sycl::buffer<Dtype> *>(single_send_buf),
                                      count,
                                      static_cast<cl::sycl::buffer<IType>**>(&single_recv_ibuf),
                                      &single_recv_icount,
                                      reinterpret_cast<cl::sycl::buffer<Dtype>**>(&single_recv_buf),
                                      &single_recv_vcount,
                                      coll_attr, stream, reqs);
    }
};
#endif

template<class Dtype>
void create_cpu_colls(std::list<std::string>& names, coll_list_t& colls)
{
    using namespace sparse_detail;
    using incremental_index_int_sparse_strategy = 
            sparse_detail::sparse_allreduce_strategy_impl<int, 
                                sparse_detail::incremental_indices_distributor>;
    using incremental_index_bfp16_sparse_strategy = 
            sparse_detail::sparse_allreduce_strategy_impl<ccl::bfp16, 
                                sparse_detail::incremental_indices_distributor>;
                                
    std::stringstream error_messages_stream;
    base_coll::comm = ccl::environment::instance().create_communicator();
    base_coll::stream = ccl::environment::instance().create_stream(ccl::stream_type::cpu, nullptr);
    for (auto names_it = names.begin(); names_it != names.end(); )
    {
        const std::string& name = *names_it;
        if (name == allgatherv_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_allgatherv_coll<Dtype>());
        }
        else if (name == allreduce_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_allreduce_coll<Dtype>());
        }
        else if (name == bcast_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_bcast_coll<Dtype>());
        }
        else if (name == reduce_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_reduce_coll<Dtype>());
        }
        else if (name == alltoall_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_alltoall_coll<Dtype>);
        }
        else if (name == alltoallv_strategy_impl::class_name())
        {
            colls.emplace_back(new cpu_alltoallv_coll<Dtype>);
        }
        else if (name.find(incremental_index_int_sparse_strategy::class_name()) != std::string::npos)
        {
            if (name.find(incremental_index_bfp16_sparse_strategy::class_name()) != std::string::npos)
            {
                if (is_bfp16_enabled() == 0)
                {
                    error_messages_stream << "BFP16 is not supported for current CPU, skipping " << name << ".\n";
                    names_it = names.erase(names_it);
                    continue;
                }
#ifdef CCL_BFP16_COMPILER
                const std::string args = name.substr(name.find(incremental_index_bfp16_sparse_strategy::class_name()) +
                                                     std::strlen(incremental_index_bfp16_sparse_strategy::class_name()));
                colls.emplace_back(new cpu_sparse_allreduce_coll<ccl::bfp16, int64_t,
                                                                 sparse_detail::incremental_indices_distributor>(args,
                                                                                  sizeof(float) / sizeof(ccl::bfp16),
                                                                                  sizeof(float) / sizeof(ccl::bfp16)));
#else
                error_messages_stream << "BFP16 is not supported by current compiler, skipping " << name << ".\n";
                names_it = names.erase(names_it);
                continue;
#endif
            }
            else
            {
                const std::string args = name.substr(name.find(incremental_index_int_sparse_strategy::class_name()) +
                                                     std::strlen(incremental_index_int_sparse_strategy::class_name()));
                colls.emplace_back(new cpu_sparse_allreduce_coll<Dtype, int>(args));
            }
        }
        else
        {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
        ++names_it;
    }

    const std::string& coll_processing_log = error_messages_stream.str();
    if (!coll_processing_log.empty())
    {
        std::cerr << "WARNING:\n" << coll_processing_log << std::endl;
    }
    
    if (colls.empty())
    {
        throw std::logic_error(std::string(__FUNCTION__) + 
                               " - empty colls, reason: " + coll_processing_log);
    }
}

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
void create_sycl_colls(std::list<std::string>& names, coll_list_t& colls)
{

    using incremental_index_int_sparse_strategy = 
            sparse_detail::sparse_allreduce_strategy_impl<int, 
                                sparse_detail::incremental_indices_distributor>;
    using incremental_index_bfp16_sparse_strategy = 
            sparse_detail::sparse_allreduce_strategy_impl<ccl::bfp16, 
                                sparse_detail::incremental_indices_distributor>;
            
    std::stringstream error_messages_stream;
    base_coll::comm = ccl::environment::instance().create_communicator();
    base_coll::stream = ccl::environment::instance().create_stream(ccl::stream_type::sycl, &sycl_queue);
    for (auto names_it = names.begin(); names_it != names.end(); )
    {
        const std::string& name = *names_it;
        
        if (name == allgatherv_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_allgatherv_coll<Dtype>());
        }
        else if (name == allreduce_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_allreduce_coll<Dtype>());
        }
        else if (name == alltoall_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_alltoall_coll<Dtype>());
        }
        else if (name == alltoallv_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_alltoallv_coll<Dtype>());
        }
        else if (name == bcast_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_bcast_coll<Dtype>());
        }
        else if (name == reduce_strategy_impl::class_name())
        {
            colls.emplace_back(new sycl_reduce_coll<Dtype>());
        }
        else if (name.find(incremental_index_int_sparse_strategy::class_name()) != std::string::npos)
        {
            // TODO case is not supported yet
            if (true)
            {
                error_messages_stream << "SYCL coll: skipping " << name << ", because it is not supported yet.\n";
                names_it = names.erase(names_it);
                continue;
            }
            
            const std::string args = name.substr(name.find(incremental_index_int_sparse_strategy::class_name()) +
                                                 std::strlen(incremental_index_int_sparse_strategy::class_name()));
            colls.emplace_back(new sycl_sparse_allreduce_coll<Dtype, int>(args));
        }
        else if (name.find(incremental_index_bfp16_sparse_strategy::class_name()) != std::string::npos)
        {
            // TODO case is not supported yet
            if (true)
            {
                error_messages_stream << "SYCL coll: skipping " << name << ", because it is not supported yet.\n";
                names_it = names.erase(names_it);
                continue;
            }
            
            if (is_bfp16_enabled() == 0)
            {
                error_messages_stream << "SYCL BFP16 is not supported for current CPU, skipping " << name << ".\n";
                names_it = names.erase(names_it);
                continue;
            }
#ifdef CCL_BFP16_COMPILER
            const std::string args = name.substr(name.find(incremental_index_bfp16_sparse_strategy::class_name()) +
                                                 std::strlen(incremental_index_bfp16_sparse_strategy::class_name()));
            colls.emplace_back(new sycl_sparse_allreduce_coll<ccl::bfp16, int64_t,
                                                              sparse_detail::incremental_indices_distributor>(args,
                                                                               sizeof(float) / sizeof(ccl::bfp16),
                                                                               sizeof(float) / sizeof(ccl::bfp16)));
#else
            error_messages_stream << "SYCL BFP16 is not supported by current compiler, skipping " << name << ".\n";
            names_it = names.erase(names_it);
            continue;
#endif
        }
        else
        {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
        
        ++names_it;
    }
    
    const std::string& coll_processing_log = error_messages_stream.str();
    if (!coll_processing_log.empty())
    {
        std::cerr << "WARNING: " << coll_processing_log << std::endl;
    }
    
    if (colls.empty())
    {
        throw std::logic_error(std::string(__FUNCTION__) + 
                               " - empty colls, reason: " + coll_processing_log);
    }
}
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
void create_colls(std::list<std::string>& names, ccl::stream_type backend_type, coll_list_t& colls)
{
    switch (backend_type)
    {
        case ccl::stream_type::cpu:
            create_cpu_colls<Dtype>(names, colls);
            break;
        case ccl::stream_type::sycl:
#ifdef CCL_ENABLE_SYCL
            create_sycl_colls<Dtype>(names, colls);
#else
            ASSERT(0, "SYCL backend is requested but CCL_ENABLE_SYCL is not defined");
#endif
            break;
        default:
            ASSERT(0, "unknown backend %d", (int)backend_type);
            break;
    }

}

void do_regular(ccl::communicator* comm,
                ccl::coll_attr& coll_attr,
                coll_list_t& colls,
                req_list_t& reqs)
{
    char* match_id = (char*)coll_attr.match_id;

    reqs.reserve(colls.size() * BUF_COUNT);

    /* warm up */
    PRINT_BY_ROOT("do warm up");
    coll_attr.to_cache = 0;
    for (size_t count = 1; count < ELEM_COUNT; count *= 2)
    {
        for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
        {
            auto& coll = colls[coll_idx];
            for (size_t buf_idx = 0; buf_idx < BUF_COUNT; buf_idx++)
            {
                // snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                //          coll->name(), coll_idx, count, buf_idx);
                // PRINT_BY_ROOT("start_coll: %s, count %zu, buf_idx %zu", coll->name(), count, buf_idx);
                coll->start(count, buf_idx, coll_attr, reqs);
            }
        }
        for (auto &req : reqs)
        {
            req->wait();
        }
        reqs.clear();
    }

    /* benchmark with multiple equal sized buffer per collective */
    PRINT_BY_ROOT("do multi-buffers benchmark");
    coll_attr.to_cache = 1;
    for (size_t count = 1; count <= ELEM_COUNT; count *= 2)
    {
        try
        {
            double t = 0;
            for (size_t iter_idx = 0; iter_idx < ITERS; iter_idx++)
            {
                for (auto& coll : colls)
                {
                    coll->prepare(count);
                }

                double t1 = when();
                for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                {
                    auto& coll = colls[coll_idx];
                    for (size_t buf_idx = 0; buf_idx < BUF_COUNT; buf_idx++)
                    {
                        snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                                 coll->name(), coll_idx, count, buf_idx);
                        coll->start(count, buf_idx, coll_attr, reqs);
                    }
                }
                for (auto &req : reqs)
                {
                    req->wait();
                }
                double t2 = when();
                t += (t2 - t1);
            }

            reqs.clear();

            for (auto& coll : colls)
            {
                coll->finalize(count);
            }
            print_timings(*comm, &t, count,
                          sizeof(DTYPE), BUF_COUNT,
                          comm->rank(), comm->size());
        }
        catch(const std::exception& ex)
        {
            ASSERT(0, "error on count %zu, reason: %s", count, ex.what());
        }
    }

    comm->barrier();

    /* benchmark with single buffer per collective */
    PRINT_BY_ROOT("do single-buffer benchmark");
    coll_attr.to_cache = 1;
    for (size_t count = BUF_COUNT; count <= SINGLE_ELEM_COUNT; count *= 2)
    {
        try
        {
            double t = 0;
            for (size_t iter_idx = 0; iter_idx < ITERS; iter_idx++)
            {
                double t1 = when();
                for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                {
                    auto& coll = colls[coll_idx];
                    snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_single_count_%zu",
                             coll->name(), coll_idx, count);
                    coll->start_single(count, coll_attr, reqs);
                }
                for (auto &req : reqs)
                {
                    req->wait();
                }
                double t2 = when();
                t += (t2 - t1);

                reqs.clear();
            }
            print_timings(*comm, &t, count,
                          sizeof(DTYPE), 1,
                          comm->rank(), comm->size());
        } catch (...)
        {
            ASSERT(0, "error on count %zu", count);
        }
    }

    PRINT_BY_ROOT("PASSED\n");
}

void do_unordered(ccl::communicator* comm,
                  ccl::coll_attr& coll_attr,
                  coll_list_t& colls,
                  req_list_t& reqs)
{
    std::set<std::string> match_ids;
    char* match_id = (char*)coll_attr.match_id;
    size_t rank = comm->rank();

    reqs.reserve(colls.size() * BUF_COUNT * (log2(ELEM_COUNT) + 1));

    PRINT_BY_ROOT("do unordered test");
    coll_attr.to_cache = 1;

    for (size_t count = 1; count <= ELEM_COUNT; count *= 2)
    {
        try
        {
            if (rank % 2)
            {
                for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                {
                    auto& coll = colls[coll_idx];
                    for (size_t buf_idx = 0; buf_idx < BUF_COUNT; buf_idx++)
                    {
                        snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                                 coll->name(), coll_idx, count, buf_idx);
                        coll->start(count, buf_idx, coll_attr, reqs);
                        match_ids.emplace(match_id);
                    }
                }
            }
            else
            {
                for (size_t coll_idx = 0; coll_idx < colls.size(); coll_idx++)
                {
                    size_t real_coll_idx = colls.size() - coll_idx - 1;
                    auto& coll = colls[real_coll_idx];
                    for (size_t buf_idx = 0; buf_idx < BUF_COUNT; buf_idx++)
                    {
                        size_t real_buf_idx = BUF_COUNT - buf_idx - 1;
                        snprintf(match_id, MATCH_ID_SIZE, "coll_%s_%zu_count_%zu_buf_%zu",
                                 coll->name(), real_coll_idx, count, real_buf_idx);
                        coll->start(count, real_buf_idx, coll_attr, reqs);
                        match_ids.insert(std::string(match_id));
                    }
                }
            }
        }
        catch (...)
        {
            ASSERT(0, "error on count %zu", count);
        }
    }

    ASSERT(match_ids.size() == reqs.size(),
           "unexpected match_ids.size %zu, expected %zu",
           match_ids.size(), reqs.size());

    try
    {
        for (auto &req : reqs)
        {
            req->wait();
        }
    }
    catch (...)
    {
        ASSERT(0, "error on coll completion");
    }


    PRINT_BY_ROOT("PASSED\n");
}


int main(int argc, char *argv[])
{
    if (argc > 4)
    {
        PRINT("%s", help_message);
        return -1;
    }

    std::string backend_str = (argc > 1) ? std::string(argv[1]) : DEFAULT_BACKEND;
    std::set<std::string> suppored_backends { "cpu" };
#ifdef CCL_ENABLE_SYCL
    suppored_backends.insert("sycl");
#endif

    std::stringstream sstream;
    if (suppored_backends.find(backend_str) == suppored_backends.end())
    {
        PRINT("unsupported backend: %s", backend_str.c_str());

        std::copy(suppored_backends.begin(), suppored_backends.end(),
                  std::ostream_iterator<std::string>(sstream, " "));
        PRINT("supported backends: %s", sstream.str().c_str());
        PRINT("%s", help_message);
        return -1;
    }

    ccl::stream_type backend_type = ccl::stream_type::cpu;
    if (backend_str == "sycl")
        backend_type = ccl::stream_type::sycl;

    std::string loop_str = (argc > 2) ? std::string(argv[2]) : DEFAULT_LOOP;
    std::set<std::string> suppored_loops { "regular", "unordered" };
    if (suppored_loops.find(loop_str) == suppored_loops.end())
    {
        PRINT("unsupported loop: %s", loop_str.c_str());

        std::copy(suppored_loops.begin(), suppored_loops.end(),
                  std::ostream_iterator<std::string>(sstream, " "));
        PRINT("supported loops: %s", sstream.str().c_str());
        PRINT("%s", help_message);
        return -1;
    }

    loop_type_t loop = LOOP_REGULAR;
    if (loop_str == "unordered")
    {
        loop = LOOP_UNORDERED;
        setenv("CCL_UNORDERED_COLL", "1", 1);
    }

    ccl::coll_attr coll_attr{};

    std::list<std::string> coll_names;
    coll_list_t colls;
    req_list_t reqs;

    char match_id[MATCH_ID_SIZE] {'\0'};
    coll_attr.match_id = match_id;


    try
    {
        coll_names = tokenize((argc == 4) ? argv[3] : DEFAULT_COLL_LIST, ',');
        create_colls<DTYPE>(coll_names, backend_type, colls);
    }
    catch (const std::runtime_error& e)
    {
        ASSERT(0, "cannot create coll objects: %s\n%s", e.what(), help_message);
    }
    catch (const std::logic_error& e)
    {
        std::cerr << "Cannot launch benchmark: " << e.what() << std::endl;
        return -1;
    }

    ccl::communicator* comm = base_coll::comm.get();
    if (colls.empty())
    {
        PRINT_BY_ROOT("%s", help_message);
        ASSERT(0, "unexpected coll list");
    }

    int check_values = 1;

    comm->barrier();

    std::copy(coll_names.begin(), coll_names.end(),
              std::ostream_iterator<std::string>(sstream, " "));

    PRINT_BY_ROOT("start colls: %s, iters: %d, buf_count: %d, ranks %zu, check_values %d, backend %s, loop %s",
                  sstream.str().c_str(), ITERS, BUF_COUNT, comm->size(), check_values,
                  backend_str.c_str(), loop_str.c_str());

    for (auto& coll : colls)
    {
        coll->check_values = check_values;
    }

    switch (loop)
    {
        case LOOP_REGULAR:
            do_regular(comm, coll_attr, colls, reqs);
            break;
        case LOOP_UNORDERED:
            do_unordered(comm, coll_attr, colls, reqs);
            break;
        default:
            ASSERT(0, "unknown loop %d", loop);
            break;
    }

    base_coll::comm.reset();
    base_coll::stream.reset();
    return 0;
}
