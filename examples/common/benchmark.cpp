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
#include <vector>

#include "base.hpp"

#define BUF_COUNT         (16)
#define ELEM_COUNT        (128)
#define SINGLE_ELEM_COUNT (BUF_COUNT * ELEM_COUNT)
#define ALIGNMENT         (2 * 1024 * 1024)
#define DTYPE             float
#define CCL_DTYPE         ccl::data_type::dt_float
#define MATCH_ID_SIZE     (256)

/* different collectives with duplications */
#define DEFAULT_COLL_LIST "allgatherv,allreduce,alltoall,alltoallv,bcast,reduce,allgatherv,allreduce,alltoall,alltoallv,bcast,reduce"

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
                                     "example:\n\tcpu regular allgatherv,allreduce\n"
                                     "example:\n\tsycl unordered bcast,reduce\n";

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
struct cpu_base_coll : virtual base_coll, private strategy
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

template<class Dtype>
void create_cpu_colls(const std::list<std::string>& names, coll_list_t& colls)
{
    base_coll::comm = ccl::environment::instance().create_communicator();
    base_coll::stream = ccl::environment::instance().create_stream(ccl::stream_type::cpu, nullptr);
    for (const auto& name : names)
    {
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
        else
        {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
    }
}

#ifdef CCL_ENABLE_SYCL
template<class Dtype>
void create_sycl_colls(const std::list<std::string>& names, coll_list_t& colls)
{
    base_coll::comm = ccl::environment::instance().create_communicator();
    base_coll::stream = ccl::environment::instance().create_stream(ccl::stream_type::sycl, &sycl_queue);
    for (const auto& name : names)
    {
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
        else
        {
            ASSERT(0, "create_colls error, unknown coll name: %s", name.c_str());
        }
    }
}
#endif /* CCL_ENABLE_SYCL */

template<class Dtype>
void create_colls(const std::list<std::string>& names, ccl::stream_type backend_type, coll_list_t& colls)
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
            ASSERT(0, "sycl backend is requested but CCL_ENABLE_SYCL is not defined");
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
        catch (...)
        {
            ASSERT(0, "error on count %zu", count);
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
