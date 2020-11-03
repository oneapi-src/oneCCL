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

#include "base.hpp"
#include "config.hpp"

#ifdef CCL_ENABLE_SYCL
#include "sycl_base.hpp"
template <typename Dtype>
using sycl_buffer_t = cl::sycl::buffer<Dtype, 1>;
#endif

struct base_coll;

using coll_list_t = std::vector<std::shared_ptr<base_coll>>;
using req_list_t = std::vector<ccl::event>;

typedef struct bench_exec_attr {

    bench_exec_attr() = default;
    template <ccl::operation_attr_id attrId, class value_t>
    struct setter {
        setter(value_t v) : val(v) {}
        template <class attr_t>
        void operator()(ccl::shared_ptr_class<attr_t>& attr) {
            attr->template set<attrId, value_t>(val);
        }
        value_t val;
    };
    struct factory {
        template <class attr_t>
        void operator()(ccl::shared_ptr_class<attr_t>& attr) {
            attr = std::make_shared<attr_t>(
                ccl::create_operation_attr<attr_t>());
        }
    };

    using supported_op_attr_t = std::tuple<ccl::shared_ptr_class<ccl::allgatherv_attr>,
                                           ccl::shared_ptr_class<ccl::allreduce_attr>,
                                           ccl::shared_ptr_class<ccl::alltoall_attr>,
                                           ccl::shared_ptr_class<ccl::alltoallv_attr>,
                                           ccl::shared_ptr_class<ccl::reduce_attr>,
                                           ccl::shared_ptr_class<ccl::broadcast_attr>,
                                           ccl::shared_ptr_class<ccl::reduce_scatter_attr>,
                                           ccl::shared_ptr_class<ccl::sparse_allreduce_attr>>;

    template <class attr_t>
    attr_t& get_attr() {
        return *(ccl_tuple_get<ccl::shared_ptr_class<attr_t>>(coll_attrs).get());
    }

    template <class attr_t>
    const attr_t& get_attr() const {
        return *(ccl_tuple_get<ccl::shared_ptr_class<attr_t>>(coll_attrs).get());
    }

    template <ccl::operation_attr_id attrId, class Value>
    typename ccl::details::ccl_api_type_attr_traits<ccl::operation_attr_id, attrId>::return_type
    set(const Value& v) {
        ccl_tuple_for_each(coll_attrs, setter<attrId, Value>(v));
        return v;
    }

    void init_all() {
        ccl_tuple_for_each(coll_attrs, factory{});
    }

    ccl::reduction reduction;

private:
    supported_op_attr_t coll_attrs;
} bench_exec_attr;

typedef struct bench_init_attr {
    size_t buf_count;
    size_t max_elem_count;
    size_t v2i_ratio;
} bench_init_attr;

/* base polymorph collective wrapper class */
struct base_coll {
    base_coll(bench_init_attr init_attr) : init_attr(init_attr) {
        send_bufs.resize(init_attr.buf_count);
        recv_bufs.resize(init_attr.buf_count);
    }

    base_coll() = delete;
    virtual ~base_coll() = default;

    virtual const char* name() const noexcept {
        return nullptr;
    };

    virtual void prepare(size_t elem_count){};
    virtual void finalize(size_t elem_count){};

    virtual ccl::datatype get_dtype() const = 0;

    virtual void start(size_t count,
                       size_t buf_idx,
                       const bench_exec_attr& attr,
                       req_list_t& reqs) = 0;

    virtual void start_single(size_t count, const bench_exec_attr& attr, req_list_t& reqs) = 0;

    /* to get buf_count from initialized private member */
    size_t get_buf_count() const noexcept {
        return init_attr.buf_count;
    }
    size_t get_max_elem_count() const noexcept {
        return init_attr.max_elem_count;
    }
    size_t get_single_buf_max_elem_count() const noexcept {
        return init_attr.buf_count * init_attr.max_elem_count;
    }

    std::vector<void*> send_bufs;
    std::vector<void*> recv_bufs;

    void* single_send_buf = nullptr;
    void* single_recv_buf = nullptr;

private:
    bench_init_attr init_attr;
};

struct host_data {
    static ccl::shared_ptr_class<ccl::communicator> comm_ptr;
    static void init(size_t size, size_t rank, ccl::shared_ptr_class<ccl::kvs_interface> kvs) {

        if (comm_ptr) {
            throw ccl::exception(std::string(__FUNCTION__) + " - reinit is not allowed");
        }

        comm_ptr = std::make_shared<ccl::communicator>(
            ccl::create_communicator(size, rank, kvs));
    }

    static void deinit() {
        comm_ptr.reset();
    }
};

ccl::shared_ptr_class<ccl::communicator> host_data::comm_ptr{};

#ifdef CCL_ENABLE_SYCL
struct device_data {

    static ccl::shared_ptr_class<ccl::communicator> comm_ptr;
    static ccl::shared_ptr_class<ccl::stream> stream_ptr;
    static cl::sycl::queue sycl_queue;

    static void init(size_t size,
                     size_t rank,
                     cl::sycl::device& device,
                     cl::sycl::context& ctx,
                     ccl::shared_ptr_class<ccl::kvs_interface> kvs) {

        if (stream_ptr or comm_ptr) {
            throw ccl::exception(std::string(__FUNCTION__) + " - reinit is not allowed");
        }

        auto ccl_dev = ccl::create_device(device);
        auto ccl_ctx = ccl::create_context(ctx);

        comm_ptr = std::make_shared<ccl::communicator>(
            ccl::create_communicator(
                size, rank,
                ccl_dev,
                ccl_ctx,
                kvs));

        sycl_queue = cl::sycl::queue(device);

        stream_ptr =
            std::make_shared<ccl::stream>(ccl::create_stream(sycl_queue));
    }

    static void deinit() {
        comm_ptr.reset();
        stream_ptr.reset();
    }
};

ccl::shared_ptr_class<ccl::communicator> device_data::comm_ptr{};
ccl::shared_ptr_class<ccl::stream> device_data::stream_ptr{};
cl::sycl::queue device_data::sycl_queue{};

#endif /* CCL_ENABLE_SYCL */
