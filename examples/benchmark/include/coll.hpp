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
#ifndef COLL_HPP
#define COLL_HPP

#include "base.hpp"
#include "config.hpp"

struct base_coll;

using coll_list_t = std::vector<std::shared_ptr<base_coll>>;
using req_list_t = std::vector<std::unique_ptr<ccl::request>>;

/* base polymorph collective wrapper class */
struct base_coll
{
    virtual ~base_coll() = default;

    virtual const char* name() const noexcept { return nullptr; };

    virtual void prepare(size_t elem_count) {};
    virtual void finalize(size_t elem_count) {};

    virtual ccl::datatype get_dtype() const = 0;

    virtual void start(size_t count, size_t buf_idx,
                       const ccl::coll_attr& attr,
                       req_list_t& reqs) = 0;

    virtual void start_single(size_t count,
                              const ccl::coll_attr& attr,
                              req_list_t& reqs) = 0;

    void* send_bufs[BUF_COUNT] = { nullptr };
    void* recv_bufs[BUF_COUNT] = { nullptr };
    void* single_send_buf = nullptr;
    void* single_recv_buf = nullptr;

    /* global communicator & stream for all collectives */
    static ccl::communicator_t comm;
    static ccl::stream_t stream;
};

ccl::communicator_t base_coll::comm;
ccl::stream_t base_coll::stream;

#endif /* COLL_HPP */
