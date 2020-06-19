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
#include "common/comm/compiler_comm_interface_dispatcher.hpp"
#include "types_generator_defines.hpp"

namespace native
{
    struct ccl_device;
}

namespace ccl
{
struct gpu_comm_attr;
struct communicator_interface : public communicator_interface_dispatcher
{
    virtual ~communicator_interface() = default;

    virtual size_t rank() const = 0;
    virtual size_t size() const = 0;

    virtual bool is_host() const noexcept = 0;
    virtual bool is_cpu() const noexcept = 0;
    virtual bool is_gpu() const noexcept = 0;
    virtual bool is_accelerator() const noexcept = 0;

    virtual comm_attr_t get_host_attr() const = 0;

    virtual bool is_ready() const = 0;

    // collectives operation declarations
    virtual void barrier(ccl::stream::impl_t& stream) = 0;

    COMM_INTERFACE_COLL_DECLARATION__VOID;
    COMM_INTERFACE_COLL_DECLARATION(char);
    COMM_INTERFACE_COLL_DECLARATION(int);
    COMM_INTERFACE_COLL_DECLARATION(int64_t);
    COMM_INTERFACE_COLL_DECLARATION(uint64_t);
    COMM_INTERFACE_COLL_DECLARATION(float);
    COMM_INTERFACE_COLL_DECLARATION(double);

#ifdef CCL_ENABLE_SYCL
    COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<char COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<int COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<int64_t COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<uint64_t COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

    COMM_INTERFACE_SPARSE_DECLARATION__VOID
    COMM_INTERFACE_SPARSE_DECLARATION(char, char);
    COMM_INTERFACE_SPARSE_DECLARATION(char, int);
    COMM_INTERFACE_SPARSE_DECLARATION(char, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DECLARATION(char, float);
    COMM_INTERFACE_SPARSE_DECLARATION(char, double);
    COMM_INTERFACE_SPARSE_DECLARATION(char, int64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(char, uint64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(int, char);
    COMM_INTERFACE_SPARSE_DECLARATION(int, int);
    COMM_INTERFACE_SPARSE_DECLARATION(int, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DECLARATION(int, float);
    COMM_INTERFACE_SPARSE_DECLARATION(int, double);
    COMM_INTERFACE_SPARSE_DECLARATION(int, int64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(int, uint64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, char);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, int);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, float);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, double);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, int64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(int64_t, uint64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, char);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, int);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, float);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, double);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, int64_t);
    COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, uint64_t);

#ifdef CCL_ENABLE_SYCL
    COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int COMMA 1>,
                                            cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int COMMA 1>,
                                            cl::sycl::buffer<ccl::bfp16 COMMA 1>);

    COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int64_t COMMA 1>,
                                            cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int64_t COMMA 1>,
                                            cl::sycl::buffer<ccl::bfp16 COMMA 1>);
#endif //CCL_ENABLE_SYCL
};
}
