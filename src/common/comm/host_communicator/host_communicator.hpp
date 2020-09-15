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
#include "common/comm/comm_interface.hpp"

class host_communicator : public ccl::communicator_interface {
public:
    friend class ccl::environment;
    using base_t = ccl::communicator_interface;
    using traits = ccl::host_communicator_traits;

    host_communicator(const ccl::comm_attr_t& attr);

    bool is_ready() const override;

    // traits
    bool is_host() const noexcept override {
        return traits::is_host();
    }

    bool is_cpu() const noexcept override {
        return traits::is_cpu();
    }

    bool is_gpu() const noexcept override {
        return traits::is_gpu();
    }

    bool is_accelerator() const noexcept override {
        return traits::is_accelerator();
    }

    // communicator interfaces implementation
    size_t rank() const override;
    size_t size() const override;

    ccl::comm_attr_t get_host_attr() const override;

#ifdef MULTI_GPU_SUPPORT
    void visit(ccl::gpu_comm_attr& comm_attr) override;
    ccl::device_group_split_type get_topology_type() const override;
    ccl::device_topology_type get_topology_class() const override;
    ccl::device_index_type get_device_path() const override;
    ccl::communicator_interface::native_device_type_ref get_device() override;
    ccl::device_comm_attr_t get_device_attr() const override;
#endif

    // collectives algo implementation
    void barrier(ccl::stream::impl_t& stream) override;
    COMM_INTERFACE_COLL_DEFINITION__VOID;
    COMM_INTERFACE_COLL_DEFINITION(char);
    COMM_INTERFACE_COLL_DEFINITION(int);
    COMM_INTERFACE_COLL_DEFINITION(int64_t);
    COMM_INTERFACE_COLL_DEFINITION(uint64_t);
    COMM_INTERFACE_COLL_DEFINITION(float);
    COMM_INTERFACE_COLL_DEFINITION(double);

#ifdef CCL_ENABLE_SYCL
    COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<char COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<int COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<int64_t COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<uint64_t COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

    COMM_INTERFACE_SPARSE_DEFINITION__VOID;
    COMM_INTERFACE_SPARSE_DEFINITION(char, char);
    COMM_INTERFACE_SPARSE_DEFINITION(char, int);
    COMM_INTERFACE_SPARSE_DEFINITION(char, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DEFINITION(char, float);
    COMM_INTERFACE_SPARSE_DEFINITION(char, double);
    COMM_INTERFACE_SPARSE_DEFINITION(char, int64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(char, uint64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(int, char);
    COMM_INTERFACE_SPARSE_DEFINITION(int, int);
    COMM_INTERFACE_SPARSE_DEFINITION(int, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DEFINITION(int, float);
    COMM_INTERFACE_SPARSE_DEFINITION(int, double);
    COMM_INTERFACE_SPARSE_DEFINITION(int, int64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(int, uint64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, char);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, int);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, float);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, double);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, int64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(int64_t, uint64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, char);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, int);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, ccl::bfp16);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, float);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, double);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, int64_t);
    COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, uint64_t);

#ifdef CCL_ENABLE_SYCL
    COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int COMMA 1>,
                                           cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int COMMA 1>,
                                           cl::sycl::buffer<ccl::bfp16 COMMA 1>);

    COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int64_t COMMA 1>,
                                           cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int64_t COMMA 1>,
                                           cl::sycl::buffer<ccl::bfp16 COMMA 1>);
#endif //CCL_ENABLE_SYCL
private:
    host_communicator* get_impl() {
        return this;
    }
    COMM_IMPL_DECLARATION
    COMM_IMPL_CLASS_DECLARATION
    COMM_IMPL_SPARSE_DECLARATION
    COMM_IMPL_SPARSE_CLASS_DECLARATION

    ccl::comm_attr_t comm_attr;
    std::shared_ptr<ccl_comm> comm_impl;
    size_t comm_rank;
    size_t comm_size;
};
