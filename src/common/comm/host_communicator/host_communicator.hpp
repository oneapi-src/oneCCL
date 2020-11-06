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

#include "common/comm/comm.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_event.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

#include "common/comm/communicator_traits.hpp"
#include "common/comm/comm_interface.hpp"
#include "types_generator_defines.hpp"

namespace ccl {

class kvs_interface;

class host_communicator : public ccl::communicator_interface {
public:
    using coll_request_t = ccl::event;
    using traits = ccl::host_communicator_traits;

    size_t rank() const override;
    size_t size() const override;

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

    bool is_ready() const override {
        return true;
    }

    const ccl::group_unique_key& get_comm_group_id() const override {
        return owner_id;
    }

    void set_comm_group_id(ccl::group_unique_key id) {
        owner_id = id;
    }

#ifdef MULTI_GPU_SUPPORT
    void visit(ccl::gpu_comm_attr& comm_attr) override;
#endif

    ccl::device_index_type get_device_path() const override;
    ccl::communicator_interface::device_t get_device() override;
    ccl::communicator_interface::context_t get_context() override;

    const ccl::comm_split_attr& get_comm_split_attr() const override {
        return comm_attr;
    }

    ccl::group_split_type get_topology_type() const override {
        throw ccl::exception(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
        return ccl::group_split_type::undetermined;
    }

    ccl::device_topology_type get_topology_class() const override {
        throw ccl::exception(std::string(__FUNCTION__) + " is not applicable for " + traits::name());
        return ccl::device_topology_type::undetermined;
    }

    ccl::communicator_interface_ptr split(const comm_split_attr& attr) override;

    // collectives operation declarations
    coll_request_t barrier(const stream::impl_value_t& op_stream,
                                const barrier_attr& attr,
                                const vector_class<event>& deps = {}) override;
    coll_request_t barrier_impl(const stream::impl_value_t& op_stream,
                                const barrier_attr& attr,
                                const vector_class<event>& deps = {});

    // communicator interfaces implementation
    DEVICE_COMM_INTERFACE_COLL_DEFINITION__VOID;
    DEVICE_COMM_INTERFACE_COLL_DEFINITION(char);
    DEVICE_COMM_INTERFACE_COLL_DEFINITION(int);
    DEVICE_COMM_INTERFACE_COLL_DEFINITION(int64_t);
    DEVICE_COMM_INTERFACE_COLL_DEFINITION(uint64_t);
    DEVICE_COMM_INTERFACE_COLL_DEFINITION(float);
    DEVICE_COMM_INTERFACE_COLL_DEFINITION(double);

#ifdef CCL_ENABLE_SYCL
    DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<char COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<int COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<int64_t COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<uint64_t COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<float COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DEFINITION(cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION__VOID;
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, char);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, int);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, float);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, double);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(char, uint64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, char);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, int);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, float);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, double);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int, uint64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, char);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, int);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, float);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, double);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(int64_t, uint64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, char);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, int);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, float);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, double);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DEFINITION(uint64_t, uint64_t);

#ifdef CCL_ENABLE_SYCL
    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int COMMA 1>,
                                                  cl::sycl::buffer<float COMMA 1>);
    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int COMMA 1>,
                                                  cl::sycl::buffer<ccl::bf16 COMMA 1>);

    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int64_t COMMA 1>,
                                                  cl::sycl::buffer<float COMMA 1>);
    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DEFINITION(cl::sycl::buffer<int64_t COMMA 1>,
                                                  cl::sycl::buffer<ccl::bf16 COMMA 1>);
#endif //CCL_ENABLE_SYCL



    DEVICE_COMM_IMPL_DECLARATION;
    DEVICE_COMM_IMPL_CLASS_DECLARATION
    DEVICE_COMM_IMPL_SPARSE_DECLARATION;
    DEVICE_COMM_IMPL_SPARSE_CLASS_DECLARATION

    host_communicator();
    host_communicator(size_t size, shared_ptr_class<kvs_interface> kvs);
    host_communicator(size_t size, size_t rank, shared_ptr_class<kvs_interface> kvs);
    host_communicator(std::shared_ptr<ccl_comm> impl);
    host_communicator(host_communicator& src) = delete;
    host_communicator(host_communicator&& src) = default;
    host_communicator& operator=(host_communicator& src) = delete;
    host_communicator& operator=(host_communicator&& src) = default;
    ~host_communicator() = default;
    std::shared_ptr<atl_wrapper> get_atl();

    // troubleshooting
    std::string to_string() const;

private:
    friend struct group_context;
    std::shared_ptr<ccl_comm> comm_impl;
    ccl::comm_split_attr comm_attr;
    size_t comm_rank;
    size_t comm_size;
    ccl::group_unique_key owner_id;
    // ccl::unified_device_type device;
    // ccl::unified_device_context_type context;

    host_communicator* get_impl() {
        return this;
    }

    void exchange_colors(std::vector<int>& colors);
    ccl_comm* create_with_color(int color,
                                ccl_comm_id_storage* comm_ids,
                                const ccl_comm* parent_comm);
}; // class host_communicator

} // namespace ccl
