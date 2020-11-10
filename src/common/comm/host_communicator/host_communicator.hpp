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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"

#include "oneapi/ccl/event.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "oneapi/ccl/coll_attr.hpp"

#include "common/comm/communicator_traits.hpp"
#include "common/comm/comm_interface.hpp"
#include "types_generator_defines.hpp"
#include "atl/atl_wrapper.h"

namespace ccl {

namespace v1 {
class kvs_interface;
}

class host_communicator : public ccl::communicator_interface {
public:
    using traits = ccl::host_communicator_traits;

    int rank() const override;
    int size() const override;

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
    ccl::communicator_interface::device_t get_device() const override;
    ccl::communicator_interface::context_t get_context() const override;

    const ccl::comm_split_attr& get_comm_split_attr() const override {
        return comm_attr;
    }

    ccl::group_split_type get_topology_type() const override {
        throw ccl::exception(std::string(__FUNCTION__) + " is not applicable for " +
                             traits::name());
        return ccl::group_split_type::undetermined;
    }

    ccl::device_topology_type get_topology_class() const override {
        throw ccl::exception(std::string(__FUNCTION__) + " is not applicable for " +
                             traits::name());
        return ccl::device_topology_type::undetermined;
    }

    ccl::communicator_interface_ptr split(const comm_split_attr& attr) override;

    // collectives operation declarations
    ccl::event barrier(const stream::impl_value_t& op_stream,
                       const barrier_attr& attr,
                       const vector_class<event>& deps = {}) override;
    ccl::event barrier_impl(const stream::impl_value_t& op_stream,
                            const barrier_attr& attr,
                            const vector_class<event>& deps = {});

    COMM_INTERFACE_COLL_METHODS(DEFINITION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DEFINITION);
#endif /* CCL_ENABLE_SYCL */

    COMM_IMPL_DECLARATION;
    COMM_IMPL_CLASS_DECLARATION
    COMM_IMPL_SPARSE_DECLARATION;
    COMM_IMPL_SPARSE_CLASS_DECLARATION

    host_communicator();
    host_communicator(int size, shared_ptr_class<kvs_interface> kvs);
    host_communicator(int size, int rank, shared_ptr_class<kvs_interface> kvs);
    host_communicator(std::shared_ptr<atl_wrapper> atl);
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
    int comm_rank;
    int comm_size;
    ccl::group_unique_key owner_id;
    // ccl::unified_device_type device;
    // ccl::unified_context_type context;

    host_communicator* get_impl() {
        return this;
    }

    void exchange_colors(std::vector<int>& colors);
    ccl_comm* create_with_color(int color,
                                ccl_comm_id_storage* comm_ids,
                                const ccl_comm* parent_comm);
}; // class host_communicator

} // namespace ccl
