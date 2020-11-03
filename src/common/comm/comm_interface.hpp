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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "common/event/event_internal/event_internal_attr_ids.hpp"
#include "common/event/event_internal/event_internal_attr_ids_traits.hpp"
#include "common/event/event_internal/event_internal.hpp"

#include "common/comm/compiler_comm_interface_dispatcher.hpp"
#include "common/comm/l0/comm_context_id.hpp"

namespace native {
struct ccl_device;
}

namespace ccl {
struct gpu_comm_attr;
class allgatherv_attr;
class allreduce_attr;
class alltoall_attr;
class alltoallv_attr;
class barrier_attr;
class broadcast_attr;
class reduce_attr;
class reduce_scatter_attr;
class sparse_allreduce_attr;
} // namespace ccl

#include "types_generator_defines.hpp"

namespace ccl {
struct communicator_interface : public communicator_interface_dispatcher {
    virtual ~communicator_interface() = default;

    virtual size_t rank() const = 0;
    virtual size_t size() const = 0;

    virtual bool is_host() const noexcept = 0;
    virtual bool is_cpu() const noexcept = 0;
    virtual bool is_gpu() const noexcept = 0;
    virtual bool is_accelerator() const noexcept = 0;

    virtual bool is_ready() const = 0;

    virtual const group_unique_key& get_comm_group_id() const = 0;

    virtual ccl::communicator_interface_ptr split(const ccl::comm_split_attr& attr) = 0;

    // collectives operation declarations
    virtual ccl::event barrier(const stream::impl_value_t& op_stream,
                                 const barrier_attr& attr,
                                 const vector_class<event>& deps = {}) = 0;

    DEVICE_COMM_INTERFACE_COLL_DECLARATION__VOID;
    DEVICE_COMM_INTERFACE_COLL_DECLARATION(char);
    DEVICE_COMM_INTERFACE_COLL_DECLARATION(int);
    DEVICE_COMM_INTERFACE_COLL_DECLARATION(int64_t);
    DEVICE_COMM_INTERFACE_COLL_DECLARATION(uint64_t);
    DEVICE_COMM_INTERFACE_COLL_DECLARATION(float);
    DEVICE_COMM_INTERFACE_COLL_DECLARATION(double);

#ifdef CCL_ENABLE_SYCL
    DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<char COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<int COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<int64_t COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<uint64_t COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<float COMMA 1>);
    DEVICE_COMM_INTERFACE_COLL_CLASS_DECLARATION(cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION__VOID
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, char);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, int);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, float);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, double);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(char, uint64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, char);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, int);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, float);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, double);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int, uint64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, char);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, int);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, float);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, double);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(int64_t, uint64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, char);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, int);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, ccl::bf16);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, float);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, double);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, int64_t);
    DEVICE_COMM_INTERFACE_SPARSE_DECLARATION(uint64_t, uint64_t);

#ifdef CCL_ENABLE_SYCL
    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int COMMA 1>,
                                                   cl::sycl::buffer<float COMMA 1>);
    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int COMMA 1>,
                                                   cl::sycl::buffer<ccl::bf16 COMMA 1>);

    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int64_t COMMA 1>,
                                                   cl::sycl::buffer<float COMMA 1>);
    DEVICE_COMM_INTERFACE_SPARSE_CLASS_DECLARATION(cl::sycl::buffer<int64_t COMMA 1>,
                                                   cl::sycl::buffer<ccl::bf16 COMMA 1>);
#endif //CCL_ENABLE_SYCL
};
} // namespace ccl
