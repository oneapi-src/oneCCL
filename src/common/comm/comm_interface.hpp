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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/event.hpp"

#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/stream.hpp"

#include "common/comm/compiler_comm_interface_dispatcher.hpp"
#include "common/comm/l0/comm_context_id.hpp"
#include "internal_types.hpp"

namespace native {
struct ccl_device;
}

namespace ccl {
namespace v1 {
class allgatherv_attr;
class allreduce_attr;
class alltoall_attr;
class alltoallv_attr;
class barrier_attr;
class broadcast_attr;
class reduce_attr;
class reduce_scatter_attr;
class sparse_allreduce_attr;
} // namespace v1

struct gpu_comm_attr;
} // namespace ccl

#include "types_generator_defines.hpp"

#define COMM_INTERFACE_COLL_METHODS(TYPE) \
\
    COMM_INTERFACE_COLL_##TYPE##__VOID; \
    COMM_INTERFACE_COLL_##TYPE(int8_t); \
    COMM_INTERFACE_COLL_##TYPE(uint8_t); \
    COMM_INTERFACE_COLL_##TYPE(int16_t); \
    COMM_INTERFACE_COLL_##TYPE(uint16_t); \
    COMM_INTERFACE_COLL_##TYPE(int32_t); \
    COMM_INTERFACE_COLL_##TYPE(uint32_t); \
    COMM_INTERFACE_COLL_##TYPE(int64_t); \
    COMM_INTERFACE_COLL_##TYPE(uint64_t); \
    COMM_INTERFACE_COLL_##TYPE(float); \
    COMM_INTERFACE_COLL_##TYPE(double); \
\
    COMM_INTERFACE_SPARSE_##TYPE##__VOID; \
    COMM_INTERFACE_SPARSE_##TYPE(int32_t, ccl::bfloat16); \
    COMM_INTERFACE_SPARSE_##TYPE(int32_t, float); \
    COMM_INTERFACE_SPARSE_##TYPE(int64_t, ccl::bfloat16); \
    COMM_INTERFACE_SPARSE_##TYPE(int64_t, float);

#define SYCL_COMM_INTERFACE_COLL_METHODS(TYPE) \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<int8_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<uint8_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<int16_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<uint16_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<int32_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<uint32_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<int64_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<uint64_t COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<ccl::float16 COMMA 1>);*/ \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<double COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_##TYPE(cl::sycl::buffer<ccl::bfloat16 COMMA 1>);*/ \
\
    COMM_INTERFACE_SPARSE_CLASS_##TYPE(cl::sycl::buffer<int32_t COMMA 1>, \
                                       cl::sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_SPARSE_CLASS_##TYPE(cl::sycl::buffer<int32_t COMMA 1>, \
                                       cl::sycl::buffer<ccl::bfloat16 COMMA 1>); \
\
    COMM_INTERFACE_SPARSE_CLASS_##TYPE(cl::sycl::buffer<int64_t COMMA 1>, \
                                       cl::sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_SPARSE_CLASS_##TYPE(cl::sycl::buffer<int64_t COMMA 1>, \
                                       cl::sycl::buffer<ccl::bfloat16 COMMA 1>);

#define COMM_INTERFACE_COLL_INSTANTIATION(COMM) \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int8_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint8_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int16_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint16_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int32_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint32_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int64_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint64_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, float); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, double); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(COMM, int32_t, float); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(COMM, int32_t, ccl::bfloat16); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(COMM, int64_t, float); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(COMM, int64_t, ccl::bfloat16);

#define SYCL_COMM_INTERFACE_COLL_INSTANTIATION(COMM) \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<int8_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<int32_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<int64_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<uint64_t COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<ccl::float16 COMMA 1>);*/ \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<double COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, cl::sycl::buffer<ccl::bfloat16 COMMA 1>);*/ \
\
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION( \
        COMM, cl::sycl::buffer<int32_t COMMA 1>, cl::sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION( \
        COMM, cl::sycl::buffer<int32_t COMMA 1>, cl::sycl::buffer<ccl::bfloat16 COMMA 1>); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION( \
        COMM, cl::sycl::buffer<int64_t COMMA 1>, cl::sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION( \
        COMM, cl::sycl::buffer<int64_t COMMA 1>, cl::sycl::buffer<ccl::bfloat16 COMMA 1>);

namespace ccl {
struct communicator_interface : public communicator_interface_dispatcher {
    virtual ~communicator_interface() = default;

    virtual int rank() const = 0;
    virtual int size() const = 0;

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

    COMM_INTERFACE_COLL_METHODS(DECLARATION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DECLARATION);
#endif /* CCL_ENABLE_SYCL */
};
} // namespace ccl
