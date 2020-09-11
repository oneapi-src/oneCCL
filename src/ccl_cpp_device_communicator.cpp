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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_aliases.hpp"

#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "oneapi/ccl/ccl_event_attr_ids.hpp"
#include "oneapi/ccl/ccl_event_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "oneapi/ccl/ccl_request.hpp"

#include "oneapi/ccl/ccl_device_communicator.hpp"
#include "common/comm/l0/comm_context_storage.hpp"

#include "common/global/global.hpp"

//TODO
#include "common/comm/comm.hpp"

#include "common/comm/l0/comm_context.hpp"
#include "device_communicator_impl.hpp"

#ifndef COMMA
#define COMMA ,
#endif

//API force instantiations for factory methods
#ifdef CCL_ENABLE_SYCL
API_DEVICE_COMM_CREATE_WO_RANK_EXPLICIT_INSTANTIATION(cl::sycl::device, cl::sycl::context)
API_DEVICE_COMM_CREATE_WITH_RANK_IN_VECTOR_EXPLICIT_INSTANTIATION(cl::sycl::device,
                                                                  cl::sycl::context)
API_DEVICE_COMM_CREATE_WITH_RANK_IN_MAP_EXPLICIT_INSTANTIATION(cl::sycl::device, cl::sycl::context)

API_DEVICE_COMM_CREATE_WO_RANK_EXPLICIT_INSTANTIATION(ccl::device_index_type, cl::sycl::context)
API_DEVICE_COMM_CREATE_WITH_RANK_IN_VECTOR_EXPLICIT_INSTANTIATION(ccl::device_index_type,
                                                                  cl::sycl::context)
API_DEVICE_COMM_CREATE_WITH_RANK_IN_MAP_EXPLICIT_INSTANTIATION(ccl::device_index_type,
                                                               cl::sycl::context)
#else
#ifdef MULTI_GPU_SUPPORT
API_DEVICE_COMM_CREATE_WO_RANK_EXPLICIT_INSTANTIATION(
    ccl::device_index_type,
    ccl::unified_device_context_type::ccl_native_t)
API_DEVICE_COMM_CREATE_WITH_RANK_IN_VECTOR_EXPLICIT_INSTANTIATION(
    ccl::device_index_type,
    ccl::unified_device_context_type::ccl_native_t)
API_DEVICE_COMM_CREATE_WITH_RANK_IN_MAP_EXPLICIT_INSTANTIATION(
    ccl::device_index_type,
    ccl::unified_device_context_type::ccl_native_t)
#endif
#endif

// API force instantiations for Operations
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(char);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int64_t);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(float);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(double);

#ifdef CCL_ENABLE_SYCL
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<char COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int64_t COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<uint64_t COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<float COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, char);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, int);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, ccl::bfp16);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, float);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, double);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, int64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, uint64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, char);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, int);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, ccl::bfp16);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, float);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, double);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, int64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, uint64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, char);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, int);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, ccl::bfp16);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, float);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, double);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, int64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, uint64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, char);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, int);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, ccl::bfp16);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, float);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, double);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, int64_t);
// API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, uint64_t);

#ifdef CCL_ENABLE_SYCL
/*
    API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int COMMA 1>,
                                                      cl::sycl::buffer<float COMMA 1>);
    API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int COMMA 1>,
                                                      cl::sycl::buffer<ccl::bfp16 COMMA 1>);

    API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int64_t COMMA 1>,
                                                      cl::sycl::buffer<float COMMA 1>);
    API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int64_t COMMA 1>,
                                                      cl::sycl::buffer<ccl::bfp16 COMMA 1>);
    */
#endif //CCL_ENABLE_SYCL
#undef COMMA
