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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

namespace v1 {

/**
 * Common operation attributes id
 */
enum class operation_attr_id : int {
    version,

    priority,
    to_cache,
    synchronous,
    match_id,

    last_value
};

/**
 * Collective attributes
 */
enum class allgatherv_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    last_value
};

enum class allreduce_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),
    reduction_fn = op_id_offset,

    last_value
};

enum class alltoall_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    last_value
};

enum class alltoallv_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    last_value
};

enum class broadcast_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    last_value
};

enum class reduce_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    reduction_fn = op_id_offset,
    last_value
};

enum class reduce_scatter_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    reduction_fn = op_id_offset,
    last_value
};

enum class sparse_allreduce_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    completion_fn = op_id_offset,
    alloc_fn,
    fn_ctx,
    coalesce_mode,
    last_value
};

enum class barrier_attr_id : int {
    op_id_offset = static_cast<typename std::underlying_type<operation_attr_id>::type>(
        operation_attr_id::last_value),

    last_value
};

} // namespace v1

using v1::operation_attr_id;
using v1::allgatherv_attr_id;
using v1::allreduce_attr_id;
using v1::alltoall_attr_id;
using v1::alltoallv_attr_id;
using v1::broadcast_attr_id;
using v1::reduce_attr_id;
using v1::reduce_scatter_attr_id;
using v1::sparse_allreduce_attr_id;
using v1::barrier_attr_id;

} // namespace ccl
