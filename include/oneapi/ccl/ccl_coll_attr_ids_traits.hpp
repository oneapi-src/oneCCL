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

namespace details {
template <class T>
class function_holder {
public:
    function_holder(T in = nullptr) : val(in) {}
    T get() const {
        return val;
    }

private:
    T val;
};
/**
 * Traits for common attributes specializations
 */
template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::version> {
    using type = ccl::library_version;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::prologue_fn> {
    using type = ccl::prologue_fn;
    using return_type = function_holder<type>;
};

template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::epilogue_fn> {
    using type = ccl::epilogue_fn;
    using return_type = function_holder<type>;
};

template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::priority> {
    using type = size_t;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::synchronous> {
    using type = int;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::to_cache> {
    using type = int;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::match_id> {
    using type = string_class;
    using return_type = type;
};

/**
 * Traits specialization for allgatherv op attributes
 */
template <>
struct ccl_api_type_attr_traits<allgatherv_attr_id, allgatherv_attr_id::vector_buf> {
    using type = int;
    using return_type = type;
};

/**
 * Traits specialization for allreduce op attributes
 */
template <>
struct ccl_api_type_attr_traits<allreduce_attr_id, allreduce_attr_id::reduction_fn> {
    using type = ccl::reduction_fn;
    using return_type = function_holder<type>;
};

/**
 * Traits specialization for alltoall op attributes
 */

/**
 * Traits specialization for alltoallv op attributes
 */

/**
 * Traits specialization for broadcast op attributes
 */

/**
 * Traits specialization for reduce op attributes
 */
template <>
struct ccl_api_type_attr_traits<reduce_attr_id, reduce_attr_id::reduction_fn> {
    using type = ccl::reduction_fn;
    using return_type = function_holder<type>;
};

/**
 * Traits specialization for reduce_scatter op attributes
 */
template <>
struct ccl_api_type_attr_traits<reduce_scatter_attr_id, reduce_scatter_attr_id::reduction_fn> {
    using type = ccl::reduction_fn;
    using return_type = function_holder<type>;
};

/**
 * Traits specialization for sparse_allreduce op attributes
 */
template <>
struct ccl_api_type_attr_traits<sparse_allreduce_attr_id, sparse_allreduce_attr_id::completion_fn> {
    using type = ccl::sparse_allreduce_completion_fn;
    using return_type = function_holder<type>;
};
template <>
struct ccl_api_type_attr_traits<sparse_allreduce_attr_id, sparse_allreduce_attr_id::alloc_fn> {
    using type = ccl::sparse_allreduce_alloc_fn;
    using return_type = function_holder<type>;
};
template <>
struct ccl_api_type_attr_traits<sparse_allreduce_attr_id, sparse_allreduce_attr_id::fn_ctx> {
    using type = const void*;
    using return_type = const void*;
};
template <>
struct ccl_api_type_attr_traits<sparse_allreduce_attr_id, sparse_allreduce_attr_id::coalesce_mode> {
    using type = ccl::sparse_coalesce_mode;
    using return_type = type;
};

/**
 * Traits specialization for barrier op attributes
 */
} // namespace details
} // namespace ccl
