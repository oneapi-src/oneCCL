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
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

// Core file with PIMPL implementation
#include "coll_attr_impl.hpp"
#include "coll/coll_attributes.hpp"

namespace ccl {
#define COMMA ,

#define API_FORCE_INSTANTIATION_SET(class_name, IN_attrType, IN_attrId, IN_Value) \
    template CCL_API \
        typename details::ccl_api_type_attr_traits<IN_attrType, IN_attrId>::return_type \
        class_name::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_FORCE_INSTANTIATION_GET(class_name, IN_attrType, IN_attrId) \
    template CCL_API const typename details::ccl_api_type_attr_traits<IN_attrType, \
                                                                      IN_attrId>::return_type& \
    class_name::get<IN_attrId>() const;

#define API_FORCE_INSTANTIATION(class_name, IN_attrType, IN_attrId, IN_Value) \
    API_FORCE_INSTANTIATION_SET(class_name, IN_attrType, IN_attrId, IN_Value) \
    API_FORCE_INSTANTIATION_GET(class_name, IN_attrType, IN_attrId)

#define COMMON_API_FORCE_INSTANTIATION(class_name) \
    API_FORCE_INSTANTIATION( \
        class_name, operation_attr_id, operation_attr_id::version, ccl::library_version) \
    API_FORCE_INSTANTIATION( \
        class_name, operation_attr_id, operation_attr_id::prologue_fn, ccl::prologue_fn) \
    API_FORCE_INSTANTIATION( \
        class_name, operation_attr_id, operation_attr_id::epilogue_fn, ccl::epilogue_fn) \
\
    API_FORCE_INSTANTIATION_SET( \
        class_name, operation_attr_id, operation_attr_id::priority, size_t) \
    API_FORCE_INSTANTIATION_SET(class_name, operation_attr_id, operation_attr_id::priority, int) \
    API_FORCE_INSTANTIATION_SET( \
        class_name, operation_attr_id, operation_attr_id::priority, unsigned int) \
    API_FORCE_INSTANTIATION_GET(class_name, operation_attr_id, operation_attr_id::priority) \
\
    API_FORCE_INSTANTIATION(class_name, operation_attr_id, operation_attr_id::synchronous, int) \
    API_FORCE_INSTANTIATION(class_name, operation_attr_id, operation_attr_id::to_cache, int) \
    API_FORCE_INSTANTIATION( \
        class_name, operation_attr_id, operation_attr_id::match_id, ccl::string_class)

/**
 * allgatherv coll attributes
 */
CCL_API allgatherv_attr::allgatherv_attr(allgatherv_attr&& src) : base_t(std::move(src)) {}

CCL_API allgatherv_attr::allgatherv_attr(const allgatherv_attr& src) : base_t(src) {}

CCL_API allgatherv_attr::allgatherv_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API allgatherv_attr::~allgatherv_attr() {}

/**
 * allreduce coll attributes
 */
CCL_API allreduce_attr::allreduce_attr(allreduce_attr&& src) : base_t(std::move(src)) {}

CCL_API allreduce_attr::allreduce_attr(const allreduce_attr& src) : base_t(src) {}

CCL_API allreduce_attr::allreduce_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API allreduce_attr::~allreduce_attr() {}

/**
 * alltoall coll attributes
 */
CCL_API alltoall_attr::alltoall_attr(alltoall_attr&& src) : base_t(std::move(src)) {}

CCL_API alltoall_attr::alltoall_attr(const alltoall_attr& src) : base_t(src) {}

CCL_API alltoall_attr::alltoall_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API alltoall_attr::~alltoall_attr() {}

/**
 * alltoallv coll attributes
 */
CCL_API alltoallv_attr::alltoallv_attr(alltoallv_attr&& src) : base_t(std::move(src)) {}

CCL_API alltoallv_attr::alltoallv_attr(const alltoallv_attr& src) : base_t(src) {}

CCL_API alltoallv_attr::alltoallv_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API alltoallv_attr::~alltoallv_attr() {}

/**
 * barrier coll attributes
 */
CCL_API barrier_attr::barrier_attr(barrier_attr&& src) : base_t(std::move(src)) {}

CCL_API barrier_attr::barrier_attr(const barrier_attr& src) : base_t(src) {}

CCL_API barrier_attr::barrier_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API barrier_attr::~barrier_attr() {}

/**
 * bcast coll attributes
 */
CCL_API broadcast_attr::broadcast_attr(broadcast_attr&& src) : base_t(std::move(src)) {}

CCL_API broadcast_attr::broadcast_attr(const broadcast_attr& src) : base_t(src) {}

CCL_API broadcast_attr::broadcast_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API broadcast_attr::~broadcast_attr() {}

/**
 * reduce coll attributes
 */
CCL_API reduce_attr::reduce_attr(reduce_attr&& src) : base_t(std::move(src)) {}

CCL_API reduce_attr::reduce_attr(const reduce_attr& src) : base_t(src) {}

CCL_API reduce_attr::reduce_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API reduce_attr::~reduce_attr() {}

/**
 * reduce_scatter coll attributes
 */
CCL_API reduce_scatter_attr::reduce_scatter_attr(reduce_scatter_attr&& src)
        : base_t(std::move(src)) {}

CCL_API reduce_scatter_attr::reduce_scatter_attr(const reduce_scatter_attr& src) : base_t(src) {}

CCL_API reduce_scatter_attr::reduce_scatter_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API reduce_scatter_attr::~reduce_scatter_attr() {}

/**
 * sparse_allreduce coll attributes
 */
CCL_API sparse_allreduce_attr::sparse_allreduce_attr(sparse_allreduce_attr&& src)
        : base_t(std::move(src)) {}

CCL_API sparse_allreduce_attr::sparse_allreduce_attr(const sparse_allreduce_attr& src)
        : base_t(src) {}

CCL_API sparse_allreduce_attr::sparse_allreduce_attr(
    const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                     operation_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API sparse_allreduce_attr::~sparse_allreduce_attr() {}

template <>
CCL_API const void* sparse_allreduce_attr::set<sparse_allreduce_attr_id::fn_ctx, const void*>(
    const void* const& v) {
    return get_impl()->set_attribute_value(
        v,
        details::ccl_api_type_attr_traits<sparse_allreduce_attr_id,
                                          sparse_allreduce_attr_id::fn_ctx>{});
}
template <>
CCL_API const void* const& sparse_allreduce_attr::get<sparse_allreduce_attr_id::fn_ctx>() const {
    return get_impl()->get_attribute_value(
        details::ccl_api_type_attr_traits<sparse_allreduce_attr_id,
                                          sparse_allreduce_attr_id::fn_ctx>{});
}

/**
 * Force instantiations
 */
COMMON_API_FORCE_INSTANTIATION(allgatherv_attr)
COMMON_API_FORCE_INSTANTIATION(allreduce_attr)
COMMON_API_FORCE_INSTANTIATION(alltoall_attr)
COMMON_API_FORCE_INSTANTIATION(alltoallv_attr)
COMMON_API_FORCE_INSTANTIATION(barrier_attr)
COMMON_API_FORCE_INSTANTIATION(broadcast_attr)
COMMON_API_FORCE_INSTANTIATION(reduce_attr)
COMMON_API_FORCE_INSTANTIATION(reduce_scatter_attr)
COMMON_API_FORCE_INSTANTIATION(sparse_allreduce_attr)

API_FORCE_INSTANTIATION(allgatherv_attr, allgatherv_attr_id, allgatherv_attr_id::vector_buf, int)
API_FORCE_INSTANTIATION(allreduce_attr,
                        allreduce_attr_id,
                        allreduce_attr_id::reduction_fn,
                        ccl::reduction_fn)
API_FORCE_INSTANTIATION(reduce_attr,
                        reduce_attr_id,
                        reduce_attr_id::reduction_fn,
                        ccl::reduction_fn)
API_FORCE_INSTANTIATION(reduce_scatter_attr,
                        reduce_scatter_attr_id,
                        reduce_scatter_attr_id::reduction_fn,
                        ccl::reduction_fn)
API_FORCE_INSTANTIATION(sparse_allreduce_attr,
                        sparse_allreduce_attr_id,
                        sparse_allreduce_attr_id::completion_fn,
                        ccl::sparse_allreduce_completion_fn)
API_FORCE_INSTANTIATION(sparse_allreduce_attr,
                        sparse_allreduce_attr_id,
                        sparse_allreduce_attr_id::alloc_fn,
                        ccl::sparse_allreduce_alloc_fn)
//API_FORCE_INSTANTIATION(sparse_allreduce_attr, sparse_allreduce_attr_id, sparse_allreduce_attr_id::fn_ctx, void*)
//template CCL_API const void * sparse_allreduce_attr::set<sparse_allreduce_attr_id::fn_ctx, void*>(void *const & v);
API_FORCE_INSTANTIATION(sparse_allreduce_attr,
                        sparse_allreduce_attr_id,
                        sparse_allreduce_attr_id::coalesce_mode,
                        ccl::sparse_coalesce_mode)

#undef API_FORCE_INSTANTIATION
#undef COMMON_API_FORCE_INSTANTIATION
#undef COMMA
} // namespace ccl
