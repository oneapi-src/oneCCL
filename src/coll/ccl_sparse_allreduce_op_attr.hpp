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
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "coll/coll_common_attributes.hpp"
namespace ccl {

class ccl_sparse_allreduce_attr_impl_t : public ccl_operation_attr_impl_t {
public:
    using base_t = ccl_operation_attr_impl_t;

    ccl_sparse_allreduce_attr_impl_t(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);

    using sparse_allreduce_completion_fn_traits =
        detail::ccl_api_type_attr_traits<sparse_allreduce_attr_id,
                                         sparse_allreduce_attr_id::completion_fn>;
    typename sparse_allreduce_completion_fn_traits::return_type set_attribute_value(
        typename sparse_allreduce_completion_fn_traits::type val,
        const sparse_allreduce_completion_fn_traits& t);
    const typename sparse_allreduce_completion_fn_traits::return_type& get_attribute_value(
        const sparse_allreduce_completion_fn_traits& id) const;

    using sparse_allreduce_alloc_fn_traits =
        detail::ccl_api_type_attr_traits<sparse_allreduce_attr_id,
                                         sparse_allreduce_attr_id::alloc_fn>;
    typename sparse_allreduce_alloc_fn_traits::return_type set_attribute_value(
        typename sparse_allreduce_alloc_fn_traits::type val,
        const sparse_allreduce_alloc_fn_traits& t);
    const typename sparse_allreduce_alloc_fn_traits::return_type& get_attribute_value(
        const sparse_allreduce_alloc_fn_traits& id) const;

    using sparse_allreduce_fn_ctx_traits =
        detail::ccl_api_type_attr_traits<sparse_allreduce_attr_id,
                                         sparse_allreduce_attr_id::fn_ctx>;
    typename sparse_allreduce_fn_ctx_traits::return_type set_attribute_value(
        typename sparse_allreduce_fn_ctx_traits::type val,
        const sparse_allreduce_fn_ctx_traits& t);
    const typename sparse_allreduce_fn_ctx_traits::return_type& get_attribute_value(
        const sparse_allreduce_fn_ctx_traits& id) const;

    using sparse_coalesce_mode_traits =
        detail::ccl_api_type_attr_traits<sparse_allreduce_attr_id,
                                         sparse_allreduce_attr_id::coalesce_mode>;
    typename sparse_coalesce_mode_traits::return_type set_attribute_value(
        typename sparse_coalesce_mode_traits::type val,
        const sparse_coalesce_mode_traits& t);
    const typename sparse_coalesce_mode_traits::return_type& get_attribute_value(
        const sparse_coalesce_mode_traits& id) const;

private:
    typename sparse_allreduce_completion_fn_traits::return_type completion_fn_val{};
    typename sparse_allreduce_alloc_fn_traits::return_type alloc_fn_val{};
    typename sparse_allreduce_fn_ctx_traits::return_type fn_ctx_val = nullptr;
    typename sparse_coalesce_mode_traits::return_type mode_val = ccl::sparse_coalesce_mode::regular;
};
} // namespace ccl
