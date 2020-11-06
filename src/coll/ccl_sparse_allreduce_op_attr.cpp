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
#include "coll/ccl_sparse_allreduce_op_attr.hpp"

namespace ccl {

ccl_sparse_allreduce_attr_impl_t::ccl_sparse_allreduce_attr_impl_t(
    const typename ccl_operation_attr_impl_t::version_traits_t::type& version)
        : base_t(version) {}

typename ccl_sparse_allreduce_attr_impl_t::sparse_allreduce_completion_fn_traits::return_type
ccl_sparse_allreduce_attr_impl_t::set_attribute_value(
    typename sparse_allreduce_completion_fn_traits::type val,
    const sparse_allreduce_completion_fn_traits& t) {
    auto old = reduction_fn_val;
    reduction_fn_val = typename sparse_allreduce_completion_fn_traits::return_type{ val };
    return typename sparse_allreduce_completion_fn_traits::return_type{ old };
}

const typename ccl_sparse_allreduce_attr_impl_t::sparse_allreduce_completion_fn_traits::return_type&
ccl_sparse_allreduce_attr_impl_t::get_attribute_value(
    const sparse_allreduce_completion_fn_traits& id) const {
    return reduction_fn_val;
}

typename ccl_sparse_allreduce_attr_impl_t::sparse_allreduce_alloc_fn_traits::return_type
ccl_sparse_allreduce_attr_impl_t::set_attribute_value(
    typename sparse_allreduce_alloc_fn_traits::type val,
    const sparse_allreduce_alloc_fn_traits& t) {
    auto old = alloc_fn_val;
    alloc_fn_val = typename sparse_allreduce_alloc_fn_traits::return_type{ val };
    return typename sparse_allreduce_alloc_fn_traits::return_type{ old };
}

const typename ccl_sparse_allreduce_attr_impl_t::sparse_allreduce_alloc_fn_traits::return_type&
ccl_sparse_allreduce_attr_impl_t::get_attribute_value(
    const sparse_allreduce_alloc_fn_traits& id) const {
    return alloc_fn_val;
}

typename ccl_sparse_allreduce_attr_impl_t::sparse_allreduce_fn_ctx_traits::return_type
ccl_sparse_allreduce_attr_impl_t::set_attribute_value(
    typename sparse_allreduce_fn_ctx_traits::type val,
    const sparse_allreduce_fn_ctx_traits& t) {
    auto old = fn_ctx_val;
    fn_ctx_val = typename sparse_allreduce_fn_ctx_traits::return_type{ val };
    return typename sparse_allreduce_fn_ctx_traits::return_type{ old };
}

const typename ccl_sparse_allreduce_attr_impl_t::sparse_allreduce_fn_ctx_traits::return_type&
ccl_sparse_allreduce_attr_impl_t::get_attribute_value(
    const sparse_allreduce_fn_ctx_traits& id) const {
    return fn_ctx_val;
}

typename ccl_sparse_allreduce_attr_impl_t::sparse_coalesce_mode_traits::return_type
ccl_sparse_allreduce_attr_impl_t::set_attribute_value(
    typename sparse_coalesce_mode_traits::type val,
    const sparse_coalesce_mode_traits& t) {
    auto old = mode_val;
    std::swap(mode_val, val);
    return typename sparse_coalesce_mode_traits::return_type{ old };
}

const typename ccl_sparse_allreduce_attr_impl_t::sparse_coalesce_mode_traits::return_type&
ccl_sparse_allreduce_attr_impl_t::get_attribute_value(const sparse_coalesce_mode_traits& id) const {
    return mode_val;
}
} // namespace ccl
