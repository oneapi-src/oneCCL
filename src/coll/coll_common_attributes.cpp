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
#include "coll/coll_common_attributes.hpp"

namespace ccl {
/**
 * Definition
 */
ccl_operation_attr_impl_t::ccl_operation_attr_impl_t(const ccl::library_version& version)
        : version(version) {}

typename ccl_operation_attr_impl_t::version_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename version_traits_t::type val,
                                               const version_traits_t& t) {
    (void)t;
    throw ccl_error("Set value for 'ccl::operation_attr_id::version' is not allowed");
    return version;
}

const typename ccl_operation_attr_impl_t::version_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const version_traits_t& id) const {
    return version;
}

/**
 * `prologue_fn` operations definitions
 */
const typename ccl_operation_attr_impl_t::prologue_fn_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const prologue_fn_traits_t& id) const {
    return prologue_fn;
}

typename ccl_operation_attr_impl_t::prologue_fn_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename prologue_fn_traits_t::type val,
                                               const prologue_fn_traits_t& t) {
    auto old = prologue_fn.get();
    prologue_fn = typename prologue_fn_traits_t::return_type{ val };
    return typename prologue_fn_traits_t::return_type{ old };
}
/**
 * `epilogue_fn` operations definitions
 */
const typename ccl_operation_attr_impl_t::epilogue_fn_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const epilogue_fn_traits_t& id) const {
    return epilogue_fn;
}

typename ccl_operation_attr_impl_t::epilogue_fn_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename epilogue_fn_traits_t::type val,
                                               const epilogue_fn_traits_t& t) {
    auto old = epilogue_fn.get();
    epilogue_fn = typename epilogue_fn_traits_t::return_type{ val };
    return typename epilogue_fn_traits_t::return_type{ old };
}

/**
 * `priority` operations definitions
 */
const typename ccl_operation_attr_impl_t::priority_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const priority_traits_t& id) const {
    return priority;
}

typename ccl_operation_attr_impl_t::priority_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename priority_traits_t::type val,
                                               const priority_traits_t& t) {
    auto old = priority;
    std::swap(priority, val);
    return old;
}

/**
 * `synchronous` operations definitions
 */
const typename ccl_operation_attr_impl_t::synchronous_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const synchronous_traits_t& id) const {
    return synchronous;
}

typename ccl_operation_attr_impl_t::synchronous_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename synchronous_traits_t::type val,
                                               const synchronous_traits_t& t) {
    auto old = synchronous;
    std::swap(synchronous, val);
    return old;
}

/**
 * `to_cache` operations definitions
 */
const typename ccl_operation_attr_impl_t::to_cache_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const to_cache_traits_t& id) const {
    return to_cache;
}

typename ccl_operation_attr_impl_t::to_cache_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename to_cache_traits_t::type val,
                                               const to_cache_traits_t& t) {
    auto old = to_cache;
    std::swap(to_cache, val);
    return old;
}

/**
 * `match_id` operations definitions
 */
const typename ccl_operation_attr_impl_t::match_id_traits_t::return_type&
ccl_operation_attr_impl_t::get_attribute_value(const match_id_traits_t& id) const {
    return match_id;
}

typename ccl_operation_attr_impl_t::match_id_traits_t::return_type
ccl_operation_attr_impl_t::set_attribute_value(typename match_id_traits_t::type val,
                                               const match_id_traits_t& t) {
    auto old = match_id;
    std::swap(match_id, val);
    return old;
}
} // namespace ccl
