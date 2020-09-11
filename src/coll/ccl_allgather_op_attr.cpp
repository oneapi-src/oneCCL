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
#include "coll/ccl_allgather_op_attr.hpp"

namespace ccl {

ccl_allgatherv_attr_impl_t::ccl_allgatherv_attr_impl_t(const base_t& base) : base_t(base) {}
ccl_allgatherv_attr_impl_t::ccl_allgatherv_attr_impl_t(
    const typename ccl_operation_attr_impl_t::version_traits_t::type& version)
        : base_t(version) {}
ccl_allgatherv_attr_impl_t::ccl_allgatherv_attr_impl_t(const ccl_allgatherv_attr_impl_t& src)
        : base_t(src) {}

typename ccl_allgatherv_attr_impl_t::vector_buf_traits_t::type
ccl_allgatherv_attr_impl_t::set_attribute_value(typename vector_buf_traits_t::type val,
                                                const vector_buf_traits_t& t) {
    auto old = vector_buf_id_val;
    std::swap(vector_buf_id_val, val);
    return old;
}

const typename ccl_allgatherv_attr_impl_t::vector_buf_traits_t::type&
ccl_allgatherv_attr_impl_t::get_attribute_value(const vector_buf_traits_t& id) const {
    return vector_buf_id_val;
}
} // namespace ccl
