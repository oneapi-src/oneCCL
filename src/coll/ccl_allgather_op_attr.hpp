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
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "coll/coll_common_attributes.hpp"
namespace ccl {

class ccl_allgatherv_attr_impl_t : public ccl_operation_attr_impl_t {
public:
    using base_t = ccl_operation_attr_impl_t;

    ccl_allgatherv_attr_impl_t(const base_t& base);
    ccl_allgatherv_attr_impl_t(
        const typename details::ccl_api_type_attr_traits<operation_attr_id,
                                                         ccl::operation_attr_id::version>::type&
            version);
    ccl_allgatherv_attr_impl_t(const ccl_allgatherv_attr_impl_t& src);

    using vector_buf_traits_t =
        details::ccl_api_type_attr_traits<allgatherv_attr_id, allgatherv_attr_id::vector_buf>;
    typename vector_buf_traits_t::type set_attribute_value(typename vector_buf_traits_t::type val,
                                                           const vector_buf_traits_t& t);

    const typename vector_buf_traits_t::type& get_attribute_value(
        const vector_buf_traits_t& id) const;

private:
    typename vector_buf_traits_t::type vector_buf_id_val = 0;
};
} // namespace ccl
