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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

// Core file with PIMPL implementation
#include "comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"

namespace ccl {

namespace v1 {

#define API_FORCE_INSTANTIATION(class_name, IN_attrId, IN_Value, OUT_Traits_Value) \
    template CCL_API IN_Value class_name::set<IN_attrId, IN_Value>(const IN_Value& v); \
\
    template CCL_API const typename OUT_Traits_Value<comm_split_attr_id, IN_attrId>::type& \
    class_name::get<IN_attrId>() const; \
\
    template CCL_API bool class_name::is_valid<IN_attrId>() const noexcept;

/**
 * comm_split_attr attributes definition
 */
CCL_API comm_split_attr::comm_split_attr(ccl_empty_attr)
        : base_t(impl_value_t(new impl_t(ccl_empty_attr::version))) {}
CCL_API comm_split_attr::comm_split_attr(comm_split_attr&& src) : base_t(std::move(src)) {}

CCL_API comm_split_attr::comm_split_attr(const comm_split_attr& src) : base_t(src) {}

CCL_API comm_split_attr::comm_split_attr(
    const typename detail::ccl_api_type_attr_traits<comm_split_attr_id,
                                                    comm_split_attr_id::version>::type& version)
        : base_t(impl_value_t(new impl_t(version))) {}

CCL_API comm_split_attr::~comm_split_attr() noexcept {}

CCL_API comm_split_attr& comm_split_attr::operator=(const comm_split_attr& src) {
    this->acc_policy_t::create(this, src);
    return *this;
}

CCL_API comm_split_attr& comm_split_attr::operator=(comm_split_attr&& src) {
    this->acc_policy_t::create(this, std::move(src));
    return *this;
}

API_FORCE_INSTANTIATION(comm_split_attr,
                        comm_split_attr_id::color,
                        int,
                        detail::ccl_api_type_attr_traits)
API_FORCE_INSTANTIATION(comm_split_attr,
                        comm_split_attr_id::group,
                        split_group,
                        detail::ccl_api_type_attr_traits)
API_FORCE_INSTANTIATION(comm_split_attr,
                        comm_split_attr_id::version,
                        ccl::library_version,
                        detail::ccl_api_type_attr_traits)

#undef API_FORCE_INSTANTIATION

} // namespace v1

} // namespace ccl
