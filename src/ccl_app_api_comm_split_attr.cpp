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
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

// Core file with PIMPL implementation
#include "common/comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"

namespace ccl {
#define COMMA ,
#define API_FORCE_INSTANTIATION(class_name, IN_attrId, IN_Value, OUT_Traits_Value) \
    template CCL_API IN_Value class_name::set<IN_attrId, IN_Value>(const IN_Value& v); \
\
    template CCL_API const typename details::OUT_Traits_Value<comm_split_attr_id, \
                                                              IN_attrId>::type& \
    class_name::get<IN_attrId>() const; \
\
    template CCL_API bool class_name::is_valid<IN_attrId>() const noexcept;

/**
 * comm_split_attr attributes definition
 */
CCL_API comm_split_attr::comm_split_attr(ccl_empty_attr)
        : base_t(std::shared_ptr<impl_t>(new impl_t(ccl_empty_attr::version))) {}
CCL_API comm_split_attr::comm_split_attr(comm_split_attr&& src) : base_t(std::move(src)) {}

CCL_API comm_split_attr::comm_split_attr(const comm_split_attr& src) : base_t(src) {}

CCL_API comm_split_attr::comm_split_attr(
    const typename details::ccl_host_split_traits<comm_split_attr_id,
                                                  comm_split_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API comm_split_attr::~comm_split_attr() noexcept {}

CCL_API comm_split_attr& comm_split_attr::operator=(const comm_split_attr& src) {
    this->get_impl() = src.get_impl();
    return *this;
}

CCL_API comm_split_attr& comm_split_attr::operator=(comm_split_attr&& src) {
    if (src.get_impl() != this->get_impl()) {
        src.get_impl().swap(this->get_impl());
        src.get_impl().reset();
    }
    return *this;
}
API_FORCE_INSTANTIATION(comm_split_attr, comm_split_attr_id::color, int, ccl_host_split_traits)
API_FORCE_INSTANTIATION(comm_split_attr,
                        comm_split_attr_id::group,
                        ccl_group_split_type,
                        ccl_host_split_traits)
API_FORCE_INSTANTIATION(comm_split_attr,
                        comm_split_attr_id::version,
                        ccl::library_version,
                        ccl_host_split_traits)

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

/**
 * device_comm_split_attr attributes definition
 */
CCL_API device_comm_split_attr::device_comm_split_attr(ccl_empty_attr)
        : base_t(std::shared_ptr<impl_t>(new impl_t(ccl_empty_attr::version))) {}
CCL_API device_comm_split_attr::device_comm_split_attr(device_comm_split_attr&& src)
        : base_t(std::move(src)) {}

CCL_API device_comm_split_attr::device_comm_split_attr(const device_comm_split_attr& src)
        : base_t(src) {}

CCL_API device_comm_split_attr::device_comm_split_attr(
    const typename details::ccl_host_split_traits<comm_split_attr_id,
                                                  comm_split_attr_id::version>::type& version)
        : base_t(std::shared_ptr<impl_t>(new impl_t(version))) {}

CCL_API device_comm_split_attr::~device_comm_split_attr() noexcept {}

CCL_API device_comm_split_attr& device_comm_split_attr::operator=(device_comm_split_attr&& src) {
    if (src.get_impl() != this->get_impl()) {
        src.get_impl().swap(this->get_impl());
        src.get_impl().reset();
    }
    return *this;
}
API_FORCE_INSTANTIATION(device_comm_split_attr,
                        comm_split_attr_id::color,
                        int,
                        ccl_device_split_traits)
API_FORCE_INSTANTIATION(device_comm_split_attr,
                        comm_split_attr_id::group,
                        device_group_split_type,
                        ccl_device_split_traits)
API_FORCE_INSTANTIATION(device_comm_split_attr,
                        comm_split_attr_id::version,
                        ccl::library_version,
                        ccl_device_split_traits)

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)/

#undef API_FORCE_INSTANTIATION
#undef COMMA
} // namespace ccl
