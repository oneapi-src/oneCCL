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
#include "device_impl.hpp"

namespace ccl {

CCL_API device::device(device&& src) : base_t(std::move(src)) {}

CCL_API device::device(const device& src) : base_t(src) {}

CCL_API device::device(impl_value_t&& impl) : base_t(std::move(impl)) {}

CCL_API device::~device() {}

CCL_API device& device::operator=(device&& src) {
    if (src.get_impl() != this->get_impl()) {
        src.get_impl().swap(this->get_impl());
        src.get_impl().reset();
    }
    return *this;
}

CCL_API device& device::operator=(const device& src) {
    if (src.get_impl() != this->get_impl()) {
        this->get_impl() = src.get_impl();
    }
    return *this;
}

CCL_API void device::build_from_params() {
    get_impl()->build_from_params();
}

CCL_API device::native_t& device::get_native()
{
    return const_cast<device::native_t&>(static_cast<const device*>(this)->get_native());
}

CCL_API const device::native_t& device::get_native() const
{
    return get_impl()->get_attribute_value(
        details::ccl_api_type_attr_traits<ccl::device_attr_id, ccl::device_attr_id::native_handle>{});
}
} // namespace ccl

API_DEVICE_CREATION_FORCE_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t)

API_DEVICE_FORCE_INSTANTIATION(ccl::device_attr_id::version, ccl::library_version);
API_DEVICE_FORCE_INSTANTIATION_GET(ccl::device_attr_id::cl_backend);
API_DEVICE_FORCE_INSTANTIATION_GET(ccl::device_attr_id::native_handle);
