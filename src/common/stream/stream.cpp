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
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/stream/stream_provider_dispatcher_impl.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

std::string to_string(const stream_type& type) {
    return stream_str_enum({ "host", "cpu", "gpu" }).choose(type, "unknown");
}

ccl_stream::ccl_stream(stream_type type,
                       stream_native_t& stream,
                       const ccl::library_version& version)
        : type(type),
          version(version) {
    native_stream = stream;

#ifdef CCL_ENABLE_SYCL
    native_streams.resize(ccl::global_data::env().worker_count);
    for (size_t idx = 0; idx < native_streams.size(); idx++) {
        native_streams[idx] = stream_native_t(stream.get_context(), stream.get_device());
    }

    backend = stream.get_device().get_backend();
#endif // CCL_ENABLE_SYCL
}

// export attributes
typename ccl_stream::version_traits_t::type ccl_stream::set_attribute_value(
    typename version_traits_t::type val,
    const version_traits_t& t) {
    (void)t;
    throw ccl::exception("set value for 'ccl::stream_attr_id::library_version' is not allowed");
    return version;
}

const typename ccl_stream::version_traits_t::return_type& ccl_stream::get_attribute_value(
    const version_traits_t& id) const {
    return version;
}

typename ccl_stream::native_handle_traits_t::return_type& ccl_stream::get_attribute_value(
    const native_handle_traits_t& id) {
    return native_stream;
}

std::string ccl_stream::to_string() const {
    std::stringstream ss;
#ifdef CCL_ENABLE_SYCL
    ss << "{ "
       << "type: " << ::to_string(type) << ", in_order: " << native_stream.is_in_order()
       << ", device: " << native_stream.get_device().get_info<cl::sycl::info::device::name>()
       << " }";
#else // CCL_ENABLE_SYCL
    ss << reinterpret_cast<void*>(native_stream.get());
#endif // CCL_ENABLE_SYCL
    return ss.str();
}
