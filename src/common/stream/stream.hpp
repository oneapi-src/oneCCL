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

#include "coll/coll_common_attributes.hpp"
#include "common/stream/stream_provider_dispatcher.hpp"
#include "common/utils/enums.hpp"
#include "common/utils/utils.hpp"
#include "internal_types.hpp"
#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"

namespace ccl {
namespace detail {
class environment;
}
} // namespace ccl

using stream_str_enum = utils::enum_to_str<utils::enum_to_underlying(stream_type::last_value)>;
std::string to_string(const stream_type& type);

class alignas(CACHELINE_SIZE) ccl_stream : public stream_provider_dispatcher {
public:
    friend class stream_provider_dispatcher;
    friend class ccl::detail::environment;

    using stream_native_t = stream_provider_dispatcher::stream_native_t;

    ccl_stream() = delete;
    ccl_stream(const ccl_stream& other) = delete;
    ccl_stream& operator=(const ccl_stream& other) = delete;

    ~ccl_stream() = default;

    using stream_provider_dispatcher::get_native_stream;

    std::string to_string() const;

    stream_type get_type() const {
        return type;
    }

    bool is_sycl_device_stream() const {
        return (type == stream_type::cpu || type == stream_type::gpu);
    }

    bool is_gpu() const {
        return type == stream_type::gpu;
    }

#ifdef CCL_ENABLE_SYCL
    cl::sycl::backend get_backend() const noexcept {
        return backend;
    }
#endif // CCL_ENBALE_SYCL

    static std::unique_ptr<ccl_stream> create(stream_native_t& native_stream,
                                              const ccl::library_version& version);

    // export attributes
    using version_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::version>;
    typename version_traits_t::return_type set_attribute_value(typename version_traits_t::type val,
                                                               const version_traits_t& t);

    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const;

    using native_handle_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id,
                                              ccl::stream_attr_id::native_handle>;
    typename native_handle_traits_t::return_type& get_attribute_value(
        const native_handle_traits_t& id);

private:
    ccl_stream(stream_type type,
               stream_native_t& native_stream,
               const ccl::library_version& version);

    stream_type type;
#ifdef CCL_ENABLE_SYCL
    cl::sycl::backend backend;
#endif // CCL_ENBALE_SYCL
    const ccl::library_version version;
};
