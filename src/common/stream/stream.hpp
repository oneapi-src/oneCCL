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

#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "common/utils/enums.hpp"
#include "common/utils/utils.hpp"
#include "common/stream/stream_provider_dispatcher.hpp"

#include "coll/coll_common_attributes.hpp"
#include "internal_types.hpp"

namespace ccl {
namespace detail {
class environment;
}
} // namespace ccl

using stream_str_enum = utils::enum_to_str<utils::enum_to_underlying(stream_type::last_value)>;
std::string to_string(const stream_type& type);

/*
ccl::status CCL_API ccl_stream_create(stream_type type,
                                          void* native_stream,
                                          ccl_stream_t* stream);
*/
class alignas(CACHELINE_SIZE) ccl_stream : public stream_provider_dispatcher {
public:
    friend class stream_provider_dispatcher;
    friend class ccl::detail::environment;
    /*
    friend ccl::status CCL_API ccl_stream_create(stream_type type,
                               void* native_stream,
                               ccl_stream_t* stream);*/
    using stream_native_t = stream_provider_dispatcher::stream_native_t;
    using stream_native_handle_t = stream_provider_dispatcher::stream_native_handle_t;

    ccl_stream() = delete;
    ccl_stream(const ccl_stream& other) = delete;
    ccl_stream& operator=(const ccl_stream& other) = delete;

    ~ccl_stream() = default;

    using stream_provider_dispatcher::get_native_stream;

    stream_type get_type() const {
        return type;
    }

    bool is_sycl_device_stream() const {
        return (type == stream_type::cpu || type == stream_type::gpu);
    }

    static std::unique_ptr<ccl_stream> create(stream_native_t& native_stream,
                                              const ccl::library_version& version);

    //Export Attributes
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

    using device_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::device>;
    typename device_traits_t::return_type& get_attribute_value(const device_traits_t& id);

    using context_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::context>;
    typename context_traits_t::return_type& get_attribute_value(const context_traits_t& id);

    typename context_traits_t::return_type& set_attribute_value(typename context_traits_t::type val,
                                                                const context_traits_t& t);
    /*
    typename context_traits_t::return_type& set_attribute_value(
        typename context_traits_t::handle_t val,
        const context_traits_t& t);
*/
    using ordinal_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::ordinal>;
    typename ordinal_traits_t::return_type set_attribute_value(typename ordinal_traits_t::type val,
                                                               const ordinal_traits_t& t);

    const typename ordinal_traits_t::return_type& get_attribute_value(
        const ordinal_traits_t& id) const;

    using index_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::index>;
    typename index_traits_t::return_type set_attribute_value(typename index_traits_t::type val,
                                                             const index_traits_t& t);

    const typename index_traits_t::return_type& get_attribute_value(const index_traits_t& id) const;

    using flags_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::flags>;
    typename flags_traits_t::return_type set_attribute_value(typename flags_traits_t::type val,
                                                             const flags_traits_t& t);

    const typename flags_traits_t::return_type& get_attribute_value(const flags_traits_t& id) const;

    using mode_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::mode>;
    typename mode_traits_t::return_type set_attribute_value(typename mode_traits_t::type val,
                                                            const mode_traits_t& t);

    const typename mode_traits_t::return_type& get_attribute_value(const mode_traits_t& id) const;

    using priority_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::priority>;
    typename priority_traits_t::return_type set_attribute_value(
        typename priority_traits_t::type val,
        const priority_traits_t& t);

    const typename priority_traits_t::return_type& get_attribute_value(
        const priority_traits_t& id) const;

    void build_from_params();

private:
    ccl_stream(stream_type type,
               stream_native_t& native_stream,
               const ccl::library_version& version);

    ccl_stream(stream_type type,
               stream_native_handle_t native_stream,
               const ccl::library_version& version);

    ccl_stream(stream_type type, const ccl::library_version& version);

    stream_type type;
    const ccl::library_version version;
    typename ordinal_traits_t::return_type ordinal_val;
    typename index_traits_t::return_type index_val;
    typename flags_traits_t::return_type flags_val;
    typename mode_traits_t::return_type mode_val;
    typename priority_traits_t::return_type priority_val;

    bool is_context_enabled{ false };
};
