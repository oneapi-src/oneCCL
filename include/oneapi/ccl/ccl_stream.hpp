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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

class ccl_stream;
namespace ccl {

/**
 * A stream object is an abstraction over CPU/GPU streams
 * Has no defined public constructor. Use ccl::environment::create_stream
 * for stream objects creation
 */
/**
 * Stream class
 */
class stream : public ccl_api_base_copyable<stream, direct_access_policy, ccl_stream> {
public:
    using base_t = ccl_api_base_copyable<stream, direct_access_policy, ccl_stream>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    /**
     * Declare native stream type
     */
    using native_t = typename details::ccl_api_type_attr_traits<ccl::stream_attr_id,
                                                                ccl::stream_attr_id::native_handle>::return_type;

    ~stream();

    stream(stream&& src);
    stream(const stream& src);
    stream& operator=(const stream& src);

    /**
     * Get specific attribute value by @attrId
     */
    template <stream_attr_id attrId>
    const typename details::ccl_api_type_attr_traits<stream_attr_id, attrId>::return_type& get()
        const;

    /**
     * Get native stream object
     */
     native_t& get_native();
     const native_t& get_native() const;
private:
    friend class environment;
    friend class communicator;
    friend struct ccl_empty_attr;
    friend struct impl_dispatch;

    template <class... attr_value_pair_t>
    friend stream create_stream_from_attr(
        typename unified_device_type::ccl_native_t device,
        typename unified_device_context_type::ccl_native_t context,
        attr_value_pair_t&&... avps);
    template <class... attr_value_pair_t>
    friend stream create_stream_from_attr(typename unified_device_type::ccl_native_t device,
                                          attr_value_pair_t&&... avps);

    stream(impl_value_t&& impl);

    /**
     *Parametrized stream creation helper
     */
    template <stream_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename details::ccl_api_type_attr_traits<stream_attr_id, attrId>::return_type set(const Value& v);

    void build_from_params();
    stream(
        const typename details::ccl_api_type_attr_traits<stream_attr_id,
                                                         stream_attr_id::version>::type& version);

    /**
     *  Factory methods
     */
    template <class native_stream_type,
              class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
    static stream create_stream(native_stream_type& native_stream);

    template <class native_stream_type,
              class native_context_type,
              class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
    static stream create_stream(native_stream_type& native_stream, native_context_type& native_ctx);

    template <class... attr_value_pair_t>
    static stream create_stream_from_attr(typename unified_device_type::ccl_native_t device,
                                          attr_value_pair_t&&... avps);

    template <class... attr_value_pair_t>
    static stream create_stream_from_attr(
        typename unified_device_type::ccl_native_t device,
        typename unified_device_context_type::ccl_native_t context,
        attr_value_pair_t&&... avps);
};

template <stream_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> details::attr_value_tripple<stream_attr_id, t, value_type> {
    return details::attr_value_tripple<stream_attr_id, t, value_type>(v);
}

/**
 * Declare extern empty attributes
 */
extern stream default_stream;
} // namespace ccl
