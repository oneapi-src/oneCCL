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

class ccl_event;
namespace ccl {

/**
 * A event object is an abstraction over CPU/GPU events
 * Has no defined public constructor. Use ccl::environment::create_event
 * for event objects creation
 */
/**
 * Stream class
 */
class event_internal : public ccl_api_base_movable<event_internal, direct_access_policy, ccl_event> {
public:
    using base_t = ccl_api_base_movable<event_internal, direct_access_policy, ccl_event>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    event_internal(event_internal&& src);
    event_internal& operator=(event_internal&& src);
    ~event_internal();

    /**
     * Get specific attribute value by @attrId
     */
    template <event_attr_id attrId>
    const typename details::ccl_api_type_attr_traits<event_attr_id, attrId>::return_type& get()
        const;

private:
    friend class environment;
    friend class communicator;
    event_internal(impl_value_t&& impl);

    /**
     *Parametrized event_internal creation helper
     */
    template <event_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename ccl::details::ccl_api_type_attr_traits<ccl::event_attr_id, attrId>::return_type set(const Value& v);

    void build_from_params();
    event_internal(const typename details::ccl_api_type_attr_traits<event_attr_id,
                                                           event_attr_id::version>::type& version);

    /**
     * Factory methods
     */
    template <class event_type,
              class = typename std::enable_if<is_event_supported<event_type>()>::type>
    static event_internal create_event(event_type& native_event);

    template <class event_handle_type,
              class = typename std::enable_if<is_event_supported<event_handle_type>()>::type>
    static event_internal create_event(event_handle_type native_event_handle,
                              typename unified_device_context_type::ccl_native_t context);

    template <class event_type, class... attr_value_pair_t>
    static event_internal create_event_from_attr(event_type& native_event_handle,
                                        typename unified_device_context_type::ccl_native_t context,
                                        attr_value_pair_t&&... avps);
};

template <event_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> details::attr_value_tripple<event_attr_id, t, value_type> {
    return details::attr_value_tripple<event_attr_id, t, value_type>(v);
}

} // namespace ccl
