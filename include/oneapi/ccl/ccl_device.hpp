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

class ccl_device_impl;
namespace ccl {

/**
 * A device object is an abstraction over CPU/GPU device
 * Has no defined public constructor. Use ccl::environment::create_device
 * for device objects creation
 */
/**
 * Stream class
 */
class device : public ccl_api_base_copyable<device, direct_access_policy, ccl_device_impl> {
public:
    using base_t = ccl_api_base_copyable<device, direct_access_policy, ccl_device_impl>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    /**
     * Declare native device type
     */
    using native_t = typename details::ccl_api_type_attr_traits<ccl::device_attr_id,
                                                                ccl::device_attr_id::native_handle>::return_type;

    device(device&& src);
    device(const device& src);
    device& operator=(device&& src);
    device& operator=(const device& src);
    ~device();

    /**
     * Get specific attribute value by @attrId
     */
    template <device_attr_id attrId>
    const typename details::ccl_api_type_attr_traits<device_attr_id, attrId>::return_type& get()
        const;

    /**
     * Get native device object
     */
     native_t& get_native();
     const native_t& get_native() const;
private:
    friend class environment;
    friend class communicator;
    device(impl_value_t&& impl);

    /**
     *Parametrized device creation helper
     */
    template <device_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename ccl::details::ccl_api_type_attr_traits<ccl::device_attr_id, attrId>::return_type set(const Value& v);

    void build_from_params();
    device(const typename details::ccl_api_type_attr_traits<device_attr_id,
                                                           device_attr_id::version>::type& version);

    /**
     * Factory methods
     */
    template <class device_type,
              class = typename std::enable_if<is_device_supported<device_type>()>::type>
    static device create_device(device_type&& native_device);

    template <class device_handle_type, class... attr_value_pair_t>
    static device create_device_from_attr(device_handle_type& native_device_handle,
                                        attr_value_pair_t&&... avps);
};

template <device_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> details::attr_value_tripple<device_attr_id, t, value_type> {
    return details::attr_value_tripple<device_attr_id, t, value_type>(v);
}

template<class DeviceType>
using rank_device_pair_t = ccl::pair_class<size_t, typename std::remove_reference<typename std::remove_cv<DeviceType>::type>::type>;

template <class device_value_type>
constexpr auto attr_val(size_t rank, device_value_type&& v)
    -> rank_device_pair_t<device_value_type>{
    return rank_device_pair_t<device_value_type>{rank, std::forward<device_value_type>(v)};
}

} // namespace ccl
