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

namespace ccl {

class ccl_host_comm_split_attr_impl;
class ccl_device_comm_split_attr_impl;
struct ccl_empty_attr;

/**
 * Host attributes
 */
class comm_split_attr : public ccl_api_base_copyable<comm_split_attr,
                                                     copy_on_write_access_policy,
                                                     ccl_host_comm_split_attr_impl> {
public:
    using base_t = ccl_api_base_copyable<comm_split_attr,
                                         copy_on_write_access_policy,
                                         ccl_host_comm_split_attr_impl>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    comm_split_attr& operator=(const comm_split_attr& src);
    comm_split_attr& operator=(comm_split_attr&& src);
    comm_split_attr(comm_split_attr&& src);
    comm_split_attr(const comm_split_attr& src);
    comm_split_attr(ccl_empty_attr);
    ~comm_split_attr() noexcept;

    /**
     * Set specific value for selft attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <comm_split_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    Value set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <comm_split_attr_id attrId>
    const typename details::ccl_host_split_traits<comm_split_attr_id, attrId>::type& get() const;

    template <comm_split_attr_id attrId>
    bool is_valid() const noexcept;

private:
    friend class environment;
    comm_split_attr(
        const typename details::ccl_host_split_traits<comm_split_attr_id,
                                                      comm_split_attr_id::version>::type& version);

    /* TODO temporary function for UT compilation: would be part of ccl::environment in final*/
    template <class... attr_value_pair_t>
    static comm_split_attr create_comm_split_attr(attr_value_pair_t&&... avps);

    // create_split_attr() is internal func of create_comm_split_attr() in which comm_split_attr constructor is called
    template <class attr_t, class... attr_value_pair_t>
    friend attr_t create_split_attr(attr_value_pair_t&&... avps);
};

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

/**
 * Device attributes
 */
class device_comm_split_attr : public ccl_api_base_copyable<device_comm_split_attr,
                                                            copy_on_write_access_policy,
                                                            ccl_device_comm_split_attr_impl> {
public:
    using base_t = ccl_api_base_copyable<device_comm_split_attr,
                                         copy_on_write_access_policy,
                                         ccl_device_comm_split_attr_impl>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    device_comm_split_attr& operator=(device_comm_split_attr&& src);
    device_comm_split_attr(device_comm_split_attr&& src);
    device_comm_split_attr(const device_comm_split_attr& src);
    device_comm_split_attr(ccl_empty_attr);
    ~device_comm_split_attr() noexcept;

    /**
     * Set specific value for selft attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <comm_split_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    Value set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <comm_split_attr_id attrId>
    const typename details::ccl_device_split_traits<comm_split_attr_id, attrId>::type& get() const;

    template <comm_split_attr_id attrId>
    bool is_valid() const noexcept;

private:
    friend class environment;
    device_comm_split_attr(
        const typename details::ccl_device_split_traits<comm_split_attr_id,
                                                        comm_split_attr_id::version>::type&
            version);

    /* TODO temporary function for UT compilation: would be part of ccl::environment in final*/
    template <class... attr_value_pair_t>
    static device_comm_split_attr create_device_comm_split_attr(attr_value_pair_t&&... avps);
};

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

template <comm_split_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> details::attr_value_tripple<comm_split_attr_id, t, value_type> {
    return details::attr_value_tripple<comm_split_attr_id, t, value_type>(v);
}
} // namespace ccl
