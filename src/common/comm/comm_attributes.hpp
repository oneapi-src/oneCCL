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
#include "ccl.hpp"

namespace ccl
{
struct host_attr_impl : public ccl_host_comm_attr_t
{
    using base_t = ccl_host_comm_attr_t;
    using base_t::comm_attr;

    host_attr_impl (const base_t& base, const ccl_version_t& lib_version);

    host_attr_impl(const host_attr_impl& src);

    constexpr static int color_default()
    {
        return 0;
    }

    int set_attribute_value(int preferred_color);
    ccl_version_t set_attribute_value(ccl_version_t);

    const int& get_attribute_value(std::integral_constant<ccl_host_attributes,
                                                   ccl_host_attributes::ccl_host_color> stub) const;
    const ccl_version_t& get_attribute_value(std::integral_constant<ccl_host_attributes,
                                                   ccl_host_attributes::ccl_host_version> stub) const;
private:
    ccl_version_t library_version;
};



#ifdef MULTI_GPU_SUPPORT
struct device_attr_impl
{
    constexpr static device_topology_class class_default()
    {
        return device_topology_class::ring_class;
    }
    constexpr static device_topology_group group_default()
    {
        return device_topology_group::thread_dev_group;
    }

    device_topology_class set_attribute_value(device_topology_class preferred_topology);
    device_topology_group set_attribute_value(device_topology_group preferred_topology);

    const device_topology_class&
        get_attribute_value(std::integral_constant<ccl_device_attributes,
                                                   ccl_device_attributes::ccl_device_preferred_topology_class> stub) const;
    const device_topology_group&
        get_attribute_value(std::integral_constant<ccl_device_attributes,
                                                   ccl_device_attributes::ccl_device_preferred_group> stub) const;
private:
    device_topology_class current_preferred_topology_class = class_default();
    device_topology_group current_preferred_topology_group = group_default();
};
#endif
}
