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
#include "common/comm/l0/device_group_routing_schema.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"
#include "common/comm/l0/gpu_device_types.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"

namespace native
{
using device_indexed_group_ptr = std::unique_ptr<specific_indexed_device_storage>;

template<ccl::device_topology_type schema_id>
struct device_community
{
    device_community(const ccl::context_comm_addr& comm_addr) :
        topology_addr(comm_addr)
    {
    }

    std::multiset<ccl::device_index_type> registered_device_id;
    ccl::context_comm_addr topology_addr;

    device_indexed_group_ptr& get_device_storage_ptr()
    {
        return devices;
    }

    specific_indexed_device_storage& get_device_storage()
    {
        auto &ptr = get_device_storage_ptr();
        if(!ptr)
        {
            abort();
        }
        return *ptr;
    }

    template<class device_t>
    indexed_device_container<device_t>& get_devices()
    {
        static native::indexed_device_container<device_t> empty;

        return devices ?
                    std::get<device_t::type_idx()>(*devices) :
                    empty;
    }

    template<class device_t>
    size_t get_device_count() const
    {
        return devices ?
                    std::get<device_t::type_idx()>(*devices).size():
                    0;
    }

    std::string to_string() const
    {
        std::stringstream result;
        result << "Topology: " << ::to_string(schema_id) << "\n";
        native::details::printer<schema_id> p;
        if(devices)
        {
            ccl_tuple_for_each(*devices, p);
            result << p.to_string();
        }
        else
        {
            result << "EMPTY";
        }
        return result.str();
    }
private:
    device_indexed_group_ptr devices;
};

namespace details
{
struct device_community_printer
{
    device_community_printer(std::ostream& out = std::cout) :
        output(out)
    {
    }

    template<ccl::device_topology_type top>
    void operator() (const device_community<top>& community)
    {
        output << "Community topology: " << community.to_string() << std::endl;
    }
    template<ccl::device_topology_type top>
    void operator() (const std::shared_ptr<device_community<top>>& community_ptr)
    {
        output << "Community topology: " << community_ptr->to_string() << std::endl;
    }
private:
    std::ostream& output;

};
}
}
