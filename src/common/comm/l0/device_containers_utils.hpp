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
#include <sstream>

#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "common/comm/l0/device_containers.hpp"

namespace native {

namespace details {
/*
struct splice_devices
{
    splice_devices(std::shared_ptr<specific_plain_device_storage>& out_process_devices) :
        total_process_devices(out_process_devices)
    {
    }

    template<class device_t>
    void operator() (plain_device_container<device_t>& in_container)
    {
        //naive merge
        auto& output_container = std::get<device_t::type_idx()>(*total_process_devices);
        output_container.insert(output_container.end(), in_container.begin(), in_container.end());
    }
    std::shared_ptr<specific_plain_device_storage>& total_process_devices;
};
*/

template <ccl::device_group_split_type group_id, ccl::device_topology_type class_id>
struct printer {
    template <class device_t>
    void operator()(const native::indexed_device_container<device_t>& container) {
        for (const auto& dev : container) {
            device_rank_descr.insert({ dev.first, dev.second->to_string() });
        }
    }

    template <class device_t>
    void operator()(const native::plain_device_container<device_t>& container) {
        for (const auto& dev : container) {
            device_rank_descr.insert(
                { dev->template get_comm_data<group_id, class_id>().rank, dev->to_string() });
        }
    }
    /*
    template<class device_t>
    void operator() (const native::indexed_device_container<native::ccl_thread_comm<device_t>>& container)
    {
        const auto& impl = container.get_read();
        const auto& inner_map = std::get<0>(impl).get();
        for(const auto& dev : inner_map)
        {
            device_rank_descr.insert({dev.first, dev.second->to_string()});
        }
    }

    void operator() (const native::indexed_device_container<native::ccl_thread_comm<native::ccl_virtual_gpu_comm>>& container)
    {
        auto ret = container.locker();
        const auto& impl = container.impl(ret);
        for(const auto& dev : impl)
        {
            device_rank_descr.insert({dev.first, dev.second->to_string()});
        }
    }*/

    std::string to_string() const {
        std::stringstream ss;
        for (auto val : device_rank_descr) {
            ss << "idx: " << val.first << "\n" << val.second << std::endl;
        }
        return ss.str();
    }
    std::map<size_t, std::string> device_rank_descr;
};
} // namespace details

} // namespace native
