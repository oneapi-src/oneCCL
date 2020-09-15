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
#include "common/comm/l0/device_community_utils.hpp"

namespace native {

template <ccl::device_topology_type schema_id>
struct device_community {
    device_community(const ccl::context_comm_addr& comm_addr)
            : community_addr(comm_addr),
              devices(new specific_indexed_device_storage()) {}

    std::multiset<ccl::device_index_type> registered_device_id;

    specific_indexed_device_storage& get_device_storage() {
        auto& ptr = get_impl();
        if (!ptr) {
            abort();
        }
        return *ptr;
    }

    template <class device_t>
    indexed_device_container<device_t>& get_devices() {
        static native::indexed_device_container<device_t> empty;

        return devices ? std::get<device_t::type_idx()>(*devices) : empty;
    }

    template <class device_t>
    size_t get_device_count() const {
        return devices ? std::get<device_t::type_idx()>(*devices).size() : 0;
    }

    template <ccl::device_group_split_type group_id>
    void register_device_by_id(const ccl::device_index_type& device_id,
                               ccl::context_comm_addr& registered_addr) {
        if (!get_impl()) {
            std::string err_str;
            {
                std::stringstream str;
                ccl_logger::format(str,
                                   "Cannot initialize comm_addr for device id: ",
                                   device_id,
                                   " on topology: ",
                                   ::to_string(group_id),
                                   ", class: ",
                                   ::to_string(schema_id),
                                   ", empty device storage has got from context");
                err_str = str.str();
            }
            LOG_ERROR(err_str);
            throw std::runtime_error(err_str);
        }

        if (registered_addr.comm_rank != 0 or registered_addr.comm_size != 0) {
            std::string err_str;
            {
                std::stringstream str;
                ccl_logger::format(str,
                                   "Cannot register_device_by_id in topology for device id: ",
                                   device_id,
                                   " on topology: ",
                                   ::to_string(group_id),
                                   ", class: ",
                                   ::to_string(schema_id),
                                   ", because topology registered already, comm addr:",
                                   registered_addr.to_string());
                err_str = str.str();
            }
            LOG_ERROR(err_str);
            throw std::runtime_error(err_str);
        }

        // find device in topology and obtain its rank/sie
        details::rank_getter<group_id, schema_id> initializer(device_id, registered_device_id);
        ccl_tuple_for_each(get_device_storage(), initializer);

        // copy shared data from community addr
        registered_addr = community_addr;

        // get individual rank from initializer
        registered_addr.comm_rank = initializer.get_assigned_rank();
    }

    const ccl::context_comm_addr& get_comm_addr() const noexcept {
        return community_addr;
    }

    template <ccl::device_group_split_type group_id>
    std::string to_string() const {
        std::stringstream result;
        result << "Topology: " << ::to_string(schema_id) << "\n";
        native::details::printer<group_id, schema_id> p;
        if (devices) {
            ccl_tuple_for_each(*devices, p);
            result << p.to_string();
        }
        else {
            result << "EMPTY";
        }
        return result.str();
    }

private:
    ccl::context_comm_addr community_addr;
    std::unique_ptr<specific_indexed_device_storage>& get_impl() {
        return devices;
    }

    std::unique_ptr<specific_indexed_device_storage> devices;
};
} // namespace native
