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
#include <vector>
#include <string>

#define private public
#define protected public
//#include "../utils.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "common/comm/l0/devices/devices_declaration.hpp"
#undef protected
#undef private

namespace stub {
struct test_device : public native::ccl_device {
    test_device(native::ccl_device::owner_ptr_t&& parent)
            : native::ccl_device(
                  reinterpret_cast<native::ccl_device::handle_t>(new native::ccl_device::handle_t),
                  std::move(parent),
                  std::false_type{}) {}

    static std::shared_ptr<test_device> create(const ccl::device_index_type& full_device_index,
                                               native::ccl_device::owner_ptr_t&& driver) {
        std::shared_ptr<test_device> dev = std::make_shared<test_device>(std::move(driver));

        dev->device_properties.type = ZE_DEVICE_TYPE_GPU;
        dev->device_properties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
        dev->device_properties.deviceId =
            std::get<ccl::device_index_enum::device_index_id>(full_device_index);
        dev->device_properties.isSubdevice = 0;

        //create default queue
        auto queue_prop = ccl_device::get_default_queue_desc();
        queue_prop.ordinal = 0;
        dev->cmd_queus.emplace(queue_prop, ccl_device::device_queue{ nullptr, dev->get_ptr() });

        //create module
        auto module_ptr = std::make_shared<ccl_device::device_module>(nullptr, dev->get_ptr());

        using mod_integer_type = typename std::underlying_type<ccl_coll_type>::type;
        using top_integer_type = typename std::underlying_type<ccl::device_group_split_type>::type;
        using top_class_integer_type =
            typename std::underlying_type<ccl::device_topology_type>::type;
        for (auto i = static_cast<mod_integer_type>(ccl_coll_type::ccl_coll_allgatherv);
             i < static_cast<mod_integer_type>(ccl_coll_type::ccl_coll_last_value);
             i++) {
            for (auto j = static_cast<top_integer_type>(ccl::device_group_split_type::thread);
                 j < static_cast<top_integer_type>(ccl::device_group_split_type::cluster);
                 j++) {
                for (auto k = static_cast<top_class_integer_type>(ccl::device_topology_type::ring);
                     k < static_cast<top_class_integer_type>(ccl::device_topology_type::a2a);
                     k++) {
                    size_t hash = native::module_hash(static_cast<ccl_coll_type>(i),
                                                      static_cast<ccl::device_group_split_type>(j),
                                                      static_cast<ccl::device_topology_type>(k));
                    dev->modules.insert({ hash, module_ptr });
                }
            }
        }
        return dev;
    }
};

struct test_subdevice : public native::ccl_subdevice {
    test_subdevice(native::ccl_subdevice::owner_ptr_t&& parent,
                   typename native::ccl_subdevice::base::owner_ptr_t&& driver)
            : native::ccl_subdevice(reinterpret_cast<native::ccl_subdevice::handle_t>(
                                        new native::ccl_subdevice::handle_t),
                                    std::move(parent),
                                    std::move(driver),
                                    std::false_type{}) {}

    static std::shared_ptr<test_subdevice> create(
        const ccl::device_index_type& full_device_index,
        native::ccl_subdevice::owner_ptr_t&& device,
        typename native::ccl_subdevice::base::owner_ptr_t&& driver) {
        std::shared_ptr<test_subdevice> subdev =
            std::make_shared<test_subdevice>(std::move(device), std::move(driver));
        subdev->device_properties.type = ZE_DEVICE_TYPE_GPU;
        subdev->device_properties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
        subdev->device_properties.deviceId =
            std::get<ccl::device_index_enum::subdevice_index_id>(full_device_index);
        subdev->device_properties.isSubdevice = 1;

        //create default queue
        auto queue_prop = ccl_subdevice::get_default_queue_desc();
        queue_prop.ordinal = 0;
        subdev->cmd_queus.emplace(queue_prop,
                                  ccl_subdevice::device_queue{ nullptr, subdev->get_ptr() });

        //create module
        auto module_ptr =
            std::make_shared<ccl_subdevice::device_module>(nullptr, subdev->get_ptr());

        using mod_integer_type = typename std::underlying_type<ccl_coll_type>::type;
        using top_integer_type = typename std::underlying_type<ccl::device_group_split_type>::type;
        using top_class_integer_type =
            typename std::underlying_type<ccl::device_topology_type>::type;
        for (auto i = static_cast<mod_integer_type>(ccl_coll_type::ccl_coll_allgatherv);
             i < static_cast<mod_integer_type>(ccl_coll_type::ccl_coll_last_value);
             i++) {
            for (auto j = static_cast<top_integer_type>(ccl::device_group_split_type::thread);
                 j < static_cast<top_integer_type>(ccl::device_group_split_type::cluster);
                 j++) {
                for (auto k = static_cast<top_class_integer_type>(ccl::device_topology_type::ring);
                     k < static_cast<top_class_integer_type>(ccl::device_topology_type::a2a);
                     k++) {
                    size_t hash = native::module_hash(static_cast<ccl_coll_type>(i),
                                                      static_cast<ccl::device_group_split_type>(j),
                                                      static_cast<ccl::device_topology_type>(k));
                    subdev->modules.insert({ hash, module_ptr });
                }
            }
        }
        return subdev;
    }
};

inline void make_stub_devices(const ccl::device_indices_t& stub_indices) {
    using namespace native;
    using namespace ccl;

    ccl_device_platform& platform = native::get_platform();
    ccl_device_platform::driver_storage_type& drivers = platform.drivers;

    for (const auto& index : stub_indices) {
        ccl::index_type driver_index = std::get<device_index_enum::driver_index_id>(index);
        if (driver_index == unused_index_value) {
            continue;
        }

        auto driver_it = drivers.find(driver_index);
        if (driver_it == drivers.end()) {
            driver_it =
                drivers
                    .emplace(driver_index,
                             ccl_device_driver::create(nullptr, driver_index, platform.get_ptr()))
                    .first;
        }

        ccl::index_type device_index = std::get<device_index_enum::device_index_id>(index);
        if (device_index == unused_index_value) {
            continue;
        }

        ccl_device_driver& driver = *(driver_it->second);
        ccl_device_driver::devices_storage_type& devices = driver.devices;
        auto device_it = devices.find(device_index);
        if (device_it == devices.end()) {
            auto dev_idx = index;
            std::get<device_index_enum::subdevice_index_id>(dev_idx) = unused_index_value;
            device_it =
                devices.emplace(device_index, test_device::create(dev_idx, driver.get_ptr())).first;
        }

        ccl::index_type subdevice_index = std::get<device_index_enum::subdevice_index_id>(index);
        if (subdevice_index == unused_index_value) {
            continue;
        }

        ccl_device& device = *(device_it->second);
        ccl_device::sub_devices_container_type& subdevices = device.get_subdevices();
        auto subdevice_it = subdevices.find(subdevice_index);
        if (subdevice_it == subdevices.end()) {
            subdevice_it =
                subdevices
                    .emplace(subdevice_index,
                             test_subdevice::create(index, device.get_ptr(), driver.get_ptr()))
                    .first;
        }
    }
}

inline void make_stub_devices(const ccl::process_device_indices_t& stub_indices) {
    for (const auto& pr_indices : stub_indices) {
        make_stub_devices(pr_indices.second);
    }
}
} // namespace stub
