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
#include "common/global/global.hpp"
#include "common/ze/ze_api_wrapper.hpp"

namespace ccl {
namespace ze {

global_data_desc::global_data_desc() {
    LOG_INFO("initializing level-zero");

    // enables driver initialization and
    // dependencies for system management
    setenv("ZES_ENABLE_SYSMAN", "1", 0);

    ZE_CALL(zeInit, (ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count{};
    ZE_CALL(zeDriverGet, (&driver_count, nullptr));
    driver_list.resize(driver_count);
    ZE_CALL(zeDriverGet, (&driver_count, driver_list.data()));
    LOG_DEBUG("found drivers: ", driver_list.size());

    context_list.resize(driver_list.size());
    for (size_t i = 0; i < driver_list.size(); ++i) {
        ze_context_desc_t desc = ze::default_context_desc;
        ZE_CALL(zeContextCreate, (driver_list.at(i), &desc, &context_list.at(i)));

        uint32_t device_count{};
        ZE_CALL(zeDeviceGet, (driver_list.at(i), &device_count, nullptr));
        std::vector<ze_device_handle_t> devs(device_count);
        ZE_CALL(zeDeviceGet, (driver_list.at(i), &device_count, devs.data()));
        for (uint32_t idx = 0; idx < device_count; idx++) {
            device_list.push_back(device_info(devs[idx], idx));
            device_handles.push_back(devs[idx]);
        }

        for (uint32_t idx = 0; idx < device_count; idx++) {
            auto dev = devs[idx];
            uint32_t subdevice_count{};
            ZE_CALL(zeDeviceGetSubDevices, (dev, &subdevice_count, nullptr));
            std::vector<ze_device_handle_t> subdevs(subdevice_count);
            ZE_CALL(zeDeviceGetSubDevices, (dev, &subdevice_count, subdevs.data()));

            for (uint32_t subdev_idx = 0; subdev_idx < subdevice_count; subdev_idx++) {
                device_list.push_back(device_info(subdevs[subdev_idx], idx));
                device_handles.push_back(subdevs[subdev_idx]);
            }
        }
    }
    LOG_DEBUG("found devices: ", device_list.size());

    cache = std::make_unique<ze::cache>(global_data::env().worker_count);

    LOG_INFO("initialized level-zero");
}

global_data_desc::~global_data_desc() {
    LOG_INFO("finalizing level-zero");

    if (!global_data::env().ze_fini_wa) {
        cache.reset();
        for (auto& context : context_list) {
            ZE_CALL(zeContextDestroy, (context));
        }
    }
    else {
        LOG_INFO("skip level-zero finalization");
    }

    context_list.clear();
    device_list.clear();
    device_handles.clear();
    driver_list.clear();

    ze_api_fini();

    LOG_INFO("finalized level-zero");
}

} // namespace ze
} // namespace ccl
