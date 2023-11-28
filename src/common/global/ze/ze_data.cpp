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
#include "common/api_wrapper/ze_api_wrapper.hpp"

namespace ccl {
namespace ze {

device_info::device_info(ze_device_handle_t dev, uint32_t parent_idx)
        : device(dev),
          parent_idx(parent_idx),
          physical_idx(fd_manager::invalid_physical_idx) {
    ze_device_properties_t dev_props = ccl::ze::default_device_props;
    zeDeviceGetProperties(device, &dev_props);
    uuid = dev_props.uuid;

#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    ze_pci_ext_properties_t pci_prop = ccl::ze::default_pci_property;
    ze_result_t ret = zeDevicePciGetPropertiesExt(dev, &pci_prop);
    if (ret == ZE_RESULT_SUCCESS) {
        pci = pci_prop.address;
    }
#endif // ZE_PCI_PROPERTIES_EXT_NAME
};

global_data_desc::global_data_desc() {
    LOG_INFO("initializing level-zero");

    // enables driver initialization and
    // dependencies for system management
    setenv("ZES_ENABLE_SYSMAN", "1", 1);

    ZE_CALL(zeInit, (ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count{};
    ZE_CALL(zeDriverGet, (&driver_count, nullptr));
    drivers.resize(driver_count);

    ZE_CALL(zeDriverGet, (&driver_count, drivers.data()));
    LOG_DEBUG("found drivers: ", drivers.size());

    CCL_THROW_IF_NOT(!drivers.empty(), "no ze drivers");

    contexts.resize(drivers.size());
    for (size_t i = 0; i < drivers.size(); ++i) {
        ze_context_desc_t desc = ze::default_context_desc;
        ZE_CALL(zeContextCreate, (drivers.at(i), &desc, &contexts.at(i)));
        CCL_THROW_IF_NOT(contexts[i], "ze context is null");

        uint32_t device_count{};
        ZE_CALL(zeDeviceGet, (drivers.at(i), &device_count, nullptr));
        std::vector<ze_device_handle_t> devs(device_count);
        ZE_CALL(zeDeviceGet, (drivers.at(i), &device_count, devs.data()));

        for (uint32_t idx = 0; idx < device_count; idx++) {
            devices.push_back(device_info(devs[idx], idx));
        }

        for (uint32_t idx = 0; idx < device_count; idx++) {
            auto dev = devs[idx];

            uint32_t subdevice_count{};
            ZE_CALL(zeDeviceGetSubDevices, (dev, &subdevice_count, nullptr));
            std::vector<ze_device_handle_t> subdevs(subdevice_count);
            ZE_CALL(zeDeviceGetSubDevices, (dev, &subdevice_count, subdevs.data()));

            for (uint32_t subdev_idx = 0; subdev_idx < subdevice_count; subdev_idx++) {
                devices.push_back(device_info(subdevs[subdev_idx], idx));
            }
        }
    }
    LOG_DEBUG("found devices: ", devices.size());

    cache = std::make_unique<ze::cache>(global_data::env().worker_count);

    topo_manager::detect_tune_port_count(devices);

    init_ipc_exchange_mode();

    LOG_INFO("initialized level-zero");
}

global_data_desc::~global_data_desc() {
    LOG_INFO("finalizing level-zero");

    if (!global_data::env().ze_fini_wa) {
        cache.reset();
        for (auto& context : contexts) {
            ZE_CALL(zeContextDestroy, (context));
        }
    }
    else {
        LOG_INFO("skip level-zero finalization");
    }

    contexts.clear();
    devices.clear();
    drivers.clear();

    LOG_INFO("finalized level-zero");
}

void global_data_desc::init_ipc_exchange_mode() {
    if (global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::pidfd &&
        ze::fd_manager::is_pidfd_supported()) {
        LOG_DEBUG("pidfd exchange mode is verified successfully");
    }
#ifdef CCL_ENABLE_DRM
    else if (global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::drmfd) {
        fd_manager = std::make_unique<ze::fd_manager>();
        // update physical_idx for each logical device, by default it is invalid
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
        for (size_t idx = 0; idx < devices.size(); idx++) {
            devices[idx].physical_idx = ccl::ze::fd_manager::get_physical_device_idx(
                fd_manager->get_physical_devices(), devices[idx].pci);
        }
#endif // ZE_PCI_PROPERTIES_EXT_NAME
        LOG_DEBUG("drmfd exchange mode is verified successfully");
    }
#endif // CCL_ENABLE_DRM
    else if (global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::none) {
        LOG_WARN("CCL_ZE_IPC_EXCHANGE is set to none."
                 " It will fail with GPU buffers and topo algorithms");
    }
    else if (global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets) {
        // TODO: remove ipc_exchange_mode::none, and warning when MLSL-2078 is done
        LOG_WARN("sockets exchange mode is set. It may cause"
                 " potential problem of 'Too many open file descriptors'");
    }
    else {
        // we must use std::cerr to see the error message because
        // comm_selector.cpp:57 create_comm_impl: EXCEPTION: ze_data was not initialized
        // has higher priority of printing the its error message
        std::cerr << "ERROR:  unexpected ipc exchange mode" << std::endl;
        throw std::runtime_error(std::string(__FUNCTION__));
    }
}

} // namespace ze
} // namespace ccl
