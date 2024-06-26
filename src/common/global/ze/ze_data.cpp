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
    total_threads = dev_props.numThreadsPerEU * dev_props.numEUsPerSubslice *
                    dev_props.numSubslicesPerSlice * dev_props.numSlices;

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
    dev_memory_manager = std::make_unique<ze::device_memory_manager>();

    topo_manager::detect_tune_port_count(devices);

    init_external_pointer_registration();

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

void global_data_desc::init_external_pointer_registration() {
    if (!global_data::env().enable_buffer_cache || drivers.empty()) {
        external_pointer_registration_enabled = false;
        return;
    }
    auto ze_driver_handle = drivers.front();

    ze_result_t res_import = zeDriverGetExtensionFunctionAddress(
        ze_driver_handle,
        "zexDriverImportExternalPointer",
        reinterpret_cast<void**>(&zexDriverImportExternalPointer));
    ze_result_t res_release = zeDriverGetExtensionFunctionAddress(
        ze_driver_handle,
        "zexDriverReleaseImportedPointer",
        reinterpret_cast<void**>(&zexDriverReleaseImportedPointer));
    if ((res_import != ZE_RESULT_SUCCESS) || (res_release != ZE_RESULT_SUCCESS)) {
        // Reset function pointers for safety
        zexDriverImportExternalPointer = nullptr;
        zexDriverReleaseImportedPointer = nullptr;
        external_pointer_registration_enabled = false;
        LOG_INFO("Can not initialize Import Extension API ",
                 "(zexDriverReleaseImportedPointer/zexDriverImportExternalPointer: ",
                 std::to_string(res_import),
                 " ",
                 std::to_string(res_release));
    }
    else {
        external_pointer_registration_enabled = true;
    }
}

void global_data_desc::import_external_pointer(void* ptr, size_t size) {
    CCL_THROW_IF_NOT(external_pointer_registration_enabled);
    auto ze_driver_handle = drivers.front();

    // TODO: maybe use this via SYCL's sycl_ext_oneapi_copy_optimize?
    ze_result_t res = zexDriverImportExternalPointer(ze_driver_handle, ptr, size);
    if (res != ZE_RESULT_SUCCESS) {
        LOG_INFO("zexDriverImportExternalPointer can not register the pointer with error: ",
                 std::to_string(res));
    }
}

void global_data_desc::release_imported_pointer(void* ptr) {
    CCL_THROW_IF_NOT(external_pointer_registration_enabled);
    auto ze_driver_handle = drivers.front();
    ze_result_t res = zexDriverReleaseImportedPointer(ze_driver_handle, ptr);
    if (res != ZE_RESULT_SUCCESS) {
        LOG_INFO("zexDriverReleaseImportPointer can not release the pointer with error: ",
                 std::to_string(res));
    }
}

} // namespace ze
} // namespace ccl
