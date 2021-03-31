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
#include <cassert>
#include <functional>

#include "oneapi/ccl/native_device_api/l0/base_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/driver.hpp"
#include "oneapi/ccl/native_device_api/l0/subdevice.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "common/log/log.hpp"

namespace native {

uint32_t get_subdevice_properties_from_handle(ccl_device::handle_t handle) {
    ze_device_properties_t device_properties;
    ze_result_t ret = zeDeviceGetProperties(handle, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("zeDeviceGetProperties failed, error: " + native::to_string(ret));
    }
    if (!(device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)) {
        CCL_THROW("invalid device type, got device, but subdevice requested");
    }
    return device_properties.subdeviceId;
}

CCL_BE_API
std::shared_ptr<ccl_subdevice> ccl_subdevice::create(handle_t handle,
                                                     owner_ptr_t&& device,
                                                     base::owner_ptr_t&& driver) {
    auto ctx = driver.lock()->get_driver_contexts();
    std::shared_ptr<ccl_subdevice> subdevice = std::make_shared<ccl_subdevice>(
        handle, std::move(device), std::move(driver), std::move(ctx));
    return subdevice;
}

CCL_BE_API
ccl_subdevice::indexed_handles ccl_subdevice::get_handles(
    const ccl_device& device,
    const ccl::device_indices_type& requested_indices) {
    uint32_t subdevices_count = 0;
    ze_result_t err = zeDeviceGetSubDevices(device.get(), &subdevices_count, nullptr);
    if (err != ZE_RESULT_SUCCESS) {
        CCL_THROW("zeDeviceGetSubDevices failed, error: " + native::to_string(err));
    }

    std::vector<ccl_subdevice::handle_t> handles;
    handles.resize(subdevices_count);

    err = zeDeviceGetSubDevices(device.get(), &subdevices_count, handles.data());
    if (err != ZE_RESULT_SUCCESS) {
        CCL_THROW("zeDeviceGetSubDevices failed for device request, error: " +
                  native::to_string(err));
    }

    //filter indices
    ccl::device_index_type owner_path = device.get_device_path();
    ccl::device_indices_type filtered_ids;
    if (!requested_indices.empty()) {
        for (const auto& index : requested_indices) {
            if ((std::get<ccl::device_index_enum::driver_index_id>(index) ==
                 std::get<ccl::device_index_enum::driver_index_id>(owner_path)) and
                (std::get<ccl::device_index_enum::device_index_id>(index) ==
                 std::get<ccl::device_index_enum::device_index_id>(owner_path))) {
                filtered_ids.insert(index);
            }
        }
        if (filtered_ids.empty()) {
            CCL_THROW("failed, nothing to get");
        }
    }

    //collect subdevice by indices
    indexed_handles ret;
    try {
        ret = detail::collect_indexed_data<ccl::device_index_enum::subdevice_index_id>(
            filtered_ids,
            handles,
            std::bind(get_subdevice_properties_from_handle, std::placeholders::_1));
    }
    catch (const std::exception& ex) {
        CCL_THROW(std::string("cannot add subdevice: ") + ex.what());
    }
    return ret;
}

void ccl_subdevice::initialize_subdevice_data() {
    ze_result_t ret = zeDeviceGetProperties(handle, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot get properties for subdevice, error: " + native::to_string(ret));
    }
}

CCL_BE_API
ccl_subdevice::ccl_subdevice(handle_t h,
                             owner_ptr_t&& device,
                             base::owner_ptr_t&& driver,
                             base::context_ptr_t&& ctx,
                             std::false_type)
        : base(h, std::move(driver), std::move(ctx), std::false_type{}),
          parent_device(std::move(device)) {}

CCL_BE_API
ccl_subdevice::ccl_subdevice(handle_t h,
                             owner_ptr_t&& device,
                             base::owner_ptr_t&& driver,
                             base::context_ptr_t&& ctx)
        : //  my_enable_shared_from_this<ccl_subdevice>(),
          base(h, std::move(driver), std::move(ctx)),
          parent_device(std::move(device)) {
    initialize_subdevice_data();
}

CCL_BE_API
ccl_subdevice::~ccl_subdevice() {
    //TODO think about orphant device

    std::shared_ptr<ccl_device> device = parent_device.lock();
    ze_context_handle_t ctxtmp;
    if (device) {
        // no need to notify driver, because ccl_device owns ccl_subdevice
        device->on_delete(handle, ctxtmp);
        device->release();
    }
}

CCL_BE_API
bool ccl_subdevice::is_subdevice() const noexcept {
    return true;
}

CCL_BE_API
ccl::index_type CCL_BE_API ccl_subdevice::get_device_id() const {
    assert((device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) && "Must be subdevice");
    return device_properties.subdeviceId;
}

CCL_BE_API
ccl::device_index_type CCL_BE_API ccl_subdevice::get_device_path() const {
    const auto device = parent_device.lock();
    if (!device) {
        CCL_THROW("cannot get get_device_path() because ccl_subdevice has no owner");
    }

    ccl::device_index_type suddevice_path = device->get_device_path();
    std::get<ccl::device_index_enum::subdevice_index_id>(suddevice_path) = get_device_id();
    return suddevice_path;
}

CCL_BE_API
std::string ccl_subdevice::to_string(const std::string& prefix) const {
    std::stringstream ss;
    ss << prefix << "SubdDevice: " << handle << std::endl;
    ss << ccl_device::to_string(prefix);
    return ss.str();
}

CCL_BE_API
std::weak_ptr<ccl_subdevice> ccl_subdevice::deserialize(
    const uint8_t** data,
    size_t& size,
    std::shared_ptr<ccl_device_platform>& out_platform) {
    //restore device
    auto device = ccl_device::deserialize(data, size, out_platform).lock();
    if (!device) {
        CCL_THROW("cannot deserialize ccl_subdevice, because owner is nullptr");
    }

    if (!device->is_subdevice()) {
        CCL_THROW("is not a subdevice");
    }

    return std::static_pointer_cast<ccl_subdevice>(device);
}

CCL_BE_API
std::ostream& operator<<(std::ostream& out, const ccl_subdevice& node) {
    out << "SubDevice: " << node.handle << "\n"
        << "parent device: "
        << "TODO\n"
        << node.to_string() << std::endl;
    return out;
}
} // namespace native
