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

#include "native_device_api/l0/base_impl.hpp"
#include "native_device_api/l0/device.hpp"
#include "native_device_api/l0/subdevice.hpp"
#include "native_device_api/l0/primitives_impl.hpp"

namespace native {

uint32_t get_subdevice_properties_from_handle(ccl_device::handle_t handle) {
    ze_device_properties_t device_properties;
    device_properties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
    ze_result_t ret = zeDeviceGetProperties(handle, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("zeDeviceGetProperties failed, error: ") +
                                 native::to_string(ret));
    }

    if (!device_properties.isSubdevice) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 "- invalid device type, got device, but subdevice requested");
    }
    return device_properties.deviceId;
}

CCL_API
std::shared_ptr<ccl_subdevice> ccl_subdevice::create(handle_t handle,
                                                     owner_ptr_t&& device,
                                                     base::owner_ptr_t&& driver) {
    std::shared_ptr<ccl_subdevice> subdevice =
        std::make_shared<ccl_subdevice>(handle, std::move(device), std::move(driver));
    return subdevice;
}

CCL_API
ccl_subdevice::indexed_handles ccl_subdevice::get_handles(
    const ccl_device& device,
    const ccl::device_indices_t& requested_indices) {
    uint32_t subdevices_count = 0;
    ze_result_t err = zeDeviceGetSubDevices(device.get(), &subdevices_count, nullptr);
    if (err != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 "zeDeviceGetSubDevices failed, error: " + native::to_string(err));
    }

    std::vector<ccl_subdevice::handle_t> handles;
    handles.resize(subdevices_count);

    err = zeDeviceGetSubDevices(device.get(), &subdevices_count, handles.data());
    if (err != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string(__FUNCTION__) +
            "zeDeviceGetSubDevices failed for device request, error: " + native::to_string(err));
    }

    //filter indices
    ccl::device_index_type owner_path = device.get_device_path();
    ccl::device_indices_t filtered_ids;
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
            throw std::runtime_error(std::string(__FUNCTION__) + " - Failed, nothing to get");
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
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - Cannot add subdevice: " + ex.what());
    }
    return ret;
}

void ccl_subdevice::initialize_subdevice_data() {
    ze_result_t ret = zeDeviceGetProperties(handle, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot get properties for subdevice, error: ") +
                                 native::to_string(ret));
    }
}

CCL_API
ccl_subdevice::ccl_subdevice(handle_t h,
                             owner_ptr_t&& device,
                             base::owner_ptr_t&& driver,
                             std::false_type)
        : base(h, std::move(driver), std::false_type{}),
          parent_device(std::move(device)) {}

CCL_API
ccl_subdevice::ccl_subdevice(handle_t h, owner_ptr_t&& device, base::owner_ptr_t&& driver)
        : //  my_enable_shared_from_this<ccl_subdevice>(),
          base(h, std::move(driver)),
          parent_device(std::move(device)) {
    initialize_subdevice_data();
}

CCL_API
ccl_subdevice::~ccl_subdevice() {
    //TODO think about orphant device

    std::shared_ptr<ccl_device> device = parent_device.lock();
    if (device) {
        // no need to notify driver, because ccl_device owns ccl_subdevice
        device->on_delete(handle);
        device->release();
    }
}

CCL_API
bool ccl_subdevice::is_subdevice() const noexcept {
    return true;
}

CCL_API
ccl::index_type CCL_API ccl_subdevice::get_device_id() const {
    assert(device_properties.isSubdevice && "Must be subdevice");
    return device_properties.subdeviceId;
}

CCL_API
ccl::device_index_type CCL_API ccl_subdevice::get_device_path() const {
    const auto device = parent_device.lock();
    if (!device) {
        throw std::runtime_error("cannot get get_device_path() because ccl_subdevice has no owner");
    }

    ccl::device_index_type suddevice_path = device->get_device_path();
    std::get<ccl::device_index_enum::subdevice_index_id>(suddevice_path) = get_device_id();
    return suddevice_path;
}

CCL_API
std::string ccl_subdevice::to_string(const std::string& prefix) const {
    std::stringstream ss;
    ss << prefix << "SubdDevice: " << handle << std::endl;
    ss << ccl_device::to_string(prefix);
    return ss.str();
}

CCL_API
size_t ccl_subdevice::serialize(std::vector<uint8_t>& out,
                                size_t from_pos,
                                size_t expected_size) const {
    // check parent existence
    const auto device = parent_device.lock();
    if (!device) {
        throw std::runtime_error("cannot serialize ccl_subdevice without owner");
    }

    constexpr size_t expected_device_bytes = sizeof(device_properties.subdeviceId);
    size_t serialized_bytes = device->serialize(
        out, from_pos, expected_device_bytes + expected_size); //resize vector inside

    // serialize from position
    uint8_t* data_start = out.data() + from_pos + serialized_bytes;
    *(reinterpret_cast<decltype(device_properties.subdeviceId)*>(data_start)) = get_device_id();
    serialized_bytes += expected_device_bytes;

    return serialized_bytes;
}

CCL_API
std::weak_ptr<ccl_subdevice> ccl_subdevice::deserialize(const uint8_t** data,
                                                        size_t& size,
                                                        ccl_device_platform& platform) {
    //restore driver
    auto device = ccl_device::deserialize(data, size, platform).lock();
    if (!device) {
        throw std::runtime_error("cannot deserialize ccl_subdevice, because owner is nullptr");
    }

    constexpr size_t expected_bytes = sizeof(device_properties.subdeviceId);
    if (size < expected_bytes) {
        throw std::runtime_error("cannot deserialize ccl_device, not enough data");
    }

    //restore subdevice index
    decltype(device_properties.subdeviceId) recovered_handle =
        *(reinterpret_cast<const decltype(device_properties.subdeviceId)*>(*data));
    size -= expected_bytes;
    *data += expected_bytes;

    //find subdevice device with requested handle
    const auto& subdevices = device->get_subdevices();

    auto it = std::find_if(
        subdevices.begin(),
        subdevices.end(),
        [recovered_handle](const typename ccl_device::sub_devices_container_type::value_type& sub) {
            return sub.second->get_device_id() == recovered_handle;
        });

    if (it == subdevices.end()) {
        throw std::runtime_error(
            std::string("cannot deserialize ccl_subdevice: orphant subddevice: ") +
            std::to_string(recovered_handle));
    }
    return it->second;
}

CCL_API
std::ostream& operator<<(std::ostream& out, const ccl_subdevice& node) {
    out << "SubDevice: " << node.handle << "\n"
        << "parent device: "
        << "TODO\n"
        << node.to_string() << std::endl;
    return out;
}
} // namespace native
