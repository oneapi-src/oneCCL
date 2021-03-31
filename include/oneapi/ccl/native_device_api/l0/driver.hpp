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

#include <iostream>
#include <map>
#include <memory>

#include "oneapi/ccl/native_device_api/l0/base.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"

namespace native {
struct ccl_device_platform;
struct ccl_device_driver;
struct ccl_device;
struct ccl_context;
struct ccl_context_holder;
struct ccl_device_driver
        : public cl_base<ze_driver_handle_t, ccl_device_platform, ccl_context_holder>,
          std::enable_shared_from_this<ccl_device_driver> {
    friend std::ostream& operator<<(std::ostream&, const ccl_device_driver&);

    using base = cl_base<ze_driver_handle_t, ccl_device_platform, ccl_context_holder>;
    using handle_t = base::handle_t;
    using device_ptr = std::shared_ptr<ccl_device>;
    using const_device_ptr = std::shared_ptr<const ccl_device>;
    using base::handle;
    using base::owner_ptr_t;

    using base::get;

    using context_storage_type = std::shared_ptr<ccl_context_holder>;
    using devices_storage_type = std::map<ccl::index_type, device_ptr>;
    using indexed_driver_handles = indexed_storage<handle_t>;

    using driver_index_type = uint32_t;

    ccl_device_driver(handle_t h,
                      driver_index_type id,
                      owner_ptr_t&& platform,
                      std::weak_ptr<ccl_context_holder>&& ctx);

    static indexed_driver_handles get_handles(
        const ccl::device_indices_type& requested_driver_indexes = ccl::device_indices_type());
    static std::shared_ptr<ccl_device_driver> create(
        handle_t h,
        driver_index_type id,
        owner_ptr_t&& platform,
        const ccl::device_mask_t& rank_device_affinity);

    static std::shared_ptr<ccl_device_driver> create(
        handle_t h,
        driver_index_type id,
        owner_ptr_t&& platform,
        const ccl::device_indices_type& rank_device_affinity = ccl::device_indices_type());

    std::shared_ptr<ccl_device_driver> get_ptr() {
        return this->shared_from_this();
    }

    context_storage_type get_driver_contexts();
    context_storage_type get_driver_contexts() const;

    driver_index_type get_driver_id() const noexcept;
    ccl::device_index_type get_driver_path() const noexcept;

    ze_driver_properties_t get_properties() const;
    const devices_storage_type& get_devices() const noexcept;
    device_ptr get_device(const ccl::device_index_type& path);
    const_device_ptr get_device(const ccl::device_index_type& path) const;

    std::shared_ptr<ccl_context> create_context();
    std::shared_ptr<ccl_context> create_context_from_handle(ccl_context::handle_t);

    std::string to_string(const std::string& prefix = std::string()) const;

    // ownership release
    void on_delete(ze_device_handle_t& sub_device_handle, ze_context_handle_t& ctx);

    // serialize/deserialize
    static constexpr size_t get_size_for_serialize() {
        return /*owner_t::get_size_for_serialize()*/
            sizeof(int) + sizeof(int) + sizeof(size_t) + detail::serialize_device_path_size;
    }

    static std::weak_ptr<ccl_device_driver> deserialize(
        const uint8_t** data,
        size_t& size,
        std::shared_ptr<ccl_device_platform>& out_platform,
        ccl::device_index_type& out_device_path);
    size_t serialize(std::vector<uint8_t>& out, size_t from_pos, size_t expected_size) const;

    // utility
    static ccl::device_mask_t create_device_mask(const std::string& str_mask,
                                                 std::ios_base::fmtflags flag = std::ios_base::hex);
    static ccl::device_indices_type get_device_indices(const ccl::device_mask_t& mask);
    static ccl::device_mask_t get_device_mask(const ccl::device_indices_type& device_idx);

    driver_index_type driver_id;

private:
    devices_storage_type devices;
    ze_driver_ipc_properties_t ipc_prop;
};
/*
template <class DeviceType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type = 0>
ccl_device_driver::device_ptr get_runtime_device(const DeviceType& device);

template <class DeviceType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type = 0>
ccl_device_driver::device_ptr get_runtime_device(DeviceType device);
*/
} // namespace native
