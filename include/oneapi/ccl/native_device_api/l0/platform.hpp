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
#include <unistd.h>

#include "oneapi/ccl/native_device_api/l0/driver.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"
#include "oneapi/ccl/native_device_api/l0/utils.hpp"

namespace native {
struct ccl_device_platform : std::enable_shared_from_this<ccl_device_platform> {
    using driver_ptr = std::shared_ptr<ccl_device_driver>;
    using const_driver_ptr = std::shared_ptr<ccl_device_driver>;
    using driver_storage_type = std::map<ccl::index_type, driver_ptr>;
    using device_affinity_per_driver = std::map<size_t, ccl::device_mask_t>;
    using context_storage_type = std::shared_ptr<ccl_context_holder>;

    using platform_id_type = size_t;

    void init_drivers(const ccl::device_indices_type& indices = ccl::device_indices_type());

    std::shared_ptr<ccl_device_platform> get_ptr() {
        return this->shared_from_this();
    }

    const_driver_ptr get_driver(ccl::index_type index) const;
    driver_ptr get_driver(ccl::index_type index);

    const driver_storage_type& get_drivers() const noexcept;

    ccl_device_driver::device_ptr get_device(const ccl::device_index_type& path);
    ccl_device_driver::const_device_ptr get_device(const ccl::device_index_type& path) const;

    std::shared_ptr<ccl_context> create_context(std::shared_ptr<ccl_device_driver> driver);
    context_storage_type get_platform_contexts();

    std::string to_string() const;
    void on_delete(ccl_device_driver::handle_t& driver_handle, ze_context_handle_t& ctx);
    void on_delete(ccl_context::handle_t& context, ze_context_handle_t& ctx);

    static std::shared_ptr<ccl_device_platform> create(
        const ccl::device_indices_type& indices = ccl::device_indices_type());

    detail::adjacency_matrix calculate_device_access_metric(
        const ccl::device_indices_type& indices = ccl::device_indices_type(),
        detail::p2p_rating_function func = detail::binary_p2p_rating_calculator) const;

    // serialize/deserialize
    static constexpr size_t get_size_for_serialize() {
        return sizeof(pid_t) + sizeof(pid_t) + sizeof(platform_id_type);
    }

    static std::weak_ptr<ccl_device_platform> deserialize(
        const uint8_t** data,
        size_t& size,
        std::shared_ptr<ccl_device_platform>& out_platform);
    size_t serialize(std::vector<uint8_t>& out, size_t from_pos, size_t expected_size) const;

    platform_id_type get_id() const noexcept;
    pid_t get_pid() const noexcept;

    static CCL_BE_API ccl_device_platform& get_platform();

private:
    ccl_device_platform(platform_id_type platform_id = 0);

    std::shared_ptr<ccl_device_platform> clone(platform_id_type id, pid_t foreign_pid) const;

    driver_storage_type drivers;
    context_storage_type context;

    platform_id_type id;
    pid_t pid;
};

extern CCL_BE_API ccl_device_platform& get_platform();

extern CCL_BE_API ccl_device_platform::driver_ptr get_driver(size_t index = 0);
} // namespace native
