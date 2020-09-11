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

#include "native_device_api/l0/driver.hpp"
#include "native_device_api/l0/utils.hpp"

namespace native {
struct ccl_device_platform : std::enable_shared_from_this<ccl_device_platform> {
    using driver_ptr = std::shared_ptr<ccl_device_driver>;
    using const_driver_ptr = std::shared_ptr<ccl_device_driver>;
    using driver_storage_type = std::map<ccl::index_type, driver_ptr>;
    using device_affinity_per_driver = std::map<size_t, ccl::device_mask_t>;

    //void init_drivers(const device_affinity_per_driver& affinities / * = device_affinity_per_driver()* /);
    void init_drivers(const ccl::device_indices_t& indices = ccl::device_indices_t());

    std::shared_ptr<ccl_device_platform> get_ptr() {
        return this->shared_from_this();
    }

    const_driver_ptr get_driver(ccl::index_type index) const;
    driver_ptr get_driver(ccl::index_type index);

    const driver_storage_type& get_drivers() const noexcept;

    ccl_device_driver::device_ptr get_device(const ccl::device_index_type& path);
    ccl_device_driver::const_device_ptr get_device(const ccl::device_index_type& path) const;

    std::string to_string() const;
    void on_delete(ccl_device_driver::handle_t& driver_handle);

    static std::shared_ptr<ccl_device_platform> create(
        const ccl::device_indices_t& indices = ccl::device_indices_t());
    //static std::shared_ptr<ccl_device_platform> create(const device_affinity_per_driver& affinities);

    details::adjacency_matrix calculate_device_access_metric(
        const ccl::device_indices_t& indices = ccl::device_indices_t(),
        details::p2p_rating_function func = details::binary_p2p_rating_calculator) const;

private:
    ccl_device_platform();

    driver_storage_type drivers;
};

//extern std::shared_ptr<ccl_device_platform> global_platform;
ccl_device_platform& get_platform();

ccl_device_platform::driver_ptr get_driver(size_t index = 0);
} // namespace native
