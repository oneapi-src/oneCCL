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
#include "oneapi/ccl/native_device_api/l0/device.hpp"

namespace native {
struct ccl_device_driver;
struct ccl_device;
struct ccl_context;

// TODO not thread-safe!!!
struct ccl_subdevice : public ccl_device {
    using base = ccl_device;
    using handle_t = ccl_device::handle_t;
    using owner_ptr_t = std::weak_ptr<ccl_device>;
    using context_ptr_t = std::weak_ptr<ccl_context>;

    using indexed_handles = indexed_storage<handle_t>;

    friend std::ostream& operator<<(std::ostream&, const ccl_subdevice& node);

    ccl_subdevice(handle_t h, owner_ptr_t&& device, base::owner_ptr_t&& driver, base::context_ptr_t&& ctx);
    virtual ~ccl_subdevice();

    // factory
    static indexed_handles get_handles(
        const ccl_device& device,
        const ccl::device_indices_t& requested_indices = ccl::device_indices_t());
    static std::shared_ptr<ccl_subdevice> create(handle_t h,
                                                 owner_ptr_t&& device,
                                                 base::owner_ptr_t&& driver);

    // properties
    bool is_subdevice() const noexcept override;
    ccl::index_type get_device_id() const override;
    ccl::device_index_type get_device_path() const override;

    // utils
    std::string to_string(const std::string& prefix = std::string()) const;

    // serialize/deserialize
    static constexpr size_t get_size_for_serialize() {
        return /*owner_t::get_size_for_serialize()*/ sizeof(size_t) +
               sizeof(device_properties.subdeviceId);
    }
    size_t serialize(std::vector<uint8_t>& out,
                     size_t from_pos,
                     size_t expected_size) const override;
    static std::weak_ptr<ccl_subdevice> deserialize(const uint8_t** data,
                                                    size_t& size,
                                                    ccl_device_platform& platform);

private:
    ccl_subdevice(handle_t h, owner_ptr_t&& device, base::owner_ptr_t&& driver, base::context_ptr_t&& ctx, std::false_type);
    void initialize_subdevice_data();
    owner_ptr_t parent_device;
};
} // namespace native
