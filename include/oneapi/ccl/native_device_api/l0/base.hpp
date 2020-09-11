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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ze_api.h>

#ifndef UT
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#endif

namespace native {
/**
 * Base RAII L0 handles wrappper
 * support serialize/deserialize concept
 */
template <class handle_type, class resource_owner>
class cl_base {
public:
    friend resource_owner;
    using self_t = cl_base<handle_type, resource_owner>;
    using handle_t = handle_type;
    using owner_t = resource_owner;
    using owner_ptr_t = std::weak_ptr<resource_owner>;

    cl_base(cl_base&& src) noexcept;
    cl_base& operator=(cl_base&& src) noexcept;
    ~cl_base() noexcept;

    // getter/setters
    std::shared_ptr<self_t> get_instance_ptr();

    handle_t release();
    handle_t& get() noexcept;
    const handle_t& get() const noexcept;
    handle_t* get_ptr() noexcept;
    const handle_t* get_ptr() const noexcept;
    const owner_ptr_t get_owner() const;

    // serialization/deserialization
    static constexpr size_t get_size_for_serialize();

    template <class... helpers>
    size_t serialize(std::vector<uint8_t>& out, size_t from_pos, const helpers&... args) const;

    template <class type, class... helpers>
    static std::shared_ptr<type> deserialize(const uint8_t** data, size_t& size, helpers&... args);

protected:
    cl_base(handle_t h, owner_ptr_t parent);

    handle_t handle;

private:
    owner_ptr_t owner;
};

template <class value_type>
using indexed_storage = std::multimap<uint32_t, value_type>;
std::ostream& operator<<(std::ostream& out, const ccl::device_index_type& index);
} // namespace native
