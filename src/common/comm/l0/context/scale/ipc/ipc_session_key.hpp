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
#include <functional>
#include <string>
#include <vector>

#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "coll/algorithms/algorithms_enum.hpp"
#include "coll/coll_param.hpp"

namespace native {

template <ccl_coll_type type>
struct ipc_invoke_params {
    ipc_invoke_params(std::vector<ccl_device::device_ipc_memory_handle>&& h,
                      const coll_param_gpu& params)
            : handles(std::move(h)),
              params{ params } {}

    static constexpr ccl_coll_type get_coll_type() {
        return type;
    }

    const coll_param_gpu& get_kernel_params() const {
        return params;
    }

    std::vector<ccl_device::device_ipc_memory_handle> handles;
    // TODO: can we guarantee that this object is not destroyed before l0 entry and
    // use const& here?
    coll_param_gpu params;
};

struct ipc_session_key {
    using hash_core_t = size_t;

    friend std::ostream& operator<<(std::ostream& out, const ipc_session_key& key) {
        out << key.to_string();
        return out;
    }

    template <class T>
    ipc_session_key(const T* src) : hash(std::hash<const T*>{}(src)) {}

    bool operator<(const ipc_session_key& other) const noexcept;

    std::string to_string() const;

private:
    hash_core_t hash;
};
} // namespace native
