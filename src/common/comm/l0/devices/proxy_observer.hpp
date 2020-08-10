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
#include "common/comm/l0/device_types.hpp"

namespace native {

template <class device_t>
class proxy_observer {
public:
    using impl = device_t;
    using type_idx_t = typename std::underlying_type<gpu_types>::type;

    static constexpr type_idx_t idx() {
        return impl::type_idx() - gpu_types::SCALING_PROXY_GPU_TYPES;
    }

    template <class... Args>
    void notify(Args&&... args) {
        get_this()->notify_impl(std::forward<Args>(args)...);
    }

    impl* get_this() {
        return static_cast<device_t*>(this);
    }

    const impl* get_this() const {
        return static_cast<const impl*>(this);
    }

private:
};
} // namespace native
