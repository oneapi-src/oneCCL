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
#include "native_device_api/l0/utils.hpp"
#include "native_device_api/l0/device.hpp"

namespace native {
namespace details {

adjacency_matrix::adjacency_matrix(std::initializer_list<typename base::value_type> init)
        : base(init) {}

cross_device_rating binary_p2p_rating_calculator(const native::ccl_device& lhs,
                                                 const native::ccl_device& rhs,
                                                 size_t weight) {
    return property_p2p_rating_calculator(lhs, rhs, 1);
}
} // namespace details
} // namespace native
