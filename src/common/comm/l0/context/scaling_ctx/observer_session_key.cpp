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
#include "common/comm/l0/context/scaling_ctx/observer_session_key.hpp"

namespace native {
namespace observer {

bool session_key::operator<(const session_key& other) const noexcept {
    return hash < other.hash;
}

std::string session_key::to_string() const {
    return std::to_string(hash);
}

} // namespace observer
} // namespace native
