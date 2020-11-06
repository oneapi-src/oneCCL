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
#include "common/event/event_internal/event_internal_impl.hpp"

namespace ccl {

event_internal::event_internal(event_internal&& src) : base_t(std::move(src)) {}

event_internal::event_internal(impl_value_t&& impl) : base_t(std::move(impl)) {}

event_internal::~event_internal() {}

event_internal& event_internal::operator=(event_internal&& src) {
    if (src.get_impl() != this->get_impl()) {
        src.get_impl().swap(this->get_impl());
        src.get_impl().reset();
    }
    return *this;
}

void event_internal::build_from_params() {
    get_impl()->build_from_params();
}
} // namespace ccl
