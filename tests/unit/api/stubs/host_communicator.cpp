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
#include "host_communicator.hpp"

class ccl_comm;
namespace ccl {
host_communicator::host_communicator(std::shared_ptr<ccl_comm> impl) : comm_impl(impl) {}

size_t host_communicator::rank() const {
    return 0;
}

size_t host_communicator::size() const {
    return 1;
}

void host_communicator::barrier_impl() {}

request_t host_communicator::barrier_impl(const barrier_attr& attr) {
    return {};
}
} // namespace ccl
