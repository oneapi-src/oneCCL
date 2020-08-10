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

#include "ccl_types.hpp"
#include "ccl_type_traits.hpp"

namespace native {

struct ccl_device;
namespace details {

/*
 * Boolean matrix represents P2P device capable connectivity 'cross_device_rating'
 * Left process GPU IDs grows by rows --->
 * Right process GPu IDs grows by columns \/
 */
using cross_device_rating = size_t;
using adjacency_list = std::map<ccl::device_index_type, cross_device_rating>;
struct adjacency_matrix : std::map<ccl::device_index_type, adjacency_list> {
    using base = std::map<ccl::device_index_type, adjacency_list>;
    adjacency_matrix() = default;
    adjacency_matrix(adjacency_matrix&&) = default;
    adjacency_matrix(const adjacency_matrix&) = default;
    adjacency_matrix& operator=(const adjacency_matrix&) = default;
    adjacency_matrix& operator=(adjacency_matrix&&) = default;
    adjacency_matrix(std::initializer_list<typename base::value_type> init);
    ~adjacency_matrix() = default;
};

/* 
 * Functor for calculation peer-to-peer device access capability:
 * Receives left-hand-side and right-hand-side devices 
 * Return cross_device_rating score
 */
using p2p_rating_function =
    std::function<cross_device_rating(const ccl_device&, const ccl_device&)>;

cross_device_rating binary_p2p_rating_calculator(const ccl_device& lhs, const ccl_device& rhs);
} // namespace details
} // namespace native
