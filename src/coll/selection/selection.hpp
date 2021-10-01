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

#include "coll/selection/selector_wrapper.hpp"

bool ccl_can_use_datatype(ccl_coll_algo algo, const ccl_selector_param& param);

bool ccl_is_direct_algo(const ccl_selector_param& param);
bool ccl_is_topo_ring_algo(const ccl_selector_param& param);
bool ccl_can_use_topo_ring_algo(const ccl_selector_param& param);
