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

#include "oneapi/ccl/types.hpp"
#include "coll/algorithms/algorithms_enum.hpp"
#include "internal_types.hpp"

#ifdef MULTI_GPU_SUPPORT
ccl::status load_gpu_module(const std::string& path,
                            ccl::device_topology_type topo_type,
                            ccl_coll_type coll_type);
#endif //MULTI_GPU_SUPPORT
