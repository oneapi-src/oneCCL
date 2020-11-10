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
#include "common/comm/l0/modules/ring/alltoallv_export_functions.hpp"
#include "common/comm/l0/modules/gpu_typed_module.hpp"

namespace native {

DEFINE_SPECIFIC_GPU_MODULE_CLASS(device_coll_module,
                                 real_gpu_typed_module,
                                 ccl_coll_alltoallv,
                                 ccl::device_topology_type::ring,
                                 ring_alltoallv_kernel,
                                 ring_alltoallv_numa_kernel);

DEFINE_SPECIFIC_GPU_MODULE_CLASS(ipc_dst_device_coll_module,
                                 ipc_gpu_typed_module,
                                 ccl_coll_alltoallv,
                                 ccl::device_topology_type::ring,
                                 ring_alltoallv_ipc,
                                 ring_alltoallv_ipc);

DEFINE_VIRTUAL_GPU_MODULE_CLASS(ccl_coll_alltoallv,
                                ccl::device_topology_type::ring,
                                ring_alltoallv_kernel,
                                ring_alltoallv_numa_kernel);
} // namespace native
