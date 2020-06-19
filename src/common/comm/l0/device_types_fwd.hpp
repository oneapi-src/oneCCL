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

namespace native
{

class ccl_gpu_comm;
class ccl_ipc_gpu_comm;
class ccl_virtual_gpu_comm;
template<class device_t>
class ccl_thread_comm;
template<class device_t>
class ccl_ipc_source_gpu_comm;
template<class device_t>
class ccl_gpu_scaleup_proxy;

#define SUPPORTED_DEVICES_DECL_LIST         ccl_gpu_comm, ccl_virtual_gpu_comm,                     \
                                            ccl_thread_comm<ccl_gpu_comm>,                          \
                                            ccl_thread_comm<ccl_virtual_gpu_comm>,                  \
                                            ccl_ipc_source_gpu_comm<ccl_gpu_comm>,                  \
                                            ccl_ipc_source_gpu_comm<ccl_virtual_gpu_comm>,          \
                                            ccl_ipc_gpu_comm,                                       \
                                            ccl_gpu_scaleup_proxy<ccl_gpu_comm>,                    \
                                            ccl_gpu_scaleup_proxy<ccl_virtual_gpu_comm>

}
