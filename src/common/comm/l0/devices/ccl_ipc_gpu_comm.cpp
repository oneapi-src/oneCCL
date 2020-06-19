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
#include <iostream>
#include <vector>
#include <set>
#include "common/comm/l0/devices/ccl_ipc_gpu_comm.hpp"
#include "sched/sched.hpp"
#include "sched/entry/l0/l0_entry.hpp"
#include "common/comm/l0/modules/specific_modules_source_data.hpp"

namespace native
{

ccl_ipc_gpu_comm::ccl_ipc_gpu_comm(ccl_device& assigned_device, size_t idx, size_t size,
                                   ccl::device_topology_type topology_type) :
        base(assigned_device, idx)
{
    /* No queue or other device-related primitives creation
     * Because related device belong to the different processes
     */
    //compile and load modules from all sources
    load_modules(specific_modules_source_data_storage::instance());

    //register in topology
    switch(topology_type)
    {
        case ccl::device_topology_type::allied_process_group_ring:
        {
            reset_rank<ccl::device_topology_type::allied_process_group_ring>(idx, size);
            break;
        }
        case ccl::device_topology_type::process_group_torn_apart_ring:
        {
            reset_rank<ccl::device_topology_type::process_group_torn_apart_ring>(idx, size);
            break;
        }
        default:
        {
            throw std::runtime_error(std::string("ccl_ipc_gpu_comm must be created") +
                                     "for process-based topology, but requested: " +
                                     std::to_string(topology_type));
        }
    }


}


std::string ccl_ipc_gpu_comm::to_string_impl() const
{
    std::string ret(name_impl());
    ret = ret + ", comm: " + comm_to_str();
    return ret;
}
}
