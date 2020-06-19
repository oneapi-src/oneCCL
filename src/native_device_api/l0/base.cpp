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
#include <algorithm>
#include <sstream>

#include "native_device_api/l0/base_impl.hpp"


namespace native
{

std::string CCL_API to_string(const ze_result_t result)
{
    switch(result)
    {
        case ZE_RESULT_SUCCESS:
            return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_NOT_READY:
            return "ZE_RESULT_NOT_READY";
        case ZE_RESULT_ERROR_UNINITIALIZED:
            return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_DEVICE_LOST:
            return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
            return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:
            return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
            return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
            return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
            return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
//        case  ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS :
//            return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
//        case ZE_RESULT_ERROR_DEVICE_IS_IN_USE :
//            return "ZE_RESULT_ERROR_DEVICE_IS_IN_USE ";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
            return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT ";
        case ZE_RESULT_ERROR_NOT_AVAILABLE:
            return "ZE_RESULT_ERROR_NOT_AVAILABLE ";
        case ZE_RESULT_ERROR_UNKNOWN :
            return "ZE_RESULT_ERROR_UNKNOWN ";
        default:
            throw std::runtime_error(std::string("Unknown ze_result_t value: ") +
                                    std::to_string(static_cast<int>(result)));
    }
    return "";
}

std::string CCL_API to_string(ze_device_type_t type)
{
    switch(type)
    {
        case ZE_DEVICE_TYPE_GPU:
            return "ZE_DEVICE_TYPE_GPU";
        case ZE_DEVICE_TYPE_FPGA:
            return "ZE_DEVICE_TYPE_FPGA";
        default:
            throw std::runtime_error(std::string("Unknown ze_device_type_t value: ") +
                                     std::to_string(static_cast<int>(type)));
    }
    return "";
}

std::string to_string(ze_memory_access_capabilities_t cap)
{
    std::string ret;
    if(cap & ZE_MEMORY_ACCESS_NONE)
    {
        ret += "ZE_MEMORY_ACCESS_NONE";
    }
    if(cap & ZE_MEMORY_ACCESS)
    {
        ret += ret.empty() ? "ZE_MEMORY_ACCESS" : "|ZE_MEMORY_ACCESS";
    }
    if(cap & ZE_MEMORY_ATOMIC_ACCESS)
    {
        ret += ret.empty() ? "ZE_MEMORY_ATOMIC_ACCESS" : "|ZE_MEMORY_ATOMIC_ACCESS";
    }
    if(cap & ZE_MEMORY_CONCURRENT_ACCESS)
    {
        ret += ret.empty() ? "ZE_MEMORY_CONCURRENT_ACCESS" : "|ZE_MEMORY_CONCURRENT_ACCESS";
    }
    if(cap & ZE_MEMORY_CONCURRENT_ATOMIC_ACCESS)
    {
        ret += ret.empty() ? "ZE_MEMORY_CONCURRENT_ATOMIC_ACCESS" : "|ZE_MEMORY_CONCURRENT_ATOMIC_ACCESS";
    }
    return ret;
}

std::string CCL_API to_string(ze_memory_type_t type)
{
    switch(type)
    {
        case ZE_MEMORY_TYPE_UNKNOWN:
            return "ZE_MEMORY_TYPE_UNKNOWN";
        case ZE_MEMORY_TYPE_HOST:
            return "ZE_MEMORY_TYPE_HOST";
        case ZE_MEMORY_TYPE_DEVICE:
            return "ZE_MEMORY_TYPE_DEVICE";
        case ZE_MEMORY_TYPE_SHARED:
            return "ZE_MEMORY_TYPE_SHARED";
        default:
            throw std::runtime_error(std::string("Unknown ze_memory_type_t value: ") +
                                     std::to_string(static_cast<int>(type)));
            break;
    }
    return "";
}

std::string CCL_API to_string(const ze_device_properties_t &device_properties)
{
    std::stringstream ss;
    ss << "name: " << device_properties.name
       << "\ntype: " << native::to_string(device_properties.type)
       << "\nvendor_id: "   << device_properties.vendorId
       << "\ndevice_id: "   << device_properties.deviceId
       << "\nuuid: " << device_properties.uuid.id
       << "\nisSubDevice: " << (bool) device_properties.isSubdevice;

    if(device_properties.isSubdevice)
    {
        ss << "\nsubdevice_id: " << device_properties.subdeviceId;
    }

    ss << "\nunifiedMemorySupported:" << (bool)device_properties.unifiedMemorySupported
       << "\nonDemandPageFaults: " << (bool)device_properties.onDemandPageFaultsSupported
       << "\ncoreClockRate: " << device_properties.coreClockRate
       << "\nmaxCommandQueues: " << device_properties.maxCommandQueues
       << "\nnumAsyncComputeEngines: " << device_properties.numAsyncComputeEngines
       << "\nnumAsyncCopyEngines: " << device_properties.numAsyncCopyEngines
       << "\nmaxCommandQueuePriority: " << device_properties.maxCommandQueuePriority
       << "\nnumThreadsPerEU: " << device_properties.numThreadsPerEU
       << "\nphysicalEUSimdWidth: " << device_properties.physicalEUSimdWidth
       << "\nnumEUsPerSubslice: " << device_properties.numEUsPerSubslice
       << "\nnumSubslicesPerSlice: " << device_properties.numSubslicesPerSlice
       << "\numSlices: " << device_properties.numSlices;
    return ss.str();
}

std::string CCL_API to_string(const ze_device_memory_properties_t &device_mem_properties)
{
    std::stringstream ss;
    ss << "maxClockRate: " << device_mem_properties.maxClockRate
       << "\nmaxlBusWidth: " << device_mem_properties.maxBusWidth
       << "\ntotalSize: " << device_mem_properties.totalSize;
    return ss.str();
}

std::string CCL_API to_string(const ze_device_memory_access_properties_t& mem_access_prop)
{
    std::stringstream ss;
    ss << "hostAllocCapabilities: " << native::to_string(mem_access_prop.hostAllocCapabilities)
       << "\ndeviceAllocCapabilities: " << native::to_string(mem_access_prop.deviceAllocCapabilities)
       << "\nsharedSingleDeviceAllocCapabilities: " << native::to_string(mem_access_prop.sharedSingleDeviceAllocCapabilities)
       << "\nsharedCrossDeviceAllocCapabilities: " << native::to_string(mem_access_prop.sharedCrossDeviceAllocCapabilities)
       << "\nsharedSystemAllocCapabilities: " << native::to_string(mem_access_prop.sharedSystemAllocCapabilities);
    return ss.str();
}

std::string CCL_API to_string(const ze_device_compute_properties_t& compute_properties)
{
    std::stringstream ss;
    ss << "maxTotalGroupSize: " << compute_properties.maxTotalGroupSize
       << "\nmaxGroupSizeX: " << compute_properties.maxGroupSizeX
       << "\nmaxGroupSizeY: " << compute_properties.maxGroupSizeY
       << "\nmaxGroupSizeZ: " << compute_properties.maxGroupSizeZ
       << "\nmaxGroupCountX: " << compute_properties.maxGroupCountX
       << "\nmaxGroupCountY: " << compute_properties.maxGroupCountY
       << "\nmaxGroupCountZ: " << compute_properties.maxGroupCountZ
       << "\nmaxSharedLocalMemory: " << compute_properties.maxSharedLocalMemory
       << "\nnumSubGroupSizes: " << compute_properties.numSubGroupSizes
       << "\nsubGroupSizes: ";
    std::copy(compute_properties.subGroupSizes, compute_properties.subGroupSizes + ZE_SUBGROUPSIZE_COUNT,
              std::ostream_iterator<uint32_t>(ss, ", "));
    return ss.str();
}

std::string CCL_API to_string(const ze_memory_allocation_properties_t &prop)
{
    std::stringstream ss;
    ss << "version: " << prop.version <<
          ", type: "   << to_string(prop.type) <<
          //", device: " << prop.device <<
          ", id: " << prop.id;
    return ss.str();
}

std::string CCL_API to_string(const ze_device_p2p_properties_t& properties)
{
    std::stringstream ss;
    ss << "version: " << properties.version <<
          ", accessSupported: "   << (bool)properties.accessSupported <<
          ", atomicsSupported: " << (bool)properties.atomicsSupported;
    return ss.str();
}

std::string CCL_API to_string(const ze_ipc_mem_handle_t& handle)
{
    std::stringstream ss;
    std::copy(handle.data, handle.data + ZE_MAX_IPC_HANDLE_SIZE,
              std::ostream_iterator<int>(ss, ", "));
    return ss.str();
}
}
