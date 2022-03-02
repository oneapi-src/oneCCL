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

#include "oneapi/ccl/config.h"

#include <dlfcn.h>

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

#include <ze_api.h>
#include <zes_api.h>

namespace ccl {

typedef struct libze_ops {
    decltype(zeInit) *zeInit;
    decltype(zeDriverGet) *zeDriverGet;
    decltype(zeDriverGetApiVersion) *zeDriverGetApiVersion;
    decltype(zeMemGetAllocProperties) *zeMemGetAllocProperties;
    decltype(zeMemGetAddressRange) *zeMemGetAddressRange;
    decltype(zeMemAllocHost) *zeMemAllocHost;
    decltype(zeMemAllocDevice) *zeMemAllocDevice;
    decltype(zeMemAllocShared) *zeMemAllocShared;
    decltype(zeMemFree) *zeMemFree;
    decltype(zeMemOpenIpcHandle) *zeMemOpenIpcHandle;
    decltype(zeMemCloseIpcHandle) *zeMemCloseIpcHandle;
    decltype(zeMemGetIpcHandle) *zeMemGetIpcHandle;
    decltype(zeDeviceGet) *zeDeviceGet;
    decltype(zeDeviceGetProperties) *zeDeviceGetProperties;
    decltype(zeDeviceCanAccessPeer) *zeDeviceCanAccessPeer;
    decltype(zeDeviceGetCommandQueueGroupProperties) *zeDeviceGetCommandQueueGroupProperties;
    decltype(zeDeviceGetP2PProperties) *zeDeviceGetP2PProperties;
    decltype(zeDeviceGetGlobalTimestamps) *zeDeviceGetGlobalTimestamps;
    decltype(zeDriverGetProperties) *zeDriverGetProperties;
    decltype(zeDriverGetIpcProperties) *zeDriverGetIpcProperties;
    decltype(zeCommandQueueCreate) *zeCommandQueueCreate;
    decltype(zeCommandQueueExecuteCommandLists) *zeCommandQueueExecuteCommandLists;
    decltype(zeCommandQueueSynchronize) *zeCommandQueueSynchronize;
    decltype(zeCommandQueueDestroy) *zeCommandQueueDestroy;
    decltype(zeCommandListCreate) *zeCommandListCreate;
    decltype(zeCommandListCreateImmediate) *zeCommandListCreateImmediate;
    decltype(zeCommandListAppendMemoryCopy) *zeCommandListAppendMemoryCopy;
    decltype(zeCommandListAppendLaunchKernel) *zeCommandListAppendLaunchKernel;
    decltype(zeCommandListAppendWaitOnEvents) *zeCommandListAppendWaitOnEvents;
    decltype(zeCommandListAppendBarrier) *zeCommandListAppendBarrier;
    decltype(zeCommandListClose) *zeCommandListClose;
    decltype(zeCommandListReset) *zeCommandListReset;
    decltype(zeCommandListDestroy) *zeCommandListDestroy;
    decltype(zeContextCreate) *zeContextCreate;
    decltype(zeContextDestroy) *zeContextDestroy;
    decltype(zeEventPoolCreate) *zeEventPoolCreate;
    decltype(zeEventCreate) *zeEventCreate;
    decltype(zeEventQueryStatus) *zeEventQueryStatus;
    decltype(zeEventHostSynchronize) *zeEventHostSynchronize;
    decltype(zeEventHostReset) *zeEventHostReset;
    decltype(zeEventHostSignal) *zeEventHostSignal;
    decltype(zeEventDestroy) *zeEventDestroy;
    decltype(zeEventPoolOpenIpcHandle) *zeEventPoolOpenIpcHandle;
    decltype(zeEventPoolCloseIpcHandle) *zeEventPoolCloseIpcHandle;
    decltype(zeEventPoolGetIpcHandle) *zeEventPoolGetIpcHandle;
    decltype(zeEventQueryKernelTimestamp) *zeEventQueryKernelTimestamp;
    decltype(zeEventPoolDestroy) *zeEventPoolDestroy;
    decltype(zeFenceHostSynchronize) *zeFenceHostSynchronize;
    decltype(zeFenceCreate) *zeFenceCreate;
    decltype(zeKernelCreate) *zeKernelCreate;
    decltype(zeKernelSetArgumentValue) *zeKernelSetArgumentValue;
    decltype(zeKernelSuggestGroupSize) *zeKernelSuggestGroupSize;
    decltype(zeKernelSetGroupSize) *zeKernelSetGroupSize;
    decltype(zeKernelDestroy) *zeKernelDestroy;
    decltype(zeModuleCreate) *zeModuleCreate;
    decltype(zeModuleDestroy) *zeModuleDestroy;
    decltype(zeModuleBuildLogGetString) *zeModuleBuildLogGetString;
    decltype(zeModuleBuildLogDestroy) *zeModuleBuildLogDestroy;
    decltype(zeDeviceGetComputeProperties) *zeDeviceGetComputeProperties;
    decltype(zeDeviceGetMemoryAccessProperties) *zeDeviceGetMemoryAccessProperties;
    decltype(zeDeviceGetMemoryProperties) *zeDeviceGetMemoryProperties;
    decltype(zeDeviceGetSubDevices) *zeDeviceGetSubDevices;
    decltype(zesDevicePciGetProperties) *zesDevicePciGetProperties;
    decltype(zesDeviceEnumFabricPorts) *zesDeviceEnumFabricPorts;
    decltype(zesFabricPortGetConfig) *zesFabricPortGetConfig;
    decltype(zesFabricPortGetProperties) *zesFabricPortGetProperties;
    decltype(zesFabricPortGetState) *zesFabricPortGetState;
} libze_ops_t;

static const char *fn_names[] = {
    "zeInit",
    "zeDriverGet",
    "zeDriverGetApiVersion",
    "zeMemGetAllocProperties",
    "zeMemGetAddressRange",
    "zeMemAllocHost",
    "zeMemAllocDevice",
    "zeMemAllocShared",
    "zeMemFree",
    "zeMemOpenIpcHandle",
    "zeMemCloseIpcHandle",
    "zeMemGetIpcHandle",
    "zeDeviceGet",
    "zeDeviceGetProperties",
    "zeDeviceCanAccessPeer",
    "zeDeviceGetCommandQueueGroupProperties",
    "zeDeviceGetP2PProperties",
    "zeDeviceGetGlobalTimestamps",
    "zeDriverGetProperties",
    "zeDriverGetIpcProperties",
    "zeCommandQueueCreate",
    "zeCommandQueueExecuteCommandLists",
    "zeCommandQueueSynchronize",
    "zeCommandQueueDestroy",
    "zeCommandListCreate",
    "zeCommandListCreateImmediate",
    "zeCommandListAppendMemoryCopy",
    "zeCommandListAppendLaunchKernel",
    "zeCommandListAppendWaitOnEvents",
    "zeCommandListAppendBarrier",
    "zeCommandListClose",
    "zeCommandListReset",
    "zeCommandListDestroy",
    "zeContextCreate",
    "zeContextDestroy",
    "zeEventPoolCreate",
    "zeEventCreate",
    "zeEventQueryStatus",
    "zeEventHostSynchronize",
    "zeEventHostReset",
    "zeEventHostSignal",
    "zeEventDestroy",
    "zeEventPoolOpenIpcHandle",
    "zeEventPoolCloseIpcHandle",
    "zeEventPoolGetIpcHandle",
    "zeEventQueryKernelTimestamp",
    "zeEventPoolDestroy",
    "zeFenceHostSynchronize",
    "zeFenceCreate",
    "zeKernelCreate",
    "zeKernelSetArgumentValue",
    "zeKernelSuggestGroupSize",
    "zeKernelSetGroupSize",
    "zeKernelDestroy",
    "zeModuleCreate",
    "zeModuleDestroy",
    "zeModuleBuildLogGetString",
    "zeModuleBuildLogDestroy",
    "zeDeviceGetComputeProperties",
    "zeDeviceGetMemoryAccessProperties",
    "zeDeviceGetMemoryProperties",
    "zeDeviceGetSubDevices",
    "zesDevicePciGetProperties",
    "zesDeviceEnumFabricPorts",
    "zesFabricPortGetConfig",
    "zesFabricPortGetProperties",
    "zesFabricPortGetState",
};

extern ccl::libze_ops_t libze_ops;

#define zeInit                                 ccl::libze_ops.zeInit
#define zeDriverGet                            ccl::libze_ops.zeDriverGet
#define zeDriverGetApiVersion                  ccl::libze_ops.zeDriverGetApiVersion
#define zeMemGetAllocProperties                ccl::libze_ops.zeMemGetAllocProperties
#define zeMemGetAddressRange                   ccl::libze_ops.zeMemGetAddressRange
#define zeMemAllocHost                         ccl::libze_ops.zeMemAllocHost
#define zeMemAllocDevice                       ccl::libze_ops.zeMemAllocDevice
#define zeMemAllocShared                       ccl::libze_ops.zeMemAllocShared
#define zeMemFree                              ccl::libze_ops.zeMemFree
#define zeMemOpenIpcHandle                     ccl::libze_ops.zeMemOpenIpcHandle
#define zeMemCloseIpcHandle                    ccl::libze_ops.zeMemCloseIpcHandle
#define zeMemGetIpcHandle                      ccl::libze_ops.zeMemGetIpcHandle
#define zeDeviceGet                            ccl::libze_ops.zeDeviceGet
#define zeDeviceGetProperties                  ccl::libze_ops.zeDeviceGetProperties
#define zeDeviceCanAccessPeer                  ccl::libze_ops.zeDeviceCanAccessPeer
#define zeDeviceGetCommandQueueGroupProperties ccl::libze_ops.zeDeviceGetCommandQueueGroupProperties
#define zeDeviceGetP2PProperties               ccl::libze_ops.zeDeviceGetP2PProperties
#define zeDeviceGetGlobalTimestamps            ccl::libze_ops.zeDeviceGetGlobalTimestamps
#define zeDriverGetProperties                  ccl::libze_ops.zeDriverGetProperties
#define zeDriverGetIpcProperties               ccl::libze_ops.zeDriverGetIpcProperties
#define zeCommandQueueCreate                   ccl::libze_ops.zeCommandQueueCreate
#define zeCommandQueueExecuteCommandLists      ccl::libze_ops.zeCommandQueueExecuteCommandLists
#define zeCommandQueueSynchronize              ccl::libze_ops.zeCommandQueueSynchronize
#define zeCommandQueueDestroy                  ccl::libze_ops.zeCommandQueueDestroy
#define zeCommandListCreate                    ccl::libze_ops.zeCommandListCreate
#define zeCommandListCreateImmediate           ccl::libze_ops.zeCommandListCreateImmediate
#define zeCommandListAppendMemoryCopy          ccl::libze_ops.zeCommandListAppendMemoryCopy
#define zeCommandListAppendLaunchKernel        ccl::libze_ops.zeCommandListAppendLaunchKernel
#define zeCommandListAppendWaitOnEvents        ccl::libze_ops.zeCommandListAppendWaitOnEvents
#define zeCommandListAppendBarrier             ccl::libze_ops.zeCommandListAppendBarrier
#define zeCommandListClose                     ccl::libze_ops.zeCommandListClose
#define zeCommandListReset                     ccl::libze_ops.zeCommandListReset
#define zeCommandListDestroy                   ccl::libze_ops.zeCommandListDestroy
#define zeContextCreate                        ccl::libze_ops.zeContextCreate
#define zeContextDestroy                       ccl::libze_ops.zeContextDestroy
#define zeEventPoolCreate                      ccl::libze_ops.zeEventPoolCreate
#define zeEventCreate                          ccl::libze_ops.zeEventCreate
#define zeEventQueryStatus                     ccl::libze_ops.zeEventQueryStatus
#define zeEventHostSynchronize                 ccl::libze_ops.zeEventHostSynchronize
#define zeEventHostReset                       ccl::libze_ops.zeEventHostReset
#define zeEventHostSignal                      ccl::libze_ops.zeEventHostSignal
#define zeEventDestroy                         ccl::libze_ops.zeEventDestroy
#define zeEventPoolOpenIpcHandle               ccl::libze_ops.zeEventPoolOpenIpcHandle
#define zeEventPoolCloseIpcHandle              ccl::libze_ops.zeEventPoolCloseIpcHandle
#define zeEventPoolGetIpcHandle                ccl::libze_ops.zeEventPoolGetIpcHandle
#define zeEventQueryKernelTimestamp            ccl::libze_ops.zeEventQueryKernelTimestamp
#define zeEventPoolDestroy                     ccl::libze_ops.zeEventPoolDestroy
#define zeFenceHostSynchronize                 ccl::libze_ops.zeFenceHostSynchronize
#define zeFenceCreate                          ccl::libze_ops.zeFenceCreate
#define zeKernelCreate                         ccl::libze_ops.zeKernelCreate
#define zeKernelSetArgumentValue               ccl::libze_ops.zeKernelSetArgumentValue
#define zeKernelSuggestGroupSize               ccl::libze_ops.zeKernelSuggestGroupSize
#define zeKernelSetGroupSize                   ccl::libze_ops.zeKernelSetGroupSize
#define zeKernelDestroy                        ccl::libze_ops.zeKernelDestroy
#define zeModuleCreate                         ccl::libze_ops.zeModuleCreate
#define zeModuleDestroy                        ccl::libze_ops.zeModuleDestroy
#define zeModuleBuildLogGetString              ccl::libze_ops.zeModuleBuildLogGetString
#define zeModuleBuildLogDestroy                ccl::libze_ops.zeModuleBuildLogDestroy
#define zeDeviceGetComputeProperties           ccl::libze_ops.zeDeviceGetComputeProperties
#define zeDeviceGetMemoryAccessProperties      ccl::libze_ops.zeDeviceGetMemoryAccessProperties
#define zeDeviceGetMemoryProperties            ccl::libze_ops.zeDeviceGetMemoryProperties
#define zeDeviceGetSubDevices                  ccl::libze_ops.zeDeviceGetSubDevices
#define zesDevicePciGetProperties              ccl::libze_ops.zesDevicePciGetProperties
#define zesDeviceEnumFabricPorts               ccl::libze_ops.zesDeviceEnumFabricPorts
#define zesFabricPortGetConfig                 ccl::libze_ops.zesFabricPortGetConfig
#define zesFabricPortGetProperties             ccl::libze_ops.zesFabricPortGetProperties
#define zesFabricPortGetState                  ccl::libze_ops.zesFabricPortGetState

bool ze_api_init();
void ze_api_fini();

} //namespace ccl

#endif //CCL_ENABLE_SYCL && CCL_ENABLE_ZE
