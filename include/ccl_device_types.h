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
#ifdef MULTI_GPU_SUPPORT
#ifndef CCL_PRODUCT_FULL
    #error "Do not include this file directly. Please include 'ccl_types.h'"
#endif

/** Device topology class. */
typedef enum
{
    ring_algo_class      = 0,
    a2a_algo_class       = 1,

    ccl_topology_class_last_value
} ccl_topology_class_t;

/** Device topology group. */
typedef enum
{
    device_group         = 0,
    thread_group         = 1,
    process_group        = 2,

    ccl_topology_group_last_value
} ccl_topology_group_t;

/** Device attributes
 *
 */
typedef enum
{
    ccl_device_preferred_topology_class,
    ccl_device_preferred_group

} ccl_device_attributes;

typedef struct
{
    ccl_host_comm_attr_t core;

    unsigned char data[];
} ccl_device_comm_attr_t;

#endif //MULTI_GPU_SUPPORT
