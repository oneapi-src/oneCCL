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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

enum class comm_split_attr_id : int {
    version,

    color,
    group, // ccl_group_split_type or device_group_split_type

    last_value
};

/**
 * Host-specific values for the 'group' split attribute
 */
enum class
    ccl_group_split_type : int { // TODO fill in this enum with the actual values in the final
        //device,
        thread,
        process,
        //socket,
        //node,
        cluster,

        last_value
    };

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

/**
 * Device-specific values for the 'group' split attribute
 */
enum class
    device_group_split_type : int { // TODO fill in this enum with the actual values in the final
        undetermined = -1,
        //device,
        thread,
        process,
        //socket,
        //node,
        cluster,

        last_value
    };

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

} // namespace ccl
