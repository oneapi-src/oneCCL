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
#include <limits.h>
#include <unistd.h>

#include "ccl_gpu_module.hpp"

#ifdef MULTI_GPU_SUPPORT
#include "common/comm/l0/modules/specific_modules_source_data.hpp"
#include "common/comm/l0/device_group_routing_schema.hpp"
#include "coll/algorithms/algorithms_enum.hpp"

ccl_status_t CCL_API register_gpu_module_source(const char* path,
                                                ccl::device_topology_type topology_class,
                                                ccl_coll_type type) {
    ccl::device_topology_type t_class = static_cast<ccl::device_topology_type>(topology_class);
    char pwd[PATH_MAX];
    char* ret = getcwd(pwd, sizeof(pwd));
    (void)ret;

    LOG_INFO("loading file contained gpu module \"",
             ccl_coll_type_to_str(type),
             "\", topology class: \"",
             to_string(t_class),
             "\" by path: ",
             path,
             ". Current path is: ",
             pwd);

    try {
        if (!path) {
            throw std::runtime_error("Path is empty");
        }

        switch (type) {
            case ccl_coll_allgatherv:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_allgatherv>(path, t_class);
                break;
            case ccl_coll_allreduce:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_allreduce>(path, t_class);
                break;
            case ccl_coll_alltoallv:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_alltoallv>(path, t_class);
                break;
            case ccl_coll_bcast:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_bcast>(path, t_class);
                break;
            case ccl_coll_reduce:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_reduce>(path, t_class);
                break;
            case ccl_coll_reduce_scatter:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_reduce_scatter>(path, t_class);
                break;
            default: 
                throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                                    " - get unexpected ccl collective type: " +
                                                    std::to_string(type));
                break;
        }
    }
    catch (const std::exception& ex) {
        LOG_ERROR("Cannot preload kernel source by path: ", path, ", error: ", ex.what());
        CCL_ASSERT(false);
        return ccl_status_runtime_error;
    }

    LOG_INFO("gpu kernel source by type \"",
             ccl_coll_type_to_str(type),
             "\", topology class: \"",
             to_string(t_class),
             "\" loaded succesfully");
    return ccl_status_success;
}

#endif //MULTI_GPU_SUPPORT
