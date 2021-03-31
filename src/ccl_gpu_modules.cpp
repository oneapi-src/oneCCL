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

ccl::status load_gpu_module(const std::string& path,
                            ccl::device_topology_type topo_type,
                            ccl_coll_type coll_type) {
    char pwd[PATH_MAX];
    char* ret = getcwd(pwd, sizeof(pwd));
    (void)ret;

    LOG_INFO("loading GPU module for collective: \"",
             ccl_coll_type_to_str(coll_type),
             "\", topology: \"",
             to_string(topo_type),
             "\" by path: ",
             path,
             ", current directory is: ",
             pwd);

    try {
        if (path.empty()) {
            throw std::runtime_error("path is empty");
        }

        switch (coll_type) {
            case ccl_coll_allgatherv:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_allgatherv>(path, topo_type);
                break;
            case ccl_coll_allreduce:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_allreduce>(path, topo_type);
                break;
            case ccl_coll_alltoallv:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_alltoallv>(path, topo_type);
                break;
            case ccl_coll_bcast:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_bcast>(path, topo_type);
                break;
            case ccl_coll_reduce:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_reduce>(path, topo_type);
                break;
            case ccl_coll_reduce_scatter:
                native::specific_modules_source_data_storage::instance()
                    .load_kernel_source<ccl_coll_reduce_scatter>(path, topo_type);
                break;
            default:
                throw std::runtime_error(
                    std::string(__PRETTY_FUNCTION__) +
                    " - unexpected collective type: " + std::to_string(coll_type));
                break;
        }
    }
    catch (const std::exception& ex) {
        LOG_ERROR("cannot load GPU module from: ", path, ", error: ", ex.what());
        CCL_ASSERT(false);
        return ccl::status::runtime_error;
    }

    LOG_INFO("GPU module for collective: \"",
             ccl_coll_type_to_str(coll_type),
             "\", topology: \"",
             to_string(topo_type),
             "\" loaded succesfully");

    return ccl::status::success;
}

#endif //MULTI_GPU_SUPPORT
