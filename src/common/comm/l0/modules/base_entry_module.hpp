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
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "supported_topologies.hpp"
#include "common/comm/l0/modules/kernel_functions.hpp"

namespace native {

//module, collection of functions
struct gpu_module_base {
    using handle = ze_module_handle_t;
    using kernel_handle = ze_kernel_handle_t;
    using imported_kernels = std::map<std::string, kernel_handle>;

    gpu_module_base(handle module_handle);
    ~gpu_module_base();

    handle get() const;
    void release();
    kernel_handle import_kernel(const std::string& name);

    handle module;
    imported_kernels functions;
};

//specific type module implementations:
//1) in-process gpu module
template <ccl_coll_type type, ccl::group_split_type topology, ccl::device_topology_type mode>
struct device_coll_module : private gpu_module_base {
    static constexpr ccl_coll_type get_coll_type() {
        return type;
    }
    static constexpr ccl::group_split_type get_topology_type() {
        return topology;
    }
    static constexpr ccl::device_topology_type get_topology_class() {
        return mode;
    }

    device_coll_module(handle module_handle) : gpu_module_base(module_handle) {}
};

//2) out-of-process gpu module
template <ccl_coll_type type, ccl::group_split_type topology, ccl::device_topology_type mode>
struct ipc_dst_device_coll_module : private gpu_module_base {
    static constexpr ccl_coll_type get_coll_type() {
        return type;
    }
    static constexpr ccl::group_split_type get_topology_type() {
        return topology;
    }
    static constexpr ccl::device_topology_type get_topology_class() {
        return mode;
    }

    ipc_dst_device_coll_module(handle module_handle) : gpu_module_base(module_handle) {}
};

//3) virtual gpu module
template <ccl_coll_type type, ccl::group_split_type topology, ccl::device_topology_type mode>
struct virtual_device_coll_module {
    static constexpr ccl_coll_type get_coll_type() {
        return type;
    }
    static constexpr ccl::group_split_type get_topology_type() {
        return topology;
    }
    static constexpr ccl::device_topology_type get_topology_class() {
        return mode;
    }

    virtual_device_coll_module(
        std::shared_ptr<device_coll_module<type, topology, mode>> real_module)
            : real_module_ref(real_module) {}
    std::shared_ptr<device_coll_module<type, topology, mode>> real_module_ref;
};

template <ccl_coll_type type, ccl::group_split_type group_id, ccl::device_topology_type class_id>
struct coll_module_traits {
    static constexpr ccl_coll_type coll_type() {
        return type;
    }
    static constexpr ccl::group_split_type group_type() {
        return group_id;
    }
    static constexpr ccl::device_topology_type topology_class() {
        return class_id;
    }
};

} // namespace native
