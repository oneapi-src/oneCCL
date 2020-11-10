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
#include <functional>

#include "supported_topologies.hpp"
#include "coll/algorithms/algorithms_enum.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "common/comm/l0/modules/modules_source_data.hpp"

namespace native {

inline std::size_t module_hash(ccl_coll_type module_type,
                               ccl::group_split_type group_id,
                               ccl::device_topology_type class_id) {
    std::string str = std::string(ccl_coll_type_to_str(module_type)) + "," + ::to_string(group_id) +
                      ::to_string(class_id);
    return std::hash<std::string>{}(str);
}

template <class communicator_type>
struct module_loader {
    template <ccl_coll_type... modules_types>
    void load_modules(const modules_src_container<modules_types...>& sources) {
        ccl_tuple_for_each(sources.get_modules(), *this);
    }

    // load specific module type
    template <ccl_coll_type module_type>
    void operator()(const typed_source_data_storage_t<module_type>& sources) {
        for (auto it = sources.begin(); it != sources.end(); ++it) {
            switch (it->first) {
                case ccl::device_topology_type::ring:
                    load_module_impl<module_type,
                                     ccl::device_topology_type::ring,
                                     ccl::group_split_type::thread,
                                     ccl::group_split_type::process,
                                     ccl::group_split_type::cluster>(it->second);
                    break;
                case ccl::device_topology_type::a2a:
                    load_module_impl<module_type,
                                     ccl::device_topology_type::a2a,
                                     ccl::group_split_type::thread,
                                     ccl::group_split_type::process,
                                     ccl::group_split_type::cluster>(it->second);
                    break;
                default:
                    throw std::runtime_error(std::string("unknown topology class: ") +
                                             std::to_string(it->first));
            }
        }
    }

private:
    communicator_type* get_this() {
        return static_cast<communicator_type*>(this);
    }

    template <ccl_coll_type module_type,
              ccl::device_topology_type class_id,
              ccl::group_split_type... topology_types>
    void load_module_impl(const source_data_t& module_data) {
        LOG_DEBUG("Started loading module \"",
                  ccl_coll_type_to_str(module_type),
                  "\" for topology: \"",
                  ::to_string(class_id),
                  "\", for: ",
                  communicator_type::name_impl())

        ze_module_desc_t module_description;
        module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        module_description.pNext = nullptr;
        module_description.format = ZE_MODULE_FORMAT_IL_SPIRV;
        module_description.inputSize =
            static_cast<uint32_t>(module_data.size()); //Ask L0: why not size_t?
        module_description.pInputModule = module_data.data();
        module_description.pBuildFlags = nullptr;
        module_description.pConstants = nullptr;

        //compile modules TODO ring only
        std::array<std::string, sizeof...(topology_types)> logs{
            get_this()->template create_module_impl<module_type, topology_types, class_id>(
                module_description)...
        };

        std::string accumulated_log;
        if (!logs.empty()) {
            accumulated_log =
                std::accumulate(logs.begin(), logs.end(), std::string("\nLoading log:\n"));
        }
        LOG_DEBUG("Finished loading module \"",
                  ccl_coll_type_to_str(module_type),
                  "\" for topology: \"",
                  ::to_string(class_id),
                  "\" for: ",
                  communicator_type::name_impl(),
                  accumulated_log);
    }
};
} // namespace native
