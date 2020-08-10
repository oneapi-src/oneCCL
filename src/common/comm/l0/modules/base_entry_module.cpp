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
#include <atomic>
#include <map>
#include <set>
#include <tuple>

#include "coll/coll.hpp"
#include "native_device_api/export_api.hpp"
#include "common/comm/l0/modules/base_entry_module.hpp"
namespace native {
gpu_module_base::gpu_module_base(handle module_handle) : module(module_handle) {}

gpu_module_base::~gpu_module_base() {
    release();
}

void gpu_module_base::release() {
    //release imported functions at first
    for (auto& f : functions) {
        zeKernelDestroy(f.second);
    }

    if (module) {
        zeModuleDestroy(module);
    }
    functions.clear();
}

gpu_module_base::handle gpu_module_base::get() const {
    return module;
}

ze_kernel_handle_t gpu_module_base::import_kernel(const std::string& name) {
    ze_kernel_desc_t desc = { ZE_KERNEL_DESC_VERSION_CURRENT, ZE_KERNEL_FLAG_NONE };
    desc.pKernelName = name.c_str();
    ze_kernel_handle_t handle;

    if (!module) {
        return nullptr;
    }

    ze_result_t result = zeKernelCreate(module, &desc, &handle);
    if (result != ZE_RESULT_SUCCESS) {
        CCL_THROW("Cannot create kernel: ", name, ", error: ", native::to_string(result));
    }

    //TODO avoid duplicates
    std::string imported_name = name + std::to_string((size_t)handle);
    functions.emplace(std::piecewise_construct,
                      std::forward_as_tuple(imported_name),
                      std::forward_as_tuple(handle));
    return handle;
}

} // namespace native
