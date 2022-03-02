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
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/ze/ze_api_wrapper.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

namespace ccl {

static void* libze_handle;
libze_ops_t libze_ops;

bool ze_api_init() {
    // lib_path specifies the name and full path to the level-zero library
    // it should be absolute and validated path
    // pointing to desired libze_loader library
    std::string lib_path = ccl::global_data::env().ze_lib_path;

    if (lib_path.empty()) {
        lib_path = "libze_loader.so";
    }

    libze_handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!libze_handle) {
        LOG_WARN("could not open level-zero library: ", lib_path.c_str(), ", error: ", dlerror());
        return false;
    }

    void** ops = (void**)((void*)&libze_ops);
    int fn_count = sizeof(fn_names) / sizeof(*fn_names);

    for (int i = 0; i < fn_count; ++i) {
        ops[i] = dlsym(libze_handle, fn_names[i]);
        if (!ops[i]) {
            LOG_WARN("dlsym is failed on: ", fn_names[i], ", error: ", dlerror());
            return false;
        }
        LOG_DEBUG("dlsym loaded of ", fn_count, " - ", i + 1, ": ", fn_names[i]);
    }

    return true;
}

void ze_api_fini() {
    if (libze_handle) {
        dlclose(libze_handle);
        libze_handle = nullptr;
    }
}

} //namespace ccl
