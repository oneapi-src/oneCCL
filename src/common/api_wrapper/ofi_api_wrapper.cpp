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
#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/ofi_api_wrapper.hpp"

namespace ccl {

lib_info_t ofi_lib_info;
ofi_lib_ops_t ofi_lib_ops;

bool ofi_api_init() {
    bool ret = true;

    ofi_lib_info.ops = &ofi_lib_ops;
    ofi_lib_info.fn_names = ofi_fn_names;

    // lib_path specifies the name and full path to the OFI library
    // it should be absolute and validated path
    // pointing to desired libfabric library
    ofi_lib_info.path = ccl::global_data::env().ofi_lib_path;

    if (ofi_lib_info.path.empty()) {
        ofi_lib_info.path = "libfabric.so.1";
    }
    LOG_DEBUG("OFI lib path: ", ofi_lib_info.path);

    load_library(ofi_lib_info);
    if (!ofi_lib_info.handle)
        ret = false;

    return ret;
}

void ofi_api_fini() {
    LOG_DEBUG("close OFI lib: handle: ", ofi_lib_info.handle);
    close_library(ofi_lib_info);
}

} //namespace ccl
