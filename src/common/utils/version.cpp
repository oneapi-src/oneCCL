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
#include "oneapi/ccl/config.h"
#include "oneapi/ccl/types.hpp"
#include "common/utils/version.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

namespace utils {

ccl::library_version get_library_version() {
    ccl::library_version version{};

    version.major = CCL_MAJOR_VERSION;
    version.minor = CCL_MINOR_VERSION;
    version.update = CCL_UPDATE_VERSION;
    version.product_status = CCL_PRODUCT_STATUS;
    version.build_date = CCL_PRODUCT_BUILD_DATE;
    version.full = CCL_PRODUCT_FULL;
    version.cl_backend_name = ccl::backend_traits::name();

    return version;
}

} // namespace utils
