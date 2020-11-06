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
#include "oneapi/ccl/ccl_types.hpp"

namespace ccl {
ccl::library_version ccl_empty_attr::version{
    CCL_MAJOR_VERSION,  CCL_MINOR_VERSION,      CCL_UPDATE_VERSION,
    CCL_PRODUCT_STATUS, CCL_PRODUCT_BUILD_DATE, CCL_PRODUCT_FULL,
};

template <class attr>
attr ccl_empty_attr::create_empty() {
    return attr{ ccl_empty_attr::version };
}
} // namespace ccl
