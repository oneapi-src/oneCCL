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
#include "base.hpp"

#ifdef MULTI_GPU_SUPPORT
#include "native_device_api/export_api.hpp"
#endif

int main() {
    auto version = ccl::environment::instance().get_version();
    std::cout << "CCL library info:\nVersion:\n"
              << "major: " << version.major << "\nminor: " << version.minor
              << "\nupdate: " << version.update << "\nProduct: " << version.product_status
              << "\nBuild date: " << version.build_date << "\nFull: " << version.full << std::endl;
#ifdef MULTI_GPU_SUPPORT
    std::cout << "Compute runtime information:\n"
              << native::get_platform().to_string() << std::endl;
#endif
    std::cout << "Compute runtime information unavailable for current version" << std::endl;
}
