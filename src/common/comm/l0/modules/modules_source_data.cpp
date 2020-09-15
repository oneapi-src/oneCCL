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
#include <stdexcept>
#include <fstream>
#include "common/comm/l0/modules/modules_source_data.hpp"

namespace native {
source_data_t load_binary_file(const std::string& source_path) {
    std::ifstream stream(source_path, std::ios::in | std::ios::binary);

    source_data_t binary_file;
    if (!stream.good()) {
        std::string error("Failed to load binary file: ");
        error += source_path;

        throw std::runtime_error(error);
    }

    size_t length = 0;
    stream.seekg(0, stream.end);
    length = static_cast<size_t>(stream.tellg());
    stream.seekg(0, stream.beg);

    binary_file.resize(length);
    stream.read(reinterpret_cast<char*>(binary_file.data()), length);
    return binary_file;
}
} // namespace native
