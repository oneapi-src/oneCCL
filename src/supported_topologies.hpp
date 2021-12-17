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

#include "common/utils/enums.hpp"
#include "internal_types.hpp"

inline std::string to_string(ccl::group_split_type type) {
    using device_group_split_type_names = ::utils::enum_to_str<
        static_cast<typename std::underlying_type<ccl::group_split_type>::type>(
            ccl::group_split_type::last_value)>;
    return device_group_split_type_names({
                                             "TG",
                                             "PG",
                                             "CG",
                                         })
        .choose(type, "INVALID_VALUE");
}
