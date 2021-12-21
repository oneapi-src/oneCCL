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
#include "atl/atl_def.h"

std::map<atl_mnic_t, std::string> mnic_type_names = { std::make_pair(ATL_MNIC_NONE, "none"),
                                                      std::make_pair(ATL_MNIC_LOCAL, "local"),
                                                      std::make_pair(ATL_MNIC_GLOBAL, "global") };

std::map<atl_mnic_offset_t, std::string> mnic_offset_names = {
    std::make_pair(ATL_MNIC_OFFSET_NONE, "none"),
    std::make_pair(ATL_MNIC_OFFSET_LOCAL_PROC_IDX, "local_proc_idx")
};

std::string to_string(atl_mnic_t type) {
    auto it = mnic_type_names.find(type);
    if (it != mnic_type_names.end()) {
        return it->second;
    }
    else {
        return "unknown";
    }
}

std::string to_string(atl_mnic_offset_t offset) {
    auto it = mnic_offset_names.find(offset);
    if (it != mnic_offset_names.end()) {
        return it->second;
    }
    else {
        return "unknown";
    }
}
