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
#include "sched/entry/copy/copy_helper.hpp"

copy_attr::copy_attr(int peer_rank,
                     size_t peer_buf_idx,
                     copy_direction direction,
                     ccl_comm* map_comm,
                     size_t in_buf_offset,
                     size_t out_buf_offset)
        : peer_rank(peer_rank),
          peer_buf_idx(peer_buf_idx),
          direction(direction),
          map_comm(map_comm),
          in_buf_offset(in_buf_offset),
          out_buf_offset(out_buf_offset) {}

copy_attr::copy_attr(copy_direction direction, size_t in_buf_offset, size_t out_buf_offset)
        : direction(direction),
          in_buf_offset(in_buf_offset),
          out_buf_offset(out_buf_offset) {}

using copy_direction_str_enum =
    utils::enum_to_str<utils::enum_to_underlying(copy_direction::d2d) + 1>;
std::string to_string(copy_direction val) {
    return copy_direction_str_enum({ "UNDEFINED", "H2H", "D2H", "H2D", "D2D" })
        .choose(val, "UNKNOWN");
}
