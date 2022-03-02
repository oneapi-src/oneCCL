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

#include "common/log/log.hpp"
#include "common/utils/utils.hpp"

using ccl_op_id_t = uint8_t;
using ccl_sched_id_t = uint16_t;
using ccl_comm_id_t = uint16_t;

class ccl_atl_tag {
public:
    ccl_atl_tag(size_t tag_bits, size_t max_tag) : tag_bits(tag_bits), max_tag(max_tag) {
        CCL_THROW_IF_NOT(tag_bits >= 32, "unexpected tag_bits ", tag_bits);

        if (max_tag == ccl::utils::pof2(max_tag) * 2 - 1)
            max_tag_mask = max_tag;
        else
            max_tag_mask = ccl::utils::pof2(max_tag) - 1;
    }

    ccl_atl_tag(const ccl_atl_tag& other) = delete;
    ccl_atl_tag(ccl_atl_tag&& other) = delete;

    ccl_atl_tag& operator=(const ccl_atl_tag& other) = delete;
    ccl_atl_tag& operator=(ccl_atl_tag&& other) = delete;

    ~ccl_atl_tag() = default;

    std::string to_string() const;

    /**
     * Generates the tag to be used by ATL communication operations
     * @param rank identifier of the rank within communicator
     * @param comm_id identifier of the communicator
     * @param sched_id identifier of the schedule within communicator
     * @param op_id local operation id, used as sub-schedule identifier
     * @return ATL communication tag
     */
    uint64_t create(int rank,
                    ccl_comm_id_t comm_id,
                    ccl_sched_id_t sched_id,
                    ccl_op_id_t op_id = 0);

private:
    /**********************************************************************************
     *  atl tag layout                                                                *
     * ********************************************************************************
     * 01234567 01234567 01234567 | 01234567 01234567 | 01234567 01234567  | 01234567 |
     *                            |                   |                    |          |
     *           rank             |      comm_id      |       sched_id     |   op_id  |
     *********************************************************************************/

    size_t tag_bits;
    size_t max_tag;
    size_t max_tag_mask;

    const int op_id_shift = 0;
    const int sched_id_shift = 8;
    const int comm_id_shift = 24;
    const int rank_shift = 40;

    const uint64_t op_id_mask = 0x00000000000000FF;
    const uint64_t sched_id_mask = 0x0000000000FFFF00;
    const uint64_t comm_id_mask = 0x000000FFFF000000;
    const uint64_t rank_mask = 0xFFFFFF0000000000;
};
