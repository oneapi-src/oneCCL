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
#include <limits>

#include "common/datatype/datatype.hpp"

ccl_datatype ccl_datatype_char;

ccl_datatype::ccl_datatype(ccl_datatype_t idx, size_t size) : m_idx(idx), m_size(size) {
    CCL_THROW_IF_NOT(m_size > 0, "unexpected datatype size ", m_size);
}

ccl_datatype_storage::ccl_datatype_storage() : custom_idx(ccl_dtype_last_value) {
    LOG_DEBUG("create datatype_storage");

    size_t size = 0;
    std::string name_str;

    for (ccl_datatype_t idx = ccl_dtype_char; idx < ccl_dtype_last_value; idx++) {
        /* fill table with predefined datatypes */
        size = (idx == ccl_dtype_char)     ? sizeof(char)
               : (idx == ccl_dtype_int)    ? sizeof(int)
               : (idx == ccl_dtype_bfp16)  ? sizeof(uint16_t)
               : (idx == ccl_dtype_float)  ? sizeof(float)
               : (idx == ccl_dtype_double) ? sizeof(double)
               : (idx == ccl_dtype_int64)  ? sizeof(int64_t)
               : (idx == ccl_dtype_uint64) ? sizeof(uint64_t)
                                           : 0;

        name_str = (idx == ccl_dtype_char)     ? "CHAR"
                   : (idx == ccl_dtype_int)    ? "INT"
                   : (idx == ccl_dtype_bfp16)  ? "BFLOAT16"
                   : (idx == ccl_dtype_float)  ? "FLOAT"
                   : (idx == ccl_dtype_double) ? "DOUBLE"
                   : (idx == ccl_dtype_int64)  ? "INT64"
                   : (idx == ccl_dtype_uint64) ? "UINT64"
                                               : 0;

        create_internal(predefined_table, idx, size, name_str);

        const ccl_datatype& dtype = get(idx);
        const std::string& dtype_name = name(dtype);

        CCL_THROW_IF_NOT(
            dtype.idx() == idx, "unexpected datatype idx ", dtype.idx(), ", expected ", idx);
        CCL_THROW_IF_NOT(
            dtype.idx() == idx, "unexpected datatype size ", dtype.size(), ", expected ", size);
        CCL_THROW_IF_NOT(!dtype_name.compare(name_str),
                         "unexpected datatype name ",
                         dtype_name,
                         ", expected ",
                         name_str);
    }

    ccl_datatype_char = get(ccl_dtype_char);
}

ccl_datatype_storage::~ccl_datatype_storage() {
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };
    predefined_table.clear();
    custom_table.clear();
}

void ccl_datatype_storage::create_internal(ccl_datatype_table_t& table,
                                           size_t idx,
                                           size_t size,
                                           const std::string& name) {
    CCL_THROW_IF_NOT(table.find(idx) == table.end(), "datatype index is busy, idx ", idx);
    table[idx] = std::make_pair(ccl_datatype(idx, size), name);
    LOG_DEBUG("created datatype idx: ", idx, ", size: ", size, ", name: ", name);
}

ccl_datatype_t ccl_datatype_storage::create(const ccl_datatype_attr_t* attr) {
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };

    while (custom_table.find(custom_idx) != custom_table.end() ||
           is_predefined_datatype(custom_idx)) {
        custom_idx++;
        if (custom_idx < 0)
            custom_idx = 0;
    }

    create_internal(custom_table,
                    custom_idx,
                    (attr) ? attr->size : 1,
                    std::string("DTYPE_") + std::to_string(custom_idx));

    return custom_idx;
}

void ccl_datatype_storage::free(ccl_datatype_t idx) {
    std::lock_guard<ccl_datatype_lock_t> lock{ guard };

    if (is_predefined_datatype(idx)) {
        CCL_THROW("attempt to free predefined datatype idx ", idx);
        return;
    }

    if (custom_table.find(idx) == custom_table.end()) {
        CCL_THROW("attempt to free non-existing datatype idx ", idx);
        return;
    }

    LOG_DEBUG("free datatype idx ", idx);
    custom_table.erase(idx);
}

const ccl_datatype& ccl_datatype_storage::get(ccl_datatype_t idx) const {
    if (is_predefined_datatype(idx)) {
        return predefined_table.find(idx)->second.first;
    }
    else {
        std::lock_guard<ccl_datatype_lock_t> lock{ guard };
        return custom_table.find(idx)->second.first;
    }
}

const std::string& ccl_datatype_storage::name(const ccl_datatype& dtype) const {
    size_t idx = dtype.idx();
    if (is_predefined_datatype(idx)) {
        return predefined_table.find(idx)->second.second;
    }
    else {
        std::lock_guard<ccl_datatype_lock_t> lock{ guard };
        return custom_table.find(idx)->second.second;
    }
}

const std::string& ccl_datatype_storage::name(ccl_datatype_t idx) const {
    return name(get(idx));
}

bool ccl_datatype_storage::is_predefined_datatype(ccl_datatype_t idx) {
    return (idx >= ccl_dtype_char && idx < ccl_dtype_last_value) ? true : false;
}
