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

#include <mutex>
#include <unordered_map>
#include <utility>

#include "ccl_types.h"
#include "common/log/log.hpp"
#include "common/utils/spinlock.hpp"

class ccl_datatype {
public:
    ccl_datatype(ccl_datatype_t idx, size_t size);
    ccl_datatype() = default;
    ~ccl_datatype() = default;
    ccl_datatype& operator=(const ccl_datatype& other) = default;

    ccl_datatype(const ccl_datatype& other) = default;

    ccl_datatype_t idx() const {
        return m_idx;
    }

    size_t size() const {
        CCL_THROW_IF_NOT(m_size > 0, "non-positive datatype size ", m_size);
        return m_size;
    }

private:
    ccl_datatype_t m_idx;
    size_t m_size;
};

/* frequently used in multiple places */
extern ccl_datatype ccl_datatype_char;

using ccl_datatype_lock_t = ccl_spinlock;

using ccl_datatype_table_t =
    std::unordered_map<ccl_datatype_t, std::pair<ccl_datatype, std::string>>;

class ccl_datatype_storage {
public:
    ccl_datatype_storage();
    ~ccl_datatype_storage();

    ccl_datatype_storage(const ccl_datatype_storage& other) = delete;
    ccl_datatype_storage& operator=(const ccl_datatype_storage& other) = delete;

    ccl_datatype_t create(const ccl_datatype_attr_t* attr);
    void free(ccl_datatype_t idx);

    const ccl_datatype& get(ccl_datatype_t idx) const;

    const std::string& name(const ccl_datatype& dtype) const;
    const std::string& name(ccl_datatype_t idx) const;

    static bool is_predefined_datatype(ccl_datatype_t idx);

private:
    void create_internal(ccl_datatype_table_t& table,
                         size_t idx,
                         size_t size,
                         const std::string& name);

    mutable ccl_datatype_lock_t guard{};

    ccl_datatype_t custom_idx;

    ccl_datatype_table_t predefined_table;
    ccl_datatype_table_t custom_table;
};
