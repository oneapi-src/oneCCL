/*
 Copyright 2016-2019 Intel Corporation
 
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

#include "ccl_types.h"
#include "common/utils/utils.hpp"

class alignas(CACHELINE_SIZE) ccl_stream
{
public:
    ccl_stream() = delete;
    ccl_stream(const ccl_stream& other) = delete;
    ccl_stream& operator=(const ccl_stream& other) = delete;

    ~ccl_stream() = default;

    ccl_stream(ccl_stream_type_t type, void* native_stream);

    ccl_stream_type_t get_type() const
    {
        return type;
    }

    void* get_native_stream() const
    {
        return native_stream;
    }

private:
    ccl_stream_type_t type;
    void* native_stream;
};
