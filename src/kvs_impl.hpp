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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_kvs.hpp"

namespace ccl {

class kvs_impl {
public:
    //STUB

    kvs::address_type addr;
};

kvs::address_type CCL_API kvs::get_address() const {
    //TODO: add logic;
    //static kvs_impl tmp;
    //return tmp.addr;
    static array_class<char, 256> tmp;
    return tmp;
}

vector_class<char> CCL_API kvs::get(const string_class& prefix, const string_class& key) const {
    //TODO: add logic;
    throw;
}

void CCL_API kvs::set(const string_class& prefix,
                      const string_class& key,
                      const vector_class<char>& data) const {
    //TODO: add logic;
    throw;
}
CCL_API kvs::~kvs() {
    //TODO: add logic;
}

CCL_API kvs::kvs(const kvs::address_type& addr) {
    //TODO: add logic;
}

CCL_API kvs::kvs() {
    //TODO: add logic;
}

} // namespace ccl
