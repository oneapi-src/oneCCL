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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"

#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"

#include "oneapi/ccl/kvs_attr_ids.hpp"
#include "oneapi/ccl/kvs_attr_ids_traits.hpp"
#include "oneapi/ccl/kvs_attr.hpp"

#include "kvs_impl.hpp"

namespace ccl {
kvs_impl::kvs_impl(const kvs_attr& attr) : inter_kvs(new internal_kvs()) {
    ;
    if (attr.is_valid<kvs_attr_id::ip_port>()) {
        std::string server_address = attr.get<kvs_attr_id::ip_port>();
        inter_kvs->set_server_address(server_address);
    }

    inter_kvs->kvs_main_server_address_reserve(addr.data());
    inter_kvs->kvs_init(addr.data());
}

kvs_impl::kvs_impl(const kvs::address_type& addr, const kvs_attr& attr)
        : inter_kvs(new internal_kvs()),
          addr(addr) {
    if (attr.is_valid<kvs_attr_id::ip_port>()) {
        std::string server_address = attr.get<kvs_attr_id::ip_port>();
        inter_kvs->set_server_address(server_address);
    }

    inter_kvs->kvs_init(addr.data());
}

kvs::address_type kvs_impl::get_addr() {
    return addr;
}

vector_class<char> kvs_impl::get(const string_class& key) {
    char ret[MAX_KVS_VAL_LENGTH];
    CCL_THROW_IF_NOT(inter_kvs->kvs_get_value_by_name_key(prefix.c_str(), key.c_str(), ret) ==
                         KVS_STATUS_SUCCESS,
                     "kvs get failed");
    size_t ret_len = strlen(ret);
    vector_class<char> ret_vec;
    if (ret_len != 0) {
        ret_vec = vector_class<char>(ret, ret + ret_len + 1);
        ret_vec[ret_len] = '\0';
    }
    else
        ret_vec = vector_class<char>('\0');
    return ret_vec;
}

void kvs_impl::set(const string_class& key, const vector_class<char>& data) {
    CCL_THROW_IF_NOT(!data.empty(), "data should have at least one element");
    CCL_THROW_IF_NOT(data.back() == '\0', "data should have terminating symbol");
    CCL_THROW_IF_NOT(data.data(), "data pointer should be non-null");
    inter_kvs->kvs_set_value(prefix.c_str(), key.c_str(), data.data());
}

std::shared_ptr<internal_kvs> kvs_impl::get() const {
    return inter_kvs;
}

namespace v1 {

kvs::address_type CCL_API kvs::get_address() const {
    return pimpl->get_addr();
}

vector_class<char> CCL_API kvs::get(const string_class& key) {
    return pimpl->get(key);
}

void CCL_API kvs::set(const string_class& key, const vector_class<char>& data) {
    pimpl->set(key, data);
}

CCL_API kvs::kvs(const kvs::address_type& addr, const kvs_attr& attr) {
    pimpl = std::unique_ptr<kvs_impl>(new kvs_impl(addr, attr));
}

CCL_API const kvs_impl& kvs::get_impl() {
    return *pimpl;
}

CCL_API kvs::kvs(const kvs_attr& attr) : pimpl(new kvs_impl(attr)) {}

CCL_API kvs::~kvs() {}

} // namespace v1

} // namespace ccl
