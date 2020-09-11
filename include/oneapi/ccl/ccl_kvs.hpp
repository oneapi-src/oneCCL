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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {

class CCL_API kvs_interface {
public:
    virtual vector_class<char> get(const string_class& prefix, const string_class& key) const = 0;

    virtual void set(const string_class& prefix,
                     const string_class& key,
                     const vector_class<char>& data) const = 0;

    virtual ~kvs_interface() = default;
};

class kvs_impl;
class CCL_API kvs final : public kvs_interface {
public:
    static constexpr size_t address_max_size = 256;
    using address_type = array_class<char, address_max_size>;

    address_type get_address() const;

    ~kvs() override;

    vector_class<char> get(const string_class& prefix, const string_class& key) const override;

    void set(const string_class& prefix,
             const string_class& key,
             const vector_class<char>& data) const override;

private:
    friend class environment;

    kvs();
    kvs(const address_type& addr);

    unique_ptr_class<kvs_impl> pimpl;
};
} // namespace ccl
