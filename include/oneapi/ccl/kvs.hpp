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
namespace detail {
class environment;
}
namespace v1 {
class communicator;
class kvs;
} // namespace v1

class base_kvs_impl;

template <class T>
const T* get_kvs_impl_typed(std::shared_ptr<ccl::v1::kvs>);

namespace v1 {

class CCL_API kvs_interface {
public:
    virtual ~kvs_interface() = default;

    virtual vector_class<char> get(const string_class& key) = 0;

    virtual void set(const string_class& key, const vector_class<char>& data) = 0;
};

class CCL_API kvs final : public kvs_interface {
public:
    static constexpr size_t address_max_size = 256;
    using address_type = array_class<char, address_max_size>;

    ~kvs() override;

    address_type get_address() const;

    vector_class<char> get(const string_class& key) override;

    void set(const string_class& key, const vector_class<char>& data) override;

private:
    friend class ccl::detail::environment;

    template <class T>
    friend const T* ccl::get_kvs_impl_typed(std::shared_ptr<kvs>);

    kvs(const kvs_attr& attr);
    kvs(const address_type& addr, const kvs_attr& attr);
    const base_kvs_impl& get_impl();

    address_type addr;
    unique_ptr_class<base_kvs_impl> pimpl;
};

} // namespace v1

using v1::kvs_interface;
using v1::kvs;

} // namespace ccl
