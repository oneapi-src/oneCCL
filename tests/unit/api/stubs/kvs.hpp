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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_aliases.hpp"

#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"

#include "oneapi/ccl/ccl_kvs.hpp"

class stub_kvs : public ccl::kvs_interface {
public:
    ccl::vector_class<char> get(const ccl::string_class& prefix,
                                const ccl::string_class& key) const override;

    void set(const ccl::string_class& prefix,
             const ccl::string_class& key,
             const ccl::vector_class<char>& data) const override;

    ~stub_kvs() = default;
};
