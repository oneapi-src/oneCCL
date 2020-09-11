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

//API headers with declaration of new API object
#define private public
#define protected public
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_aliases.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_datatype_attr_ids.hpp"
#include "oneapi/ccl/ccl_datatype_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_datatype_attr.hpp"

#include "datatype_attr_creation_impl.hpp"

// Core file with PIMPL implementation
#include "common/datatype/datatype_attr.hpp"

#undef protected
#undef private

namespace datatype_attr_suite {

TEST(datatype_attr, datatype_attr_empty_creation) {
    ccl::datatype_attr attr = ccl::create_datatype_attr();
    ASSERT_TRUE(attr.get<ccl::datatype_attr_id::version>().full != nullptr);
    ASSERT_EQ(attr.get<ccl::datatype_attr_id::size>(), 0);
}

TEST(datatype_attr, copy_datatype_attr) {
    auto attr = ccl::create_datatype_attr();
    attr.set<ccl::datatype_attr_id::size>(666);

    auto original_inner_impl_ptr = attr.get_impl().get();
    auto attr2 = attr;
    auto copied_inner_impl_ptr = attr2.get_impl().get();
    ASSERT_TRUE(original_inner_impl_ptr != copied_inner_impl_ptr);
    ASSERT_TRUE(attr.get_impl());
    ASSERT_TRUE(attr2.get<ccl::datatype_attr_id::version>().full != nullptr);
    ASSERT_EQ(attr2.get<ccl::datatype_attr_id::size>(), 666);
}

TEST(datatype_attr, move_datatype_attr) {
    /* move constructor test */
    auto orig_attr = ccl::create_datatype_attr();

    auto orig_inner_impl_ptr = orig_attr.get_impl().get();
    auto moved_attr = (std::move(orig_attr));
    auto moved_inner_impl_ptr = moved_attr.get_impl().get();

    ASSERT_EQ(orig_inner_impl_ptr, moved_inner_impl_ptr);
    ASSERT_TRUE(moved_attr.get_impl());
    ASSERT_TRUE(!orig_attr.get_impl());

    /* move assignment test*/
    auto orig_attr2 = ccl::create_datatype_attr();
    auto moved_attr2 = ccl::create_datatype_attr();
    moved_attr2 = std::move(orig_attr2);

    ASSERT_TRUE(moved_attr2.get_impl());
    ASSERT_TRUE(!orig_attr2.get_impl());
}

TEST(datatype_attr, datatype_attr_empty_size) {
    auto attr = ccl::create_datatype_attr(ccl::attr_val<ccl::datatype_attr_id::size>(123));
    ASSERT_TRUE(attr.get<ccl::datatype_attr_id::version>().full != nullptr);

    ASSERT_EQ(attr.get<ccl::datatype_attr_id::size>(), 123);

    auto old_value = attr.set<ccl::datatype_attr_id::size>(1234);
    ASSERT_EQ(attr.get<ccl::datatype_attr_id::size>(), 1234);
    ASSERT_EQ(old_value, 123);
}

} // namespace datatype_attr_suite
