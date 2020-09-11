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
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "comm_split_attr_creation_impl.hpp"

// Core file with PIMPL implementation
#include "common/comm/comm_split_common_attr.hpp"

#undef protected
#undef private

namespace host_comm_split_attr_suite {

TEST(host_comm_split_attr, host_comm_split_attr_empty_creation) {
    ccl::comm_split_attr attr = ccl::create_comm_split_attr();
    ASSERT_TRUE(attr.is_valid<ccl::comm_split_attr_id::version>());
    ASSERT_TRUE(attr.get<ccl::comm_split_attr_id::version>().full != nullptr);
    ASSERT_TRUE(!attr.is_valid<ccl::comm_split_attr_id::color>());
    ASSERT_TRUE(!attr.is_valid<ccl::comm_split_attr_id::group>());
}

TEST(host_comm_split_attr, host_comm_split_attr_color) {
    auto attr = ccl::create_comm_split_attr(ccl::attr_val<ccl::comm_split_attr_id::color>(123));
    ASSERT_TRUE(attr.get<ccl::comm_split_attr_id::version>().full != nullptr);

    ASSERT_TRUE(attr.is_valid<ccl::comm_split_attr_id::color>());
    ASSERT_EQ(attr.get<ccl::comm_split_attr_id::color>(), 123);

    auto old_value = attr.set<ccl::comm_split_attr_id::color>(1234);
    ASSERT_TRUE(attr.is_valid<ccl::comm_split_attr_id::color>());
    ASSERT_EQ(attr.get<ccl::comm_split_attr_id::color>(), 1234);
    ASSERT_EQ(old_value, 123);
}

TEST(host_comm_split_attr, host_comm_split_attr_group) {
    auto attr = ccl::create_comm_split_attr(
        ccl::attr_val<ccl::comm_split_attr_id::group>(ccl::ccl_group_split_type::thread));
    ASSERT_TRUE(attr.get<ccl::comm_split_attr_id::version>().full != nullptr);

    ASSERT_TRUE(attr.is_valid<ccl::comm_split_attr_id::group>());
    ASSERT_EQ(attr.get<ccl::comm_split_attr_id::group>(), ccl::ccl_group_split_type::thread);

    auto old_value = attr.set<ccl::comm_split_attr_id::group>(ccl::ccl_group_split_type::process);
    ASSERT_TRUE(attr.is_valid<ccl::comm_split_attr_id::group>());
    ASSERT_EQ(attr.get<ccl::comm_split_attr_id::group>(), ccl::ccl_group_split_type::process);
    ASSERT_EQ(old_value, ccl::ccl_group_split_type::thread);
}

TEST(host_comm_split_attr, copy_on_write_host_comm_split_attr) {
    auto attr = ccl::create_comm_split_attr(
        ccl::attr_val<ccl::comm_split_attr_id::group>(ccl::ccl_group_split_type::thread));

    auto original_inner_impl_ptr = attr.get_impl();

    ASSERT_EQ(attr.get<ccl::comm_split_attr_id::group>(), ccl::ccl_group_split_type::thread);

    //set new val
    attr.set<ccl::comm_split_attr_id::group>(ccl::ccl_group_split_type::process);
    ASSERT_EQ(attr.get<ccl::comm_split_attr_id::group>(), ccl::ccl_group_split_type::process);

    //make sure original impl is unchanged
    ASSERT_TRUE(original_inner_impl_ptr != attr.get_impl());
    ASSERT_EQ(std::static_pointer_cast<ccl::ccl_host_comm_split_attr_impl>(original_inner_impl_ptr)
                  ->get_attribute_value(
                      ccl::details::ccl_host_split_traits<ccl::comm_split_attr_id,
                                                          ccl::comm_split_attr_id::group>{}),
              ccl::ccl_group_split_type::thread);
}

TEST(host_comm_split_attr, copy_host_comm_split_attr) {
    auto attr = ccl::create_comm_split_attr();
    attr.set<ccl::comm_split_attr_id::color>(666);

    auto original_inner_impl_ptr = attr.get_impl().get();
    auto attr2 = attr;
    auto copied_inner_impl_ptr = attr2.get_impl().get();
    ASSERT_TRUE(original_inner_impl_ptr != copied_inner_impl_ptr);
    ASSERT_TRUE(attr.get_impl());
    ASSERT_TRUE(attr2.get<ccl::comm_split_attr_id::version>().full != nullptr);
    ASSERT_EQ(attr2.get<ccl::comm_split_attr_id::color>(), 666);
}

TEST(host_comm_split_attr, move_host_comm_split_attr) {
    /* move constructor test */
    auto orig_attr =
        ccl::create_comm_split_attr(ccl::attr_val<ccl::comm_split_attr_id::color>(667));

    auto orig_inner_impl_ptr = orig_attr.get_impl().get();
    auto moved_attr = (std::move(orig_attr));
    auto moved_inner_impl_ptr = moved_attr.get_impl().get();

    ASSERT_EQ(orig_inner_impl_ptr, moved_inner_impl_ptr);
    ASSERT_TRUE(!orig_attr.get_impl());
    ASSERT_TRUE(moved_attr.get<ccl::comm_split_attr_id::version>().full != nullptr);
    ASSERT_TRUE(moved_attr.is_valid<ccl::comm_split_attr_id::color>());
    ASSERT_EQ(moved_attr.get<ccl::comm_split_attr_id::color>(), 667);

    /* move assignment test*/
    auto orig_attr2 =
        ccl::create_comm_split_attr(ccl::attr_val<ccl::comm_split_attr_id::color>(123));

    auto moved_attr2 = ccl::create_comm_split_attr();
    moved_attr2 = std::move(orig_attr2);

    ASSERT_TRUE(!orig_attr2.get_impl());
    ASSERT_TRUE(moved_attr2.is_valid<ccl::comm_split_attr_id::color>());
    ASSERT_TRUE(moved_attr2.get<ccl::comm_split_attr_id::version>().full != nullptr);
    ASSERT_EQ(moved_attr2.get<ccl::comm_split_attr_id::color>(), 123);
}

TEST(host_comm_split_attr, host_comm_split_attr_valid) {
    auto attr = ccl::create_comm_split_attr();

    ASSERT_TRUE(!attr.is_valid<ccl::comm_split_attr_id::color>());
    ASSERT_TRUE(!attr.is_valid<ccl::comm_split_attr_id::group>());

    attr.set<ccl::comm_split_attr_id::group>(ccl::ccl_group_split_type::process);
    ASSERT_TRUE(attr.is_valid<ccl::comm_split_attr_id::group>());
    ASSERT_TRUE(!attr.is_valid<ccl::comm_split_attr_id::color>());

    auto attr2 = ccl::create_comm_split_attr();

    try {
        attr.get<ccl::comm_split_attr_id::color>();
        ASSERT_TRUE(false); // must never happen
    }
    catch (...) {
        ASSERT_TRUE(!attr.is_valid<ccl::comm_split_attr_id::color>());
    }
}

} // namespace host_comm_split_attr_suite
