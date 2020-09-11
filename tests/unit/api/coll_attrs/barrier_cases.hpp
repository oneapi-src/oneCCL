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
namespace coll_attr_suite {
TEST(coll_attr, barrier_attr_creation) {
    auto attr = ccl::create_coll_attr<ccl::barrier_attr>();

    ASSERT_TRUE(attr.get<ccl::operation_attr_id::version>().full != nullptr);
}

TEST(coll_attr, barrier_copy_attr) {
    auto attr = ccl::create_coll_attr<ccl::barrier_attr>();

    auto original_inner_impl_ptr = attr.get_impl().get();
    auto attr2 = attr;
    auto copied_inner_impl_ptr = attr2.get_impl().get();
    ASSERT_TRUE(original_inner_impl_ptr != copied_inner_impl_ptr);
    ASSERT_TRUE(attr.get_impl());
}

TEST(coll_attr, barrier_move_attr) {
    auto attr = ccl::create_coll_attr<ccl::barrier_attr>();

    auto original_inner_impl_ptr = attr.get_impl().get();

    auto attr2 = (std::move(attr));
    auto moved_inner_impl_ptr = attr2.get_impl().get();
    ASSERT_EQ(original_inner_impl_ptr, moved_inner_impl_ptr);
    ASSERT_TRUE(not attr.get_impl());
}
} // namespace coll_attr_suite
