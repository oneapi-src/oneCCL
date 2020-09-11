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

void stub_reduction(const void*, size_t, void*, size_t*, ccl::datatype, const ccl::fn_context*) {}

TEST(coll_attr, allreduce_attr_creation) {
    ccl::reduction_fn function{ nullptr };

    auto attr = ccl::create_coll_attr<ccl::allreduce_attr>(
        ccl::attr_val<ccl::allreduce_attr_id::reduction_fn>(function));

    ASSERT_TRUE(attr.get<ccl::operation_attr_id::version>().full != nullptr);
    ASSERT_EQ(attr.get<ccl::allreduce_attr_id::reduction_fn>().get(), function);
}

TEST(coll_attr, allreduce_copy_on_write_attr) {
    ccl::reduction_fn function{ nullptr };

    auto attr = ccl::create_coll_attr<ccl::allreduce_attr>(
        ccl::attr_val<ccl::allreduce_attr_id::reduction_fn>(function));

    auto original_inner_impl_ptr = attr.get_impl();

    ASSERT_EQ(attr.get<ccl::allreduce_attr_id::reduction_fn>().get(), function);

    //set new val
    {
        ccl::details::function_holder<ccl::reduction_fn> check_val{ stub_reduction };
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)stub_reduction);
        ASSERT_EQ(attr.get<ccl::allreduce_attr_id::reduction_fn>().get(), check_val.get());
    }

    //make sure original impl is unchanged
    ASSERT_TRUE(original_inner_impl_ptr != attr.get_impl());
    ASSERT_EQ(
        original_inner_impl_ptr
            ->get_attribute_value(
                ccl::details::ccl_api_type_attr_traits<ccl::allreduce_attr_id,
                                                       ccl::allreduce_attr_id::reduction_fn>{})
            .get(),
        function);
}

TEST(coll_attr, allreduce_copy_attr) {
    auto attr = ccl::create_coll_attr<ccl::allreduce_attr>(
        ccl::attr_val<ccl::allreduce_attr_id::reduction_fn>(stub_reduction));

    auto original_inner_impl_ptr = attr.get_impl().get();
    auto attr2 = attr;
    auto copied_inner_impl_ptr = attr2.get_impl().get();
    ASSERT_TRUE(original_inner_impl_ptr != copied_inner_impl_ptr);
    ASSERT_TRUE(attr.get_impl());
}

TEST(coll_attr, allreduce_move_attr) {
    auto attr = ccl::create_coll_attr<ccl::allreduce_attr>(
        ccl::attr_val<ccl::allreduce_attr_id::reduction_fn>(stub_reduction));

    auto original_inner_impl_ptr = attr.get_impl().get();

    auto attr2 = (std::move(attr));
    auto moved_inner_impl_ptr = attr2.get_impl().get();
    ASSERT_EQ(original_inner_impl_ptr, moved_inner_impl_ptr);
    ASSERT_TRUE(not attr.get_impl());
}
} // namespace coll_attr_suite
