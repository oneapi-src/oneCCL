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
TEST(coll_attr, sparse_allreduce_attr_creation) {
    const void* fn_ctx = nullptr;
    auto attr = ccl::create_coll_attr<ccl::sparse_allreduce_attr>(
        ccl::attr_val<ccl::sparse_allreduce_attr_id::fn_ctx>(fn_ctx));

    ASSERT_TRUE(attr.get<ccl::operation_attr_id::version>().full != nullptr);
    ASSERT_EQ(attr.get<ccl::sparse_allreduce_attr_id::fn_ctx>(), fn_ctx);
}

} // namespace coll_attr_suite
