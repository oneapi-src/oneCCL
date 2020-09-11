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

#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "environment.hpp"

namespace stream_suite {

TEST(stream_api, stream_from_sycl_queue) {
    auto q = cl::sycl::queue();
    auto str = ccl::stream::create_stream(q);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);
}

TEST(stream_api, stream_from_sycl_queue_handle) {
    //auto q = cl::sycl::queue();
    auto dev = cl::sycl::device();
    auto ctx = cl::sycl::context(dev);
    //cl_command_queue h = q.get();

    auto str = ccl::stream::create_stream(dev, ctx);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);
}
TEST(stream_api, stream_from_sycl_device_creation) {
    auto dev = cl::sycl::device();
    auto str = ccl::stream::create_stream_from_attr(dev);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);
}

TEST(stream_api, stream_from_sycl_device_context_creation) {
    auto dev = cl::sycl::device();
    auto ctx = cl::sycl::context(dev);
    auto str = ccl::stream::create_stream_from_attr(dev, ctx);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);
}

TEST(stream_api, stream_from_sycl_device_context_creation_with_attr) {
    auto dev = cl::sycl::device();
    auto ctx = cl::sycl::context(dev);
    auto str =
        ccl::stream::create_stream_from_attr(dev,
                                             ctx,
                                             ccl::attr_val<ccl::stream_attr_id::ordinal>(1),
                                             ccl::attr_val<ccl::stream_attr_id::priority>(100));

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);

    ASSERT_EQ(str.get<ccl::stream_attr_id::ordinal>(), 1);
    ASSERT_EQ(str.get<ccl::stream_attr_id::priority>(), 100);

    bool catched = false;
    try {
        str.set<ccl::stream_attr_id::priority>(99);
    }
    catch (const ccl::ccl_error& ex) {
        catched = true;
    }
    ASSERT_TRUE(catched);
}
} // namespace stream_suite
#undef protected
#undef private
