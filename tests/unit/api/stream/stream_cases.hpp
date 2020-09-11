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

// Core file with PIMPL implementation
#include "environment.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

namespace stream_suite {
TEST(stream_api, stream_from_device_creation) {
    auto dev = native::get_platform().get_device(ccl::from_string("[0:6459]"));
    auto str = ccl::stream::create_stream_from_attr(dev);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);

    auto assigned_dev = str.get<ccl::stream_attr_id::device>();
    ASSERT_EQ(assigned_dev.get(), dev.get());
}

TEST(stream_api, stream_from_device_context_creation) {
    auto dev = native::get_platform().get_device(ccl::from_string("[0:6459]"));
    auto ctx = std::make_shared<native::ccl_context>(); //TODO stub at moment
    auto str = ccl::stream::create_stream_from_attr(dev, ctx);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);

    auto assigned_dev = str.get<ccl::stream_attr_id::device>();
    ASSERT_EQ(assigned_dev.get(), dev.get());

    auto assigned_ctx = str.get<ccl::stream_attr_id::context>();
    ASSERT_EQ(assigned_ctx.get(), ctx.get());
}

TEST(stream_api, stream_creation_from_native) {
    auto dev = native::get_platform().get_device(ccl::from_string("[0:6459]"));
    auto queue = dev->create_cmd_queue();

    //TODO HACK
    typename ccl::unified_stream_type::ccl_native_t *s =
        new typename ccl::unified_stream_type::ccl_native_t(&queue);
    auto str = ccl::stream::create_stream(*s);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);

    auto assigned_handle = str.get<ccl::stream_attr_id::native_handle>();
    ASSERT_EQ(assigned_handle, *s);
}

#if 0
TEST(stream_api, stream_creation_from_native_handle)
{
    ze_command_queue_handle_t h;
    auto str = ccl::stream::create_stream(h);

    ASSERT_TRUE(str.get<ccl::stream_attr_id::version>().full != nullptr);

    auto assigned_handle = str.get<ccl::stream_attr_id::native_handle>();
    ASSERT_EQ(assigned_handle->get(), h);
    (void)assigned_handle;
}
#endif
} // namespace stream_suite
#undef protected
#undef private
