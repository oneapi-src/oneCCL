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
#include "oneapi/ccl/ccl_event_attr_ids.hpp"
#include "oneapi/ccl/ccl_event_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "environment.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

namespace stream_suite {

TEST(event_api, event_from_native_event_creation) {
    std::shared_ptr<native::ccl_device::device_event> nev;
    auto ev = ccl::event::create_event(nev);

    ASSERT_TRUE(ev.get<ccl::event_attr_id::version>().full != nullptr);
    auto assigned_handle = ev.get<ccl::event_attr_id::native_handle>();
    ASSERT_EQ(assigned_handle->get(), nev->get());
}

TEST(event_api, event_from_native_device_context_creation) {
    ze_event_handle_t h;
    std::shared_ptr<native::ccl_context> ctx;
    auto ev = ccl::event::create_event_from_attr(h, ctx);

    ASSERT_TRUE(ev.get<ccl::event_attr_id::version>().full != nullptr);
}

} // namespace stream_suite
#undef protected
#undef private
