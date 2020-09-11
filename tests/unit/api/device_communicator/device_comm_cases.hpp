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

#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "oneapi/ccl/ccl_event_attr_ids.hpp"
#include "oneapi/ccl/ccl_event_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_event.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "oneapi/ccl/ccl_request.hpp"

#include "coll/coll_attributes.hpp"

#include "common/comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"
#include "comm_split_attr_creation_impl.hpp"

//#include "environment.hpp"
#include "oneapi/ccl/ccl_device_communicator.hpp"
#include "common/comm/l0/comm_context_storage.hpp"

#include "event_impl.hpp"
#include "stream_impl.hpp"

#include "../stubs/kvs.hpp"

#include "common/global/global.hpp"

#include "../stubs/native_platform.hpp"

//TODO
#include "common/comm/comm.hpp"

#include "common/comm/l0/comm_context.hpp"
#include "device_communicator_impl.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

namespace device_communicator_suite {

TEST(device_communicator_api, device_comm_from_device_index) {
    size_t total_devices_size = 4;
    ccl::vector_class<ccl::device_index_type> devices{ total_devices_size,
                                                       ccl::from_string("[0:6459]") };
    auto ctx = std::make_shared<native::ccl_context>(); //TODO stub at moment
    std::shared_ptr<stub_kvs> stub_storage;

    ccl::vector_class<ccl::pair_class<size_t, ccl::device_index_type>> local_rank_device_map;
    local_rank_device_map.reserve(total_devices_size);
    size_t curr_rank = 0;
    std::transform(devices.begin(),
                   devices.end(),
                   std::back_inserter(local_rank_device_map),
                   [&curr_rank](ccl::device_index_type& val) {
                       return std::make_pair(curr_rank++, val);
                   });

    ccl::vector_class<ccl::device_communicator> comms =
        ccl::device_communicator::create_device_communicators(
            total_devices_size, local_rank_device_map, ctx, stub_storage);
    ASSERT_EQ(comms.size(), devices.size());

    for (const auto& dev_comm : comms) {
        //ASSERT_TRUE(dev_comm.is_ready());
        ASSERT_EQ(dev_comm.size(), total_devices_size);
    }
}
} // namespace device_communicator_suite
