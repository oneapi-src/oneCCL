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
#include "oneapi/ccl/ccl_aliases.hpp"
#include "oneapi/ccl/ccl_device_types.hpp"
#include "oneapi/ccl/ccl_type_traits.hpp"
#include "oneapi/ccl/ccl_types_policy.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"

#include "oneapi/ccl/ccl_coll_attr_ids.hpp"
#include "oneapi/ccl/ccl_coll_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_coll_attr.hpp"

#include "common/event/event_internal/event_internal_attr_ids.hpp"
#include "common/event/event_internal/event_internal_attr_ids_traits.hpp"
#include "common/event/event_internal/event_internal.hpp"

#include "oneapi/ccl/ccl_stream_attr_ids.hpp"
#include "oneapi/ccl/ccl_stream_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_stream.hpp"

#include "oneapi/ccl/ccl_event.hpp"
#include "oneapi/ccl/ccl_communicator.hpp"

#include "common/comm/l0/comm_context_id.hpp"
#include "common/comm/comm_interface.hpp"

namespace ccl {

class host_communicator;
struct gpu_comm_attr;
using shared_communicator_t = std::shared_ptr<host_communicator>;

class comm_group {
public:
    friend class environment;
    friend struct group_context;

    using context_t = typename unified_device_context_type::ccl_native_t;

    ~comm_group();
    /**
     * Device Communicator creation API: single communicator creation, based on @device
     */
    template <class DeviceType,
              class ContextType,
              typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                       ccl::device_index_type>::value,
                                      int>::type = 0>
    ccl::communicator_interface_ptr create_communicator_from_group(const DeviceType& device,
                                        ContextType& context,
                                        const comm_split_attr& attr = ccl_empty_attr());

    /**
     * Device Communicator creation API: single communicator creation, based on index @device_id
     */
    template <class DeviceType,
              class ContextType,
              typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                      int>::type = 0>
    ccl::communicator_interface_ptr create_communicator_from_group(const DeviceType& device_id,
                                        ContextType& context,
                                        const comm_split_attr& attr = ccl_empty_attr());

    /**
     * Device Communicator creation vectorized API:
     * multiple communicator creation, based on devices iterator @InputIt
     */
    template <class InputIt, class ContextType>
    std::vector<communicator> create_communicators_group(InputIt first,
                                                         InputIt last,
                                                         ContextType& context,
                                                         comm_split_attr attr = ccl_empty_attr());

    /**
     * Device Communicator creation vectorized API:
     * multiple communicator creation, based on devices of @Type, packed into container @Container
     */
    template <template <class...> class Container, class Type, class ContextType>
    std::vector<communicator> create_communicators_group(const Container<Type>& device_ids,
                                                         ContextType& context,
                                                         comm_split_attr attr = ccl_empty_attr());

    /**
     * Return device context allocated during group creation
     */
    //device_context_native_const_reference_t get_context() const;

    bool sync_group_size(size_t device_group_size);
    /*
    std::string to_string() const;
*/
    const group_unique_key& get_unique_id() const;

private:
    comm_group(ccl::shared_communicator_t comm,
               size_t current_device_group_size,
               size_t process_device_group_size,
               group_unique_key id);
    std::unique_ptr<gpu_comm_attr> pimpl;
};
} // namespace ccl
