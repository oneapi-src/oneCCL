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
#if 0
#include "oneapi/ccl.hpp"

#include "coll/coll_attributes.hpp"

#include "common/comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"

#include "common/comm/l0/comm_context_storage.hpp"

#include "common/event/event_internal/event_internal_impl.hpp"
#include "stream_impl.hpp"

#include "common/global/global.hpp"
#include "common/comm/comm.hpp"

#include "common/comm/l0/comm_context.hpp"
#include "oneapi/ccl/ccl_communicator.hpp"

#include "common/global/global.hpp"
#include "exec/exec.hpp"

#include "common/comm/comm_interface.hpp"

#include "oneapi/ccl/native_device_api/export_api.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif

#define CCL_CHECK_AND_THROW(result, diagnostic) \
    do { \
        if (result != ccl_status_success) { \
            throw ccl::exception(diagnostic); \
        } \
    } while (0);


namespace ccl
{

CCL_API ccl::environment::environment()
{
    static auto result = global_data::get().init();
    CCL_CHECK_AND_THROW(result, "failed to initialize CCL");
}

CCL_API ccl::environment::~environment()
{}

CCL_API ccl::environment& ccl::environment::instance()
{
    static ccl::environment env;
    return env;
}

void CCL_API ccl::environment::set_resize_fn(ccl_resize_fn_t callback)
{
    ccl_status_t result = ccl_set_resize_fn(callback);
    CCL_CHECK_AND_THROW(result, "failed to set resize callback");
    return;
}

ccl::library_version CCL_API ccl::environment::get_version() const
{
    ccl::library_version ret;
    ccl_status_t result = ccl_get_version(&ret);
    CCL_CHECK_AND_THROW(result, "failed to get version");
    return ret;
}
/*
static ccl::stream& get_empty_stream()
{
    static ccl::stream_t empty_stream  = ccl::environment::instance().create_stream();
    return empty_stream;
}
*/

/**
 * Factory methods
 */
// KVS
kvs_t CCL_API environment::create_main_kvs() const
{
    return std::shared_ptr<kvs>(new kvs);
}

kvs_t CCL_API environment::create_kvs(const kvs::addr_t& addr) const
{
    return std::shared_ptr<kvs>(new kvs(addr));
}

//Communicator
communicator CCL_API environment::create_communicator() const
{
    return communicator::create_communicator();
}

communicator CCL_API environment::create_communicator(const size_t size,
                                       shared_ptr_class<kvs_interface> kvs) const
{
    return communicator::create_communicator(size, kvs);
}

communicator CCL_API environment::create_communicator(const size_t size,
                                     const size_t rank,
                                     shared_ptr_class<kvs_interface> kvs) const
{
    return communicator::create_communicator(size, rank, kvs);
}

//Device communicator
#ifdef MULTI_GPU_SUPPORT

template <class ...attr_value_pair_t>
comm_split_attr environment::create_comm_split_attr(attr_value_pair_t&&...avps) const
{
    return comm_split_attr::create_comm_split_attr(std::forward<attr_value_pair_t>(avps)...);
}

template<class DeviceType,
             class ContextType>
vector_class<communicator> CCL_API environment::create_communicators(
        const size_t devices_size,
        const vector_class<DeviceType>& local_devices,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs) const
{
    return communicator::create_communicators(devices_size, local_devices, context, kvs);
}

template<class DeviceType,
         class ContextType>
vector_class<communicator> CCL_API environment::create_communicators(
        const size_t cluster_devices_size, /*global devics count*/
        const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs)
{
    return communicator::create_communicators(cluster_devices_size, local_rank_device_map, context, kvs);
}


template<class DeviceType,
         class ContextType>
vector_class<communicator> CCL_API environment::create_communicators(
        const size_t cluster_devices_size, /*global devics count*/
        const map_class<rank_t, DeviceType>& local_rank_device_map,
        ContextType& context,
        shared_ptr_class<kvs_interface> kvs)
{
    return communicator::create_communicators(cluster_devices_size, local_rank_device_map, context, kvs);
}


//Stream
template <class native_stream_type,
          typename T>
stream CCL_API environment::create_stream(native_stream_type& native_stream)
{
    return stream::create_stream(native_stream);
}

template <class native_stream_type, class native_context_type,
          typename T>
stream CCL_API environment::create_stream(native_stream_type& native_stream, native_context_type& native_ctx)
{
    return stream::create_stream(native_stream, native_ctx);
}

template <class ...attr_value_pair_t>
stream CCL_API environment::create_stream_from_attr(typename unified_device_type::ccl_native_t device, attr_value_pair_t&&...avps)
{
    return stream::create_stream_from_attr(device, std::forward<attr_value_pair_t>(avps)...);
}

template <class ...attr_value_pair_t>
stream CCL_API environment::create_stream_from_attr(typename unified_device_type::ccl_native_t device,
                               typename unified_device_context_type::ccl_native_t context,
                               attr_value_pair_t&&...avps)
{
    return stream::create_stream_from_attr(device, context, std::forward<attr_value_pair_t>(avps)...);
}


//Event
template <class event_type,
          typename T>
event CCL_API environment::create_event(event_type& native_event)
{
    return event::create_event(native_event);
}

template <class event_type,
          class ...attr_value_pair_t>
event CCL_API environment::create_event_from_attr(event_type& native_event_handle,
                             typename unified_device_context_type::ccl_native_t context,
                             attr_value_pair_t&&...avps)
{
    return event::create_event_from_attr(native_event_handle, context,  std::forward<attr_value_pair_t>(avps)...);
}
/*
#define STREAM_CREATOR_INSTANTIATION(type)                                                                                                           \
template ccl::stream_t CCL_API ccl::environment::create_stream(type& stream);

#ifdef CCL_ENABLE_SYCL
STREAM_CREATOR_INSTANTIATION(cl::sycl::queue)
#endif
*/
#endif //MULTI_GPU_SUPPORT
}
#include "types_generator_defines.hpp"
#include "oneapi/ccl/ccl_cpp_api_explicit_in.hpp"
#endif //0
