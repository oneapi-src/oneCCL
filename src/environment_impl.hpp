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
#include "oneapi/ccl/ccl_environment.hpp"

#include "coll/coll_attributes.hpp"

#include "common/comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"
#include "comm_split_attr_creation_impl.hpp"

#include "oneapi/ccl/ccl_communicator.hpp"

#include "common/comm/l0/comm_context_storage.hpp"

#include "event_impl.hpp"
#include "stream_impl.hpp"

#include "common/global/global.hpp"
#include "common/comm/comm.hpp"

#include "oneapi/ccl/ccl_device_communicator.hpp"

#include "oneapi/ccl/native_device_api/export_api.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif

#define CCL_CHECK_AND_THROW(result, diagnostic) \
    do { \
        if (result != ccl_status_success) { \
            throw ccl_error(diagnostic); \
        } \
    } while (0);

namespace ccl {

/*
template <class ...attr_value_pair_t>
comm_split_attr CCL_API environment::create_comm_split_attr(attr_value_pair_t&&...avps) const
{
    return comm_split_attr::create_comm_split_attr(std::forward<attr_value_pair_t>(avps)...);
}
*/
//Device communicator
#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
/*
template <class ...attr_value_pair_t>
device_comm_split_attr environment::create_device_comm_split_attr(attr_value_pair_t&&...avps) const
{
    return device_comm_split_attr::create_device_comm_split_attr(std::forward<attr_value_pair_t>(avps)...);
}
*/
template <class DeviceType, class ContextType>
vector_class<device_communicator> CCL_API
environment::create_device_communicators(const size_t devices_size,
                                         const vector_class<DeviceType>& local_devices,
                                         ContextType& context,
                                         shared_ptr_class<kvs_interface> kvs) const {
#ifdef MULTI_GPU_SUPPORT
    return device_communicator::create_device_communicators(
        devices_size, local_devices, context, kvs);
#else
    return {};
#endif
}

template <class DeviceType, class ContextType>
vector_class<device_communicator> CCL_API environment::create_device_communicators(
    const size_t comm_size,
    const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
    ContextType& context,
    shared_ptr_class<kvs_interface> kvs) const {
#ifdef MULTI_GPU_SUPPORT
    return device_communicator::create_device_communicators(
        comm_size, local_rank_device_map, context, kvs);
#else
    (void)context;
    vector_class<device_communicator> ret;
    ret.push_back(create_single_device_communicator(comm_size,
                                                    local_rank_device_map.begin()->first,
                                                    local_rank_device_map.begin()->second,
                                                    kvs));
    return ret;
#endif
}

template <class DeviceType, class ContextType>
vector_class<device_communicator> CCL_API
environment::create_device_communicators(const size_t comm_size,
                                         const map_class<rank_t, DeviceType>& local_rank_device_map,
                                         ContextType& context,
                                         shared_ptr_class<kvs_interface> kvs) const {
#ifdef MULTI_GPU_SUPPORT
    return device_communicator::create_device_communicators(
        comm_size, local_rank_device_map, context, kvs);
#else
    (void)context;
    vector_class<device_communicator> ret;
    ret.push_back(create_single_device_communicator(comm_size,
                                                    local_rank_device_map.begin()->first,
                                                    local_rank_device_map.begin()->second,
                                                    kvs));
    return ret;
#endif
}

//Stream
template <class native_stream_type, typename T>
stream CCL_API environment::create_stream(native_stream_type& native_stream) {
    return stream::create_stream(native_stream);
}

template <class native_stream_type, class native_context_type, typename T>
stream CCL_API environment::create_stream(native_stream_type& native_stream,
                                          native_context_type& native_ctx) {
    return stream::create_stream(native_stream, native_ctx);
}

//Event
template <class event_type, typename T>
event CCL_API environment::create_event(event_type& native_event) {
    return event::create_event(native_event);
}

template <class event_handle_type, typename T>
event CCL_API
environment::create_event(event_handle_type native_event_handle,
                          typename unified_device_context_type::ccl_native_t context) {
    return event::create_event(native_event_handle, context);
}
#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

template <class ccl_api_type, class... args_type>
ccl_api_type CCL_API environment::create_postponed_api_type(args_type... args) const {
    ccl::library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;
    // TODO: ccl_api_type is private constructor, so `static_cast`  fails always. Fix it
    //static_assert(std::is_constructible<ccl_api_type, args_type..., ccl::library_version>::value, "Cannot construct `ccl_api_type` from given `args_type...`");
    return ccl_api_type(std::forward<args_type>(args)..., ret);
}
} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define CREATE_OP_ATTR_INSTANTIATION(Attr) \
    template Attr CCL_API ccl::environment::create_postponed_api_type<Attr>() const;

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

#define CREATE_DEV_COMM_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::device_communicator> CCL_API \
    ccl::environment::create_device_communicators<DeviceType, ContextType>( \
        const size_t devices_size, \
        const ccl::vector_class<DeviceType>& local_devices, \
        ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs) const; \
\
    template ccl::vector_class<ccl::device_communicator> CCL_API \
    ccl::environment::create_device_communicators<DeviceType, ContextType>( \
        const size_t cluster_devices_size, \
        const ccl::vector_class<ccl::pair_class<ccl::rank_t, DeviceType>>& local_devices, \
        ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs) const; \
\
    template ccl::vector_class<ccl::device_communicator> CCL_API \
    ccl::environment::create_device_communicators<DeviceType, ContextType>( \
        const size_t cluster_devices_size, \
        const ccl::map_class<ccl::rank_t, DeviceType>& local_devices, \
        ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs) const;

#define CREATE_STREAM_INSTANTIATION(native_stream_type) \
    template ccl::stream CCL_API ccl::environment::create_stream(native_stream_type& native_stream);

#define CREATE_STREAM_EXT_INSTANTIATION(device_type, native_context_type) \
    template ccl::stream CCL_API ccl::environment::create_stream(device_type& device, \
                                                                 native_context_type& native_ctx);

#define CREATE_EVENT_INSTANTIATION(native_event_type) \
    template ccl::event CCL_API ccl::environment::create_event(native_event_type& native_event);

#define CREATE_EVENT_EXT_INSTANTIATION(event_handle_type) \
    template ccl::event CCL_API ccl::environment::create_event( \
        event_handle_type native_event_handle, \
        typename unified_device_context_type::ccl_native_t context);

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
