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
#include "oneapi/ccl/ccl_comm_split_attr_ids.hpp"
#include "oneapi/ccl/ccl_comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/ccl_comm_split_attr.hpp"
#include "oneapi/ccl/ccl_device_communicator.hpp"

#include "common/comm/l0/comm_context_id.hpp"
//TODO
/*
namespace ccl
{
struct comm_split_attr_impl
{
    constexpr static int color_default()
    {
        return 0;
    }
    ccl::library_version version;
};

struct device_attr_impl
{
    constexpr static device_topology_type class_default()
    {
        return device_topology_type::ring;
    }
    constexpr static device_group_split_type group_default()
    {
        return device_group_split_type::process;
    }
    device_topology_type current_preferred_topology_class = class_default();
    device_group_split_type current_preferred_topology_group = group_default();
};
}*/
#include "common/comm/comm_interface.hpp"

namespace ccl {
/* TODO temporary function for UT compilation: would be part of ccl::environment in final
template <class event_type,
          class ...attr_value_pair_t>
event create_event_from_attr(event_type& native_event_handle,
                             typename unified_device_context_type::ccl_native_t context,
                             attr_value_pair_t&&...avps)
{
    ccl::library_version ret {};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    event str {event::impl_value_t(new event::impl_t(native_event_handle, context, ret))};
    int expander [] {(str.template set<attr_value_pair_t::idx()>(avps.val()), 0)...};
    str.build_from_params();

    return str;
}
*/

template <class DeviceType, class ContextType>
vector_class<device_communicator> device_communicator::create_device_communicators(
    const size_t cluster_devices_size,
    const vector_class<DeviceType>& local_devices,
    ContextType& context,
    shared_ptr_class<kvs_interface> kvs) {
    vector_class<device_communicator> ret;
    throw std::runtime_error(std::string(__FUNCTION__) + " - not implemented");
    return ret;
}

using rank_t = size_t;

template <class DeviceType, class ContextType>
vector_class<device_communicator> device_communicator::create_device_communicators(
    const size_t cluster_devices_size, /*global devics count*/
    const vector_class<pair_class<rank_t, DeviceType>>& local_rank_device_map,
    ContextType& context,
    shared_ptr_class<kvs_interface> kvs) {
#ifdef MULTI_GPU_SUPPORT
    vector_class<rank_t> local_thread_ranks;
    local_thread_ranks.reserve(local_rank_device_map.size());
    std::transform(
        local_rank_device_map.begin(),
        local_rank_device_map.end(),
        std::back_inserter(local_thread_ranks),
        [](const typename vector_class<pair_class<rank_t, DeviceType>>::value_type& val) {
            return val.first;
        });
    group_context::comm_group_t thread_group =
        group_context::instance().group_by_kvs(local_thread_ranks, cluster_devices_size, kvs);

    vector_class<DeviceType> local_thread_devices;
    local_thread_devices.reserve(local_rank_device_map.size());
    std::transform(
        local_rank_device_map.begin(),
        local_rank_device_map.end(),
        std::back_inserter(local_thread_devices),
        [](const typename vector_class<pair_class<rank_t, DeviceType>>::value_type& val) {
            return val.second;
        });

    auto ret = thread_group->create_communicators(local_thread_devices);
    return ret;
#endif
    return {};
}

template <class DeviceType, class ContextType>
vector_class<device_communicator> device_communicator::create_device_communicators(
    const size_t cluster_devices_size, /*global devics count*/
    const map_class<rank_t, DeviceType>& local_rank_device_map,
    ContextType& context,
    shared_ptr_class<kvs_interface> kvs)

{
#ifdef MULTI_GPU_SUPPORT
    vector_class<rank_t> local_thread_ranks;
    local_thread_ranks.reserve(local_rank_device_map.size());
    std::transform(local_rank_device_map.begin(),
                   local_rank_device_map.end(),
                   std::back_inserter(local_thread_ranks),
                   [](const typename map_class<rank_t, DeviceType>::value_type& val) {
                       return val.first;
                   });
    group_context::comm_group_t thread_group =
        group_context::instance().group_by_kvs(local_thread_ranks, cluster_devices_size, kvs);

    vector_class<DeviceType> local_thread_devices;
    local_thread_devices.reserve(local_rank_device_map.size());
    std::transform(local_rank_device_map.begin(),
                   local_rank_device_map.end(),
                   std::back_inserter(local_thread_devices),
                   [](const typename map_class<rank_t, DeviceType>::value_type& val) {
                       return val.second;
                   });

    auto ret = thread_group->create_communicators(local_thread_devices);
    return ret;
#endif
    return {};
}

CCL_API device_communicator::device_communicator(impl_value_t&& impl) : base_t(std::move(impl)) {}

CCL_API device_communicator::device_communicator(device_communicator&& src)
        : base_t(std::move(src)) {}

CCL_API device_communicator& device_communicator::operator=(device_communicator&& src) {
    if (src.get_impl() != this->get_impl()) {
        src.get_impl().swap(this->get_impl());
        src.get_impl().reset();
    }
    return *this;
}

CCL_API ccl::device_communicator::~device_communicator() {}

CCL_API size_t ccl::device_communicator::rank() const {
    return get_impl()->rank();
}

CCL_API size_t ccl::device_communicator::size() const {
    return get_impl()->size();
}

/*CCL_API size_t ccl::device_communicator::get_group_unique_id() const
{
    return static_cast<size_t> (get_impl()->get_comm_group_id());
}*/

CCL_API ccl::device_communicator ccl::device_communicator::split(
    const ccl::device_comm_split_attr& attr) {
    if (!attr.is_valid<ccl::comm_split_attr_id::group>()) {
        throw ccl_error(std::string(__FUNCTION__) +
                        " - TODO `device_comm_split_attr`: supports `group` only");
    }
    //TODO
#ifdef MULTI_GPU_SUPPORT
    auto id = get_impl()->get_comm_group_id();
    ccl::group_context::comm_group_t my_group =
        ccl::group_context::instance().get_existing_group_by_id(id);
#ifdef CCL_ENABLE_SYCL
    return my_group->create_communicator<cl::sycl::device>(get_device(), attr);
#else
#ifdef MULTI_GPU_SUPPORT
    return my_group->create_communicator(get_impl()->get_device_path(), attr);
#endif
#endif
#else
    throw ccl_error(std::string(__FUNCTION__) + " - TODO `device_comm_split_attr`: unsupported");
    return std::move(*this);
#endif
}
/*
CCL_API ccl::comm_attr_t ccl::device_communicator::get_comm_split_attr() const
{
    return get_impl()->get_comm_split_attr();
}

CCL_API ccl::device_group_split_type ccl::device_communicator::get_device_group_split_type() const
{
    return get_impl()->get_topology_type();
}

CCL_API ccl::device_topology_type ccl::device_communicator::get_topology_class() const
{
    return get_impl()->get_topology_class();
}

*/
CCL_API ccl::device_communicator::ccl_device_t ccl::device_communicator::get_device() {
    return get_impl()->get_device();
}

CCL_API ccl::device_communicator::ccl_context_t ccl::device_communicator::get_context() {
    return get_impl()->get_context();
}

/*CCL_API bool ccl::device_communicator::is_ready() const
{
    return get_impl()->is_ready();
}*/

/* allgatherv */
//(stream) ? stream.get_impl() :222 ccl::get_empty_stream().get_impl())111;
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allgatherv(const void* send_buf,
                                     size_t send_count,
                                     void* recv_buf,
                                     const vector_class<size_t>& recv_counts,
                                     datatype dtype,
                                     stream op_stream,
                                     const allgatherv_attr& attr,
                                     const vector_class<event>& deps) {
    //static_assert(std::is_base_of<direct_access_policy<ccl_stream>, stream>::value, "stream should provide shared acces to ints pimpl member");
    return get_impl()->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, op_stream.get_impl(), attr, deps);
}

CCL_API ccl::device_communicator::request_t ccl::device_communicator::allgatherv(
    const void* send_buf,
    size_t send_count,
    const vector_class<void*>& recv_bufs,
    const vector_class<size_t>& recv_counts,
    datatype dtype,
    stream stream,
    const allgatherv_attr& attr,
    const vector_class<event>& deps) {
    return get_impl()->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allgatherv(const BufferType* send_buf,
                                     size_t send_count,
                                     BufferType* recv_buf,
                                     const vector_class<size_t>& recv_counts,
                                     stream stream,
                                     const allgatherv_attr& attr,
                                     const vector_class<event>& deps) {
    return get_impl()->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, stream.get_impl(), attr, deps);
}
////////////////////////// TODO
template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allgatherv(const BufferType* send_buf,
                                     size_t send_count,
                                     vector_class<BufferType*>& recv_bufs,
                                     const vector_class<size_t>& recv_counts,
                                     stream stream,
                                     const allgatherv_attr& attr,
                                     const vector_class<event>& deps) {
    return get_impl()->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allgatherv(const BufferObjectType& send_buf,
                                     size_t send_count,
                                     BufferObjectType& recv_buf,
                                     const vector_class<size_t>& recv_counts,
                                     stream stream,
                                     const allgatherv_attr& attr,
                                     const vector_class<event>& deps) {
    return get_impl()->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API ccl::device_communicator::allgatherv(
    const BufferObjectType& send_buf,
    size_t send_count,
    vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
    const vector_class<size_t>& recv_counts,
    stream stream,
    const allgatherv_attr& attr,
    const vector_class<event>& deps) {
    return get_impl()->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, stream.get_impl(), attr, deps);
}

/* allreduce */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allreduce(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    datatype dtype,
                                    reduction reduction,
                                    stream stream,
                                    const allreduce_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->allreduce(
        send_buf, recv_buf, count, dtype, reduction, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allreduce(const BufferType* send_buf,
                                    BufferType* recv_buf,
                                    size_t count,
                                    reduction reduction,
                                    stream stream,
                                    const allreduce_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->allreduce(
        send_buf, recv_buf, count, reduction, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::allreduce(const BufferObjectType& send_buf,
                                    BufferObjectType& recv_buf,
                                    size_t count,
                                    reduction reduction,
                                    stream stream,
                                    const allreduce_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->allreduce(
        send_buf, recv_buf, count, reduction, stream.get_impl(), attr, deps);
}

/* alltoall */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoall(const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   datatype dtype,
                                   stream stream,
                                   const alltoall_attr& attr,
                                   const vector_class<event>& deps) {
    return get_impl()->alltoall(send_buf, recv_buf, count, dtype, stream.get_impl(), attr, deps);
}

ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoall(const vector_class<void*>& send_buf,
                                   const vector_class<void*>& recv_buf,
                                   size_t count,
                                   datatype dtype,
                                   stream stream,
                                   const alltoall_attr& attr,
                                   const vector_class<event>& deps) {
    return get_impl()->alltoall(send_buf, recv_buf, count, dtype, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoall(const BufferType* send_buf,
                                   BufferType* recv_buf,
                                   size_t count,
                                   stream stream,
                                   const alltoall_attr& attr,
                                   const vector_class<event>& deps) {
    return get_impl()->alltoall(send_buf, recv_buf, count, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoall(const vector_class<BufferType*>& send_buf,
                                   const vector_class<BufferType*>& recv_buf,
                                   size_t count,
                                   stream stream,
                                   const alltoall_attr& attr,
                                   const vector_class<event>& deps) {
    return get_impl()->alltoall(send_buf, recv_buf, count, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoall(const BufferObjectType& send_buf,
                                   BufferObjectType& recv_buf,
                                   size_t count,
                                   stream stream,
                                   const alltoall_attr& attr,
                                   const vector_class<event>& deps) {
    return get_impl()->alltoall(send_buf, recv_buf, count, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoall(
    const vector_class<ccl::reference_wrapper_class<BufferObjectType>>& send_buf,
    const vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_buf,
    size_t count,
    stream stream,
    const alltoall_attr& attr,
    const vector_class<event>& deps) {
    return get_impl()->alltoall(send_buf, recv_buf, count, stream.get_impl(), attr, deps);
}

/* alltoallv */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoallv(const void* send_buf,
                                    const vector_class<size_t>& send_counts,
                                    void* recv_buf,
                                    const vector_class<size_t>& recv_counts,
                                    datatype dtype,
                                    stream stream,
                                    const alltoallv_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, stream.get_impl(), attr, deps);
}

ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoallv(const vector_class<void*>& send_bufs,
                                    const vector_class<size_t>& send_counts,
                                    const vector_class<void*>& recv_bufs,
                                    const vector_class<size_t>& recv_counts,
                                    datatype dtype,
                                    stream stream,
                                    const alltoallv_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoallv(const BufferType* send_buf,
                                    const vector_class<size_t>& send_counts,
                                    BufferType* recv_buf,
                                    const vector_class<size_t>& recv_counts,
                                    stream stream,
                                    const alltoallv_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoallv(const vector_class<BufferType*>& send_bufs,
                                    const vector_class<size_t>& send_counts,
                                    const vector_class<BufferType*>& recv_bufs,
                                    const vector_class<size_t>& recv_counts,
                                    stream stream,
                                    const alltoallv_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::alltoallv(const BufferObjectType& send_buf,
                                    const vector_class<size_t>& send_counts,
                                    BufferObjectType& recv_buf,
                                    const vector_class<size_t>& recv_counts,
                                    stream stream,
                                    const alltoallv_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoallv(
    const vector_class<ccl::reference_wrapper_class<BufferObjectType>>& send_bufs,
    const vector_class<size_t>& send_counts,
    const vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
    const vector_class<size_t>& recv_counts,
    stream stream,
    const alltoallv_attr& attr,
    const vector_class<event>& deps) {
    return get_impl()->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, stream.get_impl(), attr, deps);
}

/* barrier */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::barrier(stream stream,
                                  const barrier_attr& attr,
                                  const vector_class<event>& deps) {
    return get_impl()->barrier(stream.get_impl(), attr, deps);
}

/* bcast */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::broadcast(void* buf,
                                    size_t count,
                                    datatype dtype,
                                    size_t root,
                                    stream stream,
                                    const broadcast_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->bcast(buf, count, dtype, root, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::broadcast(BufferType* buf,
                                    size_t count,
                                    size_t root,
                                    stream stream,
                                    const broadcast_attr& attr,
                                    const vector_class<event>& deps)

{
    return get_impl()->bcast(buf, count, root, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::broadcast(BufferObjectType& buf,
                                    size_t count,
                                    size_t root,
                                    stream stream,
                                    const broadcast_attr& attr,
                                    const vector_class<event>& deps) {
    return get_impl()->bcast(buf, count, root, stream.get_impl(), attr, deps);
}

/* reduce */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::reduce(const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 datatype dtype,
                                 reduction reduction,
                                 size_t root,
                                 stream stream,
                                 const reduce_attr& attr,
                                 const vector_class<event>& deps) {
    return get_impl()->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::reduce(const BufferType* send_buf,
                                 BufferType* recv_buf,
                                 size_t count,
                                 reduction reduction,
                                 size_t root,
                                 stream stream,
                                 const reduce_attr& attr,
                                 const vector_class<event>& deps) {
    return get_impl()->reduce(
        send_buf, recv_buf, count, reduction, root, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::reduce(const BufferObjectType& send_buf,
                                 BufferObjectType& recv_buf,
                                 size_t count,
                                 reduction reduction,
                                 size_t root,
                                 stream stream,
                                 const reduce_attr& attr,
                                 const vector_class<event>& deps) {
    return get_impl()->reduce(
        send_buf, recv_buf, count, reduction, root, stream.get_impl(), attr, deps);
}

/* reduce_scatter */
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::reduce_scatter(const void* send_buf,
                                         void* recv_buf,
                                         size_t recv_count,
                                         datatype dtype,
                                         reduction reduction,
                                         stream stream,
                                         const reduce_scatter_attr& attr,
                                         const vector_class<event>& deps) {
    return get_impl()->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, stream.get_impl(), attr, deps);
}

template <class BufferType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::reduce_scatter(const BufferType* send_buf,
                                         BufferType* recv_buf,
                                         size_t recv_count,
                                         reduction reduction,
                                         stream stream,
                                         const reduce_scatter_attr& attr,
                                         const vector_class<event>& deps) {
    return get_impl()->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, stream.get_impl(), attr, deps);
}

template <class BufferObjectType, typename T>
ccl::device_communicator::request_t CCL_API
ccl::device_communicator::reduce_scatter(const BufferObjectType& send_buf,
                                         BufferObjectType& recv_buf,
                                         size_t recv_count,
                                         reduction reduction,
                                         stream stream,
                                         const reduce_scatter_attr& attr,
                                         const vector_class<event>& deps) {
    return get_impl()->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, stream.get_impl(), attr, deps);
}

/* sparse_allreduce */
// ccl::device_communicator::request_t CCL_API
// ccl::device_communicator::sparse_allreduce(const void* send_ind_buf,
//                                size_t send_ind_count,
//                                const void* send_val_buf,
//                                size_t send_val_count,
//                                void* recv_ind_buf,
//                                size_t recv_ind_count,
//                                void* recv_val_buf,
//                                size_t recv_val_count,
//                                datatype index_dtype,
//                                datatype value_dtype,
//                                reduction reduction,
//                                stream stream,
//                                const sparse_allreduce_attr& attr,
//                                const vector_class<event>& deps)
// {
//     return get_impl()->sparse_allreduce(send_ind_buf, send_ind_count,
//                                    send_val_buf, send_val_count,
//                                    recv_ind_buf, recv_ind_count,
//                                    recv_val_buf, recv_val_count,
//                                    index_dtype,
//                                    value_dtype,
//                                    reduction,
//                                    stream.get_impl(), attr, deps);
// }

// template<class index_BufferType,
//          class value_BufferType,
//          typename T>
// ccl::device_communicator::request_t CCL_API
// ccl::device_communicator::sparse_allreduce(const index_BufferType* send_ind_buf,
//                                size_t send_ind_count,
//                                const value_BufferType* send_val_buf,
//                                size_t send_val_count,
//                                index_BufferType* recv_ind_buf,
//                                size_t recv_ind_count,
//                                value_BufferType* recv_val_buf,
//                                size_t recv_val_count,
//                                reduction reduction,
//                                stream stream,
//                                const sparse_allreduce_attr& attr,
//                                const vector_class<event>& deps)
// {
//     return get_impl()->sparse_allreduce(send_ind_buf, send_ind_count,
//                                    send_val_buf, send_val_count,
//                                    recv_ind_buf, recv_ind_count,
//                                    recv_val_buf, recv_val_count,
//                                    reduction,
//                                    stream.get_impl(), attr, deps);
// }

// template<class IndexBufferObjectType,
//          class ValueBufferObjectType,
//          typename T>
// ccl::device_communicator::request_t CCL_API
// ccl::device_communicator::sparse_allreduce(const IndexBufferObjectType& send_ind_buf,
//                                size_t send_ind_count,
//                                const ValueBufferObjectType& send_val_buf,
//                                size_t send_val_count,
//                                IndexBufferObjectType& recv_ind_buf,
//                                size_t recv_ind_count,
//                                ValueBufferObjectType& recv_val_buf,
//                                size_t recv_val_count,
//                                reduction reduction,
//                                stream stream,
//                                const sparse_allreduce_attr& attr,
//                                const vector_class<event>& deps)
// {
//     return get_impl()->sparse_allreduce(send_ind_buf, send_ind_count,
//                                    send_val_buf, send_val_count,
//                                    recv_ind_buf, recv_ind_count,
//                                    recv_val_buf, recv_val_count,
//                                    reduction,
//                                    stream.get_impl(), attr, deps);
// }

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_DEVICE_COMM_CREATE_WO_RANK_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::device_communicator> \
    ccl::device_communicator::create_device_communicators( \
        const size_t comm_size, \
        const ccl::vector_class<DeviceType>& local_devices, \
        ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs);

#define API_DEVICE_COMM_CREATE_WITH_RANK_IN_VECTOR_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::device_communicator> \
    ccl::device_communicator::create_device_communicators( \
        const size_t comm_size, \
        const ccl::vector_class<ccl::pair_class<ccl::rank_t, DeviceType>>& local_rank_device_map, \
        ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs);

#define API_DEVICE_COMM_CREATE_WITH_RANK_IN_MAP_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::device_communicator> \
    ccl::device_communicator::create_device_communicators( \
        const size_t comm_size, \
        const ccl::map_class<ccl::rank_t, DeviceType>& local_rank_device_map, \
        ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs);

#define API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(BufferType) \
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::allgatherv( \
        const BufferType* send_buf, \
        size_t send_count, \
        BufferType* recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const ccl::allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::allgatherv( \
        const BufferType* send_buf, \
        size_t send_count, \
        ccl::vector_class<BufferType*>& recv_bufs, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const ccl::allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::allreduce( \
        const BufferType* send_buf, \
        BufferType* recv_buf, \
        size_t count, \
        ccl::reduction reduction, \
        ccl::stream stream, \
        const ccl::allreduce_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoall( \
        const BufferType* send_buf, \
        BufferType* recv_buf, \
        size_t count, \
        ccl::stream stream, \
        const ccl::alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoall( \
        const ccl::vector_class<BufferType*>& send_buf, \
        const ccl::vector_class<BufferType*>& recv_buf, \
        size_t count, \
        ccl::stream stream, \
        const ccl::alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoallv( \
        const BufferType* send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        BufferType* recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoallv( \
        const ccl::vector_class<BufferType*>& send_bufs, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<BufferType*>& recv_bufs, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const ccl::alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::broadcast( \
        BufferType* buf, \
        size_t count, \
        size_t root, \
        ccl::stream stream, \
        const ccl::broadcast_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::reduce( \
        const BufferType* send_buf, \
        BufferType* recv_buf, \
        size_t count, \
        ccl::reduction reduction, \
        size_t root, \
        ccl::stream stream, \
        const ccl::reduce_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::reduce_scatter( \
        const BufferType* send_buf, \
        BufferType* recv_buf, \
        size_t recv_count, \
        ccl::reduction reduction, \
        ccl::stream stream, \
        const ccl::reduce_scatter_attr& attr, \
        const ccl::vector_class<ccl::event>& deps);

#define API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(BufferObjectType) \
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::allgatherv( \
        const BufferObjectType& send_buf, \
        size_t send_count, \
        BufferObjectType& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::allgatherv( \
        const BufferObjectType& send_buf, \
        size_t send_count, \
        ccl::vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const allgatherv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::allreduce( \
        const BufferObjectType& send_buf, \
        BufferObjectType& recv_buf, \
        size_t count, \
        reduction reduction, \
        ccl::stream stream, \
        const allreduce_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoall( \
        const BufferObjectType& send_buf, \
        BufferObjectType& recv_buf, \
        size_t count, \
        ccl::stream stream, \
        const alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoall( \
        const ccl::vector_class<ccl::reference_wrapper_class<BufferObjectType>>& send_buf, \
        const ccl::vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_buf, \
        size_t count, \
        ccl::stream stream, \
        const alltoall_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoallv( \
        const BufferObjectType& send_buf, \
        const ccl::vector_class<size_t>& send_counts, \
        BufferObjectType& recv_buf, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::alltoallv( \
        const ccl::vector_class<ccl::reference_wrapper_class<BufferObjectType>>& send_bufs, \
        const ccl::vector_class<size_t>& send_counts, \
        const ccl::vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs, \
        const ccl::vector_class<size_t>& recv_counts, \
        ccl::stream stream, \
        const alltoallv_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::broadcast( \
        BufferObjectType& buf, \
        size_t count, \
        size_t root, \
        ccl::stream stream, \
        const broadcast_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::reduce( \
        const BufferObjectType& send_buf, \
        BufferObjectType& recv_buf, \
        size_t count, \
        reduction reduction, \
        size_t root, \
        ccl::stream stream, \
        const reduce_attr& attr, \
        const ccl::vector_class<ccl::event>& deps); \
\
    template ccl::device_communicator::request_t CCL_API ccl::device_communicator::reduce_scatter( \
        const BufferObjectType& send_buf, \
        BufferObjectType& recv_buf, \
        size_t recv_count, \
        reduction reduction, \
        ccl::stream stream, \
        const reduce_scatter_attr& attr, \
        const ccl::vector_class<ccl::event>& deps);

#define API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(index_type, value_type) \
    template ccl::device_communicator::coll_request_t CCL_API \
    ccl::device_communicator::sparse_allreduce(const index_type* send_ind_buf, \
                                               size_t send_ind_count, \
                                               const value_type* send_val_buf, \
                                               size_t send_val_count, \
                                               index_type* recv_ind_buf, \
                                               size_t recv_ind_count, \
                                               value_type* recv_val_buf, \
                                               size_t recv_val_count, \
                                               ccl::reduction reduction, \
                                               ccl::stream stream, \
                                               const ccl::sparse_allreduce_attr& attr, \
                                               const ccl::vector_class<ccl::event>& deps);

#define API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(index_object_type, value_object_type) \
    template ccl::device_communicator::coll_request_t CCL_API \
    ccl::device_communicator::sparse_allreduce(const index_object_type& send_ind_buf, \
                                               size_t send_ind_count, \
                                               const value_object_type& send_val_buf, \
                                               size_t send_val_count, \
                                               index_object_type& recv_ind_buf, \
                                               size_t recv_ind_count, \
                                               value_object_type& recv_val_buf, \
                                               size_t recv_val_count, \
                                               ccl::reduction reduction, \
                                               ccl::stream stream, \
                                               const ccl::sparse_allreduce_attr& attr, \
                                               const ccl::vector_class<ccl::event>& deps);
