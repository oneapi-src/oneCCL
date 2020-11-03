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
#include "oneapi/ccl/ccl_environment.hpp"
#include "oneapi/ccl/ccl_api_functions.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
#include "common/comm/comm_interface.hpp"
#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

#include "ccl_api_functions_generators.hpp"
#include "common/global/global.hpp"
#include "ccl_gpu_module.hpp"

namespace ccl {

/**
 * A structure that is a friend of the passed object
 * and which allows access to the internal representation of this object
 */
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

#ifdef MULTI_GPU_SUPPORT
/* register a gpu module */
void register_gpu_module(std::string kernel_dir_path)
{
    // allgatherv
    if (!kernel_dir_path.empty())
    {
        if(*kernel_dir_path.rbegin() != '/')
        {
            kernel_dir_path += '/';
        }
    }
    LOG_INFO("SPV Kernels found directory: ", kernel_dir_path);
    std::string kernel_path = kernel_dir_path + "ring_allgatherv.spv";
    register_gpu_module_source(kernel_path.c_str(),
                                ccl::device_topology_type::ring,
                                ccl_coll_allgatherv);
    // register__gpu_module_source("kernels/a2a_allgatherv.spv",
    //                             ccl::device_topology_type::a2a,
    //                             ccl_coll_allgatherv);
    // alltoallv
    kernel_path = kernel_dir_path + "ring_alltoallv.spv";
    register_gpu_module_source(kernel_path.c_str(),
                               ccl::device_topology_type::ring,
                               ccl_coll_alltoallv);
    // register_gpu_module_source("kernels/a2a_alltoallv.spv",
    //                             ccl::device_topology_type::a2a,
    //                             ccl_coll_alltoallv);
    // allreduce
    kernel_path = kernel_dir_path + "ring_allreduce.spv";
    register_gpu_module_source(kernel_path.c_str(),
                                ccl::device_topology_type::ring,
                                ccl_coll_allreduce);
    kernel_path = kernel_dir_path + "a2a_allreduce.spv";
    register_gpu_module_source(kernel_path.c_str(),
                                ccl::device_topology_type::a2a,
                                ccl_coll_allreduce);
    // bcast
    kernel_path = kernel_dir_path + "ring_bcast.spv";
    register_gpu_module_source(kernel_path.c_str(),
                               ccl::device_topology_type::ring,
                               ccl_coll_bcast);
    kernel_path = kernel_dir_path + "a2a_bcast.spv";
    register_gpu_module_source(kernel_path.c_str(),
                               ccl::device_topology_type::a2a,
                               ccl_coll_bcast);
    // reduce
    kernel_path = kernel_dir_path + "ring_reduce.spv";
    register_gpu_module_source(kernel_path.c_str(),
                                ccl::device_topology_type::ring,
                                ccl_coll_reduce);
    // register_gpu_module_source("kernels/a2a_reduce.spv",
    //                            ccl_topology_class_t::a2a_algo_class,
    //                            ccl_coll_reduce);
}
#endif //MULTI_GPU_SUPPORT

void CCL_API init() {
    auto& env = environment::instance();
    (void)env;
#ifdef MULTI_GPU_SUPPORT
    const auto& env_object = ccl::global_data::env();

    //WA
    if (!env_object.kernel_path.empty())
    {
        register_gpu_module(env_object.kernel_path);
    }
#endif //MULTI_GPU_SUPPORT
}

/******************** ENVIRONMENT ********************/

library_version CCL_API get_library_version() {
    return environment::instance().get_library_version();
}

/* datatype */
datatype CCL_API register_datatype(const datatype_attr& attr) {
    return environment::instance().register_datatype(attr);
}

void CCL_API deregister_datatype(datatype dtype) {
    return environment::instance().deregister_datatype(dtype);
}

size_t CCL_API get_datatype_size(datatype dtype) {
    return environment::instance().get_datatype_size(dtype);
}

/* KVS */
shared_ptr_class<kvs> CCL_API create_main_kvs() {
    return environment::instance().create_main_kvs();
}

shared_ptr_class<kvs> CCL_API create_kvs(const kvs::address_type& addr) {
    return environment::instance().create_kvs(addr);
}

/* device */
device CCL_API create_device()
{
    static empty_t empty {};
    return environment::instance().create_device(empty);
}

/* context */
context CCL_API create_context()
{
    static empty_t empty {};
    return environment::instance().create_context(empty);
}

/* stream */
stream CCL_API create_stream()
{
    return default_stream;
}

#ifdef CCL_ENABLE_SYCL
communicator create_single_device_communicator(const size_t comm_size,
                                                      const size_t rank,
                                                      const cl::sycl::device& device,
                                                      const cl::sycl::context& context,
                                                      shared_ptr_class<kvs_interface> kvs) {
    return environment::instance().create_single_device_communicator(
        comm_size, rank, device, context, kvs);
}
#endif // CCL_ENABLE_SYCL

// communicator create_single_device_communicator(const size_t world_size,
//                                     const size_t rank,
//                                     cl::sycl::queue queue,
//                                     shared_ptr_class<kvs_interface> kvs) const;

// template<class DeviceSelectorType>
// communicator create_single_device_communicator(const size_t world_size,
//                                     const size_t rank,
//                                     const DeviceSelectorType& selector,
//                                     shared_ptr_class<kvs_interface> kvs) const
// {
//     return return environment::instance().create_single_device_communicator(world_size, rank, cl::sycl::device(selector), kvs);
// }

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

vector_class<communicator> split_device_communicators(
    const vector_class<pair_class<communicator, comm_split_attr>>& attrs) {
    // TODO not implemented
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    // return environment::instance().split_device_communicators(attrs);
    return {};
}

#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

namespace preview {

/* communicator */
communicator CCL_API create_communicator() {
    return environment::instance().create_communicator();
}

communicator CCL_API create_communicator(const size_t size, shared_ptr_class<kvs_interface> kvs) {
    return environment::instance().create_communicator(size, kvs);
}

} // namespace preview

communicator CCL_API create_communicator(const size_t size,
                                         const size_t rank,
                                         shared_ptr_class<kvs_interface> kvs) {
    return environment::instance().create_communicator(size, rank, kvs);
}

/******************** COMMUNICATOR ********************/

/* allgatherv */
CCL_API event allgatherv(const void* send_buf,
                           size_t send_count,
                           void* recv_buf,
                           const vector_class<size_t>& recv_counts,
                           datatype dtype,
                           const communicator& comm,
                           const stream& op_stream,
                           const allgatherv_attr& attr,
                           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
}

CCL_API event allgatherv(const void* send_buf,
                           size_t send_count,
                           void* recv_buf,
                           const vector_class<size_t>& recv_counts,
                           datatype dtype,
                           const communicator& comm,
                           const allgatherv_attr& attr,
                           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(default_stream), attr, deps);
}

CCL_API event allgatherv(const void* send_buf,
                           size_t send_count,
                           const vector_class<void*>& recv_bufs,
                           const vector_class<size_t>& recv_counts,
                           datatype dtype,
                           const communicator& comm,
                           const stream& op_stream,
                           const allgatherv_attr& attr,
                           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, disp(op_stream), attr, deps);
}

CCL_API event allgatherv(const void* send_buf,
                           size_t send_count,
                           const vector_class<void*>& recv_bufs,
                           const vector_class<size_t>& recv_counts,
                           datatype dtype,
                           const communicator& comm,
                           const allgatherv_attr& attr,
                           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                   size_t send_count,
                   BufferType* recv_buf,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const stream& op_stream,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                   size_t send_count,
                   BufferType* recv_buf,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                   size_t send_count,
                   vector_class<BufferType*>& recv_bufs,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const stream& op_stream,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                   size_t send_count,
                   vector_class<BufferType*>& recv_bufs,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                   size_t send_count,
                   BufferObjectType& recv_buf,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const stream& op_stream,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                   size_t send_count,
                   BufferObjectType& recv_buf,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                   size_t send_count,
                   vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const stream& op_stream,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                   size_t send_count,
                   vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
                   const vector_class<size_t>& recv_counts,
                   const communicator& comm,
                   const allgatherv_attr& attr,
                   const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

/* allreduce */
CCL_API event allreduce(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          datatype dtype,
                          reduction reduction,
                          const communicator& comm,
                          const stream& op_stream,
                          const allreduce_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(op_stream), attr, deps);
}

CCL_API event allreduce(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          datatype dtype,
                          reduction reduction,
                          const communicator& comm,
                          const allreduce_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allreduce(const BufferType* send_buf,
                  BufferType* recv_buf,
                  size_t count,
                  reduction reduction,
                  const communicator& comm,
                  const stream& op_stream,
                  const allreduce_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allreduce(const BufferType* send_buf,
                  BufferType* recv_buf,
                  size_t count,
                  reduction reduction,
                  const communicator& comm,
                  const allreduce_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allreduce(const BufferObjectType& send_buf,
                  BufferObjectType& recv_buf,
                  size_t count,
                  reduction reduction,
                  const communicator& comm,
                  const stream& op_stream,
                  const allreduce_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allreduce(const BufferObjectType& send_buf,
                  BufferObjectType& recv_buf,
                  size_t count,
                  reduction reduction,
                  const communicator& comm,
                  const allreduce_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

/* alltoall */
CCL_API event alltoall(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         datatype dtype,
                         const communicator& comm,
                         const stream& op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

CCL_API event alltoall(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         datatype dtype,
                         const communicator& comm,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(default_stream), attr, deps);
}

CCL_API event alltoall(const vector_class<void*>& send_buf,
                         const vector_class<void*>& recv_buf,
                         size_t count,
                         datatype dtype,
                         const communicator& comm,
                         const stream& op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const BufferType* send_buf,
                 BufferType* recv_buf,
                 size_t count,
                 const communicator& comm,
                 const stream& op_stream,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const BufferType* send_buf,
                 BufferType* recv_buf,
                 size_t count,
                 const communicator& comm,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const vector_class<BufferType*>& send_buf,
                 const vector_class<BufferType*>& recv_buf,
                 size_t count,
                 const communicator& comm,
                 const stream& op_stream,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const vector_class<BufferType*>& send_buf,
                 const vector_class<BufferType*>& recv_buf,
                 size_t count,
                 const communicator& comm,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const BufferObjectType& send_buf,
                 BufferObjectType& recv_buf,
                 size_t count,
                 const communicator& comm,
                 const stream& op_stream,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const BufferObjectType& send_buf,
                 BufferObjectType& recv_buf,
                 size_t count,
                 const communicator& comm,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
                 const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                 size_t count,
                 const communicator& comm,
                 const stream& op_stream,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
                 const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                 size_t count,
                 const communicator& comm,
                 const alltoall_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

/* alltoallv */
CCL_API event alltoallv(const void* send_buf,
                          const vector_class<size_t>& send_counts,
                          void* recv_buf,
                          const vector_class<size_t>& recv_counts,
                          datatype dtype,
                          const communicator& comm,
                          const stream& op_stream,
                          const alltoallv_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
}

CCL_API event alltoallv(const void* send_buf,
                          const vector_class<size_t>& send_counts,
                          void* recv_buf,
                          const vector_class<size_t>& recv_counts,
                          datatype dtype,
                          const communicator& comm,
                          const alltoallv_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, disp(default_stream), attr, deps);
}

CCL_API event alltoallv(const vector_class<void*>& send_bufs,
                          const vector_class<size_t>& send_counts,
                          const vector_class<void*>& recv_bufs,
                          const vector_class<size_t>& recv_counts,
                          datatype dtype,
                          const communicator& comm,
                          const stream& op_stream,
                          const alltoallv_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, disp(op_stream), attr, deps);
}

CCL_API event alltoallv(const vector_class<void*>& send_bufs,
                          const vector_class<size_t>& send_counts,
                          const vector_class<void*>& recv_bufs,
                          const vector_class<size_t>& recv_counts,
                          datatype dtype,
                          const communicator& comm,
                          const alltoallv_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const BufferType* send_buf,
                  const vector_class<size_t>& send_counts,
                  BufferType* recv_buf,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const stream& op_stream,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const BufferType* send_buf,
                  const vector_class<size_t>& send_counts,
                  BufferType* recv_buf,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const vector_class<BufferType*>& send_bufs,
                  const vector_class<size_t>& send_counts,
                  const vector_class<BufferType*>& recv_bufs,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const stream& op_stream,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const vector_class<BufferType*>& send_bufs,
                  const vector_class<size_t>& send_counts,
                  const vector_class<BufferType*>& recv_bufs,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const BufferObjectType& send_buf,
                  const vector_class<size_t>& send_counts,
                  BufferObjectType& recv_buf,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const stream& op_stream,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const BufferObjectType& send_buf,
                  const vector_class<size_t>& send_counts,
                  BufferObjectType& recv_buf,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                  const vector_class<size_t>& send_counts,
                  const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const stream& op_stream,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                  const vector_class<size_t>& send_counts,
                  const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                  const vector_class<size_t>& recv_counts,
                  const communicator& comm,
                  const alltoallv_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

/* barrier */
CCL_API event barrier(const communicator& comm,
                        const stream& op_stream,
                        const barrier_attr& attr,
                        const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(op_stream), attr, deps);
}

CCL_API event barrier(const communicator& comm,
                        const barrier_attr& attr,
                        const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(default_stream), attr, deps);
}

/* broadcast */
CCL_API event broadcast(void* buf,
                          size_t count,
                          datatype dtype,
                          size_t root,
                          const communicator& comm,
                          const stream& op_stream,
                          const broadcast_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, dtype, root, disp(op_stream), attr, deps);
}

CCL_API event broadcast(void* buf,
                          size_t count,
                          datatype dtype,
                          size_t root,
                          const communicator& comm,
                          const broadcast_attr& attr,
                          const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, dtype, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                  size_t count,
                  size_t root,
                  const communicator& comm,
                  const stream& op_stream,
                  const broadcast_attr& attr,
                  const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                  size_t count,
                  size_t root,
                  const communicator& comm,
                  const broadcast_attr& attr,
                  const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& buf,
                  size_t count,
                  size_t root,
                  const communicator& comm,
                  const stream& op_stream,
                  const broadcast_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& buf,
                  size_t count,
                  size_t root,
                  const communicator& comm,
                  const broadcast_attr& attr,
                  const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

/* reduce */
CCL_API event reduce(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       datatype dtype,
                       reduction reduction,
                       size_t root,
                       const communicator& comm,
                       const stream& op_stream,
                       const reduce_attr& attr,
                       const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(op_stream), attr, deps);
}

CCL_API event reduce(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       datatype dtype,
                       reduction reduction,
                       size_t root,
                       const communicator& comm,
                       const reduce_attr& attr,
                       const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce(const BufferType* send_buf,
               BufferType* recv_buf,
               size_t count,
               reduction reduction,
               size_t root,
               const communicator& comm,
               const stream& op_stream,
               const reduce_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce(const BufferType* send_buf,
               BufferType* recv_buf,
               size_t count,
               reduction reduction,
               size_t root,
               const communicator& comm,
               const reduce_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce(const BufferObjectType& send_buf,
               BufferObjectType& recv_buf,
               size_t count,
               reduction reduction,
               size_t root,
               const communicator& comm,
               const stream& op_stream,
               const reduce_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce(const BufferObjectType& send_buf,
               BufferObjectType& recv_buf,
               size_t count,
               reduction reduction,
               size_t root,
               const communicator& comm,
               const reduce_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

/* reduce_scatter */
CCL_API event reduce_scatter(const void* send_buf,
                               void* recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               reduction reduction,
                               const communicator& comm,
                               const stream& op_stream,
                               const reduce_scatter_attr& attr,
                               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(op_stream), attr, deps);
}

CCL_API event reduce_scatter(const void* send_buf,
                               void* recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               reduction reduction,
                               const communicator& comm,
                               const reduce_scatter_attr& attr,
                               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce_scatter(const BufferType* send_buf,
                       BufferType* recv_buf,
                       size_t recv_count,
                       reduction reduction,
                       const communicator& comm,
                       const stream& op_stream,
                       const reduce_scatter_attr& attr,
                       const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce_scatter(const BufferType* send_buf,
                       BufferType* recv_buf,
                       size_t recv_count,
                       reduction reduction,
                       const communicator& comm,
                       const reduce_scatter_attr& attr,
                       const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce_scatter(const BufferObjectType& send_buf,
                       BufferObjectType& recv_buf,
                       size_t recv_count,
                       reduction reduction,
                       const communicator& comm,
                       const stream& op_stream,
                       const reduce_scatter_attr& attr,
                       const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce_scatter(const BufferObjectType& send_buf,
                       BufferObjectType& recv_buf,
                       size_t recv_count,
                       reduction reduction,
                       const communicator& comm,
                       const reduce_scatter_attr& attr,
                       const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(default_stream), attr, deps);
}

namespace preview {

/* sparse_allreduce */
CCL_API ccl::event sparse_allreduce(const void* send_ind_buf,
                                      size_t send_ind_count,
                                      const void* send_val_buf,
                                      size_t send_val_count,
                                      void* recv_ind_buf,
                                      size_t recv_ind_count,
                                      void* recv_val_buf,
                                      size_t recv_val_count,
                                      ccl::datatype index_dtype,
                                      ccl::datatype value_dtype,
                                      ccl::reduction reduction,
                                      const ccl::communicator& comm,
                                      const ccl::stream& op_stream,
                                      const ccl::sparse_allreduce_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    ccl::impl_dispatch disp;
    return disp(comm)->sparse_allreduce(send_ind_buf,
                                        send_ind_count,
                                        send_val_buf,
                                        send_val_count,
                                        recv_ind_buf,
                                        recv_ind_count,
                                        recv_val_buf,
                                        recv_val_count,
                                        index_dtype,
                                        value_dtype,
                                        reduction,
                                        disp(op_stream),
                                        attr,
                                        deps);
}

CCL_API ccl::event sparse_allreduce(const void* send_ind_buf,
                                      size_t send_ind_count,
                                      const void* send_val_buf,
                                      size_t send_val_count,
                                      void* recv_ind_buf,
                                      size_t recv_ind_count,
                                      void* recv_val_buf,
                                      size_t recv_val_count,
                                      ccl::datatype index_dtype,
                                      ccl::datatype value_dtype,
                                      ccl::reduction reduction,
                                      const ccl::communicator& comm,
                                      const ccl::sparse_allreduce_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    ccl::impl_dispatch disp;
    return disp(comm)->sparse_allreduce(send_ind_buf,
                                        send_ind_count,
                                        send_val_buf,
                                        send_val_count,
                                        recv_ind_buf,
                                        recv_ind_count,
                                        recv_val_buf,
                                        recv_val_count,
                                        index_dtype,
                                        value_dtype,
                                        reduction,
                                        disp(default_stream),
                                        attr,
                                        deps);
}

template <class IndexBufferType, class ValueBufferType, typename T>
ccl::event sparse_allreduce(const IndexBufferType* send_ind_buf,
                              size_t send_ind_count,
                              const ValueBufferType* send_val_buf,
                              size_t send_val_count,
                              IndexBufferType* recv_ind_buf,
                              size_t recv_ind_count,
                              ValueBufferType* recv_val_buf,
                              size_t recv_val_count,
                              ccl::reduction reduction,
                              const ccl::communicator& comm,
                              const ccl::stream& op_stream,
                              const ccl::sparse_allreduce_attr& attr,
                              const ccl::vector_class<ccl::event>& deps) {
    ccl::impl_dispatch disp;
    return disp(comm)->sparse_allreduce(send_ind_buf,
                                        send_ind_count,
                                        send_val_buf,
                                        send_val_count,
                                        recv_ind_buf,
                                        recv_ind_count,
                                        recv_val_buf,
                                        recv_val_count,
                                        reduction,
                                        disp(op_stream),
                                        attr,
                                        deps);
}

template <class IndexBufferType, class ValueBufferType, typename T>
ccl::event sparse_allreduce(const IndexBufferType* send_ind_buf,
                              size_t send_ind_count,
                              const ValueBufferType* send_val_buf,
                              size_t send_val_count,
                              IndexBufferType* recv_ind_buf,
                              size_t recv_ind_count,
                              ValueBufferType* recv_val_buf,
                              size_t recv_val_count,
                              ccl::reduction reduction,
                              const ccl::communicator& comm,
                              const ccl::sparse_allreduce_attr& attr,
                              const ccl::vector_class<ccl::event>& deps) {
    ccl::impl_dispatch disp;
    return disp(comm)->sparse_allreduce(send_ind_buf,
                                        send_ind_count,
                                        send_val_buf,
                                        send_val_count,
                                        recv_ind_buf,
                                        recv_ind_count,
                                        recv_val_buf,
                                        recv_val_count,
                                        reduction,
                                        disp(default_stream),
                                        attr,
                                        deps);
}

// template <class IndexBufferObjectType, class ValueBufferObjectType, typename T>
// ccl::event
// sparse_allreduce(const IndexBufferObjectType& send_ind_buf,
//                  size_t send_ind_count,
//                  const ValueBufferObjectType& send_val_buf,
//                  size_t send_val_count,
//                  IndexBufferObjectType& recv_ind_buf,
//                  size_t recv_ind_count,
//                  ValueBufferObjectType& recv_val_buf,
//                  size_t recv_val_count,
//                  ccl::reduction reduction,
//                  const ccl::communicator& comm,
//                  const ccl::stream& op_stream,
//                  const ccl::sparse_allreduce_attr& attr,
//                  const ccl::vector_class<ccl::event>& deps)
// {
//     ccl::impl_dispatch disp;
//     return disp(comm)->sparse_allreduce(send_ind_buf, send_ind_count,
//                                         send_val_buf, send_val_count,
//                                         recv_ind_buf, recv_ind_count,
//                                         recv_val_buf, recv_val_count,
//                                         reduction,
//                                         disp(op_stream), attr, deps);
// }
//
// template <class IndexBufferObjectType, class ValueBufferObjectType, typename T>
// ccl::event
// sparse_allreduce(const IndexBufferObjectType& send_ind_buf,
//                  size_t send_ind_count,
//                  const ValueBufferObjectType& send_val_buf,
//                  size_t send_val_count,
//                  IndexBufferObjectType& recv_ind_buf,
//                  size_t recv_ind_count,
//                  ValueBufferObjectType& recv_val_buf,
//                  size_t recv_val_count,
//                  ccl::reduction reduction,
//                  const ccl::communicator& comm,
//                  const ccl::sparse_allreduce_attr& attr,
//                  const ccl::vector_class<ccl::event>& deps)
// {
//     ccl::impl_dispatch disp;
//     return disp(comm)->sparse_allreduce(send_ind_buf, send_ind_count,
//                                         send_val_buf, send_val_count,
//                                         recv_ind_buf, recv_ind_count,
//                                         recv_val_buf, recv_val_count,
//                                         reduction,
//                                         disp(default_stream), attr, deps);
// }

} // namespace preview

// API force instantiations for Operations
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(char);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int64_t);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(float);
API_DEVICE_COMM_OP_PTR_EXPLICIT_INSTANTIATION(double);

#ifdef CCL_ENABLE_SYCL
#ifndef COMMA
#define COMMA ,
#endif
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<char COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int64_t COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<uint64_t COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<float COMMA 1>);
API_DEVICE_COMM_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<double COMMA 1>);
#undef COMMA
#endif // CCL_ENABLE_SYCL

namespace preview {

API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, char);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, int);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, ccl::bf16);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, float);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, double);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, int64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(char, uint64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, char);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, int);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, ccl::bf16);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, float);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, double);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, int64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int, uint64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, char);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, int);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, ccl::bf16);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, float);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, double);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, int64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(int64_t, uint64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, char);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, int);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, ccl::bf16);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, float);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, double);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, int64_t);
API_DEVICE_COMM_SPARSE_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t, uint64_t);

// #ifdef CCL_ENABLE_SYCL
// #ifndef COMMA
// #define COMMA ,
// #endif
// API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int COMMA 1>,
//                                                      cl::sycl::buffer<float COMMA 1>);
// API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int COMMA 1>,
//                                                      cl::sycl::buffer<ccl::bf16 COMMA 1>);

// API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int64_t COMMA 1>,
//                                                      cl::sycl::buffer<float COMMA 1>);
// API_DEVICE_COMM_SPARSE_OP_REF_EXPLICIT_INSTANTIATION(cl::sycl::buffer<int64_t COMMA 1>,
//                                                      cl::sycl::buffer<ccl::bf16 COMMA 1>);
// #undef COMMA
// #endif //CCL_ENABLE_SYCL

} // namespace preview

} // namespace ccl
