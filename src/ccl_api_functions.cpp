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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "oneapi/ccl/api_functions.hpp"
#include "comm/comm.hpp"
#include "oneapi/ccl/exception.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "ccl_api_functions_generators.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_sycl.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#endif // CCL_ENABLE_ZE || CCL_ENABLE_SYCL

namespace ccl {

namespace v1 {

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

void init(const init_attr& attr) {
    auto& env = detail::environment::instance();
    (void)env;
}

/******************** ENVIRONMENT ********************/

library_version get_library_version() {
    return detail::environment::get_library_version();
}

/* datatype */
datatype register_datatype(const datatype_attr& attr) {
    return detail::environment::instance().register_datatype(attr);
}

void deregister_datatype(datatype dtype) {
    return detail::environment::instance().deregister_datatype(dtype);
}

size_t get_datatype_size(datatype dtype) {
    return detail::environment::instance().get_datatype_size(dtype);
}

/* KVS */
shared_ptr_class<kvs> create_main_kvs(const kvs_attr& attr) {
    return detail::environment::instance().create_main_kvs(attr);
}

shared_ptr_class<kvs> create_kvs(const kvs::address_type& addr, const kvs_attr& attr) {
    return detail::environment::instance().create_kvs(addr, attr);
}

/* device */
device create_device() {
    static empty_t empty{};
    return detail::environment::instance().create_device(empty);
}

/* context */
context create_context() {
    static empty_t empty{};
    return detail::environment::instance().create_context(empty);
}

/* stream */
stream create_stream() {
    return default_stream;
}

} // namespace v1

namespace preview {

vector_class<communicator> split_communicators(
    const vector_class<pair_class<communicator, comm_split_attr>>& attrs) {
    // TODO not implemented
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    // return detail::environment::instance().split_device_communicators(attrs);
    return {};
}

/* communicator */
communicator create_communicator(const comm_attr& attr) {
    return ccl::detail::environment::instance().create_communicator(attr);
}

communicator create_communicator(const int size,
                                 shared_ptr_class<kvs_interface> kvs,
                                 const comm_attr& attr) {
    return ccl::detail::environment::instance().create_communicator(size, kvs, attr);
}

} // namespace preview

namespace v1 {

communicator create_communicator(const int size,
                                 const int rank,
                                 shared_ptr_class<kvs_interface> kvs,
                                 const comm_attr& attr) {
    return detail::environment::instance().create_communicator(size, rank, kvs, attr);
}

/******************** COMMUNICATOR ********************/

#define CHECK_DEPS(deps) \
    do { \
        if (!deps.empty()) { \
            throw ccl::exception( \
                std::string(__PRETTY_FUNCTION__) + \
                " - handling a vector of events that the operation should depend on is not implemented"); \
        } \
    } while (0)

/* allgatherv */
event allgatherv(const void* send_buf,
                 size_t send_count,
                 void* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
#if !(defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL))
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
#else // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    bool is_single_node = false;
    bool is_oversubscription = true;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
        is_oversubscription = topo_manager.has_oversubscription();
    }

    sycl::queue q = op_stream.get_native();

    if (ccl::global_data::env().skip_scheduler && q.is_in_order() && is_single_node &&
        comm.size() == ccl::global_data::get().get_local_proc_count() && !is_oversubscription &&
        (dtype == ccl::datatype::float16 || dtype == ccl::datatype::bfloat16 ||
         dtype == ccl::datatype::float32 || dtype == ccl::datatype::int32)) {
        LOG_DEBUG("|CCL_SYCL| allgatherv selects sycl-kernels send_count: ",
                  send_count,
                  ", datatype: ",
                  dtype);

        bool done = false;
        ccl::event e = allgather_sycl(q,
                                      send_buf,
                                      send_count,
                                      recv_buf,
                                      recv_counts,
                                      dtype,
                                      comm,
                                      op_stream,
                                      attr,
                                      deps,
                                      done);
        if (done) {
            if (ccl::global_data::env().enable_op_sync) {
                e.wait();
            }
            return e;
        }
    }

    LOG_DEBUG(
        "|CCL_SCHED| allgatherv using scheduler send_count: ", send_count, ", datatype: ", dtype);
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
}

event allgatherv(const void* send_buf,
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

event allgatherv(const void* send_buf,
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

event allgatherv(const void* send_buf,
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
event allreduce(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
#if !(defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL))
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(op_stream), attr, deps);
#else // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    bool is_single_node = false;
    bool is_oversubscription = true;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
        is_oversubscription = topo_manager.has_oversubscription();
    }

    sycl::queue q = op_stream.get_native();

    if (ccl::global_data::env().skip_scheduler && is_single_node && !is_oversubscription &&
        q.is_in_order() && reduction == ccl::reduction::sum &&
        comm.size() == ccl::global_data::get().get_local_proc_count() &&
        (dtype == ccl::datatype::float16 || dtype == ccl::datatype::bfloat16 ||
         dtype == ccl::datatype::float32 || dtype == ccl::datatype::int32)) {
        LOG_DEBUG(
            "|CCL_SYCL| allreduce selects sycl-kernels count: ", count, ", datatype: ", dtype);

        bool done = false;
        ccl::event e = allreduce_sycl(
            q, send_buf, recv_buf, count, dtype, reduction, comm, op_stream, attr, deps, done);
        if (done) {
            if (ccl::global_data::env().enable_op_sync) {
                e.wait();
            }
            return e;
        }
    }

    LOG_DEBUG("|CCL_SCHED| allreduce selects scheduler count: ", count, ", datatype: ", dtype);
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(op_stream), attr, deps);
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    assert(false);
}

event allreduce(const void* send_buf,
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
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
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
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

/* alltoall */
event alltoall(const void* send_buf,
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

event alltoall(const void* send_buf,
               void* recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(default_stream), attr, deps);
}

event alltoall(const vector_class<void*>& send_buf,
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
event alltoallv(const void* send_buf,
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

event alltoallv(const void* send_buf,
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

event alltoallv(const vector_class<void*>& send_bufs,
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

event alltoallv(const vector_class<void*>& send_bufs,
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
event barrier(const communicator& comm,
              const stream& op_stream,
              const barrier_attr& attr,
              const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(op_stream), attr, deps);
}

event barrier(const communicator& comm, const barrier_attr& attr, const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(default_stream), attr, deps);
}

/* broadcast */
event broadcast(void* buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, dtype, root, disp(op_stream), attr, deps);
}

event broadcast(void* buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, dtype, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                size_t count,
                int root,
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
                int root,
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
                int root,
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
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

/* reduce */
event reduce(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(op_stream), attr, deps);
}

event reduce(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             reduction reduction,
             int root,
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
             int root,
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
             int root,
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
             int root,
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
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

/* reduce_scatter */
event reduce_scatter(const void* send_buf,
                     void* recv_buf,
                     size_t recv_count,
                     datatype dtype,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
#if !(defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL))
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(op_stream), attr, deps);
#else // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    bool is_single_node = false;
    bool is_oversubscription = true;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
        is_oversubscription = topo_manager.has_oversubscription();
    }

    sycl::queue q = op_stream.get_native();

    if (ccl::global_data::env().skip_scheduler && is_single_node && q.is_in_order() &&
        !is_oversubscription && comm.size() == ccl::global_data::get().get_local_proc_count() &&
        (dtype == ccl::datatype::float16 || dtype == ccl::datatype::bfloat16 ||
         dtype == ccl::datatype::float32 || dtype == ccl::datatype::int32) &&
        reduction == ccl::reduction::sum) {
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects sycl-kernels recv_count: ",
                  recv_count,
                  ", datatype: ",
                  dtype)
        ccl::event e;
        bool done = false;
        e = reduce_scatter_sycl(
            q, send_buf, recv_buf, recv_count, dtype, reduction, comm, op_stream, done);
        if (done) {
            if (ccl::global_data::env().enable_op_sync) {
                e.wait();
            }
            return e;
        }
    }

    LOG_DEBUG("|CCL_SCHED| reduce_scatter using scheduler recv_count: ",
              recv_count,
              ", datatype: ",
              dtype);
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(op_stream), attr, deps);
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
}

event reduce_scatter(const void* send_buf,
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

/* recv */
event recv(void* recv_buf,
           size_t recv_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, dtype, peer, disp(op_stream), attr, deps);
}

event recv(void* recv_buf,
           size_t recv_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, dtype, peer, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event recv(BufferType* recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event recv(BufferType* recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event recv(BufferObjectType& recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event recv(BufferObjectType& recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(default_stream), attr, deps);
}

/* send */
event send(void* send_buf,
           size_t send_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, dtype, peer, disp(op_stream), attr, deps);
}

event send(void* send_buf,
           size_t send_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, dtype, peer, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event send(BufferType* send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event send(BufferType* send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event send(BufferObjectType& send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event send(BufferObjectType& send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(default_stream), attr, deps);
}

} // namespace v1

namespace v1 {

// API force instantiations for Operations
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int8_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint8_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int16_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint16_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int32_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint32_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int64_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t);
/*API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(ccl::float16);*/
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(float);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(double);
/*API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(ccl::bfloat16);*/

#ifdef CCL_ENABLE_SYCL
#ifndef COMMA
#define COMMA ,
#endif

API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int8_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint8_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int16_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint16_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int32_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint32_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int64_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint64_t COMMA 1>);
/*API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<ccl::float16 COMMA 1>);*/
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<float COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<double COMMA 1>);
/*API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<ccl::bfloat16 COMMA 1>);*/

#undef COMMA
#endif // CCL_ENABLE_SYCL

} // namespace v1

} // namespace ccl
