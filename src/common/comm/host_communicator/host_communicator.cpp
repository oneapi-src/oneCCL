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
#include "ccl.hpp"
#include "common/comm/host_communicator/host_communicator_impl.hpp"
#include "common/comm/comm.hpp"

#ifdef MULTI_GPU_SUPPORT
#include "native_device_api/export_api.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#endif

using namespace ccl;

host_communicator::host_communicator(const ccl::comm_attr_t& attr) :
    comm_attr(attr),
    comm_impl()
{
    // legacy implementation
    if (!attr)
    {
        comm_impl = std::shared_ptr<ccl_comm>(
              new ccl_comm(global_data.comm->rank(),
                           global_data.comm->size(),
                           global_data.comm_ids->acquire()));
        comm_attr = ccl::environment::instance().create_host_comm_attr();
    }
    else
    {
        comm_impl = std::shared_ptr<ccl_comm>(
            ccl_comm::create_with_color(attr->get_value<ccl_host_attributes::ccl_host_color>(),
                                        global_data.comm_ids.get(),
                                        global_data.comm.get()));
    }

    comm_rank = comm_impl->rank();
    comm_size = comm_impl->size();
}

bool host_communicator::is_ready() const
{
    return true;
}

size_t host_communicator::rank() const
{
    return comm_rank;
}

size_t host_communicator::size() const
{
    return comm_size;
}

ccl::comm_attr_t host_communicator::get_host_attr() const
{
    return comm_attr;
}

#ifdef MULTI_GPU_SUPPORT
void host_communicator::visit(ccl::gpu_comm_attr& comm_attr)
{
    (void)(comm_attr);
}

ccl::device_topology_type host_communicator::get_topology_type() const
{
    throw ccl::ccl_error(std::string(__FUNCTION__) + " is not applicable for " +
                             traits::name());
}

ccl::device_index_type host_communicator::get_device_path() const
{
    return ccl::device_index_type{ccl::unused_index_value, 
                                  ccl::unused_index_value,
                                  ccl::unused_index_value};
}

ccl::communicator_interface::native_device_type_ref host_communicator::get_device()
{
    throw ccl::ccl_error(std::string(__FUNCTION__) + " is not applicable for " +
                             traits::name());
#ifdef CCL_ENABLE_SYCL
    static ccl::communicator_interface::native_device_type empty;
#else
    static ccl::communicator_interface::native_device_type_ref empty;
#endif
    return empty;
}

ccl::device_comm_attr_t host_communicator::get_device_attr() const
{
    return std::dynamic_pointer_cast<ccl::ccl_device_attr>(comm_attr);
}
#endif

void host_communicator::barrier(ccl::stream::impl_t& stream/* = ccl::stream_t()*/)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_barrier_impl(comm_impl.get(), stream_ptr);
}

/* allgatherv */
ccl::communicator::coll_request_t
host_communicator::allgatherv_impl(const void* send_buf,
                                   size_t send_count,
                                   void* recv_buf,
                                   const size_t* recv_counts,
                                   ccl_datatype_t dtype,
                                   const ccl::coll_attr* attr,
                                   ccl::stream::impl_t& stream)
{
    // c-api require null stream for host-stream for backward compatibility
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_allgatherv_impl(send_buf,
                                           send_count,
                                           recv_buf,
                                           recv_counts,
                                           dtype,
                                           attr, comm_impl.get(),
                                           stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* allreduce */
ccl::communicator::coll_request_t
host_communicator::allreduce_impl(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl_datatype_t dtype,
                                  ccl::reduction reduction,
                                  const ccl::coll_attr* attr,
                                  ccl::stream::impl_t& stream)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_allreduce_impl(send_buf,
                                          recv_buf,
                                          count,
                                          dtype,
                                          static_cast<ccl_reduction_t>(reduction),
                                          attr, comm_impl.get(),
                                          stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}


/* alltoall */
ccl::communicator::coll_request_t
host_communicator::alltoall_impl(const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl_datatype_t dtype,
                                 const ccl::coll_attr* attr,
                                 ccl::stream::impl_t& stream)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_alltoall_impl(send_buf,
                                         recv_buf,
                                         count,
                                         dtype,
                                         attr, comm_impl.get(),
                                         stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}


/* alltoallv */
ccl::communicator::coll_request_t
host_communicator::alltoallv_impl(const void* send_buf,
                                  const size_t* send_counts,
                                  void* recv_buf,
                                  const size_t* recv_counts,
                                  ccl_datatype_t dtype,
                                  const ccl::coll_attr* attr,
                                  ccl::stream::impl_t& stream)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_alltoallv_impl(send_buf,
                                          send_counts,
                                          recv_buf,
                                          recv_counts,
                                          dtype,
                                          attr, comm_impl.get(),
                                          stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}


/* bcast */
ccl::communicator::coll_request_t
host_communicator::bcast_impl(void* buf,
                              size_t count,
                              ccl_datatype_t dtype,
                              size_t root,
                              const ccl::coll_attr* attr,
                              ccl::stream::impl_t& stream)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_bcast_impl(buf, count,
                                      dtype,
                                      root, attr, comm_impl.get(),
                                      stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}

/* reduce */
ccl::communicator::coll_request_t
host_communicator::reduce_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl_datatype_t dtype,
                               ccl::reduction reduction,
                               size_t root,
                               const ccl::coll_attr* attr,
                               ccl::stream::impl_t& stream)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_reduce_impl(send_buf,
                                       recv_buf,
                                       count,
                                       dtype,
                                       static_cast<ccl_reduction_t>(reduction),
                                       root, attr, comm_impl.get(),
                                       stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}


/* sparse_allreduce */
ccl::communicator::coll_request_t
host_communicator::sparse_allreduce_impl(const void* send_ind_buf, size_t send_ind_count,
                                         const void* send_val_buf, size_t send_val_count,
                                         void* recv_ind_buf, size_t recv_ind_count,
                                         void* recv_val_buf, size_t recv_val_count,
                                         ccl_datatype_t index_dtype,
                                         ccl_datatype_t value_dtype,
                                         ccl::reduction reduction,
                                         const ccl::coll_attr* attr,
                                         ccl::stream::impl_t& stream)
{
    const ccl_stream* stream_ptr =
        (stream->get_type() != static_cast<ccl_stream_type_t>(ccl::stream_type::host)) ?
        stream.get() : nullptr;

    ccl_request* req = ccl_sparse_allreduce_impl(send_ind_buf, send_ind_count,
                                                 send_val_buf, send_val_count,
                                                 recv_ind_buf, recv_ind_count,
                                                 recv_val_buf, recv_val_count,
                                                 index_dtype, value_dtype,
                                                 static_cast<ccl_reduction_t>(reduction),
                                                 attr, comm_impl.get(),
                                                 stream_ptr);
    return std::unique_ptr<ccl::host_request_impl>(new ccl::host_request_impl(req));
}
/***********************************************************************/

COMM_INTERFACE_COLL_INSTANTIATIONS(host_communicator, char);
COMM_INTERFACE_COLL_INSTANTIATIONS(host_communicator, int);
COMM_INTERFACE_COLL_INSTANTIATIONS(host_communicator, int64_t);
COMM_INTERFACE_COLL_INSTANTIATIONS(host_communicator, uint64_t);
COMM_INTERFACE_COLL_INSTANTIATIONS(host_communicator, float);
COMM_INTERFACE_COLL_INSTANTIATIONS(host_communicator, double);

#ifdef CCL_ENABLE_SYCL
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(host_communicator, cl::sycl::buffer<char COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(host_communicator, cl::sycl::buffer<int COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(host_communicator, cl::sycl::buffer<int64_t COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(host_communicator, cl::sycl::buffer<uint64_t COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(host_communicator, cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(host_communicator, cl::sycl::buffer<double COMMA 1>);
#endif //CCL_ENABLE_SYCL

COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, char);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, int);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, ccl::bfp16);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, float);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, double);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, int64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, char, uint64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, char);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, int);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, ccl::bfp16);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, float);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, double);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, int64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int, uint64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, char);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, int);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, ccl::bfp16);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, float);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, double);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, int64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, int64_t, uint64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, char);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, int);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, ccl::bfp16);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, float);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, double);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, int64_t);
COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(host_communicator, uint64_t, uint64_t);

#ifdef CCL_ENABLE_SYCL
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(host_communicator,
                                                                 cl::sycl::buffer<int COMMA 1>,
                                                                 cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(host_communicator,
                                                                 cl::sycl::buffer<int COMMA 1>,
                                                                 cl::sycl::buffer<ccl::bfp16 COMMA 1>);

    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(host_communicator,
                                                                 cl::sycl::buffer<int64_t COMMA 1>,
                                                                 cl::sycl::buffer<float COMMA 1>);
    COMM_INTERFACE_SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(host_communicator,
                                                                 cl::sycl::buffer<int64_t COMMA 1>,
                                                                 cl::sycl::buffer<ccl::bfp16 COMMA 1>);
#endif //CCL_ENABLE_SYCL
