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
#include "ccl_type_traits.hpp"
#include "common/global/global.hpp"
#include "exec/exec.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif

#define CCL_CHECK_AND_THROW(result, diagnostic)   \
  do {                                            \
      if (result != ccl_status_success)           \
      {                                           \
          throw ccl_error(diagnostic);            \
      }                                           \
  } while (0);


namespace ccl
{

static std::atomic<size_t> env_ref_counter;

class request_impl final : public request
{
public:
    explicit request_impl(ccl_request* r) : req(r)
    {
        if (!req)
        {
            // If the user calls collective with coll_attr->synchronous=1 then it will be progressed
            // in place and API will return null request. In this case mark cpp wrapper as completed,
            // all calls to wait() or test() will do nothing
            completed = true;
        }
    }

    ~request_impl() override
    {
        if (!completed)
        {
            LOG_ERROR("not completed request is destroyed");
        }
    }

    void wait() final
    {
        if (!completed)
        {
            ccl_wait_impl(global_data.executor.get(), req);
            completed = true;
        }
    }

    bool test() final
    {
        if (!completed)
        {
            completed = ccl_test_impl(global_data.executor.get(), req);
        }
        return completed;
    }

private:
    ccl_request* req = nullptr;
    bool completed = false;
};
}

CCL_API ccl::environment::environment()
{
    static auto result = ccl_init();
    env_ref_counter.fetch_add(1);
    CCL_CHECK_AND_THROW(result, "failed to initialize ccl");
}

CCL_API ccl::environment& ccl::environment::instance()
{
    static thread_local bool created = false;
    if (!created)
    {
        /* 
            environment destructor uses logger for ccl_finalize and it should be destroyed before logger,
            therefore construct thread_local logger at first follows to global/static initialization rules
        */
        LOG_INFO("created environment");
        created = true;
    }
    static thread_local std::unique_ptr<ccl::environment> env(new environment);
    return *env;
}

void CCL_API ccl::environment::set_resize_fn(ccl_resize_fn_t callback)
{
    ccl_status_t result = ccl_set_resize_fn(callback);
    CCL_CHECK_AND_THROW(result, "failed to set resize callback");
    return;
}

ccl_version_t CCL_API ccl::environment::get_version() const
{
    ccl_version_t ret;
    ccl_status_t result = ccl_get_version(&ret);
    CCL_CHECK_AND_THROW(result, "failed to get version");
    return ret;
}

ccl::communicator_t CCL_API ccl::environment::create_communicator(const ccl::comm_attr* attr/* = nullptr*/) const
{
    return communicator_t(new ccl::communicator(attr));
}

ccl::stream_t CCL_API ccl::environment::create_stream(ccl::stream_type type/* = ccl::stream_type::cpu*/,
                                                                    void* native_stream/* = nullptr*/) const
{
#ifndef CCL_ENABLE_SYCL
    if (type == ccl::stream_type::sycl)
    {
        throw ccl_error("SYCL stream is not supported in current ccl version");
    }
#endif
    return stream_t( new ccl::stream(type, native_stream));
}

CCL_API ccl::environment::~environment()
{
    if (env_ref_counter.fetch_sub(1) == 1)
    {
        auto result = ccl_finalize();
        if (result != ccl_status_success)
        {
            abort();
        }
    }
}

CCL_API ccl::stream::stream()
{
    this->stream_impl = std::make_shared<ccl_stream>(static_cast<ccl_stream_type_t>(ccl::stream_type::cpu),
                                                     nullptr /* native_stream */);
}

CCL_API ccl::stream::stream(ccl::stream_type type, void* native_stream)
{
    this->stream_impl = std::make_shared<ccl_stream>(static_cast<ccl_stream_type_t>(type),
                                                     native_stream);
}

CCL_API ccl::communicator::communicator()
{
    comm_impl = global_data.comm;
}

CCL_API ccl::communicator::communicator(const ccl::comm_attr* attr)
{
    if (!attr)
    {
        comm_impl = global_data.comm;
    }
    else
    {
        comm_impl = std::shared_ptr<ccl_comm>(
            ccl_comm::create_with_color(attr->color,
                                        global_data.comm_ids.get(),
                                        global_data.comm.get()));
    }
}

size_t CCL_API ccl::communicator::rank()
{
    return comm_impl->rank();
}

size_t CCL_API ccl::communicator::size()
{
    return comm_impl->size();
}


/* allgatherv */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::allgatherv(const void* send_buf,
                              size_t send_count,
                              void* recv_buf,
                              const size_t* recv_counts,
                              ccl::data_type dtype,
                              const ccl::coll_attr* attr,
                              const ccl::stream_t& stream)
{
    ccl_request* req = ccl_allgatherv_impl(send_buf, send_count,
                                           recv_buf, recv_counts,
                                           static_cast<ccl_datatype_t>(dtype),
                                           attr, comm_impl.get(),
                                           (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class buffer_type,
         typename T>
CCL_API ccl::communicator::coll_request_t
ccl::communicator::allgatherv(const buffer_type* send_buf,
                              size_t send_count,
                              buffer_type* recv_buf,
                              const size_t* recv_counts,
                              const ccl::coll_attr* attr,
                              const ccl::stream_t& stream)
{
    return allgatherv((const void*)send_buf, send_count,
                      (void*)recv_buf, recv_counts,
                      ccl::native_type_info<buffer_type>::ccl_datatype_value,
                      attr, stream);
}

template<class buffer_container_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::allgatherv(const buffer_container_type& send_buf,
                              size_t send_count,
                              buffer_container_type& recv_buf,
                              const size_t* recv_counts,
                              const ccl::coll_attr* attr,
                              const ccl::stream_t& stream)
{
    return allgatherv(reinterpret_cast<const void*>(&send_buf), send_count,
                      reinterpret_cast<void*>(&recv_buf), recv_counts,
                      ccl::native_type_info<buffer_container_type>::ccl_datatype_value,
                      attr, stream);
}

/* allreduce */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::allreduce(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::data_type dtype,
                             ccl::reduction reduction,
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream)
{
    ccl_request* req = ccl_allreduce_impl(send_buf, recv_buf, count,
                                          static_cast<ccl_datatype_t>(dtype),
                                          static_cast<ccl_reduction_t>(reduction),
                                          attr, comm_impl.get(),
                                          (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class buffer_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::allreduce(const buffer_type* send_buf,
                             buffer_type* recv_buf,
                             size_t count,
                             ccl::reduction reduction,
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream)
{
    return allreduce((const void*)send_buf, (void*)recv_buf, count,
                     ccl::native_type_info<buffer_type>::ccl_datatype_value,
                     reduction, attr, stream);
}

template<class buffer_container_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::allreduce(const buffer_container_type& send_buf,
                             buffer_container_type& recv_buf,
                             size_t count,
                             ccl::reduction reduction,
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream)
{
    return allreduce(reinterpret_cast<const void*>(&send_buf),
                     reinterpret_cast<void*>(&recv_buf), count,
                     ccl::native_type_info<buffer_container_type>::ccl_datatype_value,
                     reduction, attr, stream);
}

/* alltoall */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::alltoall(const void* send_buf,
                            void* recv_buf,
                            size_t count,
                            ccl::data_type dtype,
                            const ccl::coll_attr* attr,
                            const ccl::stream_t& stream)
{
    ccl_request* req = ccl_alltoall_impl(send_buf, recv_buf, count,
                                         static_cast<ccl_datatype_t>(dtype),
                                         attr, comm_impl.get(),
                                         (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class buffer_type, typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::alltoall(const buffer_type* send_buf,
                            buffer_type* recv_buf,
                            size_t count,
                            const ccl::coll_attr* attr,
                            const ccl::stream_t& stream)
{
    return alltoall((const void*)send_buf, (void*)recv_buf, count,
                    ccl::native_type_info<buffer_type>::ccl_datatype_value,
                    attr, stream);
}

template<class buffer_container_type, typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::alltoall(const buffer_container_type& send_buf,
                            buffer_container_type& recv_buf,
                            size_t count,
                            const ccl::coll_attr* attr,
                            const ccl::stream_t& stream)
{
    return alltoall(reinterpret_cast<const void*>(&send_buf),
                    reinterpret_cast<void*>(&recv_buf), count,
                    ccl::native_type_info<buffer_container_type>::ccl_datatype_value,
                    attr, stream);
}

/* alltoallv */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::alltoallv(const void* send_buf,
                             const size_t* send_counts,
                             void* recv_buf,
                             const size_t* recv_counts,
                             ccl::data_type dtype,
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream)
{
    ccl_request* req = ccl_alltoallv_impl(send_buf, send_counts, recv_buf, recv_counts,
                                          static_cast<ccl_datatype_t>(dtype),
                                          attr, comm_impl.get(),
                                          (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class buffer_type, typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::alltoallv(const buffer_type* send_buf,
                             const size_t* send_counts,
                             buffer_type* recv_buf,
                             const size_t* recv_counts,
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream)
{
    return alltoallv((const void*)send_buf, send_counts, (void*)recv_buf, recv_counts,
                     ccl::native_type_info<buffer_type>::ccl_datatype_value,
                     attr, stream);
}

template<class buffer_container_type, typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::alltoallv(const buffer_container_type& send_buf,
                             const size_t* send_counts,
                             buffer_container_type& recv_buf,
                             const size_t* recv_counts,
                             const ccl::coll_attr* attr,
                             const ccl::stream_t& stream)
{
    return alltoallv(reinterpret_cast<const void*>(&send_buf), send_counts,
                     reinterpret_cast<void*>(&recv_buf), recv_counts,
                     ccl::native_type_info<buffer_container_type>::ccl_datatype_value,
                     attr, stream);
}

/* barrier */
void CCL_API
ccl::communicator::barrier(const ccl::stream_t& stream)
{
    ccl_barrier_impl(comm_impl.get(),
                     (stream) ? stream->stream_impl.get() : nullptr);
    return;
}

/* bcast */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::bcast(void* buf,
                         size_t count,
                         ccl::data_type dtype,
                         size_t root,
                         const ccl::coll_attr* attr,
                         const ccl::stream_t& stream)
{
    ccl_request* req = ccl_bcast_impl(buf, count,
                                      static_cast<ccl_datatype_t>(dtype),
                                      root, attr, comm_impl.get(),
                                      (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class buffer_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::bcast(buffer_type* buf,
                         size_t count,
                         size_t root,
                         const ccl::coll_attr* attr,
                         const ccl::stream_t& stream)

{
    return bcast((void*)buf, count,
                 ccl::native_type_info<buffer_type>::ccl_datatype_value,
                 root, attr, stream);
}

template<class buffer_container_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::bcast(buffer_container_type& buf,
                         size_t count,
                         size_t root,
                         const ccl::coll_attr* attr,
                         const ccl::stream_t& stream)
{
    return bcast(reinterpret_cast<void*>(&buf), count,
                 ccl::native_type_info<buffer_container_type>::ccl_datatype_value,
                 root, attr, stream);
}


/* reduce */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::reduce(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          ccl::data_type dtype,
                          ccl::reduction reduction,
                          size_t root,
                          const ccl::coll_attr* attr,
                          const ccl::stream_t& stream)
{
    ccl_request* req = ccl_reduce_impl(send_buf, recv_buf, count,
                                       static_cast<ccl_datatype_t>(dtype),
                                       static_cast<ccl_reduction_t>(reduction),
                                       root, attr, comm_impl.get(),
                                     (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class buffer_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::reduce(const buffer_type* send_buf,
                          buffer_type* recv_buf,
                          size_t count,
                          ccl::reduction reduction,
                          size_t root,
                          const ccl::coll_attr* attr,
                          const ccl::stream_t& stream)
{
    return reduce((const void*)send_buf, (void*)recv_buf, count,
                  ccl::native_type_info<buffer_type>::ccl_datatype_value,
                  reduction, root, attr, stream);
}

template<class buffer_container_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::reduce(const buffer_container_type& send_buf,
                          buffer_container_type& recv_buf,
                          size_t count,
                          ccl::reduction reduction,
                          size_t root,
                          const ccl::coll_attr* attr,
                          const ccl::stream_t& stream)
{
    return reduce(reinterpret_cast<const void*>(&send_buf),
                  reinterpret_cast<void*>(&recv_buf), count,
                  ccl::native_type_info<buffer_container_type>::ccl_datatype_value,
                  reduction, root, attr, stream);
}


/* sparse_allreduce */
ccl::communicator::coll_request_t CCL_API
ccl::communicator::sparse_allreduce(const void* send_ind_buf, size_t send_ind_count,
                                    const void* send_val_buf, size_t send_val_count,
                                    void** recv_ind_buf, size_t* recv_ind_count,
                                    void** recv_val_buf, size_t* recv_val_count,
                                    ccl::data_type index_dtype,
                                    ccl::data_type value_dtype,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr,
                                    const ccl::stream_t& stream)
{
    ccl_request* req = ccl_sparse_allreduce_impl(send_ind_buf, send_ind_count,
                                                 send_val_buf, send_val_count,
                                                 recv_ind_buf, recv_ind_count,
                                                 recv_val_buf, recv_val_count,
                                                 static_cast<ccl_datatype_t>(index_dtype),
                                                 static_cast<ccl_datatype_t>(value_dtype),
                                                 static_cast<ccl_reduction_t>(reduction),
                                                 attr, comm_impl.get(),
                                                 (stream) ? stream->stream_impl.get() : nullptr);
    return std::unique_ptr<ccl::request_impl>(new ccl::request_impl(req));
}

template<class index_buffer_type,
         class value_buffer_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::sparse_allreduce(const index_buffer_type* send_ind_buf, size_t send_ind_count,
                                    const value_buffer_type* send_val_buf, size_t send_val_count,
                                    index_buffer_type** recv_ind_buf, size_t* recv_ind_count,
                                    value_buffer_type** recv_val_buf, size_t* recv_val_count,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr,
                                    const ccl::stream_t& stream)
{
    return sparse_allreduce((const void*)send_ind_buf, send_ind_count,
                            (const void*)send_val_buf, send_val_count,
                            (void**)recv_ind_buf, recv_ind_count,
                            (void**)recv_val_buf, recv_val_count,
                            ccl::native_type_info<index_buffer_type>::ccl_datatype_value,
                            ccl::native_type_info<value_buffer_type>::ccl_datatype_value,
                            reduction, attr, stream);
}

template<class index_buffer_container_type,
         class value_buffer_container_type,
         typename T>
ccl::communicator::coll_request_t CCL_API
ccl::communicator::sparse_allreduce(const index_buffer_container_type& send_ind_buf, size_t send_ind_count,
                                    const value_buffer_container_type& send_val_buf, size_t send_val_count,
                                    index_buffer_container_type** recv_ind_buf, size_t* recv_ind_count,
                                    value_buffer_container_type** recv_val_buf, size_t* recv_val_count,
                                    ccl::reduction reduction,
                                    const ccl::coll_attr* attr,
                                    const ccl::stream_t& stream)
{
    return sparse_allreduce(reinterpret_cast<const void*>(&send_ind_buf), send_ind_count,
                            reinterpret_cast<const void*>(&send_val_buf), send_val_count,
                            reinterpret_cast<void**>(recv_ind_buf), recv_ind_count,
                            reinterpret_cast<void**>(recv_val_buf), recv_val_count,
                            native_type_info<index_buffer_container_type>::ccl_datatype_value,
                            native_type_info<value_buffer_container_type>::ccl_datatype_value,
                            reduction, attr, stream);
}

/***********************************************************************/

#define COLL_EXPLICIT_INSTANTIATION(type)                   \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::allgatherv(const type* send_buf,         \
                              size_t send_count,            \
                              type* recv_buf,               \
                              const size_t* recv_counts,    \
                              const ccl::coll_attr* attr,   \
                              const ccl::stream_t& stream); \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::allreduce(const type* send_buf,          \
                             type* recv_buf,                \
                             size_t count,                  \
                             ccl::reduction reduction,      \
                             const ccl::coll_attr* attr,    \
                             const ccl::stream_t& stream);  \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::alltoall(const type* send_buf,           \
                            type* recv_buf,                 \
                            size_t count,                   \
                            const ccl::coll_attr* attr,     \
                            const ccl::stream_t& stream);   \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::alltoallv(const type* send_buf,          \
                             const size_t* send_counts,     \
                             type* recv_buf,                \
                             const size_t* recv_counts,     \
                             const ccl::coll_attr* attr,    \
                             const ccl::stream_t& stream);  \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::bcast(type* buf,                         \
                         size_t count,                      \
                         size_t root,                       \
                         const ccl::coll_attr* attr,        \
                         const ccl::stream_t& stream);      \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::reduce(const type* send_buf,             \
                          type* recv_buf,                   \
                          size_t count,                     \
                          ccl::reduction reduction,         \
                          size_t root,                      \
                          const ccl::coll_attr* attr,       \
                          const ccl::stream_t& stream);

#define COLL_EXPLICIT_CLASS_INSTANTIATION(type)             \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::allgatherv(const type& send_buf,         \
                              size_t send_count,            \
                              type& recv_buf,               \
                              const size_t* recv_counts,    \
                              const ccl::coll_attr* attr,   \
                              const ccl::stream_t& stream); \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::allreduce(const type& send_buf,          \
                             type& recv_buf,                \
                             size_t count,                  \
                             ccl::reduction reduction,      \
                             const ccl::coll_attr* attr,    \
                             const ccl::stream_t& stream);  \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::alltoall(const type& send_buf,           \
                            type& recv_buf,                 \
                            size_t count,                   \
                            const ccl::coll_attr* attr,     \
                            const ccl::stream_t& stream);   \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::alltoallv(const type& send_buf,          \
                             const size_t* send_counts,     \
                             type& recv_buf,                \
                             const size_t* recv_counts,     \
                             const ccl::coll_attr* attr,    \
                             const ccl::stream_t& stream);  \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::bcast(type& buf,                         \
                         size_t count,                      \
                         size_t root,                       \
                         const ccl::coll_attr* attr,        \
                         const ccl::stream_t& stream);      \
                                                            \
template ccl::communicator::coll_request_t CCL_API          \
ccl::communicator::reduce(const type& send_buf,             \
                          type& recv_buf,                   \
                          size_t count,                     \
                          ccl::reduction reduction,         \
                          size_t root,                      \
                          const ccl::coll_attr* attr,       \
                          const ccl::stream_t& stream);

#define SPARSE_ALLREDUCE_EXPLICIT_INSTANTIATION(index_type, value_type)                    \
template ccl::communicator::coll_request_t CCL_API                                         \
ccl::communicator::sparse_allreduce(const index_type* send_ind_buf, size_t send_ind_count, \
                                    const value_type* send_val_buf, size_t send_val_count, \
                                    index_type** recv_ind_buf, size_t* recv_ind_count,     \
                                    value_type** recv_val_buf, size_t* recv_val_count,     \
                                    ccl::reduction reduction,                              \
                                    const ccl::coll_attr* attr,                            \
                                    const ccl::stream_t& stream);
                                                                                           \
#define SPARSE_ALLREDUCE_EXPLICIT_CLASS_INSTANTIATION(index_type, value_type)              \
template ccl::communicator::coll_request_t CCL_API                                         \
ccl::communicator::sparse_allreduce(const index_type& send_ind_buf, size_t send_ind_count, \
                                    const value_type& send_val_buf, size_t send_val_count, \
                                    index_type** recv_ind_buf, size_t* recv_ind_count,     \
                                    value_type** recv_val_buf, size_t* recv_val_count,     \
                                    ccl::reduction reduction,                              \
                                    const ccl::coll_attr* attr,                            \
                                    const ccl::stream_t& stream);

#include "ccl_cpp_api_explicit_in.hpp"
