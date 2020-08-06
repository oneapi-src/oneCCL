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
#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
#include "exec/exec.hpp"

ccl_status_t ccl_set_resize_fn(ccl_resize_fn_t callback) {
    CCL_CHECK_IS_BLOCKED();
    try {
        return ccl::global_data::get().executor->create_listener(callback);
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_init() {
    try {
        ccl::global_data::get().init();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_finalize() {
    try {
        ccl::global_data::get().reset();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

#ifdef MULTI_GPU_SUPPORT
ccl_status_t CCL_API ccl_set_device_comm_attr(ccl_device_comm_attr_t* comm_attr,
                                              unsigned long attribute,
                                              ...) {
    if (!comm_attr) {
        return ccl_status_invalid_arguments;
    }

    //TODO
    return ccl_status_invalid_arguments;
}
#endif //MULTI_GPU_SUPPORT

ccl_status_t CCL_API ccl_get_version(ccl_version_t* version) {
    if (!version) {
        return ccl_status_invalid_arguments;
    }

    version->major = CCL_MAJOR_VERSION;
    version->minor = CCL_MINOR_VERSION;
    version->update = CCL_UPDATE_VERSION;
    version->product_status = CCL_PRODUCT_STATUS;
    version->build_date = CCL_PRODUCT_BUILD_DATE;
    version->full = CCL_PRODUCT_FULL;

    return ccl_status_success;
}

ccl_status_t CCL_API ccl_wait(ccl_request_t req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            LOG_ERROR("empty request");
            return ccl_status_success;
        }

        auto request = static_cast<ccl_request*>(req);
        ccl_wait_impl(ccl::global_data::get().executor.get(), request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_test(ccl_request_t req, int* is_completed) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            LOG_ERROR("empty request");
            if (is_completed) {
                *is_completed = 1;
            }
            return ccl_status_success;
        }

        auto request = static_cast<ccl_request*>(req);
        auto completed = ccl_test_impl(ccl::global_data::get().executor.get(), request);
        *is_completed = static_cast<int>(completed);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_comm_create(ccl_comm_t* comm, const ccl_comm_attr_t* attr) {
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(comm);
    try {
        ccl::global_data& data = ccl::global_data::get();
        ccl_comm* comm_ptr = nullptr;

        if (!attr) {
            LOG_DEBUG("create communicator as copy of global communicator");
            comm_ptr = new ccl_comm(data.comm->rank(), data.comm->size(), data.comm_ids->acquire());
        }
        else {
            LOG_DEBUG("create communicator with coll_attr");
            comm_ptr =
                ccl_comm::create_with_color(attr->color, data.comm_ids.get(), data.comm.get());
        }

        *comm = static_cast<void*>(comm_ptr);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_comm_free(ccl_comm_t comm) {
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(comm);
    LOG_DEBUG("free communicator ", comm);
    try {
        delete static_cast<ccl_comm*>(comm);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_get_comm_rank(ccl_comm_t comm, size_t* rank) {
    CCL_CHECK_IS_BLOCKED();
    if (!rank)
        return ccl_status_invalid_arguments;

    try {
        auto comm_ptr = (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get();
        *rank = comm_ptr->rank();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_get_comm_size(ccl_comm_t comm, size_t* size) {
    CCL_CHECK_IS_BLOCKED();
    if (!size)
        return ccl_status_invalid_arguments;

    try {
        auto comm_ptr = (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get();
        *size = comm_ptr->size();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_datatype_create(ccl_datatype_t* dtype, const ccl_datatype_attr_t* attr) {
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(dtype);
    LOG_DEBUG("create datatype");
    try {
        *dtype = ccl::global_data::get().dtypes->create(attr);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_get_datatype_size(ccl_datatype_t dtype, size_t* size) {
    CCL_CHECK_IS_BLOCKED();
    if (!size)
        return ccl_status_invalid_arguments;

    try {
        *size = ccl::global_data::get().dtypes->get(dtype).size();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_datatype_free(ccl_datatype_t dtype) {
    CCL_CHECK_IS_BLOCKED();
    LOG_DEBUG("free datatype ", dtype);
    try {
        ccl::global_data::get().dtypes->free(dtype);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_stream_create(ccl_stream_type_t type, void* native_stream, ccl_stream_t* stream) {
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(stream);
    try {
        LOG_DEBUG("create stream by type: ", type);
#ifdef MULTI_GPU_SUPPORT
#ifdef CCL_ENABLE_SYCL
        *stream = static_cast<void*>(
            stream_provider_dispatcher::create(*static_cast<cl::sycl::queue*>(native_stream))
                .release());
#else
        *stream = static_cast<void*>(stream_provider_dispatcher::create(
                                         *static_cast<ze_command_queue_handle_t*>(native_stream))
                                         .release());
#endif
#else
#ifdef CCL_ENABLE_SYCL
        if (type != ccl_stream_host) {
            *stream = static_cast<void*>(
                stream_provider_dispatcher::create(*static_cast<cl::sycl::queue*>(native_stream))
                    .release());
        }
        else
#endif
        {
            *stream =
                static_cast<void*>(stream_provider_dispatcher::create(native_stream).release());
        }

        //for legacy stream: override type for 'host' related queue
        static_cast<ccl_stream*>(*stream)->type = type;
#endif
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_stream_free(ccl_stream_t stream) {
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(stream);
    LOG_DEBUG("free stream ", stream);
    try {
        delete static_cast<const ccl_stream*>(stream);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_allgatherv(const void* send_buf,
                                    size_t send_count,
                                    void* recv_buf,
                                    const size_t* recv_counts,
                                    ccl_datatype_t dtype,
                                    const ccl_coll_attr_t* attr,
                                    ccl_comm_t comm,
                                    ccl_stream_t stream,
                                    ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_allgatherv_impl(
            send_buf,
            send_count,
            recv_buf,
            recv_counts,
            dtype,
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_allreduce(const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl_datatype_t dtype,
                                   ccl_reduction_t reduction,
                                   const ccl_coll_attr_t* attr,
                                   ccl_comm_t comm,
                                   ccl_stream_t stream,
                                   ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_allreduce_impl(
            send_buf,
            recv_buf,
            count,
            dtype,
            reduction,
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_alltoall(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl_datatype_t dtype,
                                  const ccl_coll_attr_t* attr,
                                  ccl_comm_t comm,
                                  ccl_stream_t stream,
                                  ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_alltoall_impl(
            send_buf,
            recv_buf,
            count,
            dtype,
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_alltoallv(const void* send_buf,
                                   const size_t* send_counts,
                                   void* recv_buf,
                                   const size_t* recv_counts,
                                   ccl_datatype_t dtype,
                                   const ccl_coll_attr_t* attr,
                                   ccl_comm_t comm,
                                   ccl_stream_t stream,
                                   ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_alltoallv_impl(
            send_buf,
            send_counts,
            recv_buf,
            recv_counts,
            dtype,
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_barrier(ccl_comm_t comm, ccl_stream_t stream) {
    try {
        ccl_barrier_impl((comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
                         static_cast<const ccl_stream*>(stream));
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_bcast(void* buf,
                               size_t count,
                               ccl_datatype_t dtype,
                               size_t root,
                               const ccl_coll_attr_t* attr,
                               ccl_comm_t comm,
                               ccl_stream_t stream,
                               ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_bcast_impl(
            buf,
            count,
            dtype,
            root,
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_reduce(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl_datatype_t dtype,
                                ccl_reduction_t reduction,
                                size_t root,
                                const ccl_coll_attr_t* attr,
                                ccl_comm_t comm,
                                ccl_stream_t stream,
                                ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_reduce_impl(
            send_buf,
            recv_buf,
            count,
            dtype,
            reduction,
            root,
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_sparse_allreduce(const void* send_ind_buf,
                                          size_t send_ind_count,
                                          const void* send_val_buf,
                                          size_t send_val_count,
                                          void* recv_ind_buf,
                                          size_t recv_ind_count,
                                          void* recv_val_buf,
                                          size_t recv_val_count,
                                          ccl_datatype_t index_dtype,
                                          ccl_datatype_t value_dtype,
                                          ccl_reduction_t reduction,
                                          const ccl_coll_attr_t* attr,
                                          ccl_comm_t comm,
                                          ccl_stream_t stream,
                                          ccl_request_t* req) {
    CCL_CHECK_IS_BLOCKED();
    try {
        if (!req) {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_sparse_allreduce_impl(
            send_ind_buf,
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
            attr,
            (comm) ? static_cast<ccl_comm*>(comm) : ccl::global_data::get().comm.get(),
            static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}
