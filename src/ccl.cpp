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
#include "coll/algorithms/allreduce/allreduce_2d.hpp"
#include "coll/selection/selection.hpp"
#include "comp/bfp16/bfp16_utils.h"
#include "common/comm/atl_tag.hpp"
#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "parallelizer/parallelizer.hpp"
#include "unordered_coll/unordered_coll.hpp"

ccl_status_t ccl_set_resize_fn(ccl_resize_fn_t callback)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        return global_data.executor->create_listener(callback);
    }
    COMMON_CATCH_BLOCK();
}

void ccl_init_resize_dependent_objects(ccl_global_data& gl_data)
{
    global_data.dtypes = std::unique_ptr<ccl_datatype_storage>(new ccl_datatype_storage());

    gl_data.sched_cache = std::unique_ptr<ccl_sched_cache>(new ccl_sched_cache());

    if (env_data.enable_fusion)
    {
        /* create fusion_manager before executor because service_worker uses fusion_manager */
        gl_data.fusion_manager =
            std::unique_ptr<ccl_fusion_manager>(new ccl_fusion_manager());
    }

    global_data.executor = std::unique_ptr<ccl_executor>(new ccl_executor());

    gl_data.comm_ids = std::unique_ptr<ccl_comm_id_storage>(new ccl_comm_id_storage(ccl_comm::max_comm_count));

    gl_data.comm = std::make_shared<ccl_comm>(gl_data.executor->get_global_proc_idx(),
                                              gl_data.executor->get_global_proc_count(),
                                              gl_data.comm_ids->acquire(true));

    if (env_data.enable_unordered_coll)
    {
        gl_data.unordered_coll_manager =
            std::unique_ptr<ccl_unordered_coll_manager>(new ccl_unordered_coll_manager());
    }

    gl_data.allreduce_2d_builder =
            std::unique_ptr<ccl_allreduce_2d_builder>(new ccl_allreduce_2d_builder());

    gl_data.atl_tag =
        std::unique_ptr<ccl_atl_tag>(new ccl_atl_tag(gl_data.executor->get_atl_attr().tag_bits,
                                                     gl_data.executor->get_atl_attr().max_tag));
    if (env_data.default_resizable)
        ccl_set_resize_fn(nullptr);
}

void ccl_init_resize_independent_objects(ccl_global_data& gl_data)
{
    gl_data.parallelizer = std::unique_ptr<ccl_parallelizer>(new ccl_parallelizer(env_data.worker_count));

    gl_data.algorithm_selector =
        std::unique_ptr<ccl_algorithm_selector_wrapper<CCL_COLL_LIST>>(
                new ccl_algorithm_selector_wrapper<CCL_COLL_LIST>());

    gl_data.algorithm_selector->init();

    if (gl_data.executor->get_global_proc_idx() == 0)
        gl_data.algorithm_selector->print();

    gl_data.default_coll_attr.reset(new ccl_coll_attr_t{});
    memset(gl_data.default_coll_attr.get(), 0, sizeof(ccl_coll_attr_t));

    gl_data.bfp16_impl_type = ccl_bfp16_get_impl_type();

    if (gl_data.bfp16_impl_type != ccl_bfp16_none)
    {
        LOG_INFO("BFP16 is enabled through ",
            (gl_data.bfp16_impl_type == ccl_bfp16_avx512bf) ? "AVX512-BF" : "AVX512-F");
    }
    else
    {
#ifdef CCL_BFP16_COMPILER
        LOG_INFO("BFP16 is disabled on HW level");
#else
        LOG_INFO("BFP16 is disabled on compiler level");
#endif
    }
}

void ccl_reset_resize_dependent_objects(ccl_global_data& gl_data)
{
    gl_data.atl_tag.reset();
    gl_data.allreduce_2d_builder.reset();
    gl_data.unordered_coll_manager.reset();
    gl_data.comm.reset();
    gl_data.comm_ids.reset();
    gl_data.fusion_manager.reset();
    gl_data.sched_cache.reset();
    gl_data.dtypes.reset();
}

void ccl_reset_resize_independent_objects(ccl_global_data& gl_data)
{
    gl_data.parallelizer.reset();
    gl_data.algorithm_selector.reset();
    gl_data.default_coll_attr.reset();
}

ccl_status_t ccl_init()
{
    try
    {
        ccl_env_parse();

        ccl_init_resize_dependent_objects(global_data);
        ccl_init_resize_independent_objects(global_data);

        if (global_data.executor->get_global_proc_idx() == 0)
            ccl_env_print();

        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_finalize()
{
    try
    {
        /*
            executor is resize_dependent object but out of regular reset procedure
            exector is responsible for resize logic and has own multi-step reset
        */
        global_data.executor.reset();

        ccl_reset_resize_dependent_objects(global_data);
        ccl_reset_resize_independent_objects(global_data);

        return ccl_status_success;
    }

    COMMON_CATCH_BLOCK();
}

#ifdef MULTI_GPU_SUPPORT
ccl_status_t CCL_API ccl_set_device_comm_attr(ccl_device_comm_attr_t* comm_attr, unsigned long attribute, ...)
{
    if (!comm_attr)
    {
        return ccl_status_invalid_arguments;
    }

    //TODO
    return ccl_status_invalid_arguments;
}
#endif //MULTI_GPU_SUPPORT

ccl_status_t CCL_API ccl_get_version(ccl_version_t* version)
{
    if (!version)
    {
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

ccl_status_t CCL_API ccl_wait(ccl_request_t req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            LOG_ERROR("empty request");
            return ccl_status_success;
        }

        auto request = static_cast<ccl_request*>(req);
        ccl_wait_impl(global_data.executor.get(), request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_test(ccl_request_t req, int* is_completed)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            LOG_ERROR("empty request");
            if (is_completed)
            {
                *is_completed = 1;
            }
            return ccl_status_success;
        }

        auto request = static_cast<ccl_request*>(req);
        auto completed = ccl_test_impl(global_data.executor.get(), request);
        *is_completed = static_cast<int>(completed);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_comm_create(ccl_comm_t* comm, const ccl_comm_attr_t* attr)
{
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(comm);
    try
    {
        ccl_comm* comm_ptr = nullptr;
        if (!attr)
        {
            LOG_DEBUG("create communicator as copy of global communicator");
            comm_ptr = new ccl_comm(global_data.comm->rank(),
                                    global_data.comm->size(),
                                    global_data.comm_ids->acquire());
        }
        else
        {
            LOG_DEBUG("create communicator with coll_attr");
            comm_ptr = ccl_comm::create_with_color(attr->color,
                                                   global_data.comm_ids.get(),
                                                   global_data.comm.get());
        }

        *comm = static_cast<void*>(comm_ptr);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_comm_free(ccl_comm_t comm)
{
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(comm);
    LOG_DEBUG("free communicator ", comm);
    try
    {
        delete static_cast<ccl_comm*>(comm);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_get_comm_rank(ccl_comm_t comm, size_t* rank)
{
    CCL_CHECK_IS_BLOCKED();
    if (!rank)
        return ccl_status_invalid_arguments;

    try
    {
        auto comm_ptr = (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get();
        *rank = comm_ptr->rank();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_get_comm_size(ccl_comm_t comm, size_t* size)
{
    CCL_CHECK_IS_BLOCKED();
    if (!size)
        return ccl_status_invalid_arguments;

    try
    {
        auto comm_ptr = (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get();
        *size = comm_ptr->size();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_datatype_create(ccl_datatype_t* dtype, const ccl_datatype_attr_t* attr)
{
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(dtype);
    LOG_DEBUG("create datatype");
    try
    {
        *dtype = global_data.dtypes->create(attr);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_get_datatype_size(ccl_datatype_t dtype, size_t* size)
{
    CCL_CHECK_IS_BLOCKED();
    if (!size)
        return ccl_status_invalid_arguments;

    try
    {
        *size = global_data.dtypes->get(dtype).size();
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_datatype_free(ccl_datatype_t dtype)
{
    CCL_CHECK_IS_BLOCKED();
    LOG_DEBUG("free datatype ", dtype);
    try
    {
        global_data.dtypes->free(dtype);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_stream_create(ccl_stream_type_t type,
                               void* native_stream,
                               ccl_stream_t* stream)
{
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(stream);
    try
    {
        LOG_DEBUG("create stream by type: ", type);
#ifdef MULTI_GPU_SUPPORT
    #ifdef CCL_ENABLE_SYCL
            *stream = static_cast<void*>(stream_provider_dispatcher::create(*static_cast<cl::sycl::queue*>(native_stream)).release());
    #else
            *stream = static_cast<void*>(stream_provider_dispatcher::create(*static_cast<ze_command_queue_handle_t*>(native_stream)).release());
    #endif
#else
    #ifdef CCL_ENABLE_SYCL
        if( type != ccl_stream_host)
        {
            *stream = static_cast<void*>(stream_provider_dispatcher::create(*static_cast<cl::sycl::queue*>(native_stream)).release());
        }
        else
    #endif
        {
            *stream = static_cast<void*>(stream_provider_dispatcher::create(native_stream).release());
        }

        //for legacy stream: override type for 'host' related queue
        static_cast<ccl_stream*>(*stream)->type = type;
#endif
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t ccl_stream_free(ccl_stream_t stream)
{
    CCL_CHECK_IS_BLOCKED();
    CCL_ASSERT(stream);
    LOG_DEBUG("free stream ", stream);
    try
    {
        delete static_cast<const ccl_stream*>(stream);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_allgatherv(
    const void* send_buf,
    size_t send_count,
    void* recv_buf,
    const size_t* recv_counts,
    ccl_datatype_t dtype,
    const ccl_coll_attr_t* attr,
    ccl_comm_t comm,
    ccl_stream_t stream,
    ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_allgatherv_impl(send_buf, send_count, recv_buf, recv_counts, dtype, attr,
                                           (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                           static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_allreduce(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl_datatype_t dtype,
    ccl_reduction_t reduction,
    const ccl_coll_attr_t* attr,
    ccl_comm_t comm,
    ccl_stream_t stream,
    ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_allreduce_impl(send_buf, recv_buf, count, dtype, reduction, attr,
                                          (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                          static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_alltoall(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl_datatype_t dtype,
    const ccl_coll_attr_t* attr,
    ccl_comm_t comm,
    ccl_stream_t stream,
    ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_alltoall_impl(send_buf, recv_buf, count, dtype, attr,
                                         (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                         static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_alltoallv(
    const void* send_buf,
    const size_t* send_counts,
    void* recv_buf,
    const size_t* recv_counts,
    ccl_datatype_t dtype,
    const ccl_coll_attr_t* attr,
    ccl_comm_t comm,
    ccl_stream_t stream,
    ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_alltoallv_impl(send_buf, send_counts, recv_buf, recv_counts, dtype, attr,
                                          (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                          static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_barrier(ccl_comm_t comm, ccl_stream_t stream)
{
    try
    {
        ccl_barrier_impl((comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                         static_cast<const ccl_stream*>(stream));
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_bcast(
    void* buf,
    size_t count,
    ccl_datatype_t dtype,
    size_t root,
    const ccl_coll_attr_t* attr,
    ccl_comm_t comm,
    ccl_stream_t stream,
    ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_bcast_impl(buf, count, dtype, root, attr,
                                      (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                      static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_reduce(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl_datatype_t dtype,
    ccl_reduction_t reduction,
    size_t root,
    const ccl_coll_attr_t* attr,
    ccl_comm_t comm,
    ccl_stream_t stream,
    ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_reduce_impl(send_buf, recv_buf, count, dtype, reduction, root, attr,
                                       (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                       static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}

ccl_status_t CCL_API ccl_sparse_allreduce(const void* send_ind_buf, size_t send_ind_count,
                                          const void* send_val_buf, size_t send_val_count,
                                          void* recv_ind_buf, size_t recv_ind_count,
                                          void* recv_val_buf, size_t recv_val_count,
                                          ccl_datatype_t index_dtype,
                                          ccl_datatype_t value_dtype,
                                          ccl_reduction_t reduction,
                                          const ccl_coll_attr_t* attr,
                                          ccl_comm_t comm,
                                          ccl_stream_t stream,
                                          ccl_request_t* req)
{
    CCL_CHECK_IS_BLOCKED();
    try
    {
        if (!req)
        {
            return ccl_status_invalid_arguments;
        }
        auto request = ccl_sparse_allreduce_impl(send_ind_buf, send_ind_count,
                                                 send_val_buf, send_val_count,
                                                 recv_ind_buf, recv_ind_count,
                                                 recv_val_buf, recv_val_count,
                                                 index_dtype, value_dtype,
                                                 reduction, attr,
                                                 (comm) ? static_cast<ccl_comm*>(comm) : global_data.comm.get(),
                                                 static_cast<const ccl_stream*>(stream));
        *req = static_cast<ccl_request_t>(request);
        return ccl_status_success;
    }
    COMMON_CATCH_BLOCK();
}
