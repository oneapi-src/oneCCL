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
#include "environment_impl.hpp"
#include "common/global/global.hpp"
#include "exec/exec.hpp"

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
#include "common/comm/l0/comm_context.hpp"
#include "common/comm/comm_interface.hpp"
#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

//#include "ccl.h"    //TODO datatypes

#include <memory>

#include "common/comm/single_device_communicator/single_device_communicator.hpp"

namespace ccl {
CCL_API ccl::environment::environment() {
    static auto result = global_data::get().init();
    CCL_CHECK_AND_THROW(result, "failed to initialize CCL");
}

CCL_API ccl::environment::~environment() {}

CCL_API ccl::environment& ccl::environment::instance() {
    static ccl::environment env;
    return env;
}

// void CCL_API ccl::environment::set_resize_fn(ccl_resize_fn_t callback)
// {
//     ccl_status_t result = ccl_set_resize_fn(callback);
//     CCL_CHECK_AND_THROW(result, "failed to set resize callback");
//     return;
// }

ccl::library_version CCL_API ccl::environment::get_library_version() const {
    ccl::library_version ret;

    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

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
shared_ptr_class<kvs> CCL_API environment::create_main_kvs() const {
    return std::shared_ptr<kvs>(new kvs);
}

shared_ptr_class<kvs> CCL_API environment::create_kvs(const kvs::address_type& addr) const {
    return std::shared_ptr<kvs>(new kvs(addr));
}

// device
device CCL_API environment::create_device(empty_t empty) const
{
    static typename ccl::unified_device_type::ccl_native_t default_native_device;
    return device::create_device(default_native_device);
}

// context
context CCL_API environment::create_context(empty_t empty) const
{
    static typename ccl::unified_device_context_type::ccl_native_t default_native_context;
    return context::create_context(default_native_context);
}

ccl::datatype CCL_API environment::register_datatype(const ccl::datatype_attr& attr) {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    LOG_DEBUG("register datatype");

    return ccl::global_data::get().dtypes->create(attr);
}

void CCL_API environment::deregister_datatype(ccl::datatype dtype) {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    LOG_DEBUG("deregister datatype");

    ccl::global_data::get().dtypes->free(dtype);
}

size_t CCL_API environment::get_datatype_size(ccl::datatype dtype) const {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    return ccl::global_data::get().dtypes->get(dtype).size();
}

} // namespace ccl

#ifdef CCL_ENABLE_SYCL
ccl::communicator CCL_API ccl::environment::create_single_device_communicator(
    const size_t comm_size,
    const size_t rank,
    const cl::sycl::device& device,
    const cl::sycl::context& context,
    ccl::shared_ptr_class<ccl::kvs_interface> kvs) const {
    LOG_TRACE("Create single device communicator from SYCL device");

    std::shared_ptr<ikvs_wrapper> kvs_wrapper(new users_kvs(kvs));
    std::shared_ptr<atl_wrapper> atl =
        std::shared_ptr<atl_wrapper>(new atl_wrapper(comm_size, { rank }, kvs_wrapper));

    ccl::comm_split_attr attr = create_comm_split_attr(
        ccl::attr_val<ccl::comm_split_attr_id::group>(ccl::group_split_type::undetermined));
    ccl::communicator_interface_ptr impl =
        ccl::communicator_interface::create_communicator_impl(device, context, rank, comm_size, attr, atl);

    //TODO use gpu_comm_attr to automatically visit()
    auto single_dev_comm = std::dynamic_pointer_cast<single_device_communicator>(impl);
    //single_dev_comm->set_context(context);
    return ccl::communicator(std::move(impl));
}

#endif

//Communicator
ccl::communicator CCL_API ccl::environment::create_communicator() const {
    return ccl::communicator::create_communicator();
}

ccl::communicator CCL_API ccl::environment::create_communicator(const size_t size,
                                                      ccl::shared_ptr_class<ccl::kvs_interface> kvs) const {
    return ccl::communicator::create_communicator(size, kvs);
}

ccl::communicator CCL_API ccl::environment::create_communicator(const size_t size,
                                                      const size_t rank,
                                                      ccl::shared_ptr_class<ccl::kvs_interface> kvs) const {
    return ccl::communicator::create_communicator(size, rank, kvs);
}

/***************************TypeGenerations*********************************************************/
namespace ccl {
template <>
stream CCL_API environment::create_postponed_api_type<
    stream,
    typename unified_device_type::ccl_native_t,
    typename unified_device_context_type::ccl_native_t>(
    typename unified_device_type::ccl_native_t device,
    typename unified_device_context_type::ccl_native_t context) const {
    library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    return stream{ stream_provider_dispatcher::create(device, context, ret) };
}
template <>
stream CCL_API
environment::create_postponed_api_type<stream,
                                            typename unified_device_type::ccl_native_t>(
    typename unified_device_type::ccl_native_t device) const {
    library_version ret{};
    ret.major = CCL_MAJOR_VERSION;
    ret.minor = CCL_MINOR_VERSION;
    ret.update = CCL_UPDATE_VERSION;
    ret.product_status = CCL_PRODUCT_STATUS;
    ret.build_date = CCL_PRODUCT_BUILD_DATE;
    ret.full = CCL_PRODUCT_FULL;

    return stream{ stream_provider_dispatcher::create(device, ret) };
}
}
CREATE_OP_ATTR_INSTANTIATION(ccl::allgatherv_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::allreduce_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::alltoall_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::alltoallv_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::broadcast_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::reduce_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::reduce_scatter_attr)
CREATE_OP_ATTR_INSTANTIATION(ccl::sparse_allreduce_attr)

CREATE_OP_ATTR_INSTANTIATION(ccl::comm_split_attr)

CREATE_OP_ATTR_INSTANTIATION(ccl::datatype_attr)

CREATE_DEV_COMM_INSTANTIATION(ccl::device, ccl::context)
CREATE_DEV_COMM_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t, typename ccl::unified_device_context_type::ccl_native_t)
CREATE_DEV_COMM_INSTANTIATION(ccl::device_index_type, typename ccl::unified_device_context_type::ccl_native_t)

CREATE_STREAM_INSTANTIATION(typename ccl::unified_stream_type::ccl_native_t)
CREATE_STREAM_EXT_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t, typename ccl::unified_device_context_type::ccl_native_t)

CREATE_CONTEXT_INSTANTIATION(typename ccl::unified_device_context_type::ccl_native_t)
CREATE_DEVICE_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t)

/*
CREATE_EVENT_INSTANTIATION(cl::sycl::event)
CREATE_EVENT_EXT_INSTANTIATION(cl_event)
*/
