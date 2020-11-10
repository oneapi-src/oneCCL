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
#include "common/utils/version.hpp"

#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)
#include "common/comm/l0/comm_context.hpp"
#include "common/comm/comm_interface.hpp"
#endif //#if defined(MULTI_GPU_SUPPORT) || defined(CCL_ENABLE_SYCL)

#include <memory>

#include "common/comm/single_device_communicator/single_device_communicator.hpp"

namespace ccl {

namespace detail {

CCL_API environment::environment() {
    static auto result = global_data::get().init();
    CCL_CHECK_AND_THROW(result, "failed to initialize CCL");
}

CCL_API environment::~environment() {}

CCL_API environment& environment::instance() {
    static environment env;
    return env;
}

ccl::library_version CCL_API environment::get_library_version() {
    return utils::get_library_version();
}

/******************** KVS ********************/

shared_ptr_class<kvs> environment::create_main_kvs(const kvs_attr& attr) const {
    return std::shared_ptr<kvs>(new kvs(attr));
}

shared_ptr_class<kvs> environment::create_kvs(const kvs::address_type& addr,
                                              const kvs_attr& attr) const {
    return std::shared_ptr<kvs>(new kvs(addr, attr));
}

/******************** DEVICE ********************/

device environment::create_device(empty_t empty) const {
    static typename ccl::unified_device_type::ccl_native_t default_native_device;
    return device::create_device(default_native_device);
}

/******************** CONTEXT ********************/

context environment::create_context(empty_t empty) const {
    static typename ccl::unified_context_type::ccl_native_t default_native_context;
    return context::create_context(default_native_context);
}

/******************** DATATYPE ********************/

ccl::datatype environment::register_datatype(const datatype_attr& attr) {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    LOG_DEBUG("register datatype");

    return ccl::global_data::get().dtypes->create(attr);
}

void environment::deregister_datatype(ccl::datatype dtype) {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    LOG_DEBUG("deregister datatype");

    ccl::global_data::get().dtypes->free(dtype);
}

size_t environment::get_datatype_size(ccl::datatype dtype) const {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    return ccl::global_data::get().dtypes->get(dtype).size();
}

/******************** STREAM ********************/

stream CCL_API environment::create_stream(typename unified_device_type::ccl_native_t device) {
    auto version = utils::get_library_version();
    return stream{ stream_provider_dispatcher::create(device, version) };
}

stream CCL_API environment::create_stream(typename unified_device_type::ccl_native_t device,
                                          typename unified_context_type::ccl_native_t context) {
    auto version = utils::get_library_version();
    return stream{ stream_provider_dispatcher::create(device, context, version) };
}

/******************** COMMUNICATOR ********************/

#ifdef CCL_ENABLE_SYCL
communicator environment::create_single_device_communicator(
    const int comm_size,
    const int rank,
    const cl::sycl::device& device,
    const cl::sycl::context& context,
    ccl::shared_ptr_class<kvs_interface> kvs) const {
    LOG_TRACE("Create single device communicator from SYCL device");

    std::shared_ptr<ikvs_wrapper> kvs_wrapper(new users_kvs(kvs));
    std::shared_ptr<atl_wrapper> atl =
        std::shared_ptr<atl_wrapper>(new atl_wrapper(comm_size, { rank }, kvs_wrapper));

    comm_split_attr attr = create_comm_split_attr(attr_val<comm_split_attr_id::group>(
        split_group::cluster /*group_split_type::undetermined*/));
    ccl::communicator_interface_ptr impl = ccl::communicator_interface::create_communicator_impl(
        device, context, rank, comm_size, attr, atl);

    //TODO use gpu_comm_attr to automatically visit()
    auto single_dev_comm = std::dynamic_pointer_cast<single_device_communicator>(impl);
    //single_dev_comm->set_context(context);
    return communicator(std::move(impl));
}
#endif

communicator environment::create_communicator(const comm_attr& attr) const {
    return communicator::create_communicator(attr);
}

communicator environment::create_communicator(const size_t size,
                                              ccl::shared_ptr_class<kvs_interface> kvs,
                                              const comm_attr& attr) const {
    return communicator::create_communicator(size, kvs, attr);
}

communicator environment::create_communicator(const size_t size,
                                              const int rank,
                                              ccl::shared_ptr_class<kvs_interface> kvs,
                                              const comm_attr& attr) const {
    return communicator::create_communicator(size, rank, kvs, attr);
}

} // namespace detail

} // namespace ccl

/******************** TypeGenerations ********************/

CREATE_DEV_COMM_INSTANTIATION(ccl::device, ccl::context)
CREATE_DEV_COMM_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t,
                              typename ccl::unified_context_type::ccl_native_t)
CREATE_DEV_COMM_INSTANTIATION(ccl::device_index_type,
                              typename ccl::unified_context_type::ccl_native_t)

CREATE_STREAM_INSTANTIATION(typename ccl::unified_stream_type::ccl_native_t)
CREATE_STREAM_EXT_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t,
                                typename ccl::unified_context_type::ccl_native_t)

CREATE_CONTEXT_INSTANTIATION(typename ccl::unified_context_type::ccl_native_t)
CREATE_DEVICE_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t)
