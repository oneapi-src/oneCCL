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
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_kvs.hpp"
#include "common/log/log.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/comm_context.hpp"

namespace ccl {

/*
 *  Single device communicator creation
 */
template <class DeviceType,
          class ContextType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type>
ccl::communicator_interface_ptr ccl::comm_group::create_communicator_from_group(
    const DeviceType& device,
    ContextType& context,
    const ccl::comm_split_attr& attr /* = comm_device_attr_t()*/) {
#ifdef CCL_ENABLE_SYCL
    static_assert(std::is_same<DeviceType, cl::sycl::device>::value,
                  "ccl::comm_group::create_communicator_from_group() - supports SYCL devices at now");
#endif

    ccl::communicator_interface_ptr impl;
    //TODO -S- temporary solution to support single device case
    auto device_count_per_process = pimpl->get_expected_process_device_size();
    LOG_DEBUG("Create communicator from device, expected devices per process: ",
              device_count_per_process);
    auto host_comm = pimpl->get_host_communicator();
    if (device_count_per_process == 1) /* special single device case */
    {
        LOG_TRACE("Create single device communicator from SYCL device");
        //TODO
        ccl::comm_split_attr single_dev_attr = attr;
        single_dev_attr.set<ccl::comm_split_attr_id::group>(
            ccl::group_split_type::undetermined);
        impl = ccl::communicator_interface::create_communicator_impl(
            device, context, host_comm->rank(), host_comm->size(), single_dev_attr, host_comm->get_atl());
    }
    else {
        // multiple device case
        impl = ccl::communicator_interface::create_communicator_impl(
            device, context, pimpl->thread_id, host_comm->rank(), attr, host_comm->get_atl());

        // registering device in group - is non blocking operation, until it is not the last device
        pimpl->sync_register_communicator(impl);
    }
    return impl;
}

template <class DeviceType,
          class ContextType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type>
ccl::communicator_interface_ptr ccl::comm_group::create_communicator_from_group(
    const DeviceType& device_id,
    ContextType& context,
    const ccl::comm_split_attr& attr /* = nullptr*/) {
    LOG_TRACE("Create communicator from id: ", device_id);
    auto host_comm = pimpl->get_host_communicator();

    ccl::communicator_interface_ptr impl = ccl::communicator_interface::create_communicator_impl(
        device_id, context, pimpl->thread_id, host_comm->rank(), attr, host_comm->get_atl());
    // registering device in group - is non blocking operation, until it is not the last device
    pimpl->sync_register_communicator(impl);
    return impl;
}

/**
 *  Multiple device communicators creation vectorized API implementation
 */
template <class InputIt, class ContextType>
std::vector<ccl::communicator> ccl::comm_group::create_communicators_group(
    InputIt first,
    InputIt last,
    ContextType& context,
    ccl::comm_split_attr attr /* = nullptr*/) {
    /*
    static_assert(not std::is_same<InputIt, typename ccl::vector_class<cl::sycl::device>::const_iterator>::value, "SYCL");
*/
    using iterator_value_type = typename std::iterator_traits<InputIt>::value_type;
    /*
    using expected_value_type = typename unified_device_type::device_t;
    static_assert(std::is_same<iterator_value_type, expected_value_type>::value,
                  "Not valid InputIt in create_communicators");
*/
    size_t indices_count = std::distance(first, last);
    LOG_TRACE("Create device communicators from index iterators type, count: ", indices_count);

    std::vector<ccl::communicator> comms;
    comms.reserve(indices_count);
    std::transform(
        first, last, std::back_inserter(comms), [this, attr, &context](const iterator_value_type& device_id) {
            return ccl::communicator(create_communicator_from_group<iterator_value_type, ContextType>(device_id, context, attr));
        });
    return comms;
}

template <template <class...> class Container, class Type, class ContextType>
std::vector<ccl::communicator> ccl::comm_group::create_communicators_group(
    const Container<Type>& device_ids,
    ContextType& context,
    ccl::comm_split_attr attr /* = nullptr*/) {
    //static_assert(not std::is_same<Type, cl::sycl::device>::value, "SYCL cont");
    //static_assert(std::is_same<Type, ccl::device_index_type>::value, "Invalid Type in create_communicators");
    LOG_TRACE("Create device communicators from index type, count: ",
              device_ids.size(),
              ". Redirect to iterators version");
    return this->create_communicators_group<typename Container<Type>::const_iterator, ContextType>(
        device_ids.begin(), device_ids.end(), context, attr);
}
/*
 ccl::comm_group::device_context_native_const_reference_t ccl::comm_group::get_context() const
{
    //TODO use PIMPL as context provider
    static unified_device_context_type context;
    return context.get();
}
*/
} // namespace ccl

/***************************************************************************************************/
#define COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(type, context_type) \
    template ccl::vector_class<ccl::communicator> ccl::comm_group::create_communicators_group( \
        const type& devices, context_type& ctx, ccl::comm_split_attr attr);

#define COMM_CREATOR_INDEXED_INSTANTIATION_TYPE(type, context_type) \
    template ccl::communicator_interface_ptr ccl::comm_group::create_communicator_from_group( \
        const type& device, context_type& context, const ccl::comm_split_attr& attr);
