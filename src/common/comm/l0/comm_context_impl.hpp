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
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/comm_context.hpp"
#include "common/comm/comm_interface.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"

namespace ccl {
/**
 *  Create communicator API:
 */
/*
 ccl::device_comm_split_attr ccl::comm_group::create_device_comm_split_attr()
{
    // TODO
    const auto& host_comm = pimpl->get_host_communicator();
    return ccl::device_comm_split_attr{new ccl::ccl_device_attr(*(host_comm->get_comm_split_attr()))};
}
*/
/*
 *  Single device communicator creation
 */
template <class DeviceType,
          typename std::enable_if<not std::is_same<typename std::remove_cv<DeviceType>::type,
                                                   ccl::device_index_type>::value,
                                  int>::type>
ccl::device_communicator ccl::comm_group::create_communicator(
    const DeviceType& device,
    const ccl::device_comm_split_attr& attr /* = comm_device_attr_t()*/) {
#ifdef CCL_ENABLE_SYCL
    static_assert(std::is_same<DeviceType, cl::sycl::device>::value,
                  "ccl::comm_group::create_communicator() - supports SYCL devices at now");
#endif

    ccl::communicator_interface_ptr impl;
    //TODO -S- temporary solution to support single device case
    auto device_count_per_process = pimpl->get_expected_process_device_size();
    LOG_DEBUG("Create communicator from device, expected devices per process: ",
              device_count_per_process);
    if (device_count_per_process == 1) /* special single device case */
    {
        LOG_TRACE("Create single device communicator from SYCL device");
        auto host_comm = pimpl->get_host_communicator();
        //TODO
        ccl::device_comm_split_attr single_dev_attr = attr;
        single_dev_attr.set<ccl::comm_split_attr_id::group>(
            ccl::device_group_split_type::undetermined);
        impl = ccl::communicator_interface::create_communicator_impl(
            device, host_comm->rank(), host_comm->size(), single_dev_attr);
    }
    else {
        // multiple device case
        impl = ccl::communicator_interface::create_communicator_impl(
            device, pimpl->thread_id, pimpl->get_host_communicator()->rank(), attr);

        // registering device in group - is non blocking operation, until it is not the last device
        pimpl->sync_register_communicator(impl);
    }
    return device_communicator(std::move(impl));
}

template <class DeviceType,
          typename std::enable_if<std::is_same<typename std::remove_cv<DeviceType>::type,
                                               ccl::device_index_type>::value,
                                  int>::type>
ccl::device_communicator ccl::comm_group::create_communicator(
    const DeviceType& device_id,
    const ccl::device_comm_split_attr& attr /* = nullptr*/) {
    LOG_TRACE("Create communicator from id: ", device_id);

    ccl::communicator_interface_ptr impl = ccl::communicator_interface::create_communicator_impl(
        device_id, pimpl->thread_id, pimpl->get_host_communicator()->rank(), attr);
    // registering device in group - is non blocking operation, until it is not the last device
    pimpl->sync_register_communicator(impl);
    return ccl::device_communicator(std::move(impl));
}

/**
 *  Multiple device communicators creation vectorized API implementation
 */
template <class InputIt>
std::vector<ccl::device_communicator> ccl::comm_group::create_communicators(
    InputIt first,
    InputIt last,
    ccl::device_comm_split_attr attr /* = nullptr*/) {
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

    std::vector<ccl::device_communicator> comms;
    comms.reserve(indices_count);
    std::transform(
        first, last, std::back_inserter(comms), [this, attr](const iterator_value_type& device_id) {
            return create_communicator<iterator_value_type>(device_id, attr);
        });
    return comms;
}

template <template <class...> class Container, class Type>
std::vector<ccl::device_communicator> ccl::comm_group::create_communicators(
    const Container<Type>& device_ids,
    ccl::device_comm_split_attr attr /* = nullptr*/) {
    //static_assert(not std::is_same<Type, cl::sycl::device>::value, "SYCL cont");
    //static_assert(std::is_same<Type, ccl::device_index_type>::value, "Invalid Type in create_communicators");
    LOG_TRACE("Create device communicators from index type, count: ",
              device_ids.size(),
              ". Redirect to iterators version");
    return this->create_communicators<typename Container<Type>::const_iterator>(
        device_ids.begin(), device_ids.end(), attr);
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
#define COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(type) \
    template ccl::vector_class<ccl::device_communicator> ccl::comm_group::create_communicators( \
        const type& devices, ccl::device_comm_split_attr attr);

#define COMM_CREATOR_INDEXED_INSTANTIATION_TYPE(type) \
    template ccl::device_communicator ccl::comm_group::create_communicator( \
        const type& device, const ccl::device_comm_split_attr& attr);
