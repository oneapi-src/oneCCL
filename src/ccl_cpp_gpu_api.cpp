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
#include <stdexcept>

#include "ccl.hpp"
#include "ccl_type_traits.hpp"
#include "common/global/global.hpp"
#include "exec/exec.hpp"

#include "common/comm/comm_attributes.hpp"
#include "common/comm/comm_interface.hpp"
#include "common/comm/host_communicator/host_communicator.hpp"
#include "common/comm/l0/gpu_comm_attr.hpp"
#include "common/comm/l0/device_community.hpp"

#include "native_device_api/export_api.hpp"
#include "native_device_api/compiler_ccl_wrappers_dispatcher.hpp"

#ifdef CCL_ENABLE_SYCL
#include <CL/sycl.hpp>
#endif

std::ostream& operator<<(std::ostream& out, const ccl::device_index_type& index)
{
    out << ccl::to_string(index);
    return out;
}

namespace ccl
{

CCL_API
std::string to_string(const device_index_type& device_id)
{
    std::stringstream ss;
    ss << "["
        << std::get<ccl::device_index_enum::driver_index_id>(device_id)
        << ":"
        << std::get<ccl::device_index_enum::device_index_id>(device_id)
        << ":";

    auto subdevice_id = std::get<ccl::device_index_enum::subdevice_index_id>(device_id);
    if (subdevice_id == ccl::unused_index_value)
    {
        ss << "*";
    }
    else
    {
        ss << std::get<ccl::device_index_enum::subdevice_index_id>(device_id);
    }
    ss << "]";

    return ss.str();
}

CCL_API
device_index_type from_string(const std::string& device_id_str)
{
    std::string::size_type from_pos = device_id_str.find('[');
    if(from_pos == std::string::npos)
    {
        throw std::invalid_argument(std::string("Cannot get ccl::device_index_type from input: ") +
                                    device_id_str);

    }

    if(device_id_str.size() == 1)
    {
         throw std::invalid_argument(std::string("Cannot get ccl::device_index_type from input, too less: ") +
                                    device_id_str);
    }
    from_pos ++;

    device_index_type path(unused_index_value, unused_index_value, unused_index_value);

    size_t cur_index = 0;
    do
    {
        std::string::size_type to_pos = device_id_str.find(':', from_pos);
        std::string::size_type count = (to_pos != std::string::npos ?
                                            to_pos - from_pos : std::string::npos);
        std::string index_string(device_id_str, from_pos, count);
        switch(cur_index)
        {
            case device_index_enum::driver_index_id:
            {
                auto index = std::atoll(index_string.c_str());
                if (index < 0)
                {
                    throw std::invalid_argument(
                                std::string("Cannot get ccl::device_index_type from input, "
                                            "driver index invalid: ") + device_id_str);
                }
                std::get<device_index_enum::driver_index_id>(path) = index;
                break;
            }
            case device_index_enum::device_index_id:
            {
                auto index = std::atoll(index_string.c_str());
                if (index < 0)
                {
                    throw std::invalid_argument(
                                std::string("Cannot get ccl::device_index_type from input, "
                                            "device index invalid: ") + device_id_str);
                }
                std::get<device_index_enum::device_index_id>(path) = index;
                break;
            }
            case device_index_enum::subdevice_index_id:
            {
                auto index = std::atoll(index_string.c_str());
                std::get<device_index_enum::subdevice_index_id>(path) = index < 0 ? unused_index_value : index;
                break;
            }
            default:
                 throw std::invalid_argument(
                                std::string("Cannot get ccl::device_index_type from input, "
                                            "unsupported format: ") + device_id_str);
        }

        cur_index ++;
        if(device_id_str.size() > to_pos)
        {
            to_pos ++;
        }
        from_pos = to_pos;
    }
    while( from_pos < device_id_str.size());

    return path;
}
// TODO Separate definitions into different files
#ifdef CCL_ENABLE_SYCL
CCL_API generic_device_type<CCL_ENABLE_SYCL_TRUE>::generic_device_type(device_index_type id,
                                                                       cl::sycl::info::device_type type/* = info::device_type::gpu*/)
    :device()
{
    LOG_DEBUG("To find SYCL device by index: ", id, ", type: ",
              static_cast<typename std::underlying_type<cl::sycl::info::device_type>::type>(type));
    auto platforms = cl::sycl::platform::get_platforms();
    LOG_DEBUG("Found SYCL plalforms: ", platforms.size());
    for(const auto& platform : platforms)
    {
        auto devices = platform.get_devices(type);
        LOG_DEBUG("SYCL Platform id: ", (platform.is_host() ? nullptr : platform.get()), " found devices: ", devices.size());
        auto it = std::find_if(devices.begin(), devices.end(), [id] (const cl::sycl::device& dev)
        {
            return id == native::get_runtime_device(dev)->get_device_path();
        });

        if(it != devices.end())
        {
            device = *it;
            return;
        }
    }

    throw std::runtime_error(std::string("Cannot find device by id: ") + ccl::to_string(id));
}

generic_device_type<CCL_ENABLE_SYCL_TRUE>::generic_device_type(const cl::sycl::device &in_device)
     :device(in_device)
{
}

device_index_type generic_device_type<CCL_ENABLE_SYCL_TRUE>::get_id() const noexcept
{
    return native::get_runtime_device(device)->get_device_path();
}

typename generic_device_type<CCL_ENABLE_SYCL_TRUE>::native_reference_t
generic_device_type<CCL_ENABLE_SYCL_TRUE>::get() noexcept
{
    return device;
}


generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::native_reference_t
    generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::get() noexcept
{
    return const_cast<generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::native_reference_t>(static_cast<const generic_device_context_type<CCL_ENABLE_SYCL_TRUE>*>(this)->get());
}

generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::native_const_reference_t
    generic_device_context_type<CCL_ENABLE_SYCL_TRUE>::get() const noexcept
{
    //TODO
    static cl::sycl::context ctx;
    return ctx;
}

#else
generic_device_type<CCL_ENABLE_SYCL_FALSE>::generic_device_type(device_index_type id)
 :device(id)
{
}

device_index_type generic_device_type<CCL_ENABLE_SYCL_FALSE>::get_id() const noexcept
{
    return device;
}

typename generic_device_type<CCL_ENABLE_SYCL_FALSE>::native_reference_t
generic_device_type<CCL_ENABLE_SYCL_FALSE>::get() noexcept
{
    return native::get_runtime_device(device);

}
generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::native_reference_t
    generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::get() noexcept
{
    return const_cast<generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::native_reference_t>(static_cast<const generic_device_context_type<CCL_ENABLE_SYCL_FALSE>*>(this)->get());
}

generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::native_const_reference_t
    generic_device_context_type<CCL_ENABLE_SYCL_FALSE>::get() const noexcept
{
    //TODO
    return native::get_platform();
}

#endif


struct group_context
{
    using group_unique_key = std::shared_ptr<ccl_comm>;
    std::map<group_unique_key, comm_group_t> communicator_group_map;

    ccl_spinlock mutex;
};

group_context global_ctx;
}

/* GPU communicator attributes
 */
CCL_API ccl::ccl_device_attr::ccl_device_attr(const ccl::ccl_host_attr& src) :
 base_t(src),
 pimpl(new ccl::device_attr_impl())
{
}

CCL_API ccl::ccl_device_attr::~ccl_device_attr() noexcept
{
}

template<ccl_device_attributes attrId,
             class Value,
             typename T>
CCL_API Value ccl::ccl_device_attr::set_value(const Value& v)
{
    return pimpl->set_attribute_value(v);
}

template<ccl_device_attributes attrId>
CCL_API const typename ccl::ccl_device_attributes_traits<attrId>::type& ccl::ccl_device_attr::get_value() const
{
    return pimpl->get_attribute_value(
            std::integral_constant<ccl_device_attributes, attrId> {});
}

/* Global Environment*/
template<class stream_native_type, typename T>
CCL_API ccl::stream_t ccl::environment::create_stream(stream_native_type& s)
{
    return ccl::stream_t(new ccl::stream(stream_provider_dispatcher::create(s)));
}

CCL_API ccl::comm_group_t ccl::environment::create_comm_group(size_t current_device_group_size, size_t process_device_group_size,
                                                              ccl::shared_communicator_t parent_comm /* = ccl::shared_communicator_t()*/)
{
    if (!parent_comm)
    {
        //use global communicator by default
        ccl::shared_communicator_t(ccl::environment::instance().create_communicator()).swap(parent_comm);
    }

    ccl::comm_group_t group;
    {
        // register group slot in global context table, based on communicator id
        auto host_comm_impl = std::dynamic_pointer_cast<host_communicator>(parent_comm->pimpl);
        if (!host_comm_impl)
        {
            throw ccl::ccl_error(std::string(__FUNCTION__) + " - failed, invalid host communicator type");
        }

        group_context::group_unique_key unique_id = host_comm_impl->comm_impl;
        if (!unique_id)
        {
            throw ccl::ccl_error(std::string(__FUNCTION__) + " - failed, host communicator is empty");
        }

        std::unique_lock<ccl_spinlock> lock(global_ctx.mutex);
        auto ctx_it = global_ctx.communicator_group_map.find(unique_id);
        if(ctx_it == global_ctx.communicator_group_map.end())
        {
            group.reset(new ccl::comm_group(parent_comm,
                                            current_device_group_size,
                                            process_device_group_size));
            global_ctx.communicator_group_map.insert({
                                                        unique_id,
                                                        group
                                                     });
        }
        else
        {
            group = ctx_it->second;
        }
    }

    // sync existing group: blocking operation - wait for all groups
    group->pimpl->sync_group_size(current_device_group_size);
    return group;
}

CCL_API ccl::comm_group::comm_group(ccl::shared_communicator_t parent_comm,
                                    size_t current_device_group_size, size_t process_device_group_size):
    pimpl(new ccl::gpu_comm_attr(parent_comm, current_device_group_size, process_device_group_size))
{
};

/**
 *  Create communicator API:
 */
CCL_API ccl::device_comm_attr_t ccl::comm_group::create_device_comm_attr()
{
    // TODO
    const auto& host_comm = pimpl->get_host_communicator();
    return ccl::device_comm_attr_t{new ccl::ccl_device_attr(*(host_comm->get_host_attr()))};
}
/*
 *  Single device communicator creation
 */
template <class DeviceType,
          typename std::enable_if<std::is_class<typename std::remove_cv<DeviceType>::type>::value,
                                      int>::type>
CCL_API ccl::communicator_t ccl::comm_group::create_communicator(const DeviceType& device,
                                                                     ccl::device_comm_attr_t attr/* = comm_device_attr_t()*/)
{
    LOG_TRACE("Create communicator from device");
    ccl::communicator_interface_ptr impl =
            ccl::communicator_interface::create_communicator_impl(device,
                                                                  pimpl->thread_id,
                                                                  pimpl->ccl_communicator->rank(),
                                                                  attr);
    // registering device in group - is non blocking operation, until it is not the last device
    pimpl->sync_register_communicator(impl);
    return ccl::communicator_t(new ccl::communicator(impl));
}

template <class DeviceType,
          typename std::enable_if<not std::is_class<typename std::remove_cv<DeviceType>::type>::value,
                                      int>::type>
CCL_API ccl::communicator_t ccl::comm_group::create_communicator(DeviceType device_id,
                                                                    ccl::device_comm_attr_t attr/* = nullptr*/)
{
    LOG_TRACE("Create communicator from id: ", device_id);

    ccl::communicator_interface_ptr impl = ccl::communicator_interface::create_communicator_impl(device_id,
                                                                                                 pimpl->thread_id,
                                                                                                 pimpl->ccl_communicator->rank(),
                                                                                                 attr);
    // registering device in group - is non blocking operation, until it is not the last device
    pimpl->sync_register_communicator(impl);
    return ccl::communicator_t(new ccl::communicator(impl));
}

/**
 *  Multiple device communicators creation vectorized API implementation
 */
template<class InputIt>
CCL_API std::vector<ccl::communicator_t> ccl::comm_group::create_communicators(InputIt first, InputIt last,
                                                                                   ccl::device_comm_attr_t attr/* = nullptr*/)
{

    using iterator_value_type = typename std::iterator_traits<InputIt>::value_type;
/*
    using expected_value_type = typename unified_device_type::device_t;
    static_assert(std::is_same<iterator_value_type, expected_value_type>::value,
                  "Not valid InputIt in create_communicators");
*/
    size_t indices_count = std::distance(first, last);
    LOG_TRACE("Create device communicators from index iterators type, count: ", indices_count);

    std::vector<ccl::communicator_t> comms;
    comms.reserve(indices_count);
    std::transform(first, last, std::back_inserter(comms), [this, attr](const iterator_value_type& device_id)
    {
        return create_communicator(device_id, attr);
    });
    return comms;
}

template<template<class...> class Container, class Type>
CCL_API std::vector<ccl::communicator_t> ccl::comm_group::create_communicators(const Container<Type>& device_ids,
                                                                                   ccl::device_comm_attr_t attr/* = nullptr*/)
{
    //static_assert(std::is_same<Type, ccl::device_index_type>::value, "Invalid Type in create_communicators");
    LOG_TRACE("Create device communicators from index type, count: ", device_ids.size(),
              ". Redirect to iterators version");
    return create_communicators(device_ids.begin(), device_ids.end(), attr);
}

CCL_API ccl::comm_group::device_context_native_const_reference_t ccl::comm_group::get_context() const
{
    //TODO use PIMPL as context provider
    static unified_device_context_type context;
    return context.get();
}


/***********************************************************************/
#define DEVICE_ATTRIBUTE_INSTANTIATION(ATTR_ID, VALUE_TYPE)                                        \
template                                                                                           \
VALUE_TYPE CCL_API ccl::ccl_device_attr::set_value<ATTR_ID, VALUE_TYPE>(VALUE_TYPE&& v);           \
template                                                                                           \
CCL_API const VALUE_TYPE& ccl::ccl_device_attr::get_value<ATTR_ID>() const;


#define STREAM_CREATOR_INSTANTIATION(type)                                                         \
template ccl::stream_t CCL_API ccl::environment::create_stream(type& stream);

#define COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(type)                                         \
template std::vector<ccl::communicator_t>                                                          \
CCL_API ccl::comm_group::create_communicators(const type& device_ids,                              \
                                              ccl::device_comm_attr_t attr);

// device attribute instantiations
DEVICE_ATTRIBUTE_INSTANTIATION(ccl_device_preferred_topology_class,
                               typename ccl::ccl_device_attributes_traits<ccl_device_preferred_topology_class>::type);
DEVICE_ATTRIBUTE_INSTANTIATION(ccl_device_preferred_group,
                               typename ccl::ccl_device_attributes_traits<ccl_device_preferred_group>::type);


// stream instantiations
STREAM_CREATOR_INSTANTIATION(ze_command_queue_handle_t)
#ifdef CCL_ENABLE_SYCL
    STREAM_CREATOR_INSTANTIATION(cl::sycl::queue)
#endif

// container-based method force-instantiation will trigger ALL other methods instantiations
COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(std::vector<ccl::device_index_type>);
COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(std::list<ccl::device_index_type>);
COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(ccl::device_indices_t);
#ifdef CCL_ENABLE_SYCL
    COMM_CREATOR_INDEXED_INSTANTIATION_CONTAINER(cl::sycl::vector_class<cl::sycl::device>);
#endif
