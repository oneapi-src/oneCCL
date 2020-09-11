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
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/stream/stream_provider_dispatcher_impl.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "unified_context_impl.hpp"

#ifdef MULTI_GPU_SUPPORT
#ifdef CCL_ENABLE_SYCL
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    cl::sycl::queue& native_stream,
    const ccl::library_version& version);
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    cl_command_queue& native_stream_handle,
    const ccl::library_version& version);
#else
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    std::shared_ptr<native::ccl_device::device_queue>& native_stream,
    const ccl::library_version& version);
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    ze_command_queue_handle_t& native_stream_handle,
    const ccl::library_version& version);
#endif
#else
#ifdef CCL_ENABLE_SYCL
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    cl::sycl::queue& native_stream,
    const ccl::library_version& version);
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    cl_command_queue& native_stream,
    const ccl::library_version& version);
#else
template std::unique_ptr<ccl_stream> stream_provider_dispatcher::create(
    void*& native_stream,
    const ccl::library_version& version);
#endif
#endif

void ccl_stream::build_from_params() {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("error");
    }
    try {
#ifdef CCL_ENABLE_SYCL
        if (is_context_enabled) {
            stream_native_t stream_candidate{ native_context, native_device };
            std::swap(stream_candidate,
                      native_stream); //TODO USE attributes fro sycl queue construction
        }
        else {
            stream_native_t stream_candidate{ native_device };
            std::swap(stream_candidate,
                      native_stream); //TODO USE attributes fro sycl queue construction
        }
#else
        ze_command_queue_desc_t descr =
            stream_native_device_t::element_type::get_default_queue_desc();

        //TODO use attributes
        native_device->create_cmd_queue(descr);
#endif
    }
    catch (const std::exception& ex) {
        throw ccl::ccl_error(std::string("Cannot build ccl_stream from params: ") + ex.what());
    }
    creation_is_postponed = false;
}

//Export Attributes
typename ccl_stream::version_traits_t::type ccl_stream::set_attribute_value(
    typename version_traits_t::type val,
    const version_traits_t& t) {
    (void)t;
    throw ccl::ccl_error("Set value for 'ccl::stream_attr_id::library_version' is not allowed");
    return version;
}

const typename ccl_stream::version_traits_t::return_type& ccl_stream::get_attribute_value(
    const version_traits_t& id) const {
    return version;
}

typename ccl_stream::native_handle_traits_t::return_type& ccl_stream::get_attribute_value(
    const native_handle_traits_t& id) {
    /*
    if (!native_stream_set)
    {
        throw  ccl::ccl_error("native stream is not set");
    }
*/
    return native_stream;
}

typename ccl_stream::device_traits_t::return_type& ccl_stream::get_attribute_value(
    const device_traits_t& id) {
    return native_device;
}

typename ccl_stream::context_traits_t::return_type& ccl_stream::get_attribute_value(
    const context_traits_t& id) {
    return native_context;
}

typename ccl_stream::context_traits_t::return_type& ccl_stream::set_attribute_value(
    typename context_traits_t::type val,
    const context_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::context'`for constructed stream");
    }
    std::swap(native_context, val);
    return native_context;
}

typename ccl_stream::context_traits_t::return_type& ccl_stream::set_attribute_value(
    typename context_traits_t::handle_t val,
    const context_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::context'`for constructed stream");
    }
    native_context = ccl::unified_device_context_type{ val }.get(); //context_traits_t::type
    is_context_enabled = true;
    return native_context;
}

typename ccl_stream::ordinal_traits_t::type ccl_stream::set_attribute_value(
    typename ordinal_traits_t::type val,
    const ordinal_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::ordinal'`for constructed stream");
    }
    auto old = ordinal_val;
    std::swap(ordinal_val, val);
    return old;
}

const typename ccl_stream::ordinal_traits_t::return_type& ccl_stream::get_attribute_value(
    const ordinal_traits_t& id) const {
    return ordinal_val;
}

typename ccl_stream::index_traits_t::type ccl_stream::set_attribute_value(
    typename index_traits_t::type val,
    const index_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::index'`for constructed stream");
    }
    auto old = index_val;
    std::swap(index_val, val);
    return old;
}

const typename ccl_stream::index_traits_t::return_type& ccl_stream::get_attribute_value(
    const index_traits_t& id) const {
    return index_val;
}

typename ccl_stream::flags_traits_t::type ccl_stream::set_attribute_value(
    typename flags_traits_t::type val,
    const flags_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::flags'`for constructed stream");
    }
    auto old = flags_val;
    std::swap(flags_val, val);
    return old;
}

const typename ccl_stream::flags_traits_t::return_type& ccl_stream::get_attribute_value(
    const flags_traits_t& id) const {
    return flags_val;
}

typename ccl_stream::mode_traits_t::type ccl_stream::set_attribute_value(
    typename mode_traits_t::type val,
    const mode_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::mode'`for constructed stream");
    }
    auto old = mode_val;
    std::swap(mode_val, val);
    return old;
}

const typename ccl_stream::mode_traits_t::return_type& ccl_stream::get_attribute_value(
    const mode_traits_t& id) const {
    return mode_val;
}

typename ccl_stream::priority_traits_t::type ccl_stream::set_attribute_value(
    typename priority_traits_t::type val,
    const priority_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("Cannot set 'ccl::stream_attr_id::priority'`for constructed stream");
    }
    auto old = priority_val;
    std::swap(priority_val, val);
    return old;
}

const typename ccl_stream::priority_traits_t::return_type& ccl_stream::get_attribute_value(
    const priority_traits_t& id) const {
    return priority_val;
}
