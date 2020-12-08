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
#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/stream/stream_provider_dispatcher_impl.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "unified_context_impl.hpp"

ccl_stream::ccl_stream(stream_type type,
                       stream_native_t& stream,
                       const ccl::library_version& version)
        : type(type),
          version(version) {
    native_stream = stream;

#ifdef CCL_ENABLE_SYCL
    native_streams.resize(ccl::global_data::env().worker_count);
    for (size_t idx = 0; idx < native_streams.size(); idx++) {
        native_streams[idx] = stream_native_t(stream.get_context(), stream.get_device());
    }
#endif /* CCL_ENABLE_SYCL */
}

ccl_stream::ccl_stream(stream_type type,
                       stream_native_handle_t handle,
                       const ccl::library_version& version)
        : type(type),
          version(version) {
    creation_is_postponed = true;
    (void)handle;
    throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + " - unsupported ");
}

ccl_stream::ccl_stream(stream_type type, const ccl::library_version& version)
        : type(type),
          version(version) {
    creation_is_postponed = true;
    LOG_DEBUG("Scheduled postponed stream creation");
}

void ccl_stream::build_from_params() {
    if (!creation_is_postponed) {
        throw ccl::exception(std::string(__FUNCTION__) +
                             " - incorrect usage, stream is not sheduled for postponed creation");
    }

    type = stream_type::host;
    try {
#ifdef CCL_ENABLE_SYCL
        if (native_context.first) {
            if (!native_device.first) {
                throw ccl::exception(
                    std::string(__FUNCTION__) +
                    " - incorrect usage, not enough parameters for stream creation: "
                    " context is available, but device is not. Both required");
            }

            LOG_DEBUG("create stream from device & context");
            stream_native_t stream_candidate{ native_context.second, native_device.second };
            std::swap(stream_candidate,
                      native_stream); //TODO USE attributes from sycl queue construction
        }
        else if (native_device.first) {
            LOG_DEBUG("create stream from device only");
            stream_native_t stream_candidate{ native_device.second };
            std::swap(stream_candidate,
                      native_stream); //TODO USE attributes from sycl queue construction

            native_context.second = native_stream.get_context();
            native_context.first = true;
        }
        else {
            throw ccl::exception(std::string(__FUNCTION__) +
                                 " - incorrect usage, not enough parameters for stream creation: "
                                 " context is empty and device is empty too.");
        }

        //override type
        if (native_stream.get_device().is_host()) {
            type = stream_type::host;
        }
        else if (native_stream.get_device().is_cpu()) {
            type = stream_type::cpu;
        }
        else if (native_stream.get_device().is_gpu()) {
            type = stream_type::gpu;
        }
        else {
            throw ccl::invalid_argument(
                "CORE",
                "create_stream",
                std::string("Unsupported SYCL queue's device type for postponed creation:\n") +
                    native_stream.get_device().template get_info<cl::sycl::info::device::name>() +
                    std::string("Supported types: host, cpu, gpu"));
        }
        LOG_INFO("SYCL queue type from postponed creation: ",
                 static_cast<int>(type),
                 " device: ",
                 native_stream.get_device().template get_info<cl::sycl::info::device::name>());
#else
#ifdef MULTI_GPU_SUPPORT
        ze_command_queue_desc_t descr =
            stream_native_device_t::element_type::get_default_queue_desc();

        //TODO use attributes....
        //Create from device & context
        if (native_context.first) {
            if (!native_device.first) {
                throw ccl::exception(
                    std::string(__FUNCTION__) +
                    " - incorrect usage, not enough parameters for stream creation: "
                    " context is available, but device is not. Both required");
            }

            LOG_DEBUG("create stream from device & context");
            auto stream_candidate =
                native_device.second->create_cmd_queue(native_context.second, descr);
            native_stream = std::make_shared<typename ccl::unified_stream_type::impl_t>(
                std::move(stream_candidate));
        }
        else if (native_device.first) {
            LOG_DEBUG("create stream from device only");

            auto stream_candidate = native_device.second->create_cmd_queue({}, descr);
            native_stream = std::make_shared<typename ccl::unified_stream_type::impl_t>(
                std::move(stream_candidate));

            native_context.second = native_stream->get_ctx().lock();
            native_context.first = true;
        }
        else {
            throw ccl::exception(std::string(__FUNCTION__) +
                                 " - incorrect usage, not enough parameters for stream creation: "
                                 " context is empty and device is empty too.");
        }

        type = stream_type::gpu;
#endif
#endif
    }
    catch (const std::exception& ex) {
        throw ccl::exception(std::string("Cannot build ccl_stream from params: ") + ex.what());
    }
    creation_is_postponed = false;
}

//Export Attributes
typename ccl_stream::version_traits_t::type ccl_stream::set_attribute_value(
    typename version_traits_t::type val,
    const version_traits_t& t) {
    (void)t;
    throw ccl::exception("Set value for 'ccl::stream_attr_id::library_version' is not allowed");
    return version;
}

const typename ccl_stream::version_traits_t::return_type& ccl_stream::get_attribute_value(
    const version_traits_t& id) const {
    return version;
}

typename ccl_stream::native_handle_traits_t::return_type& ccl_stream::get_attribute_value(
    const native_handle_traits_t& id) {
    if (creation_is_postponed) {
        throw ccl::exception(std::string(__FUNCTION__) + " - stream is not properly created yet");
    }

    return native_stream;
}

typename ccl_stream::device_traits_t::return_type& ccl_stream::get_attribute_value(
    const device_traits_t& id) {
    if (!native_device.first) {
        throw ccl::exception(std::string(__FUNCTION__) + " - stream has no native device");
    }
    return native_device.second;
}

typename ccl_stream::context_traits_t::return_type& ccl_stream::get_attribute_value(
    const context_traits_t& id) {
    if (!native_context.first) {
        throw ccl::exception(std::string(__FUNCTION__) + " - stream has no native context");
    }
    return native_context.second;
}

typename ccl_stream::context_traits_t::return_type& ccl_stream::set_attribute_value(
    typename context_traits_t::type val,
    const context_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::context'`for constructed stream");
    }
    std::swap(native_context.second, val);
    native_context.first = true;
    return native_context.second;
}
/*
typename ccl_stream::context_traits_t::return_type& ccl_stream::set_attribute_value(
    typename context_traits_t::handle_t val,
    const context_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::context'`for constructed stream");
    }
    native_context.second = ccl::unified_context_type{ val }.get(); //context_traits_t::type
    native_context.first = true;
    return native_context.second;
}*/

typename ccl_stream::ordinal_traits_t::type ccl_stream::set_attribute_value(
    typename ordinal_traits_t::type val,
    const ordinal_traits_t& t) {
    if (!creation_is_postponed) {
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::ordinal'`for constructed stream");
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
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::index'`for constructed stream");
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
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::flags'`for constructed stream");
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
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::mode'`for constructed stream");
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
        throw ccl::exception("Cannot set 'ccl::stream_attr_id::priority'`for constructed stream");
    }
    auto old = priority_val;
    std::swap(priority_val, val);
    return old;
}

const typename ccl_stream::priority_traits_t::return_type& ccl_stream::get_attribute_value(
    const priority_traits_t& id) const {
    return priority_val;
}
