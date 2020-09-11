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
#include "common/event/event.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

ccl_event::ccl_event(event_native_t& event, const ccl::library_version& version)
        : version(version),
          native_event(event) {}

ccl_event::ccl_event(event_native_handle_t event,
                     event_native_context_t context,
                     const ccl::library_version& version)
        : version(version) {
#ifdef CCL_ENABLE_SYCL
    native_event = event_native_t{ event, context };
#else
    //TODO
    throw;
#endif
}

void ccl_event::build_from_params() {
    if (!creation_is_postponed) {
        throw ccl::ccl_error("error");
    }
#ifdef CCL_ENABLE_SYCL
    /* TODO unavailbale??
    event_native_t event_candidate{native_context};
    std::swap(event_candidate, native_event); //TODO USE attributes fro sycl queue construction
    */

    throw ccl::ccl_error("build_from_attr is not availbale for sycl::event");
#else

    //TODO use attributes

#endif
    creation_is_postponed = false;
}

//Export Attributes
typename ccl_event::version_traits_t::type ccl_event::set_attribute_value(
    typename version_traits_t::type val,
    const version_traits_t& t) {
    (void)t;
    throw ccl::ccl_error("Set value for 'ccl::event_attr_id::library_version' is not allowed");
    return version;
}

const typename ccl_event::version_traits_t::return_type& ccl_event::get_attribute_value(
    const version_traits_t& id) const {
    return version;
}

typename ccl_event::native_handle_traits_t::return_type& ccl_event::get_attribute_value(
    const native_handle_traits_t& id) {
    return native_event;
}

typename ccl_event::context_traits_t::return_type& ccl_event::get_attribute_value(
    const context_traits_t& id) {
    return native_context;
}

typename ccl_event::command_type_traits_t::type ccl_event::set_attribute_value(
    typename command_type_traits_t::type val,
    const command_type_traits_t& t) {
    auto old = command_type_val;
    std::swap(command_type_val, val);
    return old;
}

const typename ccl_event::command_type_traits_t::return_type& ccl_event::get_attribute_value(
    const command_type_traits_t& id) const {
    return command_type_val;
}

typename ccl_event::command_execution_status_traits_t::type ccl_event::set_attribute_value(
    typename command_execution_status_traits_t::type val,
    const command_execution_status_traits_t& t) {
    auto old = command_execution_status_val;
    std::swap(command_execution_status_val, val);
    return old;
}

const typename ccl_event::command_execution_status_traits_t::return_type&
ccl_event::get_attribute_value(const command_execution_status_traits_t& id) const {
    return command_execution_status_val;
}
