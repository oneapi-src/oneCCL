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
#include <algorithm>
#include <iterator>
#include <sstream>

#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "oneapi/ccl/native_device_api/l0/platform.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "common/log/log.hpp"

#ifndef gettid
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)
#endif

namespace native {
struct once_init {
    once_init() {
        platform_impl = ccl_device_platform::create();
    }
    std::shared_ptr<ccl_device_platform> platform_impl;
};

CCL_BE_API ccl_device_platform::driver_ptr get_driver(size_t index) {
    return get_platform().get_driver(index);
}

CCL_BE_API ccl_device_platform& get_platform() {
    static once_init init;
    return *init.platform_impl;
}

CCL_BE_API ccl_device_platform& ccl_device_platform::get_platform() {
    return native::get_platform();
}

CCL_BE_API std::shared_ptr<ccl_device_platform> ccl_device_platform::create(
    const ccl::device_indices_type& indices /* = device_indices_per_driver()*/) {
    std::shared_ptr<ccl_device_platform> platform(
        new ccl_device_platform(indices.size() /*platform ID by devices count*/));
    platform->init_drivers(indices);
    return platform;
}

CCL_BE_API ccl_device_platform::ccl_device_platform(platform_id_type platform_id) {
    // initialize Level-Zero driver
    ze_result_t ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (ret != ZE_RESULT_SUCCESS) {
        CCL_THROW("cannot initialize L0: " + native::to_string(ret) +
                  ", hint: add user into `video` group");
    }
    context = std::make_shared<ccl_context_holder>();

    id = platform_id;
    pid = getpid();
}

CCL_BE_API void ccl_device_platform::init_drivers(
    const ccl::device_indices_type& driver_device_affinities /* = device_indices_per_driver()*/) {
    try {
        auto collected_drivers_list = ccl_device_driver::get_handles(driver_device_affinities);
        for (const auto& val : collected_drivers_list) {
            if (driver_device_affinities.empty()) {
                drivers.emplace(val.first,
                                ccl_device_driver::create(val.second, val.first, get_ptr()));
            }
            else {
                //collect device_index only for drvier specific index
                ccl::device_indices_type per_driver_index;
                for (const auto& affitinity : driver_device_affinities) {
                    if (std::get<ccl::device_index_enum::driver_index_id>(affitinity) ==
                        val.first) {
                        per_driver_index.insert(affitinity);
                    }
                }

                drivers.emplace(
                    val.first,
                    ccl_device_driver::create(val.second, val.first, get_ptr(), per_driver_index));
            }
        }
    }
    catch (const std::exception& ex) {
        std::stringstream ss;
        ss << "Cannot create drivers by indices: ";
        for (const auto& index : driver_device_affinities) {
            ss << index << ", ";
        }

        ss << "\nError: " << ex.what();
        throw;
    }
}

CCL_BE_API
ccl_device_platform::context_storage_type ccl_device_platform::get_platform_contexts() {
    return context;
}

CCL_BE_API std::shared_ptr<ccl_context> ccl_device_platform::create_context(
    std::shared_ptr<ccl_device_driver> driver) {
    return driver->create_context();
}

void CCL_BE_API ccl_device_platform::on_delete(ze_driver_handle_t& sub_device_handle,
                                               ze_context_handle_t& context) {
    // status = zeContextDestroy(context);
    // assert(status == ZE_RESULT_SUCCESS);
}

void CCL_BE_API ccl_device_platform::on_delete(ze_context_handle_t& handle,
                                               ze_context_handle_t& context) {}

CCL_BE_API ccl_device_platform::const_driver_ptr ccl_device_platform::get_driver(
    ccl::index_type index) const {
    auto it = drivers.find(index);
    if (it == drivers.end()) {
        CCL_THROW("no driver by index: " + std::to_string(index) + " on platform");
    }
    return it->second;
}

CCL_BE_API ccl_device_platform::driver_ptr ccl_device_platform::get_driver(ccl::index_type index) {
    auto it = drivers.find(index);
    if (it == drivers.end()) {
        CCL_THROW("no driver by index: " + std::to_string(index) + " on platform");
    }
    return it->second;
}

CCL_BE_API const ccl_device_platform::driver_storage_type& ccl_device_platform::get_drivers()
    const noexcept {
    return drivers;
}

CCL_BE_API ccl_device_driver::device_ptr ccl_device_platform::get_device(
    const ccl::device_index_type& path) {
    return std::const_pointer_cast<ccl_device>(
        static_cast<const ccl_device_platform*>(this)->get_device(path));
}

CCL_BE_API ccl_device_driver::const_device_ptr ccl_device_platform::get_device(
    const ccl::device_index_type& path) const {
    ccl::index_type driver_idx = std::get<ccl::device_index_enum::driver_index_id>(path);
    auto it = drivers.find(driver_idx);
    if (it == drivers.end()) {
        CCL_THROW("incorrect driver requested: " + ccl::to_string(path) +
                  ". Total driver count: " + std::to_string(drivers.size()));
    }

    return it->second->get_device(path);
}

std::string CCL_BE_API ccl_device_platform::to_string() const {
    std::stringstream out;
    out << "Platform:\n{\n";
    std::string driver_prefix = "\t";
    out << driver_prefix << "PlatformID: " << id << "\n";
    out << driver_prefix << "PID: " << pid << "\n\n";
    for (const auto& driver_pair : drivers) {
        out << driver_pair.second->to_string(driver_prefix) << std::endl;
    }
    out << "\n}";
    return out.str();
}

detail::adjacency_matrix ccl_device_platform::calculate_device_access_metric(
    const ccl::device_indices_type& indices,
    detail::p2p_rating_function func) const {
    detail::adjacency_matrix result;

    try {
        // diagonal matrix, assume symmetric cross device access
        for (typename ccl::device_indices_type::const_iterator lhs_it = indices.begin();
             lhs_it != indices.end();
             ++lhs_it) {
            for (typename ccl::device_indices_type::const_iterator rhs_it = lhs_it;
                 rhs_it != indices.end();
                 ++rhs_it) {
                ccl_device_driver::const_device_ptr lhs_dev = get_device(*lhs_it);
                ccl_device_driver::const_device_ptr rhs_dev = get_device(*rhs_it);

                detail::cross_device_rating rating = func(*lhs_dev, *rhs_dev);
                result[*lhs_it][*rhs_it] = rating;
                result[*rhs_it][*lhs_it] = rating;
            }
        }
    }
    catch (const std::exception& ex) {
        CCL_THROW(std::string("cannot calculate_device_access_metric, error: ") + ex.what() +
                  "\nCurrent platform info:\n" + to_string());
    }
    return result;
}

CCL_BE_API std::shared_ptr<ccl_device_platform> ccl_device_platform::clone(
    platform_id_type id,
    pid_t foreign_pid) const {
    //TODO make swallow copy
    std::shared_ptr<ccl_device_platform> ret(new ccl_device_platform(id));
    ret->pid = foreign_pid;
    return ret;
}

CCL_BE_API std::weak_ptr<ccl_device_platform> ccl_device_platform::deserialize(
    const uint8_t** data,
    size_t& size,
    std::shared_ptr<ccl_device_platform>& out_platform) {
    constexpr size_t expected_bytes = sizeof(pid_t) + sizeof(pid_t) + sizeof(platform_id_type);
    if (!*data or size < expected_bytes) {
        CCL_THROW("cannot deserialize ccl_device_platform. Not enough data, required: " +
                  std::to_string(expected_bytes) + ", got: " + std::to_string(size));
    }

    pid_t sender_pid = *(reinterpret_cast<const pid_t*>(*data));
    pid_t sender_tid = *(reinterpret_cast<const pid_t*>(*data + sizeof(sender_pid)));
    platform_id_type sender_id = *(
        reinterpret_cast<const platform_id_type*>(*data + sizeof(sender_pid) + sizeof(sender_tid)));

    //make clone of Global Platform for IPC communications
    auto& global_platform = get_platform();
    out_platform = global_platform.clone(sender_id, sender_pid /*, sender_tid*/);

    size -= expected_bytes;
    *data += expected_bytes;

    return global_platform.get_ptr();
}

CCL_BE_API size_t ccl_device_platform::serialize(std::vector<uint8_t>& out,
                                                 size_t from_pos,
                                                 size_t expected_size) const {
    pid_t src_process_tid = gettid();
    constexpr size_t expected_platform_bytes =
        sizeof(pid) + sizeof(src_process_tid) + sizeof(platform_id_type);

    //prepare continuous vector
    out.resize(from_pos + expected_size + expected_platform_bytes);

    //append to the offset
    uint8_t* data_start = out.data() + from_pos;
    *(reinterpret_cast<pid_t*>(data_start)) = pid;
    *(reinterpret_cast<pid_t*>(data_start + sizeof(pid))) = src_process_tid;
    *(reinterpret_cast<platform_id_type*>(data_start + sizeof(pid) + sizeof(src_process_tid))) = id;

    return expected_platform_bytes;
}

CCL_BE_API ccl_device_platform::platform_id_type ccl_device_platform::get_id() const noexcept {
    return id;
}

CCL_BE_API pid_t ccl_device_platform::get_pid() const noexcept {
    return pid;
}

} // namespace native
