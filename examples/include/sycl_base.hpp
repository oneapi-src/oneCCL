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
#pragma once

#include <algorithm>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#if __has_include(<CL/sycl/property_list.hpp>)
#include <CL/sycl/property_list.hpp>
#elif __has_include(<sycl/property_list.hpp>)
#include <sycl/property_list.hpp>
#else
#error "Unsupported compiler"
#endif
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <set>
#include <string>
#include <numeric>

#include "base.hpp"
#include "base_utils.hpp"
#include "oneapi/ccl.hpp"

#if defined(__INTEL_LLVM_COMPILER)
#if (__INTEL_LLVM_COMPILER < 20230000)
#define CCL_USE_SYCL121_API 1
#else // (__INTEL_LLVM_COMPILER < 20230000)
#define CCL_USE_SYCL121_API 0
#endif // (__INTEL_LLVM_COMPILER < 20230000)
#elif defined(__LIBSYCL_MAJOR_VERSION)
#if (__LIBSYCL_MAJOR_VERSION < 6)
#define CCL_USE_SYCL121_API 1
#else // (__LIBSYCL_MAJOR_VERSION < 6)
#define CCL_USE_SYCL121_API 0
#endif // (__LIBSYCL_MAJOR_VERSION < 6)
#else // __INTEL_LLVM_COMPILER || __LIBSYCL_MAJOR_VERSION
#error "Unsupported compiler"
#endif

using namespace std;
using namespace sycl;
using namespace sycl::access;

namespace ccl {
#if CCL_USE_SYCL121_API
const auto cpu_selector_v = ::sycl::cpu_selector{};
const auto gpu_selector_v = ::sycl::gpu_selector{};
const auto default_selector_v = ::sycl::default_selector{};
#else // CCL_USE_SYCL121_API
inline const auto& cpu_selector_v = ::sycl::cpu_selector_v;
inline const auto& gpu_selector_v = ::sycl::gpu_selector_v;
inline const auto& default_selector_v = ::sycl::default_selector_v;
#endif // CCL_USE_SYCL121_API
} // namespace ccl

/* help functions for sycl-specific base implementation */
inline bool has_gpu() {
    vector<device> devices = device::get_devices();
    for (const auto& device : devices) {
        if (device.is_gpu()) {
            return true;
        }
    }
    return false;
}

inline bool has_accelerator() {
    vector<device> devices = device::get_devices();
    for (const auto& device : devices) {
        if (device.is_accelerator()) {
            return true;
        }
    }
    return false;
}

inline bool check_sycl_usm(queue& q, usm::alloc alloc_type) {
    bool ret = true;

    device d = q.get_device();

    if ((alloc_type == usm::alloc::host) && (d.is_gpu() || d.is_accelerator()))
        ret = false;

    if ((alloc_type == usm::alloc::device) && !(d.is_gpu() || d.is_accelerator()))
        ret = false;

    if (!ret) {
        cout << "incompatible device type and USM type\n";
    }

    return ret;
}

inline std::string get_preferred_gpu_platform_name() {
    std::string result;

    std::string filter = "level-zero";
    char* env = getenv("SYCL_DEVICE_FILTER");
    if (env) {
        if (std::strstr(env, "level_zero")) {
            filter = "level-zero";
        }
        else if (std::strstr(env, "opencl")) {
            filter = "opencl";
        }
        else {
            throw std::runtime_error("invalid device filter: " + std::string(env));
        }
    }

    auto plaform_list = sycl::platform::get_platforms();

    for (const auto& platform : plaform_list) {
        auto devices = platform.get_devices();
        auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](const sycl::device& d) {
            return d.is_gpu();
        });

        if (gpu_dev == devices.end()) {
            // cout << "platform [" << platform_name
            //      << "] does not contain GPU devices, skipping\n";
            continue;
        }

        auto platform_name = platform.get_info<sycl::info::platform::name>();
        std::string platform_name_low_case;
        platform_name_low_case.resize(platform_name.size());

        std::transform(
            platform_name.begin(), platform_name.end(), platform_name_low_case.begin(), ::tolower);

        if (platform_name_low_case.find(filter) == std::string::npos) {
            // cout << "platform [" << platform_name
            //      << "] does not match with requested "
            //      << filter << ", skipping\n";
            continue;
        }

        result = platform_name;
    }

    if (result.empty())
        throw std::runtime_error("can not find preferred GPU platform");

    return result;
}

inline std::vector<sycl::device> create_sycl_gpu_devices(bool select_root_devices) {
    constexpr char prefix[] = "-- ";

    std::vector<sycl::device> result;
    auto plaform_list = sycl::platform::get_platforms();
    auto preferred_platform_name = get_preferred_gpu_platform_name();

    std::stringstream ss;
    std::stringstream ss_warn;

    for (const auto& platform : plaform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        if (platform_name.compare(preferred_platform_name) != 0) {
            continue;
        }

        auto device_list = platform.get_devices();
        for (const auto& device : device_list) {
            auto device_name = device.get_info<cl::sycl::info::device::name>();

            if (!device.is_gpu()) {
                ss_warn << prefix << "device [" << device_name << "] is not GPU, skipping\n";
                continue;
            }

            if (select_root_devices) {
                result.push_back(device);
                continue;
            }

            auto part_props = device.get_info<info::device::partition_properties>();

            if (std::find(part_props.begin(),
                          part_props.end(),
                          info::partition_property::partition_by_affinity_domain) ==
                part_props.end()) {
                ss_warn << prefix << "device [" << device_name
                        << "] does not support partition by affinity domain"
                        << ", use root device\n";
                result.push_back(device);
                continue;
            }

            auto part_affinity_domains =
                device.get_info<info::device::partition_affinity_domains>();

            if (std::find(part_affinity_domains.begin(),
                          part_affinity_domains.end(),
                          info::partition_affinity_domain::next_partitionable) ==
                part_affinity_domains.end()) {
                ss_warn << prefix << "device [" << device_name
                        << "] does not support next_partitionable affinity domain"
                        << ", use root device\n";
                result.push_back(device);
                continue;
            }

            auto sub_devices =
                device.create_sub_devices<info::partition_property::partition_by_affinity_domain>(
                    info::partition_affinity_domain::next_partitionable);

            size_t sub_devices_max =
                device.template get_info<info::device::partition_max_sub_devices>();
            if (sub_devices.size() != sub_devices_max) {
                ss_warn << prefix << "device [" << device_name << "] expected " << sub_devices_max
                        << " sub-devices, but got " << sub_devices.size();
            }

            if (sub_devices.empty()) {
                ss_warn << prefix << "device [" << device_name << "] does not provide sub-devices"
                        << ", use root device\n";
                result.push_back(device);
                continue;
            }

            result.insert(result.end(), sub_devices.begin(), sub_devices.end());
        }
    }

    if (result.empty()) {
        throw std::runtime_error("no GPU devices found");
    }

    ss << "preferred platform: " << preferred_platform_name << ", found: " << result.size()
       << " GPU device(s)\n";
    ss << ss_warn.str();
    printf("%s", ss.str().c_str());

    return result;
}

inline std::vector<sycl::queue> create_sycl_queues(const std::string& device_type,
                                                   const std::vector<int>& ranks,
                                                   bool select_root_devices = false,
                                                   const sycl::property_list& queue_props = {}) {
    std::vector<sycl::device> devices;

    try {
        if (device_type.compare("gpu") == 0) {
            if (!has_gpu()) {
                throw std::runtime_error("GPU is requested but not available");
            }

            /* GPU type has special handling to cover multi-tile case */
            devices = create_sycl_gpu_devices(select_root_devices);
        }
        else {
            if (device_type.compare("cpu") == 0) {
                devices.push_back(device(ccl::cpu_selector_v));
            }
            else if (device_type.compare("default") == 0) {
                if (!has_accelerator()) {
                    devices.push_back(device(ccl::default_selector_v));
                }
                else {
                    devices.push_back(device(ccl::cpu_selector_v));
                    cout
                        << "Accelerator is the first in device list, but unavailable for multiprocessing "
                        << " cpu_selector has been created instead of default_selector.\n";
                }
            }
            else {
                throw std::runtime_error("Please provide device type: cpu | gpu | default");
            }
        }
    }
    catch (...) {
        throw std::runtime_error("No devices of requested type available");
    }

    if (devices.empty()) {
        throw std::runtime_error("No devices of requested type available");
    }

    int global_rank = 0, local_rank = 0;
    int global_size = 0, local_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    std::stringstream error_msg;

    if (local_rank > global_rank) {
        error_msg << "Local rank should be less or equal to global rank (local_rank: " << local_rank
                  << ", global_rank: " << global_rank << ")";
        throw std::runtime_error(error_msg.str());
    }

    if (local_size > global_size) {
        error_msg << "Local size should be less or equal to global size (local_size: " << local_size
                  << ", global_size: " << global_size << ")";
        throw std::runtime_error(error_msg.str());
    }

    if (ranks.size() != 1) {
        error_msg << "Unexpected number of device ranks: " << ranks.size();
        throw std::runtime_error(error_msg.str());
    }

    if (ranks[0] != global_rank) {
        error_msg << "Unexpected device rank: " << ranks[0] << ", expected: " << global_rank;
        throw std::runtime_error(error_msg.str());
    }

    // use local rank for device selection
    std::vector<int> local_ranks(1, local_rank);

    std::vector<sycl::device> rank_devices;
    for (size_t idx = 0; idx < local_ranks.size(); idx++) {
        rank_devices.push_back(devices[local_ranks[idx] % devices.size()]);
    }

    if (rank_devices.empty()) {
        throw std::runtime_error("No devices of requested type available for specified ranks");
    }

    sycl::context ctx;

    try {
        ctx = sycl::context(rank_devices);
    }
    catch (sycl::exception&) {
        size_t preferred_idx = (local_ranks.back() / local_ranks.size()) % devices.size();
        cout << "Can not create context from all rank devices of type: " << device_type
             << ", create context from single device, idx " << preferred_idx << "\n";
        ctx = sycl::context(devices[preferred_idx]);
    }

    auto exception_handler = [&](exception_list elist) {
        for (exception_ptr const& e : elist) {
            try {
                rethrow_exception(e);
            }
            catch (std::exception const& e) {
                cout << "failure\n";
            }
        }
    };

    auto ctx_devices = ctx.get_devices();

    if (ctx_devices.empty()) {
        throw std::runtime_error("No devices of requested type available in context");
    }

    std::vector<sycl::queue> queues;

    cout << "Created context from devices of type: " << device_type << "\n";
    cout << "Devices [" << ctx_devices.size() << "]:\n";

    for (size_t idx = 0; idx < ctx_devices.size(); idx++) {
        cout << "[" << idx << "]: [" << ctx_devices[idx].get_info<info::device::name>() << "]\n";
        queues.push_back(sycl::queue(ctx_devices[idx], exception_handler, queue_props));
    }

    return queues;
}

inline bool create_sycl_queue(const std::string& type,
                              int rank,
                              queue& q,
                              const property_list& queue_props = {}) {
    if (type == "gpu" || type == "cpu" || type == "host" || type == "default") {
        try {
            std::vector<int> ranks = { rank };
            q = create_sycl_queues(type, ranks, false, queue_props)[0];
            return true;
        }
        catch (std::exception& e) {
            cerr << e.what() << "\n";
            return false;
        }
    }
    else {
        cerr << "Unknown device type: " << type << ", please provide: cpu | gpu | host | default\n";
        return false;
    }
}

inline bool create_sycl_queue(int argc, char* argv[], int rank, queue& q) {
    return create_sycl_queue(((argc >= 2) ? argv[1] : "unknown"), rank, q, {});
}

inline bool handle_exception(queue& q) {
    try {
        q.wait_and_throw();
    }
    catch (std::exception const& e) {
        cout << "Caught synchronous SYCL exception:\n" << e.what() << "\n";
        return false;
    }
    return true;
}

inline usm::alloc usm_alloc_type_from_string(const string& str) {
    const map<string, usm::alloc> names{ {
        { "host", usm::alloc::host },
        { "device", usm::alloc::device },
        { "shared", usm::alloc::shared },
    } };

    auto it = names.find(str);
    if (it == names.end()) {
        stringstream ss;
        ss << "Invalid USM type requested: " << str << "\nSupported types are:\n";
        for (const auto& v : names) {
            ss << v.first << ", ";
        }
        throw std::runtime_error(ss.str());
    }
    return it->second;
}

inline std::pair<usm::alloc, std::string> take_usm_type(const int argc, char* str_type) {
    std::map<usm::alloc, std::string> map_usm_type;
    auto usm_alloc_type = usm::alloc::shared;
    auto str_usm_alloc_type = "shared";
    if (argc > 1) {
        str_usm_alloc_type = str_type;
        usm_alloc_type = usm_alloc_type_from_string(str_usm_alloc_type);
    }

    return std::make_pair(usm_alloc_type, str_usm_alloc_type);
}

template <typename T>
struct buf_allocator {
    const size_t alignment = 64;

    buf_allocator(queue& q) : q(q) {}

    buf_allocator(const buf_allocator&) = delete;
    buf_allocator(buf_allocator&&) = default;

    ~buf_allocator() {
        for (auto& ptr : memory_storage) {
            sycl::free(ptr, q);
        }
    }

    T* allocate(size_t count, usm::alloc alloc_type) {
        T* ptr = nullptr;
        if (alloc_type == usm::alloc::host)
            ptr = aligned_alloc_host<T>(alignment, count, q);
        else if (alloc_type == usm::alloc::device)
            ptr = aligned_alloc_device<T>(alignment, count, q);
        else if (alloc_type == usm::alloc::shared)
            ptr = aligned_alloc_shared<T>(alignment, count, q);
        else
            throw std::runtime_error(string(__PRETTY_FUNCTION__) + " - unexpected alloc_type");

        if (!ptr) {
            throw std::runtime_error(string(__PRETTY_FUNCTION__) + " - failed to allocate buffer");
        }

        auto it = memory_storage.find(ptr);
        if (it != memory_storage.end()) {
            throw std::runtime_error(string(__PRETTY_FUNCTION__) +
                                     " - allocator already owns this pointer");
        }
        memory_storage.insert(ptr);

        auto pointer_type = sycl::get_pointer_type(ptr, q.get_context());
        if (pointer_type != alloc_type)
            throw std::runtime_error(string(__PRETTY_FUNCTION__) + " - pointer_type " +
                                     std::to_string((int)pointer_type) +
                                     " doesn't match with requested " +
                                     std::to_string((int)alloc_type));

        return ptr;
    }

    void deallocate(T* ptr) {
        auto it = memory_storage.find(ptr);
        if (it == memory_storage.end()) {
            throw std::runtime_error(string(__PRETTY_FUNCTION__) +
                                     " - allocator doesn't own this pointer");
        }
        free(ptr, q);
        memory_storage.erase(it);
    }

    queue q;
    set<T*> memory_storage;
};
