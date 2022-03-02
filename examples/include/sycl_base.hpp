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
#include <CL/sycl.hpp>
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

using namespace std;
using namespace sycl;
using namespace sycl::access;

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
                                                   bool select_root_devices = false) {
    std::vector<sycl::device> devices;

    try {
        if (device_type.compare("gpu") == 0) {
            if (!has_gpu()) {
                throw std::runtime_error("GPU is requested but not available.");
            }

            /* GPU type has special handling to cover multi-tile case */
            devices = create_sycl_gpu_devices(select_root_devices);
        }
        else {
            unique_ptr<device_selector> selector;

            if (device_type.compare("cpu") == 0) {
                selector.reset(new cpu_selector());
            }
            else if (device_type.compare("host") == 0) {
                selector.reset(new host_selector());
            }
            else if (device_type.compare("default") == 0) {
                if (!has_accelerator()) {
                    selector.reset(new default_selector());
                }
                else {
                    selector.reset(new host_selector());
                    cout
                        << "Accelerator is the first in device list, but unavailable for multiprocessing "
                        << " host_selector has been created instead of default_selector.\n";
                }
            }
            else {
                throw std::runtime_error("Please provide device type: cpu | gpu | host | default");
            }
            devices.push_back(sycl::device(*selector));
        }
    }
    catch (...) {
        throw std::runtime_error("No devices of requested type available");
    }

    if (devices.empty()) {
        throw std::runtime_error("No devices of requested type available");
    }

    std::vector<sycl::device> rank_devices;

    for (size_t idx = 0; idx < ranks.size(); idx++) {
        rank_devices.push_back(devices[ranks[idx] % devices.size()]);
    }

    if (rank_devices.empty()) {
        throw std::runtime_error("No devices of requested type available for specified ranks");
    }

    sycl::context ctx;

    try {
        ctx = sycl::context(rank_devices);
    }
    catch (sycl::exception&) {
        size_t preferred_idx = (ranks.back() / ranks.size()) % devices.size();
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
        queues.push_back(sycl::queue(ctx_devices[idx], exception_handler));
    }

    return queues;
}

inline bool create_sycl_queue(int argc, char* argv[], int rank, queue& q) {
    if (argc >= 2) {
        try {
            std::vector<int> ranks = { rank };
            q = create_sycl_queues(argv[1], ranks)[0];
            return true;
        }
        catch (std::exception& e) {
            cerr << e.what() << "\n";
            return false;
        }
    }
    else {
        cerr << "Please provide device type: cpu | gpu | host | default\n";
        return false;
    }
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
