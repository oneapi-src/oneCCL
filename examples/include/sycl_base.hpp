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
#ifndef SYCL_BASE_HPP
#define SYCL_BASE_HPP

#include <CL/sycl.hpp>
#include <iostream>
#include <string>
#include <iostream>
#include <map>
#include <mpi.h>
#include <set>
#include <string>

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
        cout << "Incompatible device type and USM type\n";
    }

    return ret;
}

inline bool create_sycl_queue(int argc,
                              char* argv[],
                              queue& q) {

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

    unique_ptr<device_selector> selector;
    if (argc >= 2) {
        if (strcmp(argv[1], "cpu") == 0) {
            selector.reset(new cpu_selector());
        }
        else if (strcmp(argv[1], "gpu") == 0) {
            if (has_gpu()) {
                selector.reset(new gpu_selector());
            }
            else if (has_accelerator()) {
                selector.reset(new host_selector());
                cout
                    << "Accelerator is the first in device list, but unavailable for multiprocessing, host_selector has been created instead of default_selector.\n";
            }
            else {
                selector.reset(new default_selector());
                cout
                    << "GPU is unavailable, default_selector has been created instead of gpu_selector.\n";
            }
        }
        else if (strcmp(argv[1], "host") == 0) {
            selector.reset(new host_selector());
        }
        else if (strcmp(argv[1], "default") == 0) {
            if (!has_accelerator()) {
                selector.reset(new default_selector());
            }
            else {
                selector.reset(new host_selector());
                cout
                    << "Accelerator is the first in device list, but unavailable for multiprocessing, host_selector has been created instead of default_selector.\n";
            }
        }
        else {
            cerr << "Please provide device type: cpu | gpu | host | default\n";
            return false;
        }
        q = queue(*selector, exception_handler);
        cout << "Requested device type: " << argv[1] << "\nRunning on "
                  << q.get_device().get_info<info::device::name>() << "\n";
    }
    else {
        cerr << "Please provide device type: cpu | gpu | host | default\n";
        return false;
    }
    return true;
}

bool handle_exception(queue& q) {
    try {
        q.wait_and_throw();
    }
    catch (std::exception const& e) {
        cout << "Caught synchronous SYCL exception:\n" << e.what() << "\n";
        return false;
    }
    return true;
}

usm::alloc usm_alloc_type_from_string(const string& str) {
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

template <typename  T>
struct buf_allocator {

    const size_t alignment = 64;

    buf_allocator(queue& q)
        : q(q)
    {}

    ~buf_allocator() {
        for (auto& ptr : memory_storage) {
            cl::sycl::free(ptr, q);
        }
    }

    T* allocate(size_t count, usm::alloc alloc_type) {
        T* ptr = nullptr;
        if (alloc_type == usm::alloc::host)
            ptr = aligned_alloc_host<T>(alignment, count, q);
        else if (alloc_type == usm::alloc::device)
            ptr =  aligned_alloc_device<T>(alignment, count, q);
        else if (alloc_type == usm::alloc::shared)
            ptr = aligned_alloc_shared<T>(alignment, count, q);
        else
            throw std::runtime_error(string(__PRETTY_FUNCTION__) + "unexpected alloc_type");

        auto it = memory_storage.find(ptr);
        if (it != memory_storage.end()) {
            throw std::runtime_error(string(__PRETTY_FUNCTION__) +
                                        " - allocator already owns this pointer");
        }
        memory_storage.insert(ptr);

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

template <class data_native_type, usm::alloc... types>
struct usm_polymorphic_allocator {
    using native_type = data_native_type;
    using allocator_types = tuple<usm_allocator<native_type, types>...>;
    using integer_usm_type = typename underlying_type<usm::alloc>::type;
    using self_t = usm_polymorphic_allocator<data_native_type, types...>;

    usm_polymorphic_allocator(queue& q)
            : allocators{ make_tuple(usm_allocator<native_type, types>(q)...) } {}

    ~usm_polymorphic_allocator() {
        for (auto& v : memory_storage) {
            data_native_type* mem = v.first;
            deallocate(mem, v.second.size, v.second.type);
        }
    }

private:
    struct alloc_info {
        size_t size;
        usm::alloc type;
    };
    map<data_native_type*, alloc_info> memory_storage;

    struct alloc_impl {
        alloc_impl(native_type** out_ptr, size_t count, usm::alloc type, self_t* parent)
                : out_usm_memory_pointer(out_ptr),
                  size(count),
                  alloc_index(0),
                  requested_alloc_type(type),
                  owner(parent) {}

        template <class specific_allocator>
        void operator()(specific_allocator& al) {
            if (alloc_index++ == static_cast<integer_usm_type>(requested_alloc_type)) {
                *out_usm_memory_pointer = al.allocate(size);

                alloc_info info{ size, requested_alloc_type };
                owner->memory_storage.emplace(*out_usm_memory_pointer, info);
            }
        }
        native_type** out_usm_memory_pointer;
        size_t size{};
        int alloc_index{};
        usm::alloc requested_alloc_type;
        self_t* owner;
    };

    struct dealloc_impl {
        dealloc_impl(native_type** in_ptr, size_t count, usm::alloc type, self_t* parent)
                : in_usm_memory_pointer(in_ptr),
                  size(count),
                  alloc_index(0),
                  requested_alloc_type(type),
                  owner(parent) {}

        template <class specific_allocator>
        void operator()(specific_allocator& al) {
            if (alloc_index++ == static_cast<integer_usm_type>(requested_alloc_type)) {
                auto it = owner->memory_storage.find(*in_usm_memory_pointer);
                if (it == owner->memory_storage.end()) {
                    throw std::runtime_error(string(__PRETTY_FUNCTION__) +
                                             " - not owns memory object");
                }

                al.deallocate(*in_usm_memory_pointer, size);
                *in_usm_memory_pointer = nullptr;

                owner->memory_storage.erase(it);
            }
        }
        native_type** in_usm_memory_pointer;
        size_t size;
        int alloc_index;
        usm::alloc requested_alloc_type;
        self_t* owner;
    };

public:
    allocator_types allocators;

    native_type* allocate(size_t size, usm::alloc type) {
        native_type* ret = nullptr;
        ccl_tuple_for_each(allocators, alloc_impl{ &ret, size, type, this });
        return ret;
    }

    void deallocate(native_type* in_ptr, size_t size, usm::alloc type) {
        ccl_tuple_for_each(allocators, dealloc_impl{ &in_ptr, size, type, this });
    }
};

#endif /* SYCL_BASE_HPP */
