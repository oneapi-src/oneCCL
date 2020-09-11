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
#include <stdexcept>

#include "native_device_api/l0/primitives.hpp"
#include "native_device_api/l0/base_impl.hpp"

namespace native {

#define TEMPLATE_DECL_ARG class elem_t, class resource_owner
#define TEMPLATE_DEF_ARG  elem_t, resource_owner

struct ccl_device;
namespace detail {
void copy_memory_to_device_sync_unsafe(void* dst,
                                       const void* src,
                                       size_t size,
                                       std::weak_ptr<ccl_device> device_weak);
}

template <TEMPLATE_DECL_ARG>
memory<TEMPLATE_DEF_ARG>::memory(elem_t* h, size_t count, std::weak_ptr<resource_owner>&& owner)
        : base(h, std::move(owner)),
          elem_count(count) {}

template <TEMPLATE_DECL_ARG>
elem_t* memory<TEMPLATE_DEF_ARG>::get() noexcept {
    return base::get();
}

template <TEMPLATE_DECL_ARG>
const elem_t* memory<TEMPLATE_DEF_ARG>::get() const noexcept {
    return base::get();
}

template <TEMPLATE_DECL_ARG>
size_t memory<TEMPLATE_DEF_ARG>::count() const noexcept {
    return elem_count;
}

template <TEMPLATE_DECL_ARG>
size_t memory<TEMPLATE_DEF_ARG>::size() const noexcept {
    return count() * sizeof(elem_t);
}

template <TEMPLATE_DECL_ARG>
void memory<TEMPLATE_DEF_ARG>::enqueue_write_sync(const std::vector<elem_t>& src) {
    if (count() < src.size()) {
        throw std::length_error(
            std::string(__PRETTY_FUNCTION__) +
            "\nCannot process 'enqueue_write_sync', because memory has not enough size" +
            ", expected: " + std::to_string(count()) +
            ", requested: " + std::to_string(src.size()));
    }

    try {
        detail::copy_memory_to_device_sync_unsafe(
            get(), src.data(), src.size() * sizeof(elem_t), get_owner());
    }
    catch (const std::exception& ex) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + "\n" + ex.what());
    }
}

template <TEMPLATE_DECL_ARG>
void memory<TEMPLATE_DEF_ARG>::enqueue_write_sync(
    typename std::vector<elem_t>::const_iterator first,
    typename std::vector<elem_t>::const_iterator last) {
    size_t requested_size = std::distance(first, last);
    if (count() < requested_size) {
        throw std::length_error(
            std::string(__PRETTY_FUNCTION__) +
            "\nCannot process 'enqueue_write_sync', because memory has not enough size" +
            ", expected: " + std::to_string(count()) +
            ", requested: " + std::to_string(requested_size));
    }

    try {
        detail::copy_memory_to_device_sync_unsafe(get(), &(*first), requested_size, get_owner());
    }
    catch (const std::exception& ex) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + "\n" + ex.what());
    }
}

template <TEMPLATE_DECL_ARG>
template <int N>
void memory<TEMPLATE_DEF_ARG>::enqueue_write_sync(const std::array<elem_t, N>& src) {
    if (count() < N) {
        throw std::length_error(
            std::string(__PRETTY_FUNCTION__) +
            "\nCannot process 'enqueue_write_sync', because memory has not enough size" +
            ", expected: " + std::to_string(count()) + ", requested: " + std::to_string(N));
    }

    try {
        detail::copy_memory_to_device_sync_unsafe(
            get(), src.data(), N * sizeof(elem_t), get_owner());
    }
    catch (const std::exception& ex) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + "\n" + ex.what());
    }
}
template <TEMPLATE_DECL_ARG>
void memory<TEMPLATE_DEF_ARG>::enqueue_write_sync(const elem_t* src, size_t src_elem_count) {
    if (!src) {
        throw std::invalid_argument(
            std::string(__PRETTY_FUNCTION__) +
            "\nCannot process 'enqueue_write_sync', because 'src' is 'nullptr'");
    }

    if (count() < src_elem_count * sizeof(elem_t)) {
        throw std::length_error(
            std::string(__PRETTY_FUNCTION__) +
            "\nCannot process 'enqueue_write_sync', because memory has not enough size" +
            ", expected: " + std::to_string(count()) +
            ", requested: " + std::to_string(src_elem_count * sizeof(elem_t)));
    }

    try {
        detail::copy_memory_to_device_sync_unsafe(
            get(), src, src_elem_count * sizeof(elem_t), get_owner());
    }
    catch (const std::exception& ex) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + "\n" + ex.what());
    }
}

template <TEMPLATE_DECL_ARG>
std::vector<elem_t> memory<TEMPLATE_DEF_ARG>::enqueue_read_sync(
    size_t src_elem_count /* = 0*/) const {
    if (src_elem_count == 0) {
        src_elem_count = count();
    }

    std::vector<elem_t> dst;
    size_t actual_size = std::min(src_elem_count, count());
    try {
        dst.resize(actual_size);
        detail::copy_memory_to_device_sync_unsafe(
            dst.data(), get(), actual_size * sizeof(elem_t), get_owner());
    }
    catch (const std::exception& ex) {
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + "\n" + ex.what());
    }
    return dst;
}

#undef TEMPLATE_DEF_ARG
#undef TEMPLATE_DECL_ARG
} // namespace native
