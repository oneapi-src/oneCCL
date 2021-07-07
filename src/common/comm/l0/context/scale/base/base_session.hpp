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
#include <functional>
#include <string>
#include <vector>

#include "oneapi/ccl.hpp"
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"

#include "coll/algorithms/algorithms_enum.hpp"
#include "common/comm/l0/modules/supported_modules.hpp"
#include "coll/coll_param.hpp"

namespace native {
namespace observer {
using counter_t = uint64_t;

struct producer_description {
    size_t rank;
    size_t comm_size;
    counter_t staged_buffer_elem_count;

    std::shared_ptr<ccl_context> context;
    ccl_device& device;
    ccl_device::device_cmd_list immediate_list; //TODO make persisten
};

struct context_descr {
    context_descr(const coll_param_gpu& kernel_params) : kernel_params(kernel_params) {}

    using host_mem_ptr_t = ccl_context::host_memory_ptr<uint8_t>;
    using host_mem_ptr_cntr_t = ccl_context::host_memory_ptr<counter_t>;
    using dev_mem_ptr_t = ccl_device::device_memory_ptr<uint8_t>;
    using dev_mem_ptr_cntr_t = ccl_device::device_memory_ptr<counter_t>;

    // produced by kernel
    host_mem_ptr_t host_mem_producer;
    host_mem_ptr_cntr_t host_mem_producer_counter;
    size_t host_consumed_bytes;
    size_t host_expected_bytes;

    // consumed by kernel
    dev_mem_ptr_t dev_mem_consumer;
    dev_mem_ptr_cntr_t dev_mem_consumer_counter;
    size_t device_produced_bytes;

    // (TODO consider using 'recv_buff' from collective entry)
    // to reduce copy iterations
    // TODO: rename
    dev_mem_ptr_cntr_t producer_aggregated_memory_offset;

    void init_host_dev_fields();

    void init(size_t staged_buffer_elem_count,
              size_t observer_domain_index,
              size_t observer_domain_count,
              std::shared_ptr<ccl_context>& context,
              ccl_device& device);

    void reset_counters(size_t observer_domain_index, size_t observer_domain_count);

private:
    // TODO: can we guarantee that this object is not destroyed before invoke_params and
    // use const& here?
    coll_param_gpu kernel_params;
};

template <ccl_coll_type coll_type>
struct invoke_params {
    static constexpr ccl_coll_type get_coll_type() {
        return coll_type;
    }

    invoke_params(producer_description&& in_producer_params, const coll_param_gpu& kernel_params)
            : in_params(std::move(in_producer_params)),
              kernel_params(kernel_params),
              out_params(kernel_params),
              valid(false) {}

    void set_out_params(const context_descr& src) {
        out_params = src;
        valid = true;
    }

    bool is_valid() const {
        return valid;
    }

    const producer_description& get_producer_params() const {
        return in_params;
    }

    producer_description& get_producer_params() {
        return in_params;
    }

    const coll_param_gpu& get_kernel_params() const {
        return kernel_params;
    }

    const context_descr& get_ctx_params() const {
        if (!is_valid()) {
            throw std::runtime_error("observer invocation params are not ready");
        }
        return out_params;
    }

private:
    producer_description in_params;
    // TODO: can we guarantee that this object is not destroyed before l0 entry and
    // use const& here?
    coll_param_gpu kernel_params;
    context_descr out_params;
    bool valid;
};

struct session_key {
    using hash_core_t = size_t;

    friend std::ostream& operator<<(std::ostream& out, const session_key& key) {
        out << key.to_string();
        return out;
    }

    template <class T>
    session_key(const T* src) : hash(std::hash<const T*>{}(src)) {}

    bool operator<(const session_key& other) const noexcept {
        return hash < other.hash;
    }

    std::string to_string() const {
        return std::to_string(hash);
    }

private:
    hash_core_t hash;
};

struct session_notification {
    session_notification(void* addr, size_t size_bytes)
            : host_src_ptr(addr),
              src_size_bytes(size_bytes) {}
    void* host_src_ptr;
    size_t src_size_bytes;
};

} // namespace observer
} // namespace native
