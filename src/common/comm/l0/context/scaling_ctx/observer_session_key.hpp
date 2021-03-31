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

#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/context.hpp"
#include "coll/algorithms/algorithms_enum.hpp"

namespace native {
namespace observer {
using counter_type = uint64_t;
struct producer_description {
    size_t world_rank;
    size_t world_size;
    counter_type staged_buffer_elem_count;

    std::shared_ptr<ccl_context> context;
    ccl_device& device;
    ccl_device::device_cmd_list immediate_list; //TODO make persisten
};

//TODO looks like these structure is specific for allreduce only
template <ccl_coll_type type, class native_data_type>
struct context_description {
    // produced by kernel
    ccl_context::host_memory_ptr<native_data_type> numa_staged_memory;
    ccl_context::host_memory_ptr<counter_type> staged_memory_size_counter;

    // consumed by kernel
    // (TODO consider using 'recv_buff' from collective entry)
    // to reduce copy iterations
    ccl_device::device_memory_ptr<counter_type> producer_aggregated_memory_offset;

    ccl_device::device_memory_ptr<native_data_type> total_producers_aggregated_memory;
    ccl_device::device_memory_ptr<counter_type> total_producers_aggregated_size_counter;

    void init(size_t staged_buffer_elem_count,
              size_t observer_domain_index,
              size_t observer_domain_count,
              std::shared_ptr<ccl_context>& context,
              ccl_device& device) {
        // create staged mem in host context
        ze_host_mem_alloc_desc_t host_descr = ccl_context::get_default_host_alloc_desc();
        host_descr.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;

        numa_staged_memory = context->template alloc_memory<native_data_type>(
            staged_buffer_elem_count,
            /*TODO use page size*/ sizeof(native_data_type),
            host_descr);

        // create staged mem counter in host context
        staged_memory_size_counter = context->template alloc_memory<counter_type>(
            1, /*TODO use page size*/ sizeof(counter_type), host_descr);

        ze_device_mem_alloc_desc_t mem_descr = ccl_device::get_default_mem_alloc_desc();

        // create total aggregated memory in device context
        mem_descr.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED;
        total_producers_aggregated_memory = device.template alloc_memory_ptr<native_data_type>(
            staged_buffer_elem_count * observer_domain_count,
            sizeof(native_data_type),
            context,
            mem_descr);

        // create offset in device context
        mem_descr.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
        producer_aggregated_memory_offset = device.template alloc_memory_ptr<counter_type>(
            1, sizeof(counter_type), context, mem_descr);

        // create aggregated counter in device context
        total_producers_aggregated_size_counter = device.template alloc_memory_ptr<counter_type>(
            1, sizeof(counter_type), context, mem_descr);

        // init values
        reset_staged_counters(observer_domain_index, observer_domain_count);
    }

    void reset_staged_counters(size_t observer_domain_index, size_t observer_domain_count) {
        counter_type filled_counter_value = 0;
        staged_memory_size_counter->enqueue_write_sync(&filled_counter_value, 1);

        filled_counter_value = observer_domain_index * numa_staged_memory->count();
        ;
        producer_aggregated_memory_offset->enqueue_write_sync(&filled_counter_value, 1);

        filled_counter_value = 0;
        total_producers_aggregated_size_counter->enqueue_write_sync(&filled_counter_value, 1);
    }
};

template <ccl_coll_type type, class kernel_params>
struct invoke_params {
    using kernel_params_t = kernel_params;

    static constexpr ccl_coll_type get_coll_type() {
        return type;
    }

    invoke_params(producer_description&& in)
            : in_params(std::move(in)),
              out_params(),
              valid(false) {}

    void set_out_params(
        const context_description<type, typename kernel_params_t::native_type>& src) {
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

    const context_description<type, typename kernel_params_t::native_type>& get_ctx_params() const {
        if (!is_valid()) {
            throw std::runtime_error("observer invocation params are not ready");
        }
        return out_params;
    }

private:
    producer_description in_params;
    context_description<type, typename kernel_params_t::native_type> out_params;
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

    bool operator<(const session_key& other) const noexcept;

    std::string to_string() const;

private:
    hash_core_t hash;
};
} // namespace observer
} // namespace native
