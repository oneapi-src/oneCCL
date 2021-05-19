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
#include "oneapi/ccl/native_device_api/l0/base_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"

#include "common/comm/l0/context/scale/base/base_session.hpp"

namespace native {
namespace observer {

void context_descr::init_host_dev_fields() {
    host_mem_producer = nullptr;
    host_mem_producer_counter = nullptr;
    host_consumed_bytes = 0;
    host_expected_bytes = 0;

    dev_mem_consumer = nullptr;
    dev_mem_consumer_counter = nullptr;
    device_produced_bytes = 0;
}

void context_descr::init(size_t staged_buffer_elem_count,
                         size_t observer_domain_index,
                         size_t observer_domain_count,
                         std::shared_ptr<ccl_context>& context,
                         ccl_device& device) {
    // set all fields by 0
    init_host_dev_fields();

    /* HOST */
    // create staged mem in host context (Host memory allocation descriptor)
    ze_host_mem_alloc_desc_t host_descr = ccl_context::get_default_host_alloc_desc();
    host_descr.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;

    // host mem buf
    host_mem_producer = context->template alloc_memory<uint8_t>(
        staged_buffer_elem_count * ccl::get_datatype_size(kernel_params.get_datatype()),
        /*TODO use page size*/ ccl::get_datatype_size(kernel_params.get_datatype()),
        host_descr);

    // create staged mem counter in host context (host mem buf counter)
    host_mem_producer_counter = context->template alloc_memory<counter_t>(
        1, /*TODO use page size*/ sizeof(counter_t), host_descr);

    host_expected_bytes =
        staged_buffer_elem_count * ccl::get_datatype_size(kernel_params.get_datatype());

    /* DEVICE */
    ze_device_mem_alloc_desc_t mem_descr = ccl_device::get_default_mem_alloc_desc();

    // create total aggregated memory in device context
    mem_descr.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    dev_mem_consumer = device.template alloc_memory_ptr<uint8_t>(
        (staged_buffer_elem_count * observer_domain_count) *
            ccl::get_datatype_size(kernel_params.get_datatype()),
        ccl::get_datatype_size(kernel_params.get_datatype()),
        context,
        mem_descr);

    // create offset in device context
    mem_descr.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
    producer_aggregated_memory_offset =
        device.template alloc_memory_ptr<counter_t>(1, sizeof(counter_t), context, mem_descr);

    // create aggregated counter in device context
    dev_mem_consumer_counter =
        device.template alloc_memory_ptr<counter_t>(1, sizeof(counter_t), context, mem_descr);

    /* COUNTERS */
    reset_counters(observer_domain_index, observer_domain_count);
}

void context_descr::reset_counters(size_t observer_domain_index, size_t observer_domain_count) {
    counter_t filled_counter_value = 0;

    host_mem_producer_counter->enqueue_write_sync(&filled_counter_value, 1);

    filled_counter_value = observer_domain_index * host_mem_producer->count();

    producer_aggregated_memory_offset->enqueue_write_sync(&filled_counter_value, 1);

    filled_counter_value = 0;
    dev_mem_consumer_counter->enqueue_write_sync(&filled_counter_value, 1);
}

} // namespace observer
} // namespace native
