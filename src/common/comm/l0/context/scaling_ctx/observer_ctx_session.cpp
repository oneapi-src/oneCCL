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
#include <sstream>

#include "common/comm/l0/context/scaling_ctx/observer_ctx_session.hpp"
#include "common/log/log.hpp"

namespace native {
namespace observer {

session::session()
        : host_producer_memory(nullptr),
          host_producer_ready_bytes(nullptr),
          host_consumed_bytes(0),
          host_expected_bytes(0),

          device_consumer_total_memory(nullptr),
          device_consumer_ready_bytes(nullptr),
          device_produced_bytes(0),
          copy_immediate_list() {}

std::string session::to_string() const {
    std::stringstream ss;
    ss << "sess: " << reinterpret_cast<const void*>(this);
    return ss.str();
}

size_t session::get_send_tag() const {
    return send_tag;
}

size_t session::produce_data(void** out_chunk, size_t& out_chunk_size) {
    //read ready flag
    size_t old_consumed = host_consumed_bytes;
    int total_produced = *host_producer_ready_bytes;

    size_t to_consume = total_produced - old_consumed;
    if (to_consume) {
        //fence
        LOG_TRACE(to_string(),
                  " - bytes produced: ",
                  total_produced,
                  ", previously bytes consumed: ",
                  old_consumed);
        std::atomic_thread_fence(std::memory_order::memory_order_seq_cst);

        // do not read data here!
        *out_chunk = (static_cast<uint8_t*>(host_producer_memory) + old_consumed);

        //check finalize
        host_consumed_bytes = to_consume;
    }

    out_chunk_size = to_consume;
    return to_consume;
}

bool session::consume_data(size_t observer_domain_index, void* in_chunk, size_t in_chunk_size) {
    /* TODO create event
     * ze_event_handle_t mem_event {};
     */

    ze_result_t res = zeCommandListAppendMemoryCopy(
        copy_immediate_list,
        (static_cast<uint8_t*>(device_consumer_total_memory) + device_produced_bytes),
        in_chunk,
        in_chunk_size,
        /*mem_event*/ nullptr,
        0,
        nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string(
                "cannot append copy NUMA host to device memory for partial result, error: ") +
            native::to_string(res));
    }
    device_produced_bytes += in_chunk_size;

    res = zeCommandListAppendMemoryCopy(copy_immediate_list,
                                        device_consumer_ready_bytes,
                                        &device_produced_bytes,
                                        sizeof(device_produced_bytes),
                                        nullptr,
                                        1,
                                        /*&mem_event*/ nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string("cannot append copy NUMA host to device memory for ready bytes, error: ") +
            native::to_string(res));
    }
    return device_produced_bytes == host_expected_bytes;
}

size_t session_table::get_unique_tag() {
    static std::atomic<size_t> tag_counter{ 1 };
    return tag_counter.fetch_add(1);
}

std::string session_table::to_string() const {
    std::stringstream ss;
    ss << "sessions count: " << sessions.size() << std::endl;
    for (const auto& val : sessions) {
        ss << "[" << val.first << ", " << reinterpret_cast<void*>(val.second.get()) << "]\n"
           << val.second->to_string() << std::endl;
    }
    return ss.str();
}
} // namespace observer
} // namespace native
