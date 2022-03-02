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

#include "common/utils/buffer.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

class ze_a2a_allgatherv_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_ALLGATHERV";
    }

    const char* name() const override {
        return class_name();
    }

    virtual std::string name_ext() const override {
        std::stringstream out;
        out << name() << " ";
        out << "send size: " << send_count;
        return out.str();
    }

    explicit ze_a2a_allgatherv_entry(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     size_t send_count,
                                     ccl_buffer recv_buf,
                                     const size_t* recv_counts,
                                     const ccl_datatype& dtype,
                                     ccl_comm* comm,
                                     std::vector<ze_event_handle_t> wait_events = {},
                                     size_t peer_buf_idx = 0,
                                     size_t peer_buf_offset = 0);

    void init_ze_hook() override;

    void update() override;

    static void fill_list(const ze_base_entry* entry,
                          int comm_rank,
                          void* send_buf,
                          void* recv_buf,
                          const std::vector<ccl_buffer>& peer_recv_bufs,
                          int peer_count,
                          size_t copy_bytes,
                          size_t offset_bytes,
                          bool is_inplace,
                          std::vector<ze_event_handle_t>& copy_events,
                          ze_event_handle_t wait_event = nullptr);

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "comm ", comm->to_string(), "\n");
    }

private:
    static constexpr size_t event_group_count{ 1 }; // copy phase

    const ccl_buffer send_buf;
    const size_t send_count;
    const ccl_buffer recv_buf;
    const std::vector<size_t> recv_counts;
    const ccl_datatype dtype;
    const size_t peer_buf_idx;
    const size_t peer_buf_offset;
    const int peer_count;

    std::vector<ze_event_handle_t> copy_events;
};
