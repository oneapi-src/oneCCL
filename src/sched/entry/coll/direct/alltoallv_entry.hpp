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

#include "sched/entry/coll/direct/base_coll_entry.hpp"

class alltoallv_entry : public base_coll_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "alltoallv";
    }

    alltoallv_entry() = delete;
    alltoallv_entry(ccl_sched* sched,
                    const ccl_buffer send_buf,
                    const size_t* send_cnts,
                    ccl_buffer recv_buf,
                    const size_t* recv_cnts,
                    const ccl_datatype& dtype,
                    ccl_comm* comm)
            : base_coll_entry(sched),
              send_buf(send_buf),
              send_cnts(send_cnts),
              recv_buf(recv_buf),
              recv_cnts(recv_cnts),
              dtype(dtype),
              comm(comm),
              send_bytes(nullptr),
              recv_bytes(nullptr),
              send_offsets(nullptr),
              recv_offsets(nullptr),
              sum_send_bytes(0),
              sum_recv_bytes(0) {}

    void start() override {
        size_t dt_size = dtype.size();
        int comm_size = comm->size();
        int i;
        sum_recv_bytes = 0;
        sum_send_bytes = 0;

        if (!send_bytes && !recv_bytes && !send_offsets && !recv_offsets) {
            send_bytes = static_cast<int*>(CCL_MALLOC(comm_size * sizeof(int), "send_bytes"));
            recv_bytes = static_cast<int*>(CCL_MALLOC(comm_size * sizeof(int), "recv_bytes"));
            send_offsets = static_cast<int*>(CCL_MALLOC(comm_size * sizeof(int), "send_offsets"));
            recv_offsets = static_cast<int*>(CCL_MALLOC(comm_size * sizeof(int), "recv_offsets"));
        }

        send_bytes[0] = send_cnts[0] * dt_size;
        recv_bytes[0] = recv_cnts[0] * dt_size;
        send_offsets[0] = 0;
        recv_offsets[0] = 0;
        sum_send_bytes = send_bytes[0];
        sum_recv_bytes = recv_bytes[0];

        for (i = 1; i < comm_size; i++) {
            send_bytes[i] = send_cnts[i] * dt_size;
            recv_bytes[i] = recv_cnts[i] * dt_size;
            send_offsets[i] =
                send_offsets[i - 1] + send_bytes[i - 1]; // treat buffers as char buffers
            recv_offsets[i] = recv_offsets[i - 1] + recv_bytes[i - 1];
            sum_send_bytes += send_bytes[i];
            sum_recv_bytes += recv_bytes[i];
        }

        LOG_DEBUG("alltoallv entry req ", &req, ", sum_send_bytes ", sum_send_bytes);

        atl_status_t atl_status = comm->atl->atl_ep_alltoallv(sched->bin->get_atl_ep(),
                                                              send_buf.get_ptr(sum_send_bytes),
                                                              send_bytes,
                                                              send_offsets,
                                                              recv_buf.get_ptr(sum_recv_bytes),
                                                              recv_bytes,
                                                              recv_offsets,
                                                              &req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("alltoallv entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_started;
    }

    void update() override {
        int req_status;
        atl_status_t atl_status =
            comm->atl->atl_ep_check(sched->bin->get_atl_ep(), &req_status, &req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("alltoallv entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req_status) {
            status = ccl_sched_entry_status_complete;
        }
    }

    ~alltoallv_entry() {
        CCL_FREE(send_bytes);
        CCL_FREE(recv_bytes);
        CCL_FREE(send_offsets);
        CCL_FREE(recv_offsets);
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", send_cnts ",
                           send_cnts,
                           ", send_buf ",
                           send_buf,
                           ", send_bytes ",
                           send_bytes,
                           ", send_offsets ",
                           send_offsets,
                           ", recv_cnts ",
                           recv_cnts,
                           ", recv_buf ",
                           recv_buf,
                           ", recv_bytes ",
                           recv_bytes,
                           ", recv_offsets ",
                           recv_offsets,
                           ", comm_id ",
                           sched->get_comm_id(),
                           ", req ",
                           &req,
                           "\n");
    }

private:
    ccl_buffer send_buf;
    const size_t* send_cnts;
    ccl_buffer recv_buf;
    const size_t* recv_cnts;
    ccl_datatype dtype;
    ccl_comm* comm;
    atl_req_t req{};

    int* send_bytes;
    int* recv_bytes;
    int* send_offsets;
    int* recv_offsets;
    size_t sum_send_bytes;
    size_t sum_recv_bytes;
};
