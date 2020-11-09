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

class reduce_entry : public base_coll_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "REDUCE";
    }

    reduce_entry() = delete;
    reduce_entry(ccl_sched* sched,
                 const ccl_buffer send_buf,
                 ccl_buffer recv_buf,
                 size_t cnt,
                 const ccl_datatype& dtype,
                 ccl::reduction reduction,
                 int root,
                 ccl_comm* comm)
            : base_coll_entry(sched),
              send_buf(send_buf),
              recv_buf(recv_buf),
              cnt(cnt),
              dtype(dtype),
              op(reduction),
              root(root),
              comm(comm) {
        //TODO: Add way to using MPI communicator
        CCL_UNUSED(this->comm);
    }

    void start() override {
        LOG_DEBUG("REDUCE entry req ", &req, ", cnt ", cnt);
        size_t bytes = cnt * dtype.size();
        atl_status_t atl_status = comm->atl->atl_ep_reduce(sched->bin->get_atl_ep(),
                                                           send_buf.get_ptr(bytes),
                                                           recv_buf.get_ptr(bytes),
                                                           cnt,
                                                           root,
                                                           static_cast<atl_datatype_t>(dtype.idx()),
                                                           static_cast<atl_reduction_t>(op),
                                                           &req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("REDUCE entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_started;
    }

    void update() override {
        int req_status;
        atl_status_t atl_status =
            comm->atl->atl_ep_check(sched->bin->get_atl_ep(), &req_status, &req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("REDUCE entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req_status)
            status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", cnt ",
                           cnt,
                           ", send_buf ",
                           send_buf,
                           ", recv_buf ",
                           recv_buf,
                           ", op ",
                           ccl_reduction_to_str(op),
                           ", root ",
                           root,
                           ", comm_id ",
                           sched->get_comm_id(),
                           ", req ",
                           &req,
                           "\n");
    }

private:
    ccl_buffer send_buf;
    ccl_buffer recv_buf;
    size_t cnt;
    ccl_datatype dtype;
    ccl::reduction op;
    int root;
    ccl_comm* comm;
    atl_req_t req{};
};
