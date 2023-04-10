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
#include "common/stream/stream.hpp"
#include "comp/comp.hpp"
#include "sched/entry/ze/ze_alltoallv_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_cache.hpp"
#include "sched/queue/queue.hpp"

#include <string>
#include <sstream>

using namespace ccl;
using namespace ccl::ze;

ze_alltoallv_entry::ze_alltoallv_entry(ccl_sched* sched,
                                       std::vector<ccl_buffer> send_bufs,
                                       std::vector<ccl_buffer> recv_bufs,
                                       std::vector<size_t> counts,
                                       size_t buf_idx_start,
                                       const ccl_datatype& dtype,
                                       ccl_comm* comm,
                                       std::vector<ze_event_handle_t> wait_events)
        : ze_base_entry(sched, comm, 1 /* request additional events */, wait_events),
          send_bufs(send_bufs),
          recv_bufs(recv_bufs),
          counts(counts),
          buf_idx_start(buf_idx_start),
          dtype(dtype) {}

void ze_alltoallv_entry::init_ze_hook() {
    auto comm_size = comm->size();
    auto comm_rank = comm->rank();

    std::vector<void*> in_bufs(send_bufs.size());
    std::vector<void*> out_bufs(recv_bufs.size());

    ccl::global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
    kernel_name =
        "alltoallv_kernel_" + std::to_string(comm_size) + "_" + to_string(dtype.idx()) + "_custom";
    LOG_DEBUG("kernel name: ", kernel_name);

    kernels.emplace_back(module, kernel_name, worker_idx);
    kernels.back().calculate_group_size(counts[comm_rank]);

    for (int idx = 0; idx < comm_size; idx++) {
        const auto global_rank = comm->get_global_rank(idx);
        if (idx == comm_rank) {
            // fill local send buf
            in_bufs[global_rank] = send_bufs[global_rank].get_ptr();
        }
        else {
            ccl_buffer buf{};
            sched->get_memory().handle_manager.get(
                idx, buf_idx_start + comm->get_global_rank(comm_rank), buf, comm);
            CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
            in_bufs[global_rank] = buf.get_ptr();
        }
    }

    // recv bufs are local bufs
    for (int idx = 0; idx < comm_size; idx++) {
        const int peer_rank = (comm_rank + idx + 1) % comm_size;
        const auto global_rank = comm->get_global_rank(idx);
        const auto global_peer_rank = comm->get_global_rank(peer_rank);
        if (global_rank != global_peer_rank) {
            out_bufs[global_peer_rank] = recv_bufs[global_peer_rank].get_ptr();
        }
        else {
            out_bufs[global_rank] = recv_bufs[global_rank].get_ptr();
        }
    }

    // sets in/out bufs for kernel
    int arg_idx = 0;
    for (int idx = 0; idx < comm_size; idx++) {
        const auto global_rank = comm->get_global_rank(idx);
        ZE_CALL(zeKernelSetArgumentValue,
                (kernels.back().get_kernel(), arg_idx, dtype.size(), &(in_bufs[global_rank])));
        arg_idx++;
        ZE_CALL(zeKernelSetArgumentValue,
                (kernels.back().get_kernel(), arg_idx, dtype.size(), &(out_bufs[global_rank])));
        arg_idx++;
        ZE_CALL(zeKernelSetArgumentValue,
                (kernels.back().get_kernel(), arg_idx, sizeof(unsigned long), &(counts[idx])));
        arg_idx++;
    }

    ZE_CALL(zeKernelSetArgumentValue,
            (kernels.back().get_kernel(), 3 * comm_size, sizeof(int), &comm_size));

    ZE_CALL(zeCommandListAppendLaunchKernel,
            (ze_base_entry::get_comp_list(),
             kernels.back().get_kernel(),
             kernels.back().get_group_count(),
             ze_base_entry::entry_event,
             0,
             nullptr));
}

void ze_alltoallv_entry::finalize_ze_hook() {
    LOG_DEBUG("finalize");
}

void ze_alltoallv_entry::start() {
    size_t kernel_counter = 0;
    if (global_data::env().enable_kernel_sync) {
        kernel_counter = global_data::get().ze_data->kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
    }
    else {
        global_data::get().ze_data->kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_alltoallv_entry::update() {
    ze_base_entry::update();

    if (global_data::env().enable_kernel_sync && global_data::get().ze_data->kernel_counter > 0) {
        global_data::get().ze_data->kernel_counter--;
    }
}

std::string ze_alltoallv_entry::name_ext() const {
    std::stringstream out;
    out << name();
    return out.str();
}

void ze_alltoallv_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", send_bufs ",
                       &send_bufs,
                       ", recv_bufs ",
                       &recv_bufs,
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}
