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
#include "coll/algorithms/utils/sycl_coll_base.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/allreduce/sycl/allreduce_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

ccl::event allreduce_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   ccl::reduction reduction,
                                   ccl_comm* comm,
                                   ccl::vector_class<ccl::event>& deps,
                                   bool& done,
                                   bool copy_to_host,
                                   bool is_cpu_buffers) {
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size()) {
            LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
                     count * ccl_dtype.size(),
                     " bytes. Falling back. TODO: chunking/pipelining");
            done = false;
            ccl::event e;
            return e;
        }

        scaleout_send_buf = MPI_IN_PLACE;
        scaleout_recv_buf = comm->get_scaleout_host_buf();
        auto ev = q.memcpy(scaleout_recv_buf,
                           send_buf == MPI_IN_PLACE ? recv_buf : send_buf,
                           count * ccl_dtype.size());
    }
    else if (!is_cpu_buffers) {
        // TODO: check if I_MPI_OFFLOAD is set, then let the scaleout allreduce go through.
        LOG_WARN("copy_to_host=false with a GPU buffer. "
                 "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        // TODO: determine whether we want to fallback or not. For now, no.
        // done = false;
        // ccl::event e;
        // return e;
    }

    auto op_end = q.submit([=](sycl::handler& h) {
        h.host_task([=]() {
            // call ccl::wrapper for MPI/OFI.
            int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
            atl_req_t req;
            std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
            ATL_CALL_THROW_IF_ERROR(atl_comm->allreduce(ep_idx,
                                                        scaleout_send_buf,
                                                        scaleout_recv_buf,
                                                        count,
                                                        ccl_dtype.atl_datatype(),
                                                        static_cast<atl_reduction_t>(reduction),
                                                        req));

            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
            if (!req.is_completed) {
                // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));

                // TODO: if it is determined that running atl_comm->allreduce from inside allreduce_entry (i.e. the sched) is WAY faster than running it from out here, how about checking how the schedule does progress()?
                //       allreduce_entry::update() does a simple check():     atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);
                //       Experimentally, it doesn't seem to make any difference.
            }
            else {
                // The operation was probably blocking, since it finished really quickly
            }
        });
    });

    if (copy_to_host) {
        op_end = q.memcpy(recv_buf, scaleout_recv_buf, count * ccl_dtype.size());
    }

    done = true;
    return ccl::event::create_from_native(op_end);
}
