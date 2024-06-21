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
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"

namespace ccl {
namespace v1 {

ccl::event allgather_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event e;
    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);

    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    for (uint32_t i = 0; i < recv_counts.size(); i++) {
        if (send_count != recv_counts[i]) {
            LOG_ERROR("Allgatherv only supports the case when all recv_counts are the same");
            done = false;
            return e;
        }
        assert(send_count == recv_counts[i]);
    }

    if (world == 1) {
        sycl::event e_cp;
        if (send_buf != recv_buf) {
            e_cp = q.memcpy(recv_buf, send_buf, send_count * ccl_dtype.size());
        }
        return ccl::event::create_from_native(e_cp);
    }

    if (!ccl::global_data::env().sycl_esimd) {
        if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold) {
#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLGATHERV_SMALL");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
            e = allgatherv_small(send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLGATHERV_LARGE");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);
            e = allgatherv_large(send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold &&
        has_all_vertices_connected) {
        init_allgatherv_small(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLGATHERV_SMALL");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_small(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
    }
    else if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_medium_threshold &&
             !is_single_tile) {
        init_allgatherv_medium(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLGATHERV_MEDIUM");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_medium(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
    }
    else if (!is_single_tile) {
        if (send_count % 2 == 0 || ccl_dtype.size() >= 4) {
            // TODO: rewrite comments in way small
            LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);
            if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold) {
                // TODO: implement
                CCL_THROW("Not implemented");
            }
            else {
                e = allgatherv_large(
                    send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
            }
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
        }
        else {
            LOG_DEBUG(
                "[", rank, "] allgatherv selects ccl scheduler send_count: ", send_count, ", datatype: ", dtype);
            done = false;
        }
    }
    else {
        done = false;
    }

    return e;
}

ccl::event allgather_sycl(sycl::queue& q,
                          const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          const ccl::communicator& comm,
                          const stream& op_stream,
                          const allgatherv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done) {
    ccl::impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* comm_ = (ccl_comm*)(disp_comm.get());
    ccl_stream* global_stream = get_stream_ptr(disp(op_stream));

    return allgather_sycl_single_node(
        q, send_buf, send_count, recv_buf, recv_counts, dtype, comm_, global_stream, deps, done);
}

} // namespace v1
} // namespace ccl
