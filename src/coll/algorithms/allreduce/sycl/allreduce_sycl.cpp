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
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

ccl::event allreduce_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl::reduction reduction,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event e;
    done = true;

    uint32_t world = global_comm->size();
    int rank = global_comm->rank();

    world = global_comm->get_node_comm()->size();
    rank = global_comm->get_node_comm()->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
            });
        }
        else {
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected =
        global_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| is_single_tile: ",
              is_single_tile,
              ", has_all_vertices_connected: ",
              has_all_vertices_connected);

    if (!sycl_use_esimd(global_comm)) {
        if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_small_threshold) {
#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_SMALL");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking small allreduce kernel, count:", count, " datatype: ", dtype);
            e = allreduce_small(
                send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps);
            LOG_DEBUG(
                "invoking small allreduce kernel, count:", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_LARGE");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking large allreduce kernel, count:", count, " datatype: ", dtype);
            e = allreduce_large(
                send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps);
            LOG_DEBUG(
                "invoking large allreduce kernel, count:", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    // ESIMD
    if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_small_threshold &&
        has_all_vertices_connected) {
        init_allreduce_small(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_SMALL");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_small(dtype, q, send_buf, recv_buf, count, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        if (done)
            return e;
        // continue to medium kernel if not done
    }
    if ((count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_medium_threshold ||
         (global_comm->size() == 2 && !ccl::global_data::env().sycl_allreduce_tmp_buf)) &&
        !is_single_tile) { // medium message sizes
        init_allreduce_medium(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_MEDIUM");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG(
            "|CCL_SYCL| allreduce selects medium kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_medium(dtype, q, send_buf, recv_buf, count, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects medium kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
    }
    else if (!is_single_tile) { // large message sizes
        init_allreduce_large(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_LARGE");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects large kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_large(dtype, q, send_buf, recv_buf, count, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects large kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
    }
    else {
        done = false;
    }

    return e;
}

event allreduce_sycl(sycl::queue& q,
                     const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const allreduce_attr& attr,
                     const vector_class<event>& deps,
                     bool& done) {
    done = true;
    ccl::impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = static_cast<ccl_comm*>(disp_comm.get());
    ccl_stream* global_stream = get_stream_ptr(disp(op_stream));

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return allreduce_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, reduction, global_comm, global_stream, deps, done);
    }

    CCL_THROW_IF_NOT(q.is_in_order(), "SYCL queue must be in-order");

    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();

    // TODO: Sycl allgatherv does not support counts that are non-divisible by the node_comm size.
    //       Once this support is enabled, the algorithm will be simplified.

    // TODO: Chunks to reduce_scatter and allgatherv must be aligned to 128bytes for performance.
    //       Keep the remainder an odd number if necessary
    size_t counts_per_rank = count / node_comm->size();
    size_t remainder_count = count % node_comm->size();

    {
        size_t line_size = ccl::global_data::env().sycl_kernels_line_size;
        CCL_THROW_IF_NOT(!(line_size % ccl_dtype.size()),
                         "datatype size not divisible by line_size=",
                         line_size);

        size_t counts_per_line = line_size / ccl_dtype.size();
        size_t total_lines = count / counts_per_line;
        remainder_count = count % counts_per_line;

        size_t lines_per_rank = total_lines / node_comm->size();
        size_t remainder_lines = total_lines % node_comm->size();

        counts_per_rank = lines_per_rank * counts_per_line;
        remainder_count += remainder_lines * counts_per_line;

        CCL_THROW_IF_NOT(count == remainder_count + (counts_per_rank * node_comm->size()),
                         "Incorrect calculations for lines_per_rank.");
    }

    if (ccl_dtype.size() < sizeof(int) && (counts_per_rank % 2)) {
        // TODO: The SYCL algorithms do not support pointers/counts that would be odd, for sizes
        //       less that 4 bytes. Since we know that counts_per_rank is odd, we would end up with
        //       an odd pointer.
        CCL_THROW("Unexpected non-even counts_per_rank. Not falling back.");

        // fallback
        done = false;
        return ev;
    }

    LOG_DEBUG("allreduce_sycl count=",
              count,
              " counts_per_rank=",
              counts_per_rank,
              " remainder_count=",
              remainder_count);

    // ===== STEP 1:  COUNTS DIVISIBLE BY NODE COUNT =====
    if (counts_per_rank) {
        const size_t tmp_recv_count_offset = node_comm->rank() * counts_per_rank;
        void* tmp_recv_ptr = ((char*)recv_buf) + (tmp_recv_count_offset * ccl_dtype.size());

        // ----- Scaleup Reduce Scatter Inplace Phase -----
        ev = reduce_scatter_sycl_single_node(q,
                                             send_buf,
                                             tmp_recv_ptr,
                                             counts_per_rank,
                                             dtype,
                                             reduction,
                                             node_comm,
                                             global_stream,
                                             deps,
                                             done);
        if (!done) {
            LOG_INFO("allreduce_sycl reduce_scatter was not done -- falling back");
            // fallback
            return ev;
        }

        // ----- Scaleout Allreduce Phase -----
        auto r2r_comm = global_comm->get_r2r_comm();
        void* scaleout_allreduce_ptr = tmp_recv_ptr;
        if (r2r_comm->size() > 1) {
            std::vector<event> evs;
            evs.push_back(std::move(ev));
            ev = allreduce_scaleout_sycl(q,
                                         MPI_IN_PLACE,
                                         scaleout_allreduce_ptr,
                                         counts_per_rank,
                                         dtype,
                                         reduction,
                                         r2r_comm.get(),
                                         evs,
                                         done,
                                         true /*copy_to_host*/);
            if (!done) {
                LOG_INFO("allreduce_sycl scaleout was not done -- falling back");
                return ev;
            }
        }

        // ----- Scaleup Allgatherv Inplace Phase -----
        {
            int node_size = node_comm->size();
            std::vector<size_t> recv_counts(node_size, counts_per_rank);

            std::vector<event> evs;
            evs.push_back(std::move(ev));
            ev = allgather_sycl_single_node(q,
                                            tmp_recv_ptr,
                                            recv_counts[node_comm->rank()], //send_count,
                                            recv_buf,
                                            recv_counts,
                                            dtype,
                                            node_comm,
                                            global_stream,
                                            evs,
                                            done);
            if (!done) {
                // fallback
                LOG_INFO("allreduce_sycl allgatherv was not done -- falling back");
                return ev;
            }
        }
    }

    // ===== STEP 2:  REMAINDER OF COUNTS DIVISIBLE BY NODE COUNT =====
    if (remainder_count) {
        LOG_INFO("Allgatherv would not work with irregular recv_counts vector. "
                 "Using CPU-side algorithm for the remainder count=",
                 remainder_count);

        size_t remainder_offset_count = count - remainder_count;
        size_t remainder_offset_bytes = remainder_offset_count * ccl_dtype.size();
        auto remainder_send_buf = ptr_offset(send_buf, remainder_offset_bytes);
        auto remainder_recv_buf = ptr_offset(recv_buf, remainder_offset_bytes);

        std::vector<event> evs;
        evs.push_back(std::move(ev));
        ev = allreduce_scaleout_sycl(q,
                                     remainder_send_buf,
                                     remainder_recv_buf,
                                     remainder_count,
                                     dtype,
                                     reduction,
                                     global_comm,
                                     evs,
                                     done,
                                     true /*copy_to_host*/);
        if (!done) {
            LOG_INFO("allreduce_sycl allreduce_scaleout_sycl for remainder count"
                     " was not done -- falling back");
            return ev;
        }
    }

    return ev;
}

} // namespace v1
} // namespace ccl
