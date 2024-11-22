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
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: allgatherv");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, send_count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: allgatherv");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    if (!ccl::global_data::env().sycl_esimd) {
        if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold) {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allgatherv_small", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
            e = allgatherv_small(send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allgatherv_large", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);
            e = allgatherv_large(send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    // ESIMD
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_small_threshold &&
        has_all_vertices_connected) {
        init_allgatherv_small(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_small", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_small(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects small kernel, count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done)
            return e;
    }
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().sycl_allgatherv_medium_threshold &&
        !is_single_tile) {
        init_allgatherv_medium(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_medium", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype);
        e = run_allgatherv_medium(dtype, q, send_buf, send_count, recv_buf, recv_counts, done);
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects medium kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_large", "send_size", send_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| invoking large allgatherv: count: ", send_count, " datatype: ", dtype);

#if defined(CCL_SYCL_VEC_SUPPORT_FP16) && defined(CCL_SYCL_VEC_SUPPORT_BF16)
        e = allgatherv_large(send_buf, send_count, recv_buf, recv_counts, dtype, comm, global_stream, deps);
#else
        // allgatherv_large is sycl::vec based algorithm
        // when 16-bit datatypes are not supported, gather by int16 instead
        ccl::datatype new_dtype = ccl::datatype::int16;
        size_t new_send_count = send_count * ccl_dtype.size() / 2;
        ccl::vector_class<size_t> new_recv_counts;
        for (size_t i = 0; i < recv_counts.size(); i++) {
            new_recv_counts.push_back(recv_counts[i] * ccl_dtype.size() / 2);
        }
        e = allgatherv_large(
            send_buf, new_send_count, recv_buf, new_recv_counts, new_dtype, comm, global_stream, deps);
#endif // defined(CCL_SYCL_VEC_SUPPORT_FP16) && defined(CCL_SYCL_VEC_SUPPORT_BF16)
        LOG_DEBUG(
            "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }

    return e;
}

ccl::event allgatherv_sycl_multi_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();

    size_t total_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    // check threshold to disable scaleout algorithm
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi ||
        total_count * ccl_dtype.size() > ccl::global_data::env().sycl_allgatherv_scaleout_threshold) {
        // fallback
        done = false;
        return ev;
    }

    LOG_DEBUG("allgatherv_sycl send_count=", send_count);

    std::vector<size_t> recv_scaleout_counts(r2r_comm->size());
    std::vector<size_t> scaleout_offsets(r2r_comm->size());
    size_t total_scaleout_count = 0;
    for (int i = 0; i < r2r_comm->size(); i++) {
        const int global_rank = r2r_comm->get_global_rank(i);
        recv_scaleout_counts[i] = recv_counts[global_rank];
        scaleout_offsets[i] = total_scaleout_count * ccl_dtype.size();
        total_scaleout_count += recv_counts[global_rank];
    }

    void* scaleout_buf;
    int to_free = 0;
    if (total_scaleout_count * ccl_dtype.size() <= global_comm->get_scaleout_device_buf_size()) {
        scaleout_buf = global_comm->get_scaleout_device_buf(q);
    }
    else {
        scaleout_buf = sycl::malloc_device(total_scaleout_count * ccl_dtype.size(), q);
        to_free = 1;
    }

    // ----- Scaleout Allgatherv Phase -----
    std::vector<event> evs;

    if (r2r_comm->size() > 1) {
        ev = allgatherv_scaleout_sycl(
            q, send_buf, send_count, scaleout_buf, recv_scaleout_counts, dtype, r2r_comm, deps, done, true, false);
        if (!done) {
            LOG_INFO("allgatherv_sycl scaleout was not done -- falling back");
            return ev;
        }

        evs.push_back(std::move(ev));
    }

    // ----- Scaleup Allgatherv Inplace Phase -----
    {
        size_t r2r_size = r2r_comm->size();
        size_t node_size = node_comm->size();

        std::vector<size_t> node_offsets(r2r_size);
        node_offsets[0] = 0;
        auto first = recv_counts.begin();
        auto last = first + node_size;
        for (int i = 1; i < r2r_size; i++) {
            node_offsets[i] = std::accumulate(first, last, node_offsets[i - 1]);
            first += node_size;
            last += node_size;
        }

        for (int i = 0; i < r2r_size; i++) {
            auto count_begin = recv_counts.begin() + i * node_size;
            auto count_end = count_begin + node_size;
            std::vector<size_t> counts(count_begin, count_end);

            ev = allgather_sycl_single_node(q,
                                            (char*)scaleout_buf + scaleout_offsets[i],
                                            recv_scaleout_counts[i],
                                            (char*)recv_buf + node_offsets[i] * ccl_dtype.size(),
                                            counts,
                                            dtype,
                                            global_comm,
                                            global_stream,
                                            evs,
                                            done);

            if (!done) {
                // fallback
                LOG_ERROR("allgatherv_sycl allgatherv single node was not done -- falling back");
                return ev;
            }
        }
    }

    if (to_free) {
        auto sycl_ev = ev.get_native();
        auto e = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_ev);
            h.host_task([=]() {
                sycl::free(scaleout_buf, q);
            });
        });
        ev = ccl::event::create_from_native(e);
    }
    else {
        global_comm->put_scaleout_device_buf(scaleout_buf);
    }

    return ev;
}

ccl::event allgather_sycl(sycl::queue& q,
                          const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* op_stream,
                          const allgatherv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done) {
    for (const size_t rc : recv_counts) {
        if (rc != send_count) {
            CCL_THROW(
                "|CCL_SYCL| Allgatherv Sycl kernel is called with non-equal receive counts, fallback to schedule-based implementation");
        }
    }

    if (send_count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node) {
        LOG_DEBUG("is_single_node");
        return allgather_sycl_single_node(
            q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, deps, done);
    }

    return allgatherv_sycl_multi_node(
        q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl
