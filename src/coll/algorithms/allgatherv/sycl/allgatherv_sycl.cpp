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
// why large count written in opposite way vs small, medium
#include "coll/algorithms/allgatherv/sycl/allgatherv_large_sycl.hpp"

ccl::event allgatherv_impl(const void* send_buf,
                           size_t send_count,
                           void* recv_buf,
                           const ccl::vector_class<size_t>& recv_counts,
                           ccl::datatype dtype,
                           const ccl::communicator& comm,
                           const ccl::stream& op_stream,
                           const ccl::allgatherv_attr& attr,
                           const ccl::vector_class<ccl::event>& deps) {
#ifdef CCL_ENABLE_ITT
    std::string itt_string = "CCL_ALLGATHERV_SYCL " + std::to_string(send_count);
    __itt_event coll_create_itt_event = ccl::profile::itt::event_get(itt_string.c_str());
    ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl::event e;
    if (send_count * ccl_dtype.size() <= ccl::global_data::env().allgatherv_small_size_threshold) {
        // I don't know what it is
    }
    else {
        e = allgatherv_large(send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
    }
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::event_end(coll_create_itt_event);
#endif //CCL_ENABLE_ITT
    return e;
}

namespace ccl {
namespace v1 {

struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

ccl::event allgather_sycl(sycl::queue q,
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
    ccl::event e;
    done = true;

    uint32_t world = comm.size();
    int rank = comm.rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    ccl::impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    ccl_stream* global_stream = get_stream_ptr(disp(op_stream));
    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = global_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);

    for (uint32_t i = 0; i < recv_counts.size(); i++) {
        if (send_count != recv_counts[i]) {
            LOG_ERROR("Allgatherv only supports the case when all recv_counts are the same");
            done = false;
            return e;
        }
        assert(send_count == recv_counts[i]);
    }

    if (world == 1) {
        sycl::event e_1;
        if (send_buf != recv_buf) {
            e_1 = q.memcpy(recv_buf, send_buf, send_count * ccl_dtype.size());
        }
        return ccl::event::create_from_native(e_1);
    }

    if (send_count * ccl_dtype.size() <= ccl::global_data::env().allgatherv_small_size_threshold &&
        has_all_vertices_connected) {
        init_allgatherv_small(dtype, q, global_comm, global_stream, rank, world);

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
    else if (send_count * ccl_dtype.size() <= ccl::global_data::env().allgatherv_medium_size_threshold &&
             !is_single_tile) {
        init_allgatherv_medium(dtype, q, global_comm, global_stream, rank, world);

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
            return allgatherv_impl(
                send_buf, send_count, recv_buf, recv_counts, dtype, comm, op_stream, attr, deps);
            LOG_DEBUG(
                "|CCL_SYCL| allgatherv selects large kernel: count: ", send_count, " datatype: ", dtype, " done");
        }
        else {
            LOG_DEBUG(
                "[", rank, "] allgatherv selects ccl scheduler send_count: ", send_count, ", datatype: ", dtype);
            return disp(comm)->allgatherv(
                send_buf, send_count, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
            LOG_DEBUG("[",
                      rank,
                      "] allgatherv selects ccl scheduler send_count: ",
                      send_count,
                      ", datatype: ",
                      dtype,
                      " done");
        }
    }
    else {
        done = false;
    }

    return e;
}

} // namespace v1
} // namespace ccl
