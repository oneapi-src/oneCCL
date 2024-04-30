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

namespace ccl {
namespace v1 {

struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

event allreduce_sycl(sycl::queue q,
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
    ccl::event e;
    done = true;

    uint32_t world = comm.size();
    int rank = comm.rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (world == 1) {
        sycl::event sycl_e;
        if (send_buf != recv_buf) {
            sycl_e = q.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
        }
        return ccl::event::create_from_native(sycl_e);
    }

    ccl::impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    ccl_stream* global_stream = get_stream_ptr(disp(op_stream));
    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected =
        global_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| is_single_tile: ",
              is_single_tile,
              ", has_all_vertices_connected: ",
              has_all_vertices_connected);

    if (count * ccl_dtype.size() <= ccl::global_data::env().allreduce_small_size_threshold &&
        has_all_vertices_connected) {
        init_allreduce_small(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_SMALL");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_small(dtype, q, send_buf, recv_buf, count);
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
    }
    else if ((count * ccl_dtype.size() <= ccl::global_data::env().allreduce_medium_size_threshold ||
              (global_comm->size() == 2 && !ccl::global_data::env().allreduce_use_tmp_buf)) &&
             !is_single_tile) { // medium message sizes
        init_allreduce_medium(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_ALLREDUCE_MEDIUM");
        ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        LOG_DEBUG(
            "|CCL_SYCL| allreduce selects medium kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_medium(dtype, q, send_buf, recv_buf, count);
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
        e = run_allreduce_large(dtype, q, send_buf, recv_buf, count);
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

} // namespace v1
} // namespace ccl
