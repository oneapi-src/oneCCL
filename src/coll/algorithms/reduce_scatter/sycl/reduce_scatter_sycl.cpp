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
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

ccl::event reduce_scatter_sycl(sycl::queue q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               reduction reduction,
                               const ccl::communicator& comm,
                               const stream& op_stream,
                               bool& done) {
    ccl::event e;
    done = true;

    uint32_t world = comm.size();
    int rank = comm.rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (world == 1) {
        sycl::event sycl_e;
        if (send_buf != recv_buf) {
            sycl_e = q.memcpy(recv_buf, send_buf, recv_count * ccl_dtype.size());
        }
        return ccl::event::create_from_native(sycl_e);
    }

    ccl::impl_dispatch disp;
    std::shared_ptr<ccl::comm_interface> disp_comm = disp(comm);
    ccl_comm* global_comm = (ccl_comm*)(disp_comm.get());
    ccl_stream* global_stream = get_stream_ptr(disp(op_stream));

    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = global_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);

    if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().reduce_scatter_small_size_threshold &&
        has_all_vertices_connected) {
        if ((recv_count * ccl_dtype.size()) % 4 == 0 || recv_count * ccl_dtype.size() == 2) {
            init_reduce_scatter_small(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_REDUCE_SCATTER_SMALL");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG(
                "|CCL_SYCL| reduce_scatter selects small kernel, recv_count:", recv_count, " datatype: ", dtype);
            e = run_reduce_scatter_small(dtype, q, send_buf, recv_buf, recv_count, done);
            LOG_DEBUG("|CCL_SYCL| reduce_scatter selects small kernel, recv_count:",
                      recv_count,
                      " datatype: ",
                      dtype,
                      "done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }
        else {
            done = false;
        }
    }
    else if (recv_count * world * ccl_dtype.size() <=
                 ccl::global_data::env().reduce_scatter_medium_size_threshold &&
             !is_single_tile) {
        if ((recv_count * ccl_dtype.size()) % 4 == 0) {
            init_reduce_scatter_medium(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_REDUCE_SCATTER_MEDIUM");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| reduce_scatter selects medium kernel: count:", recv_count, " datatype: ", dtype);
            e = run_reduce_scatter_medium(dtype, q, send_buf, recv_buf, recv_count, done);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }
        else {
            done = false;
        }
    }
    else if (!is_single_tile) {
        if ((recv_count * ccl_dtype.size()) % 4 == 0) {
            init_reduce_scatter_large(dtype, q, global_comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
            __itt_event coll_create_itt_event = ccl::profile::itt::event_get("CCL_REDUCE_SCATTER_LARGE");
            ccl::profile::itt::event_start(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| reduce_scatter selects large kernel: count:", recv_count, " datatype: ", dtype);
            e = run_reduce_scatter_large(dtype, q, send_buf, recv_buf, recv_count, done);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::event_end(coll_create_itt_event);
#endif // CCL_ENABLE_ITT
        }
        else {
            done = false;
        }
    }
    else {
        done = false;
    }
    return e;
}

} // namespace v1
} // namespace ccl
