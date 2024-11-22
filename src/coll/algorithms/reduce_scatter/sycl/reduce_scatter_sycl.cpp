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
//#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_ring.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

ccl::event reduce_scatter_sycl_single_node(sycl::queue& q,
                                           const void* send_buf,
                                           void* recv_buf,
                                           size_t recv_count,
                                           datatype dtype,
                                           reduction reduction,
                                           ccl_comm* comm,
                                           ccl_stream* global_stream,
                                           const vector_class<event>& deps,
                                           bool& done) {
    ccl::event e;
    done = true;

    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: reduce_scatter");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, recv_count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: reduce_scatter");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);

    if (!ccl::global_data::env().sycl_esimd) {
        if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().sycl_reduce_scatter_small_threshold) {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "reduce_scatter_small", "send_size", recv_count * world * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking small reduce_scatter: recv_count:", recv_count, " datatype: ", dtype);
            e = reduce_scatter_small(send_buf, recv_buf, recv_count, dtype, reduction, comm, global_stream, deps);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }
        else {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "reduce_scatter_large", "send_size", recv_count * world * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking large reduce_scatter: recv_count:", recv_count, " datatype: ", dtype);
            e = reduce_scatter_large(send_buf, recv_buf, recv_count, dtype, reduction, comm, global_stream, deps);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        }

        return e;
    }

    // ESIMD
    if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().sycl_reduce_scatter_small_threshold &&
        has_all_vertices_connected) {
        init_reduce_scatter_small(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("reduce_scatter_small", "send_size", recv_count * world * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects small kernel, recv_count:", recv_count, " datatype: ", dtype);
        e = run_reduce_scatter_small(dtype, q, send_buf, recv_buf, recv_count, done);
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects small kernel, recv_count:",
                  recv_count,
                  " datatype: ",
                  dtype,
                  "done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done)
            return e;
    }
    if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().sycl_reduce_scatter_medium_threshold &&
        !is_single_tile) {
        init_reduce_scatter_medium(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("reduce_scatter_medium", "send_size", recv_count * world * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects medium kernel: count:", recv_count, " datatype: ", dtype);
        e = run_reduce_scatter_medium(dtype, q, send_buf, recv_buf, recv_count, done);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else if (!is_single_tile) {
        init_reduce_scatter_large(dtype, q, comm, global_stream, rank, world);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("reduce_scatter_large", "send_size", recv_count * world * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects large kernel: count:", recv_count, " datatype: ", dtype);
        e = run_reduce_scatter_large(dtype, q, send_buf, recv_buf, recv_count, done);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
        done = false;
    }
    return e;
}

// calculate the location to copy from for a given index
static inline int pred_loc(int idx, int N, int M) {
    return idx == M * N - 1 ? idx : (M * idx) % (M * N - 1);
}

template <typename T>
inline void copy_kernel(void* out, const void* in) {
    *(T*)out = *(T*)in;
}

template <typename T, int vec_size>
void inline copy_vec(const void* in, void* out, int out_idx, int b_idx, int in_idx, int recv_count) {
    using AT = sycl::vec<T, vec_size>;
    const size_t packed_count = recv_count / vec_size;
    if (b_idx < packed_count) {
        int offset = b_idx * vec_size;
        T* in_ptr = (T*)in + in_idx * recv_count + offset;
        T* out_ptr = (T*)out + out_idx * recv_count + offset;
        copy_kernel<AT>(out_ptr, in_ptr);
    }
    else {
        int offset = vec_size * packed_count + (b_idx - packed_count);
        T* in_ptr = (T*)in + in_idx * recv_count + offset;
        T* out_ptr = (T*)out + out_idx * recv_count + offset;
        if (offset < recv_count) {
            *out_ptr = *in_ptr;
        }
    }
}

// single kernel with sycl::vec
template <typename T, int vec_size, int SGS>
sycl::event transposeT(sycl::queue& q,
                       const void* send_buf,
                       void* out_buf,
                       size_t recv_count,
                       datatype dtype,
                       int nodes,
                       int ppn,
                       std::vector<sycl::event>& dep_events) {
    sycl::event e;
    constexpr int wg_size = SGS;
    constexpr int sg_size = SGS;
    size_t vec_count = recv_count / vec_size + recv_count % vec_size;
    int kernel_threads = vec_count * nodes * ppn;
    int kernel_size = (kernel_threads + wg_size - 1) / wg_size * wg_size;
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.parallel_for(sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
            const size_t idx = it.get_global_linear_id();
            if (idx >= kernel_threads)
                return;
            int a = idx / vec_count;
            int b = idx % vec_count;
            int a_from = pred_loc(a, nodes, ppn);
            copy_vec<T, vec_size>(send_buf, out_buf, a, b, a_from, recv_count);
        });
    });
    return e;
}

// multiple sycl kernel version
static sycl::event transpose(sycl::queue& q,
                             const void* send_buf,
                             void* out_buf,
                             size_t recv_count,
                             datatype dtype,
                             int nodes,
                             int ppn,
                             std::vector<sycl::event>& dep_events) {
    sycl::event e;

    // nothing to do
    if (nodes == 1 || ppn == 1) {
        return e;
    }

/* almost same performance */
#define USE_OOO_QUEUE 0
#if USE_OOO_QUEUE
    // out of order queue
    static sycl::queue q_worker(q.get_device());
    std::vector<sycl::event> evs;
#else
    sycl::queue q_worker = q;
#endif // USE_OOO_QUEUE

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t chunk_size = recv_count * ccl_dtype.size();
    int n = 0;
    for (int i = 0; i < ppn; i++) {
        for (int j = 0; j < nodes; j++) {
            char* src = (char*)send_buf + (j * ppn + i) * chunk_size;
            char* dest = (char*)out_buf + n * chunk_size;
            n++;
            e = q_worker.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(dest, src, chunk_size);
            });
#if USE_OOO_QUEUE
            evs.push_back(std::move(e));
#endif // USE_OOO_QUEUE
        }
    }
#if USE_OOO_QUEUE
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(evs);
        h.single_task([]() {});
    });
#endif // USE_OOO_QUEUE
    return e;
}

static sycl::event rearrange(sycl::queue& q,
                             const void* send_buf,
                             void* staging_buf,
                             size_t recv_count,
                             datatype dtype,
                             int nodes,
                             int ppn,
                             std::vector<sycl::event>& dep_events) {
    sycl::event e;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t total_size = recv_count * ccl_dtype.size() * nodes * ppn;

    if (total_size <= 536870912) {
        // single sycl kernel
        int align4 = ccl_dtype.size() >= 4 || is_aligned(send_buf, recv_count * ccl_dtype.size(), 4);
        auto lambda = [&]<typename T>() {
            if (recv_count <= 65536) {
                // can not use full vector size (8) if not 4-byte aligned
                if (align4) {
                    constexpr int vec_size = 8 / sizeof(T);
                    return transposeT<T, vec_size, 8>(
                        q, send_buf, staging_buf, recv_count, dtype, nodes, ppn, dep_events);
                }
                else {
                    constexpr int vec_size = 4 / sizeof(T);
                    return transposeT<T, vec_size, 32>(
                        q, send_buf, staging_buf, recv_count, dtype, nodes, ppn, dep_events);
                }
            }
            else {
                if (align4) {
                    constexpr int vec_size = 8 / sizeof(T);
                    return transposeT<T, vec_size, 32>(
                        q, send_buf, staging_buf, recv_count, dtype, nodes, ppn, dep_events);
                }
                else {
                    constexpr int vec_size = 4 / sizeof(T);
                    return transposeT<T, vec_size, 32>(
                        q, send_buf, staging_buf, recv_count, dtype, nodes, ppn, dep_events);
                }
            }
        };
        e = invoke_scaleout(lambda, dtype);
    }
    else {
        // multiple sycl kernels
        e = transpose(q, send_buf, staging_buf, recv_count, dtype, nodes, ppn, dep_events);
    }
    return e;
}

//#define PRINT_TIMING

ccl::event reduce_scatter_sycl_multi_node(sycl::queue& q,
                                          const void* send_buf,
                                          void* recv_buf,
                                          size_t recv_count,
                                          datatype dtype,
                                          reduction reduction,
                                          ccl_comm* comm,
                                          ccl_stream* global_stream,
                                          const vector_class<event>& deps,
                                          bool& done) {
    event ev;
    sycl::event e;
    done = true;

    int rank = comm->rank();
    int world = comm->size();
    std::shared_ptr<ccl_comm> r2r_comm = comm->get_r2r_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    if (r2r_comm->size() == 1) {
        LOG_DEBUG("reduce_scatter calls single node");
        return reduce_scatter_sycl_single_node(
            q, send_buf, recv_buf, recv_count, dtype, reduction, comm, global_stream, deps, done);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t total_size = recv_count * ccl_dtype.size() * world;

    // check threshold to disable scaleout algorithm
    if (recv_count * ccl_dtype.size() > ccl::global_data::env().sycl_reduce_scatter_scaleout_threshold) {
        // fallback
        done = false;
        return ev;
    }

    // only do scale-out or small message sizes
    if (node_comm->size() == 1 || (comm->size() < 4 && total_size <= 131072)) {
        // for small scale, call MPI directly
        bool direct = false;
        if ((r2r_comm->size() < 4 && recv_count * ccl_dtype.size() * r2r_comm->size() <= 131072) &&
            ccl::global_data::env().atl_transport != ccl_atl_ofi) {
            direct = true;
        }
        ev = reduce_scatter_scaleout_sycl(
            q, send_buf, recv_buf, recv_count, dtype, reduction, comm, deps, done, direct);
        return ev;
    }

    bool __attribute__((unused)) in_place = (recv_buf == (char*)send_buf + recv_count * rank * ccl_dtype.size());
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

#ifdef PRINT_TIMING
    cpu_timer<1> ctimer;
    ctimer.start(0);
#endif // PRINT_TIMING

    // rearrange data to a staging buffer of same size
    void* staging_buf;
    int to_free = 0;
    if (total_size <= comm->get_scaleout_device_buf_size()) {
        staging_buf = comm->get_scaleout_device_buf(q);
    }
    else {
        staging_buf = sycl::malloc_device(total_size, q);
        to_free = 1;
    }

    e = rearrange(q, send_buf, staging_buf, recv_count, dtype, r2r_comm->size(), node_comm->size(), dep_events);

#ifdef PRINT_TIMING
    e.wait();
    q.wait(); // for multiple kernels with out-of-order queue
    ctimer.stop(0);
    fprintf(stderr, "[%d] rearrange takes %f us on data: %ld \n", rank, ctimer.get_us(0), recv_count * world);

    ctimer.start(0);
#endif // PRINT_TIMING

    // scale up on each node
    std::vector<event> evs;
    ev = ccl::event::create_from_native(e);
    evs.push_back(std::move(ev));
    size_t scaleup_recv_count = recv_count * r2r_comm->size();
    // doing scale up in place might not be good if not 4-byte aligned
    constexpr bool do_inplace = false;
    void* scaleup_buf =
        do_inplace ? (char*)staging_buf + scaleup_recv_count * node_comm->rank() * ccl_dtype.size() : staging_buf;
    ev = reduce_scatter_sycl_single_node(q,
                                         staging_buf,
                                         scaleup_buf,
                                         scaleup_recv_count,
                                         dtype,
                                         reduction,
                                         node_comm.get(),
                                         global_stream,
                                         evs,
                                         done);
    if (!done) {
        // fallback
        LOG_INFO("allreduce_sycl allgatherv was not done -- falling back");
        return ev;
    }

#ifdef PRINT_TIMING
    ev.wait();
    ctimer.stop(0);
    fprintf(stderr, "[%d] scale up takes %f us on %ld\n", rank, ctimer.get_us(0), recv_count * r2r_comm->size());

    ctimer.start(0);
#endif // PRINT_TIMING

    // scale out
    evs.clear();
    evs.push_back(std::move(ev));
    bool direct = false;
    if (recv_count * ccl_dtype.size() * r2r_comm->size() <=
            ccl::global_data::env().sycl_reduce_scatter_scaleout_direct_threshold &&
        ccl::global_data::env().atl_transport != ccl_atl_ofi) {
        direct = true;
    }
    ev = reduce_scatter_scaleout_sycl(
        q, scaleup_buf, recv_buf, recv_count, dtype, reduction, r2r_comm.get(), evs, done, direct);

#ifdef PRINT_TIMING
    ev.wait();
    ctimer.stop(0);
    fprintf(stderr,
            "[%d] scale out takes %f us on %ld count node %d\n",
            rank,
            ctimer.get_us(0),
            recv_count,
            r2r_comm->size());
#endif // PRINT_TIMING

    if (to_free) {
        e = q.submit([=](sycl::handler& h) {
            h.host_task([=]() {
                sycl::free(staging_buf, q);
            });
        });
        ev = ccl::event::create_from_native(e);
    }
    else {
        comm->put_scaleout_device_buf(staging_buf);
    }

    return ev;
}

ccl::event reduce_scatter_sycl(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               reduction reduction,
                               ccl_comm* comm,
                               ccl_stream* op_stream,
                               const reduce_scatter_attr& attr,
                               const vector_class<event>& deps,
                               bool& done) {
    if (recv_count == 0) {
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

    // ESIMD scale-up does not support sub-communicators
    if (ccl::global_data::env().sycl_esimd) {
        int ppn = is_single_node ? comm->size() : comm->get_node_comm()->size();
        if (ppn != ccl::global_data::get().get_local_proc_count() && ppn != 1) {
            done = false;
            ccl::event ev;
            return ev;
        }
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("reduce_scatter is_single_node");
        return reduce_scatter_sycl_single_node(
            q, send_buf, recv_buf, recv_count, dtype, reduction, comm, op_stream, deps, done);
    }

    CCL_THROW_IF_NOT(q.is_in_order(), "SYCL queue must be in-order");

    return reduce_scatter_sycl_multi_node(
        q, send_buf, recv_buf, recv_count, dtype, reduction, comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl
