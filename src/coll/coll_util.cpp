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
#include "coll/coll_util.hpp"
#include "coll/selection/selection.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "sched/entry/ze/ze_event_signal_entry.hpp"
#include "sched/entry/ze/ze_event_wait_entry.hpp"
#include "sched/entry/ze/ze_pt2pt_barrier_entry.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

namespace ccl {

void add_coll_entry(ccl_sched* sched, const ccl_coll_param& param) {
    ccl_selector_param selector_param;

    if (param.ctype == ccl_coll_send || param.ctype == ccl_coll_recv) {
        coll_entry::build_sched(sched, param);
    }
    else {
        selector_param.ctype = param.ctype;
        selector_param.count = param.count;
        if (param.ctype == ccl_coll_allgatherv) {
            selector_param.count = param.send_count;
        }
        selector_param.recv_counts =
            const_cast<size_t*>(reinterpret_cast<const size_t*>(param.recv_counts.data()));
        selector_param.dtype = param.dtype;
        selector_param.comm = param.comm;
        selector_param.stream = param.stream;
        selector_param.buf = (param.send_buf) ? param.send_buf.get_ptr() : param.recv_buf.get_ptr();
        selector_param.is_vector_buf = sched->coll_attr.is_vector_buf;
#ifdef CCL_ENABLE_SYCL
        selector_param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
        selector_param.hint_algo = param.hint_algo;
        selector_param.peer_rank = param.peer_rank;
        selector_param.is_scaleout = param.is_scaleout;

#ifdef CCL_ENABLE_SYCL
        if (ccl_is_device_side_algo(selector_param) &&
            (global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::none)) {
            std::string available_ipc_modes{};
            for (auto& ipc_exchange_name : ipc_exchange_names) {
                if (ipc_exchange_name.second == "none") {
                    continue;
                }
                available_ipc_modes += ipc_exchange_name.second + " ";
            }

            CCL_THROW("ERROR: CCL_ZE_IPC_EXCHANGE is set to none, ",
                      "CCL_ZE_IPC_EXCHANGE must be set explicitly: ",
                      available_ipc_modes,
                      ". Hint: OneCCL build may not have support of drmfd");
        }
#endif // CCL_ENABLE_SYCL

        if (ccl_is_device_side_algo(selector_param)) {
            sched->strict_order = true;
        }

        if ((ccl::global_data::env().atl_transport == ccl_atl_mpi) &&
            ccl_is_direct_algo(selector_param)) {
            /* entry directly into schedule due to performance reasons */
            coll_entry::build_sched(sched, param);
        }
        else {
            entry_factory::create<coll_entry>(sched, param);
        }
    }
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)

void add_wait_events(ccl_sched* sched, const std::vector<ze_event_handle_t>& wait_events) {
    if (wait_events.size() > 0) {
        entry_factory::create<ze_event_wait_entry>(sched, wait_events);
        sched->add_barrier();
    }
}

void add_signal_event(ccl_sched* sched, ze_event_handle_t signal_event) {
    if (signal_event) {
        entry_factory::create<ze_event_signal_entry>(sched, signal_event);
        sched->add_barrier();
    }
}

ze_event_handle_t add_signal_event(ccl_sched* sched) {
    auto signal_event = sched->get_memory().event_manager->create();
    add_signal_event(sched, signal_event);
    return signal_event;
}

// This version of add_comm_barrier without events should only be used in this file, as
// it does not deal with ze_events. While developing collectives, use the other version
// of add_comm_barrier instead.
void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      ze_event_pool_handle_t ipc_pool,
                      size_t ipc_event_idx) {
    sched->add_barrier();
    if (ipc_pool && global_data::env().enable_ze_barrier) {
        entry_factory::create<ze_barrier_entry>(sched, comm, ipc_pool, ipc_event_idx);
    }
    else {
        ccl_coll_param barrier_param{};
        barrier_param.ctype = ccl_coll_barrier;
        barrier_param.comm = comm;

        /* TODO: optimize p2p based barrier */
        //barrier_param.hint_algo.barrier = ccl_coll_barrier_ring;

        add_coll_entry(sched, barrier_param);
    }
    sched->add_barrier();
}

void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      const std::vector<ze_event_handle_t>& wait_events,
                      ze_event_handle_t& out_event,
                      ze_event_pool_handle_t ipc_pool,
                      size_t ipc_event_idx) {
    sched->add_barrier();
    out_event = sched->get_memory().event_manager->create();
    if (sched->use_single_list) {
        add_wait_events(sched, wait_events);
    }
    add_comm_barrier(sched, comm, ipc_pool, ipc_event_idx);
    add_signal_event(sched, out_event);
    sched->add_barrier();
}

void add_handle_exchange(ccl_sched* sched,
                         ccl_comm* comm,
                         const std::vector<ze_event_handle_t>& wait_events,
                         ze_event_handle_t& out_event,
                         const std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                         int skip_rank,
                         ze_event_pool_handle_t pool,
                         size_t event_idx,
                         const ccl::utils::pt2pt_handle_exchange_info& info) {
    if (!wait_events.empty()) {
        ccl::add_wait_events(sched, wait_events);
    }
    if (sched->coll_attr.to_cache) {
        sched->set_entry_exec_mode(ccl_sched_entry_exec_once);
        entry_factory::create<ze_handle_exchange_entry>(sched, comm, in_buffers, skip_rank, info);
        sched->add_barrier();
        sched->set_entry_exec_mode(ccl_sched_entry_exec_regular);

        if (sched->coll_param.ctype == ccl_coll_recv || sched->coll_param.ctype == ccl_coll_send) {
            // the entry emulates 'alignment' between send and recv op-s
            // the 'alignment' is needed to avoid situations when send or recv
            // can be faster executed (e.g. initialize the send buffer), not
            // waiting the previous pair of op-s to be finished. it must be as
            // entry because 'alignment' must be in the schedule with other
            // entries. such 'alignment' should be exectuted before other
            // entries that's why sched->add_barrier is used.
            entry_factory::create<ze_pt2pt_barrier_entry>(sched, comm, info.peer_rank);
            sched->add_barrier();
        }
        else {
            // TODO: no need barrier for the first iteration where ze_handle_exchange_entry exists
            add_comm_barrier(sched, comm, {}, out_event, pool, event_idx);
        }
    }
    else {
        entry_factory::create<ze_handle_exchange_entry>(sched, comm, in_buffers, skip_rank, info);
        sched->add_barrier();
        out_event = ccl::add_signal_event(sched);
    }
}

ze_event_handle_t add_coll(ccl_sched* sched,
                           const ccl_coll_param& param,
                           const std::vector<ze_event_handle_t>& wait_events) {
    if (sched->use_single_list) {
        ccl::add_wait_events(sched, wait_events);
    }
    if (ccl::global_data::env().ze_multi_workers) {
        ccl_coll_attr attr{};
        ccl_coll_param coll_param;
        switch (param.ctype) {
            case ccl_coll_allreduce: {
                coll_param = ccl_coll_param::create_allreduce_param(param.send_buf.get_src(),
                                                                    param.recv_buf.get_src(),
                                                                    param.count,
                                                                    param.dtype.idx(),
                                                                    param.reduction,
                                                                    attr,
                                                                    param.comm,
                                                                    param.stream);
                break;
            }
            case ccl_coll_reduce_scatter: {
                coll_param = ccl_coll_param::create_reduce_scatter_param(param.send_buf.get_src(),
                                                                         param.recv_buf.get_src(),
                                                                         param.count,
                                                                         param.dtype.idx(),
                                                                         param.reduction,
                                                                         attr,
                                                                         param.comm,
                                                                         param.stream);
                break;
            }
            case ccl_coll_reduce: {
                coll_param = ccl_coll_param::create_reduce_param(param.send_buf.get_src(),
                                                                 param.recv_buf.get_src(),
                                                                 param.count,
                                                                 param.dtype.idx(),
                                                                 param.reduction,
                                                                 param.root,
                                                                 attr,
                                                                 param.comm,
                                                                 param.stream);
                break;
            }
            case ccl_coll_alltoallv: {
                coll_param = ccl_coll_param::create_alltoallv_param(param.send_buf.get_src(),
                                                                    param.send_counts.data(),
                                                                    param.recv_buf.get_src(),
                                                                    param.recv_counts.data(),
                                                                    param.dtype.idx(),
                                                                    attr,
                                                                    param.comm,
                                                                    param.stream);
                break;
            }
            case ccl_coll_allgatherv: {
                coll_param = ccl_coll_param::create_allgatherv_param(param.send_buf.get_src(),
                                                                     param.send_count,
                                                                     param.recv_buf.get_src(),
                                                                     param.recv_counts.data(),
                                                                     param.dtype.idx(),
                                                                     attr,
                                                                     param.comm,
                                                                     param.stream);
                break;
            }
            default: CCL_THROW("unexpected coll type", ccl_coll_type_to_str(param.ctype));
        }
        LOG_DEBUG("scaleout/multi_workers: created params for: ",
                  ccl_coll_type_to_str(param.ctype),
                  " coll");
        // pass the scale-out selection param through factory
        coll_param.is_scaleout = param.is_scaleout;
        ccl_sched_create_param sched_param(sched->sched_id, coll_param);
        entry_factory::create<subsched_entry>(sched, 0, sched_param, "SCALEOUT");
    }
    else {
        coll_entry::build_sched(sched, param);
    }
    sched->add_barrier();

    auto out_event = ccl::add_signal_event(sched);
    return out_event;
}

ze_event_handle_t add_copy_entry(ccl_buffer src,
                                 ccl_buffer dst,
                                 const size_t count,
                                 const ccl_datatype dtype,
                                 const copy_attr& copy_attr,
                                 ccl_sched* sched,
                                 const std::vector<ze_event_handle_t>& wait_events) {
    LOG_DEBUG("topo/scale_out/intra: use ze_copy_entry");
    auto entry =
        entry_factory::create<ze_copy_entry>(sched, src, dst, count, dtype, copy_attr, wait_events);
    return entry->entry_event;
}

ze_event_handle_t add_copy_entry_with_offset(std::vector<ccl_buffer> bufs,
                                             ccl_buffer buf,
                                             const std::vector<size_t> counts,
                                             ccl_comm* comm,
                                             const ccl_datatype dtype,
                                             const copy_attr& copy_attr,
                                             ccl_sched* sched,
                                             const std::vector<ze_event_handle_t>& wait_events,
                                             bool is_skip_own_rank = false) {
    const size_t counts_size = comm->size();
    CCL_THROW_IF_NOT(bufs.size() == counts_size,
                     "buffers number is different from the number of counts");
    size_t offset = 0;
    std::vector<ze_event_handle_t> out_events;
    // number of not skipped s/r_counts, helps calculate the offset
    for (size_t idx = 0; idx < counts_size; idx++) {
        if (counts[idx] == 0) {
            continue;
        }

        if (!is_skip_own_rank || idx != size_t(comm->rank())) {
            ccl_buffer src = bufs[idx];
            ccl_buffer dst = buf + offset;
            // reverse the function logic (src <=> dst)
            if (copy_attr.direction == copy_direction::h2d) {
                src = buf + offset;
                dst = bufs[idx];
            }
            auto out_event =
                add_copy_entry(src, dst, counts[idx], dtype, copy_attr, sched, wait_events);
            out_events.push_back(out_event);
        }
        offset += counts[idx] * dtype.size();
    }
    LOG_DEBUG("add_copy_entry_with_offset done");

    add_wait_events(sched, out_events);
    return add_signal_event(sched);
}

ze_event_handle_t fill_scaleout_coll_param(const ccl_coll_param& in_coll_param,
                                           ccl_coll_param& out_coll_param,
                                           ccl_sched* sched,
                                           const std::vector<ze_event_handle_t>& wait_events) {
    ze_event_handle_t out_event{};
    // general case
    size_t host_buf_size = 0;
    // alltoallv case
    size_t a2av_send_bytes = 0;
    size_t a2av_recv_bytes = 0;
    // for simplicity
    ccl_coll_type ctype = out_coll_param.ctype;
    size_t counts_size = out_coll_param.comm->size();
    size_t dtype_size = out_coll_param.dtype.size();

    // calculate counts
    if (ctype == ccl_coll_alltoallv) {
        size_t a2av_send_count = std::accumulate(out_coll_param.send_counts.begin(),
                                                 out_coll_param.send_counts.end(),
                                                 ccl::utils::initial_count_value);
        size_t a2av_recv_count = std::accumulate(out_coll_param.recv_counts.begin(),
                                                 out_coll_param.recv_counts.end(),
                                                 ccl::utils::initial_count_value);
        a2av_send_bytes = a2av_send_count * dtype_size;
        a2av_recv_bytes = a2av_recv_count * dtype_size;
    }
    else if (ctype == ccl_coll_alltoall || ctype == ccl_coll_allgatherv) {
        // assume sum of send_counts and recv_counts are equal
        host_buf_size = std::accumulate(out_coll_param.recv_counts.begin(),
                                        out_coll_param.recv_counts.end(),
                                        ccl::utils::initial_count_value) *
                        dtype_size;
        LOG_DEBUG("alltoall(v)/allgatherv scale_out host buf size: ", host_buf_size);
    }
    else if (ctype == ccl_coll_reduce_scatter) {
        host_buf_size = out_coll_param.count * counts_size * dtype_size;
    }
    else {
        host_buf_size = out_coll_param.count * dtype_size;
    }

    // allocate receive and send (in out-of-place case) buffers
    if (ctype == ccl_coll_alltoallv) {
        ccl::alloc_param a2av_send_alloc_param(
            a2av_send_bytes, ccl::buffer_type::regular, ccl::buffer_place::host);
        ccl::alloc_param a2av_recv_alloc_param(
            a2av_recv_bytes, ccl::buffer_type::regular, ccl::buffer_place::host);
        out_coll_param.send_buf = sched->alloc_buffer(a2av_send_alloc_param);
        out_coll_param.recv_buf = sched->alloc_buffer(a2av_recv_alloc_param);
    }
    else {
        CCL_THROW_IF_NOT(host_buf_size != invalid_host_buf_size,
                         "unexpected the size of buffer in scaleout phase");
        ccl::alloc_param alloc_param(
            host_buf_size, ccl::buffer_type::regular, ccl::buffer_place::host);
        out_coll_param.send_buf = sched->alloc_buffer(alloc_param);
        out_coll_param.recv_buf = out_coll_param.send_buf;
    }

    // transform array of buffers in contiguous buffer with offsets
    if (ctype == ccl_coll_alltoallv) {
        auto send_out_event = add_copy_entry_with_offset(in_coll_param.send_scale_out_bufs,
                                                         out_coll_param.send_buf,
                                                         out_coll_param.send_counts,
                                                         out_coll_param.comm,
                                                         out_coll_param.dtype,
                                                         copy_attr(copy_direction::d2h),
                                                         sched,
                                                         wait_events);
        out_event = add_copy_entry_with_offset(in_coll_param.recv_scale_out_bufs,
                                               out_coll_param.recv_buf,
                                               out_coll_param.recv_counts,
                                               out_coll_param.comm,
                                               out_coll_param.dtype,
                                               copy_attr(copy_direction::d2h),
                                               sched,
                                               std::vector<ze_event_handle_t>{ send_out_event });
    }
    else if (ctype == ccl_coll_alltoall) {
        out_event = add_copy_entry_with_offset(in_coll_param.send_scale_out_bufs,
                                               out_coll_param.send_buf,
                                               out_coll_param.send_counts,
                                               out_coll_param.comm,
                                               out_coll_param.dtype,
                                               copy_attr(copy_direction::d2h),
                                               sched,
                                               wait_events);
    }
    else if (ctype == ccl_coll_allgatherv) {
        size_t offset =
            std::accumulate(out_coll_param.recv_counts.begin(),
                            out_coll_param.recv_counts.begin() + out_coll_param.comm->rank(),
                            ccl::utils::initial_count_value) *
            dtype_size;
        out_event = add_copy_entry(in_coll_param.send_buf,
                                   out_coll_param.send_buf + offset,
                                   out_coll_param.send_count,
                                   out_coll_param.dtype,
                                   copy_attr(copy_direction::d2h),
                                   sched,
                                   wait_events);
    }
    else if (ctype == ccl_coll_reduce_scatter) {
        out_event = add_copy_entry(in_coll_param.send_buf,
                                   out_coll_param.send_buf,
                                   out_coll_param.count * counts_size,
                                   out_coll_param.dtype,
                                   copy_attr(copy_direction::d2h),
                                   sched,
                                   wait_events);
    }
    else {
        out_event = add_copy_entry(in_coll_param.send_buf,
                                   out_coll_param.send_buf,
                                   out_coll_param.count,
                                   out_coll_param.dtype,
                                   copy_attr(copy_direction::d2h),
                                   sched,
                                   wait_events);
    }
    return out_event;
}

void add_scaleout(ccl_sched* sched,
                  const ccl_coll_param& in_coll_param,
                  const bool is_single_node,
                  const std::vector<ze_event_handle_t>& in_wait_events,
                  ze_event_handle_t& out_event,
                  const copy_attr& h2d_copy_attr,
                  ccl_comm* global_comm,
                  ccl_buffer global_recv_buf,
                  int global_root) {
    std::vector<ze_event_handle_t> wait_events{ in_wait_events };
    ccl_coll_param coll_param(in_coll_param);
    out_event = nullptr;

    bool multi_node =
        (!is_single_node && (coll_param.count != 0 || coll_param.recv_counts.size() != 0));
    bool enable_hmem = (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);
    bool do_h2d_copy =
        ((coll_param.ctype == ccl_coll_allreduce || coll_param.ctype == ccl_coll_reduce_scatter ||
          coll_param.ctype == ccl_coll_alltoallv || coll_param.ctype == ccl_coll_alltoall ||
          coll_param.ctype == ccl_coll_allgatherv) &&
         multi_node && !enable_hmem) ||
        (coll_param.ctype == ccl_coll_reduce && coll_param.comm->rank() == coll_param.root);

    if (multi_node) {
        if (!enable_hmem) {
            LOG_DEBUG("topo/scale_out: use host_", ccl_coll_type_to_str(coll_param.ctype));

            // mostly initialize contiguous send/recv buffers from array of input buffers
            out_event = fill_scaleout_coll_param(in_coll_param, coll_param, sched, wait_events);
            utils::clear_and_push_back(wait_events, out_event);
            sched->add_barrier();

            LOG_DEBUG("topo/scale_out: ze_copy_entry of D2H for ",
                      ccl_coll_type_to_str(coll_param.ctype),
                      " done");
        }
        // pass the scale-out selection param directly
        coll_param.is_scaleout = true;
        // do inplace collective

        out_event = ccl::add_coll(sched, coll_param, wait_events);
        utils::clear_and_push_back(wait_events, out_event);
    }

    if (!do_h2d_copy)
        return;

    ccl_buffer src_copy_buf = coll_param.recv_buf;
    ccl_buffer dst_copy_buf = in_coll_param.recv_buf;

    if (in_coll_param.ctype == ccl_coll_reduce) {
        if (!multi_node)
            src_copy_buf = in_coll_param.recv_buf;
        dst_copy_buf = (global_comm->rank() == global_root) ? global_recv_buf : ccl_buffer();
    }

    if (coll_param.ctype == ccl_coll_alltoallv || coll_param.ctype == ccl_coll_alltoall ||
        coll_param.ctype == ccl_coll_allgatherv) {
        out_event = add_copy_entry_with_offset(in_coll_param.recv_scale_out_bufs,
                                               coll_param.recv_buf,
                                               coll_param.recv_counts,
                                               coll_param.comm,
                                               coll_param.dtype,
                                               h2d_copy_attr,
                                               sched,
                                               wait_events,
                                               coll_param.ctype == ccl_coll_allgatherv);
    }
    else {
        out_event = add_copy_entry(src_copy_buf,
                                   dst_copy_buf,
                                   coll_param.count,
                                   coll_param.dtype,
                                   h2d_copy_attr,
                                   sched,
                                   wait_events);
    }
    sched->add_barrier();

    LOG_DEBUG("topo/scale_out: ze_copy_entry of H2D for ",
              ccl_coll_type_to_str(coll_param.ctype),
              " done");
}

bool is_queue_in_order(const ccl_stream* s) {
    return s != nullptr && s->is_sycl_device_stream() && s->get_native_stream().is_in_order();
}

void enable_sycl_output_barrier_in_order_queue(const ccl_stream* s) {
    LOG_DEBUG("CCL_SYCL_OUTPUT_EVENT: ", ccl::global_data::env().enable_sycl_output_event);
    if (is_queue_in_order(s)) {
        ccl::global_data::env().enable_sycl_output_event = 1;
    }
    LOG_DEBUG("CCL_SYCL_OUTPUT_EVENT is set to 1");
}

#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

} // namespace ccl
