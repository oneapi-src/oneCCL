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
#pragma once

#include <memory>
#include <mutex>
#include <list>
#include <vector>

#include "atl.h"
#include "common/comm/atl_tag.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"

class ccl_executor;

class atl_wrapper {
public:
    static void set_internal_env(const atl_attr_t& attr);
    static void set_exec(ccl_executor* exec);

    ~atl_wrapper();
    atl_wrapper();
    atl_wrapper(std::shared_ptr<ikvs_wrapper> k);
    atl_wrapper(int total_rank_count,
                const std::vector<int>& ranks,
                std::shared_ptr<ikvs_wrapper> k);

    atl_status_t atl_main_addr_reserve(char* main_addr) {
        if (!pmi)
            return ATL_STATUS_UNSUPPORTED;

        return pmi->pmrt_main_addr_reserve(main_addr);
        ;
    }

    atl_status_t atl_finalize() {
        if (pmi)
            pmi->pmrt_finalize();

        return transport->atl_finalize();
    }

    atl_status_t atl_update() {
        return transport->atl_update(pmi);
    }

    atl_status_t atl_wait_notification() {
        if (!pmi)
            return ATL_STATUS_UNSUPPORTED;

        return pmi->pmrt_wait_notification();
    }

    atl_status_t atl_set_resize_function(atl_resize_fn_t fn) {
        if (!pmi)
            return ATL_STATUS_UNSUPPORTED;

        return pmi->pmrt_set_resize_function(fn);
    }

    atl_proc_coord_t* atl_get_proc_coord() {
        return transport->atl_get_proc_coord();
    }

    atl_status_t atl_mr_reg(const void* buf, size_t len, atl_mr_t** mr) {
        return transport->atl_mr_reg(buf, len, mr);
    }

    atl_status_t atl_mr_dereg(atl_mr_t* mr) {
        return transport->atl_mr_dereg(mr);
    }

    atl_status_t atl_ep_send(size_t ep_idx,
                             const void* buf,
                             size_t len,
                             int dst_proc_idx,
                             uint64_t tag,
                             atl_req_t* req) {
        return transport->atl_ep_send(eps[ep_idx], buf, len, dst_proc_idx, tag, req);
    }

    atl_status_t atl_ep_recv(size_t ep_idx,
                             void* buf,
                             size_t len,
                             int src_proc_idx,
                             uint64_t tag,
                             atl_req_t* req) {
        return transport->atl_ep_recv(eps[ep_idx], buf, len, src_proc_idx, tag, req);
    }

    atl_status_t atl_ep_probe(size_t ep_idx,
                              int src_proc_idx,
                              uint64_t tag,
                              int* found,
                              size_t* recv_len) {
        return transport->atl_ep_probe(eps[ep_idx], src_proc_idx, tag, found, recv_len);
    }

    atl_status_t atl_ep_allgatherv(size_t ep_idx,
                                   const void* send_buf,
                                   size_t send_len,
                                   void* recv_buf,
                                   const int* recv_lens,
                                   const int* offsets,
                                   atl_req_t* req) {
        return transport->atl_ep_allgatherv(
            eps[ep_idx], send_buf, send_len, recv_buf, recv_lens, offsets, req);
    }

    atl_status_t atl_ep_allreduce(size_t ep_idx,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t len,
                                  atl_datatype_t dtype,
                                  atl_reduction_t op,
                                  atl_req_t* req) {
        return transport->atl_ep_allreduce(eps[ep_idx], send_buf, recv_buf, len, dtype, op, req);
    }

    atl_status_t atl_ep_alltoall(size_t ep_idx,
                                 const void* send_buf,
                                 void* recv_buf,
                                 int len,
                                 atl_req_t* req) {
        return transport->atl_ep_alltoall(eps[ep_idx], send_buf, recv_buf, len, req);
    }

    atl_status_t atl_ep_alltoallv(size_t ep_idx,
                                  const void* send_buf,
                                  const int* send_lens,
                                  const int* send_offsets,
                                  void* recv_buf,
                                  const int* recv_lens,
                                  const int* recv_offsets,
                                  atl_req_t* req) {
        return transport->atl_ep_alltoallv(
            eps[ep_idx], send_buf, send_lens, send_offsets, recv_buf, recv_lens, recv_offsets, req);
    }

    atl_status_t atl_ep_barrier(size_t ep_idx, atl_req_t* req) {
        return transport->atl_ep_barrier(eps[ep_idx], req);
    }

    atl_status_t atl_ep_bcast(size_t ep_idx, void* buf, size_t len, int root, atl_req_t* req) {
        return transport->atl_ep_bcast(eps[ep_idx], buf, len, root, req);
    }

    atl_status_t atl_ep_reduce(size_t ep_idx,
                               const void* send_buf,
                               void* recv_buf,
                               size_t len,
                               int root,
                               atl_datatype_t dtype,
                               atl_reduction_t op,
                               atl_req_t* req) {
        return transport->atl_ep_reduce(eps[ep_idx], send_buf, recv_buf, len, root, dtype, op, req);
    }

    atl_status_t atl_ep_reduce_scatter(size_t ep_idx,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t recv_len,
                                       atl_datatype_t dtype,
                                       atl_reduction_t op,
                                       atl_req_t* req) {
        return transport->atl_ep_reduce_scatter(
            eps[ep_idx], send_buf, recv_buf, recv_len, dtype, op, req);
    }

    atl_status_t atl_ep_read(size_t ep_idx,
                             void* buf,
                             size_t len,
                             atl_mr_t* mr,
                             uint64_t addr,
                             uintptr_t remote_key,
                             int dst_proc_idx,
                             atl_req_t* req) {
        return transport->atl_ep_read(
            eps[ep_idx], buf, len, mr, addr, remote_key, dst_proc_idx, req);
    }

    atl_status_t atl_ep_write(size_t ep_idx,
                              const void* buf,
                              size_t len,
                              atl_mr_t* mr,
                              uint64_t addr,
                              uintptr_t remote_key,
                              int dst_proc_idx,
                              atl_req_t* req) {
        return transport->atl_ep_write(
            eps[ep_idx], buf, len, mr, addr, remote_key, dst_proc_idx, req);
    }

    atl_status_t atl_ep_wait(size_t ep_idx, atl_req_t* req) {
        return transport->atl_ep_wait(eps[ep_idx], req);
    }

    atl_status_t atl_ep_wait_all(size_t ep_idx, atl_req_t* req, size_t count) {
        return transport->atl_ep_wait_all(eps[ep_idx], req, count);
    }

    atl_status_t atl_ep_cancel(size_t ep_idx, atl_req_t* req) {
        return transport->atl_ep_cancel(eps[ep_idx], req);
    }

    atl_status_t atl_ep_poll(size_t ep_idx) {
        return transport->atl_ep_poll(eps[ep_idx]);
    }

    atl_status_t atl_ep_check(size_t ep_idx, int* is_completed, atl_req_t* req) {
        return transport->atl_ep_check(eps[ep_idx], is_completed, req);
    }

    size_t get_threads_per_process() {
        return threads_per_process;
    }

    size_t get_ranks_per_process() {
        return ranks_per_process;
    }

    int get_rank() {
        return rank;
    }

    int get_size() {
        return size;
    }

    /*
     * TODO: Temporary change.
     * Need to define correct to unique id
     */
    size_t get_id() {
        return 0;
    }

    /* static ATL attr for all transport instances
       actual values generated by executor */
    static atl_attr_t attr;

    std::unique_ptr<ccl_atl_tag> tag;

private:
    int rank;
    int size;

    size_t threads_per_process;
    size_t ranks_per_process;

    std::shared_ptr<iatl> transport;
    std::unique_ptr<ipmi> pmi;
    atl_ep_t** eps = nullptr;

    void init_transport();
};
