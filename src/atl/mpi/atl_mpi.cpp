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
#include "atl_mpi.h"
#include "atl_mpi.c"

atl_status_t atl_mpi::atl_set_env(const atl_attr_t& attr) {
    return atl_mpi_set_env(attr);
}

atl_status_t atl_mpi::atl_init(int* argc,
                               char*** argv,
                               atl_attr_t* attr,
                               const char* main_addr,
                               std::unique_ptr<ipmi>& pmi) {
    (void)pmi;
    inited = true;
    return atl_mpi_init(argc, argv, attr, &ctx, main_addr);
}

atl_status_t atl_mpi::atl_finalize() {
    is_finalized = true;
    return atl_mpi_finalize(ctx);
}

atl_status_t atl_mpi::atl_update(std::unique_ptr<ipmi>& pmi) {
    (void)pmi;
    return ATL_STATUS_UNSUPPORTED;
}

atl_ep_t** atl_mpi::atl_get_eps() {
    return ctx->eps;
}

atl_proc_coord_t* atl_mpi::atl_get_proc_coord() {
    return &(ctx->coord);
}

int atl_mpi::atl_is_resize_enabled() {
    return ctx->is_resize_enabled;
}

atl_status_t atl_mpi::atl_mr_reg(const void* buf, size_t len, atl_mr_t** mr) {
    return atl_mpi_mr_reg(ctx, buf, len, mr);
}

atl_status_t atl_mpi::atl_mr_dereg(atl_mr_t* mr) {
    return atl_mpi_mr_dereg(ctx, mr);
}

atl_status_t atl_mpi::atl_ep_send(atl_ep_t* ep,
                                  const void* buf,
                                  size_t len,
                                  int dst_proc_idx,
                                  uint64_t tag,
                                  atl_req_t* req) {
    return atl_mpi_ep_send(ep, buf, len, dst_proc_idx, tag, req);
}

atl_status_t atl_mpi::atl_ep_recv(atl_ep_t* ep,
                                  void* buf,
                                  size_t len,
                                  int src_proc_idx,
                                  uint64_t tag,
                                  atl_req_t* req) {
    return atl_mpi_ep_recv(ep, buf, len, src_proc_idx, tag, req);
}

atl_status_t atl_mpi::atl_ep_probe(atl_ep_t* ep,
                                   int src_proc_idx,
                                   uint64_t tag,
                                   int* found,
                                   size_t* recv_len) {
    return atl_mpi_ep_probe(ep, src_proc_idx, tag, found, recv_len);
}

atl_status_t atl_mpi::atl_ep_allgatherv(atl_ep_t* ep,
                                        const void* send_buf,
                                        size_t send_len,
                                        void* recv_buf,
                                        const int* recv_lens,
                                        const int* offsets,
                                        atl_req_t* req) {
    return atl_mpi_ep_allgatherv(ep, send_buf, send_len, recv_buf, recv_lens, offsets, req);
}

atl_status_t atl_mpi::atl_ep_allreduce(atl_ep_t* ep,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t len,
                                       atl_datatype_t dtype,
                                       atl_reduction_t op,
                                       atl_req_t* req) {
    return atl_mpi_ep_allreduce(ep, send_buf, recv_buf, len, dtype, op, req);
}

atl_status_t atl_mpi::atl_ep_alltoall(atl_ep_t* ep,
                                      const void* send_buf,
                                      void* recv_buf,
                                      int len,
                                      atl_req_t* req) {
    return atl_mpi_ep_alltoall(ep, send_buf, recv_buf, len, req);
}

atl_status_t atl_mpi::atl_ep_alltoallv(atl_ep_t* ep,
                                       const void* send_buf,
                                       const int* send_lens,
                                       const int* send_offsets,
                                       void* recv_buf,
                                       const int* recv_lens,
                                       const int* recv_offsets,
                                       atl_req_t* req) {
    return atl_mpi_ep_alltoallv(
        ep, send_buf, send_lens, send_offsets, recv_buf, recv_lens, recv_offsets, req);
}

atl_status_t atl_mpi::atl_ep_barrier(atl_ep_t* ep, atl_req_t* req) {
    return atl_mpi_ep_barrier(ep, req);
}

atl_status_t atl_mpi::atl_ep_bcast(atl_ep_t* ep, void* buf, size_t len, int root, atl_req_t* req) {
    return atl_mpi_ep_bcast(ep, buf, len, root, req);
}

atl_status_t atl_mpi::atl_ep_reduce(atl_ep_t* ep,
                                    const void* send_buf,
                                    void* recv_buf,
                                    size_t len,
                                    int root,
                                    atl_datatype_t dtype,
                                    atl_reduction_t op,
                                    atl_req_t* req) {
    return atl_mpi_ep_reduce(ep, send_buf, recv_buf, len, root, dtype, op, req);
}

atl_status_t atl_mpi::atl_ep_reduce_scatter(atl_ep_t* ep,
                                            const void* send_buf,
                                            void* recv_buf,
                                            size_t recv_len,
                                            atl_datatype_t dtype,
                                            atl_reduction_t op,
                                            atl_req_t* req) {
    return atl_mpi_ep_reduce_scatter(ep, send_buf, recv_buf, recv_len, dtype, op, req);
}

atl_status_t atl_mpi::atl_ep_read(atl_ep_t* ep,
                                  void* buf,
                                  size_t len,
                                  atl_mr_t* mr,
                                  uint64_t addr,
                                  uintptr_t remote_key,
                                  int dst_proc_idx,
                                  atl_req_t* req) {
    return atl_mpi_ep_read(ep, buf, len, mr, addr, remote_key, dst_proc_idx, req);
}

atl_status_t atl_mpi::atl_ep_write(atl_ep_t* ep,
                                   const void* buf,
                                   size_t len,
                                   atl_mr_t* mr,
                                   uint64_t addr,
                                   uintptr_t remote_key,
                                   int dst_proc_idx,
                                   atl_req_t* req) {
    return atl_mpi_ep_write(ep, buf, len, mr, addr, remote_key, dst_proc_idx, req);
}

atl_status_t atl_mpi::atl_ep_wait(atl_ep_t* ep, atl_req_t* req) {
    return atl_mpi_ep_wait(ep, req);
}

atl_status_t atl_mpi::atl_ep_wait_all(atl_ep_t* ep, atl_req_t* req, size_t count) {
    return atl_mpi_ep_wait_all(ep, req, count);
}

atl_status_t atl_mpi::atl_ep_cancel(atl_ep_t* ep, atl_req_t* req) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t atl_mpi::atl_ep_poll(atl_ep_t* ep) {
    return atl_mpi_ep_poll(ep);
}

atl_status_t atl_mpi::atl_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) {
    return atl_mpi_ep_check(ep, is_completed, req);
}
atl_mpi::~atl_mpi() {
    if (!is_finalized)
        atl_finalize();
}
