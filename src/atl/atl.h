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
#include <stddef.h>
#include <stdint.h>
#include <memory>

#include "atl_def.h"
#include "util/pm/pm_rt.h"

/*
atl_status_t atl_init(const char* transport_name,
                      int* argc,
                      char*** argv,
                      atl_attr_t* att,
                      atl_ctx_t** ctx,
                      const char* main_addr);

void atl_main_addr_reserv(char* main_addr);

static inline atl_status_t atl_finalize(atl_ctx_t* ctx) {
    return ctx->ops->finalize(ctx);
}

static inline atl_status_t atl_update(atl_ctx_t* ctx) {
    return ctx->ops->update(ctx);
}

static inline atl_status_t atl_wait_notification(atl_ctx_t* ctx) {
    return ctx->ops->wait_notification(ctx);
}

static inline atl_status_t atl_set_resize_function(atl_ctx_t* ctx, atl_resize_fn_t fn) {
    return ctx->ops->set_resize_function(fn);
}

static inline atl_ep_t** atl_get_eps(atl_ctx_t* ctx) {
    return ctx->eps;
}

static inline atl_proc_coord_t* atl_get_proc_coord(atl_ctx_t* ctx) {
    return &(ctx->coord);
}

static inline int atl_is_resize_enabled(atl_ctx_t* ctx) {
    return ctx->is_resize_enabled;
}

static inline atl_status_t atl_mr_reg(atl_ctx_t* ctx, const void* buf, size_t len, atl_mr_t** mr) {
    return ctx->mr_ops->mr_reg(ctx, buf, len, mr);
}

static inline atl_status_t atl_mr_dereg(atl_ctx_t* ctx, atl_mr_t* mr) {
    return ctx->mr_ops->mr_dereg(ctx, mr);
}

static inline atl_status_t atl_ep_send(atl_ep_t* ep,
                                       const void* buf,
                                       size_t len,
                                       size_t dst_proc_idx,
                                       uint64_t tag,
                                       atl_req_t* req) {
    return ep->p2p_ops->send(ep, buf, len, dst_proc_idx, tag, req);
}

static inline atl_status_t atl_ep_recv(atl_ep_t* ep,
                                       void* buf,
                                       size_t len,
                                       size_t src_proc_idx,
                                       uint64_t tag,
                                       atl_req_t* req) {
    return ep->p2p_ops->recv(ep, buf, len, src_proc_idx, tag, req);
}

static inline atl_status_t atl_ep_probe(atl_ep_t* ep,
                                        size_t src_proc_idx,
                                        uint64_t tag,
                                        int* found,
                                        size_t* recv_len) {
    return ep->p2p_ops->probe(ep, src_proc_idx, tag, found, recv_len);
}

static inline atl_status_t atl_ep_allgatherv(atl_ep_t* ep,
                                             const void* send_buf,
                                             size_t send_len,
                                             void* recv_buf,
                                             const int* recv_lens,
                                             const int* offsets,
                                             atl_req_t* req) {
    return ep->coll_ops->allgatherv(ep, send_buf, send_len, recv_buf, recv_lens, offsets, req);
}

static inline atl_status_t atl_ep_allreduce(atl_ep_t* ep,
                                            const void* send_buf,
                                            void* recv_buf,
                                            size_t len,
                                            atl_datatype_t dtype,
                                            atl_reduction_t op,
                                            atl_req_t* req) {
    return ep->coll_ops->allreduce(ep, send_buf, recv_buf, len, dtype, op, req);
}

static inline atl_status_t atl_ep_alltoall(atl_ep_t* ep,
                                           const void* send_buf,
                                           void* recv_buf,
                                           int len,
                                           atl_req_t* req) {
    return ep->coll_ops->alltoall(ep, send_buf, recv_buf, len, req);
}

static inline atl_status_t atl_ep_alltoallv(atl_ep_t* ep,
                                            const void* send_buf,
                                            const int* send_lens,
                                            const int* send_offsets,
                                            void* recv_buf,
                                            const int* recv_lens,
                                            const int* recv_offsets,
                                            atl_req_t* req) {
    return ep->coll_ops->alltoallv(
        ep, send_buf, send_lens, send_offsets, recv_buf, recv_lens, recv_offsets, req);
}

static inline atl_status_t atl_ep_barrier(atl_ep_t* ep, atl_req_t* req) {
    return ep->coll_ops->barrier(ep, req);
}

static inline atl_status_t atl_ep_bcast(atl_ep_t* ep,
                                        void* buf,
                                        size_t len,
                                        size_t root,
                                        atl_req_t* req) {
    return ep->coll_ops->bcast(ep, buf, len, root, req);
}

static inline atl_status_t atl_ep_reduce(atl_ep_t* ep,
                                         const void* send_buf,
                                         void* recv_buf,
                                         size_t len,
                                         size_t root,
                                         atl_datatype_t dtype,
                                         atl_reduction_t op,
                                         atl_req_t* req) {
    return ep->coll_ops->reduce(ep, send_buf, recv_buf, len, root, dtype, op, req);
}

static inline atl_status_t atl_ep_read(atl_ep_t* ep,
                                       void* buf,
                                       size_t len,
                                       atl_mr_t* mr,
                                       uint64_t addr,
                                       uintptr_t remote_key,
                                       size_t dst_proc_idx,
                                       atl_req_t* req) {
    return ep->rma_ops->read(ep, buf, len, mr, addr, remote_key, dst_proc_idx, req);
}

static inline atl_status_t atl_ep_write(atl_ep_t* ep,
                                        const void* buf,
                                        size_t len,
                                        atl_mr_t* mr,
                                        uint64_t addr,
                                        uintptr_t remote_key,
                                        size_t dst_proc_idx,
                                        atl_req_t* req) {
    return ep->rma_ops->write(ep, buf, len, mr, addr, remote_key, dst_proc_idx, req);
}

static inline atl_status_t atl_ep_wait(atl_ep_t* ep, atl_req_t* req) {
    return ep->comp_ops->wait(ep, req);
}

static inline atl_status_t atl_ep_wait_all(atl_ep_t* ep, atl_req_t* req, size_t count) {
    return ep->comp_ops->wait_all(ep, req, count);
}

static inline atl_status_t atl_ep_cancel(atl_ep_t* ep, atl_req_t* req) {
    return ep->comp_ops->cancel(ep, req);
}

static inline atl_status_t atl_ep_poll(atl_ep_t* ep) {
    return ep->comp_ops->poll(ep);
}

static inline atl_status_t atl_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) {
    return ep->comp_ops->check(ep, is_completed, req);
}
*/
#ifdef __cplusplus
class iatl {
public:
    virtual ~iatl() = default;

    virtual atl_status_t atl_init(int* argc,
                                  char*** argv,
                                  atl_attr_t* att,
                                  const char* main_addr,
                                  std::unique_ptr<ipmi>& pmi) = 0;

    virtual atl_status_t atl_finalize() = 0;

    virtual atl_status_t atl_update(std::unique_ptr<ipmi>& pmi) = 0;

    virtual atl_ep_t** atl_get_eps() = 0;

    virtual atl_proc_coord_t* atl_get_proc_coord() = 0;

    virtual int atl_is_resize_enabled() = 0;

    virtual atl_status_t atl_mr_reg(const void* buf, size_t len, atl_mr_t** mr) = 0;

    virtual atl_status_t atl_mr_dereg(atl_mr_t* mr) = 0;

    virtual atl_status_t atl_ep_send(atl_ep_t* ep,
                                     const void* buf,
                                     size_t len,
                                     size_t dst_proc_idx,
                                     uint64_t tag,
                                     atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_recv(atl_ep_t* ep,
                                     void* buf,
                                     size_t len,
                                     size_t src_proc_idx,
                                     uint64_t tag,
                                     atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_probe(atl_ep_t* ep,
                                      size_t src_proc_idx,
                                      uint64_t tag,
                                      int* found,
                                      size_t* recv_len) = 0;

    virtual atl_status_t atl_ep_allgatherv(atl_ep_t* ep,
                                           const void* send_buf,
                                           size_t send_len,
                                           void* recv_buf,
                                           const int* recv_lens,
                                           const int* offsets,
                                           atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_allreduce(atl_ep_t* ep,
                                          const void* send_buf,
                                          void* recv_buf,
                                          size_t len,
                                          atl_datatype_t dtype,
                                          atl_reduction_t op,
                                          atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_alltoall(atl_ep_t* ep,
                                         const void* send_buf,
                                         void* recv_buf,
                                         int len,
                                         atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_alltoallv(atl_ep_t* ep,
                                          const void* send_buf,
                                          const int* send_lens,
                                          const int* send_offsets,
                                          void* recv_buf,
                                          const int* recv_lens,
                                          const int* recv_offsets,
                                          atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_barrier(atl_ep_t* ep, atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_bcast(atl_ep_t* ep,
                                      void* buf,
                                      size_t len,
                                      size_t root,
                                      atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_reduce(atl_ep_t* ep,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t len,
                                       size_t root,
                                       atl_datatype_t dtype,
                                       atl_reduction_t op,
                                       atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_reduce_scatter(atl_ep_t* ep,
                                               const void* send_buf,
                                               void* recv_buf,
                                               size_t recv_len,
                                               atl_datatype_t dtype,
                                               atl_reduction_t op,
                                               atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_read(atl_ep_t* ep,
                                     void* buf,
                                     size_t len,
                                     atl_mr_t* mr,
                                     uint64_t addr,
                                     uintptr_t remote_key,
                                     size_t dst_proc_idx,
                                     atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_write(atl_ep_t* ep,
                                      const void* buf,
                                      size_t len,
                                      atl_mr_t* mr,
                                      uint64_t addr,
                                      uintptr_t remote_key,
                                      size_t dst_proc_idx,
                                      atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_wait(atl_ep_t* ep, atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_wait_all(atl_ep_t* ep, atl_req_t* req, size_t count) = 0;

    virtual atl_status_t atl_ep_cancel(atl_ep_t* ep, atl_req_t* req) = 0;

    virtual atl_status_t atl_ep_poll(atl_ep_t* ep) = 0;

    virtual atl_status_t atl_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) = 0;
};
#endif
