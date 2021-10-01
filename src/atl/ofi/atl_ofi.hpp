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
#include <iostream>
#include <memory>
#include <rdma/fi_domain.h>
#include <unordered_map>

#include "atl.h"
#include "atl_ofi_helper.hpp"
#include "common/utils/hash.hpp"

class atl_ofi final : public iatl {
public:
    atl_ofi() = default;

    ~atl_ofi() override;

    static atl_status_t atl_set_env(const atl_attr_t& attr);

    atl_status_t atl_init(int* argc,
                          char*** argv,
                          atl_attr_t* attr,
                          const char* main_addr,
                          std::unique_ptr<ipmi>& pmi) override;

    atl_status_t atl_update(std::unique_ptr<ipmi>& pmi) override;

    atl_ep_t** atl_get_eps() override;

    atl_proc_coord_t* atl_get_proc_coord() override;

    atl_status_t atl_mr_reg(const void* buf, size_t len, atl_mr_t** mr) override;

    atl_status_t atl_mr_dereg(atl_mr_t* mr) override;

    atl_status_t atl_ep_send(atl_ep_t* ep,
                             const void* buf,
                             size_t len,
                             int dst_proc_idx,
                             uint64_t tag,
                             atl_req_t* req) override;

    atl_status_t atl_ep_recv(atl_ep_t* ep,
                             void* buf,
                             size_t len,
                             int src_proc_idx,
                             uint64_t tag,
                             atl_req_t* req) override;

    atl_status_t atl_ep_probe(atl_ep_t* ep,
                              int src_proc_idx,
                              uint64_t tag,
                              int* found,
                              size_t* recv_len) override;

    atl_status_t atl_ep_allgatherv(atl_ep_t* ep,
                                   const void* send_buf,
                                   size_t send_len,
                                   void* recv_buf,
                                   const int* recv_lens,
                                   const int* offsets,
                                   atl_req_t* req) override;

    atl_status_t atl_ep_allreduce(atl_ep_t* ep,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t len,
                                  atl_datatype_t dtype,
                                  atl_reduction_t op,
                                  atl_req_t* req) override;

    atl_status_t atl_ep_alltoall(atl_ep_t* ep,
                                 const void* send_buf,
                                 void* recv_buf,
                                 int len,
                                 atl_req_t* req) override;

    atl_status_t atl_ep_alltoallv(atl_ep_t* ep,
                                  const void* send_buf,
                                  const int* send_lens,
                                  const int* send_offsets,
                                  void* recv_buf,
                                  const int* recv_lens,
                                  const int* recv_offsets,
                                  atl_req_t* req) override;

    atl_status_t atl_ep_barrier(atl_ep_t* ep, atl_req_t* req) override;

    atl_status_t atl_ep_bcast(atl_ep_t* ep,
                              void* buf,
                              size_t len,
                              int root,
                              atl_req_t* req) override;

    atl_status_t atl_ep_reduce(atl_ep_t* ep,
                               const void* send_buf,
                               void* recv_buf,
                               size_t len,
                               int root,
                               atl_datatype_t dtype,
                               atl_reduction_t op,
                               atl_req_t* req) override;

    atl_status_t atl_ep_reduce_scatter(atl_ep_t* ep,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t recv_len,
                                       atl_datatype_t dtype,
                                       atl_reduction_t op,
                                       atl_req_t* req) override;

    atl_status_t atl_ep_read(atl_ep_t* ep,
                             void* buf,
                             size_t len,
                             atl_mr_t* mr,
                             uint64_t addr,
                             uintptr_t remote_key,
                             int dst_proc_idx,
                             atl_req_t* req) override;

    atl_status_t atl_ep_write(atl_ep_t* ep,
                              const void* buf,
                              size_t len,
                              atl_mr_t* mr,
                              uint64_t addr,
                              uintptr_t remote_key,
                              int dst_proc_idx,
                              atl_req_t* req) override;

    atl_status_t atl_ep_wait(atl_ep_t* ep, atl_req_t* req) override;

    atl_status_t atl_ep_wait_all(atl_ep_t* ep, atl_req_t* req, size_t count) override;

    atl_status_t atl_ep_cancel(atl_ep_t* ep, atl_req_t* req) override;

    atl_status_t atl_ep_poll(atl_ep_t* ep) override;

    atl_status_t atl_ep_check(atl_ep_t* ep, int* is_completed, atl_req_t* req) override;

    atl_status_t atl_finalize() override;

    bool is_inited() override {
        return inited;
    }

private:
    atl_status_t atl_ep_progress(atl_ep_t* ep);
    void atl_process_comps(atl_ep_t* ep, struct fi_cq_tagged_entry* entries, ssize_t ret);
    atl_status_t atl_prov_ep_handle_cq_err(atl_ofi_prov_ep_t* ep);

    atl_ctx_t* ctx = nullptr;

    class mr_cache {
    public:
        mr_cache() = default;
        ~mr_cache();

        void clear();
        void get(fid_domain* domain, void* buf, size_t bytes, fid_mr** mr);
        void push(fid_mr* mr);

    private:
        size_t mr_key = 0;

        using key_t = typename std::tuple<fid_domain*, void*, size_t>;
        using value_t = fid_mr*;
        std::unordered_multimap<key_t, value_t, ccl::utils::tuple_hash> cache{};
    };

    class fi_cache {
    public:
        fi_cache() = default;
        fi_cache(const fi_cache&) = delete;
        fi_cache& operator=(const fi_cache&) = delete;
        ~fi_cache();

        void clear();

        void init(size_t instance_count, int enable_hmem);
        void get(size_t idx, fid_domain* domain, void* buf, size_t bytes, fid_mr** mr);
        void push(size_t idx, fid_mr* mr);

    private:
        int enable_hmem;
        std::vector<mr_cache> memory_regions;
    };

    fi_cache cache{};

    bool is_finalized{ false };
    bool inited{ false };
};
