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

#include <iostream>
#include <memory>
#include <rdma/fi_domain.h>
#include <unordered_map>

#include "atl/atl_base_transport.hpp"
#include "atl/ofi/atl_ofi_helper.hpp"
#include "common/utils/hash.hpp"

class atl_ofi : public atl_base_transport {
public:
    atl_ofi() = default;
    ~atl_ofi();

    atl_status_t init(int* argc,
                      char*** argv,
                      atl_attr_t* attr,
                      const char* main_addr,
                      std::shared_ptr<ipmi> pmi) override;

    atl_status_t update(std::shared_ptr<ipmi> pmi) override;

    atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr) override;

    atl_status_t mr_dereg(atl_mr_t* mr) override;

    atl_status_t send(atl_ep_t& ep,
                      const void* buf,
                      size_t len,
                      int dst_proc_idx,
                      uint64_t tag,
                      atl_req_t& req) override;

    atl_status_t recv(atl_ep_t& ep,
                      void* buf,
                      size_t len,
                      int src_proc_idx,
                      uint64_t tag,
                      atl_req_t& req) override;

    atl_status_t probe(atl_ep_t& ep,
                       int src_proc_idx,
                       uint64_t tag,
                       int* found,
                       size_t* recv_len) override;

    atl_status_t allgatherv(atl_ep_t& ep,
                            const void* send_buf,
                            size_t send_len,
                            void* recv_buf,
                            const int* recv_lens,
                            const int* offsets,
                            atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t allreduce(atl_ep_t& ep,
                           const void* send_buf,
                           void* recv_buf,
                           size_t len,
                           atl_datatype_t dtype,
                           atl_reduction_t op,
                           atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t alltoall(atl_ep_t& ep,
                          const void* send_buf,
                          void* recv_buf,
                          int len,
                          atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t alltoallv(atl_ep_t& ep,
                           const void* send_buf,
                           const int* send_lens,
                           const int* send_offsets,
                           void* recv_buf,
                           const int* recv_lens,
                           const int* recv_offsets,
                           atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t barrier(atl_ep_t& ep, atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t bcast(atl_ep_t& ep, void* buf, size_t len, int root, atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t reduce(atl_ep_t& ep,
                        const void* send_buf,
                        void* recv_buf,
                        size_t len,
                        int root,
                        atl_datatype_t dtype,
                        atl_reduction_t op,
                        atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t reduce_scatter(atl_ep_t& ep,
                                const void* send_buf,
                                void* recv_buf,
                                size_t recv_len,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t& req) override {
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t read(atl_ep_t& ep,
                      void* buf,
                      size_t len,
                      atl_mr_t* mr,
                      uint64_t addr,
                      uintptr_t remote_key,
                      int dst_proc_idx,
                      atl_req_t& req) override;

    atl_status_t write(atl_ep_t& ep,
                       const void* buf,
                       size_t len,
                       atl_mr_t* mr,
                       uint64_t addr,
                       uintptr_t remote_key,
                       int dst_proc_idx,
                       atl_req_t& req) override;

    atl_status_t wait(atl_ep_t& ep, atl_req_t& req) override;

    atl_status_t wait_all(atl_ep_t& ep, std::vector<atl_req_t>& reqs, size_t count) override;

    atl_status_t cancel(atl_ep_t& ep, atl_req_t& req) override;

    atl_status_t poll(atl_ep_t& ep) override;

    atl_status_t check(atl_ep_t& ep, atl_req_t& req) override;

    atl_proc_coord_t create_proc_coord(atl_ep_t& ep) override {
        return coord;
    }

    void comms_free(std::vector<atl_ep_t>& eps) override {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    }

    atl_status_t comm_split(const std::vector<atl_ep_t>& base_eps,
                            std::vector<atl_ep_t>& eps,
                            size_t color,
                            int local_idx) override {
        throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
        return ATL_STATUS_UNSUPPORTED;
    }

    atl_status_t get_rank2rank_map(std::shared_ptr<ipmi> pmi,
                                   std::vector<int>& rank2rank_map) override;

    std::string to_string() override;

    atl_status_t finalize(int global_idx = 0) override;

    static void set_env(const atl_attr_t& attr);

private:
    atl_status_t progress_ep(atl_ep_t& ep);
    void process_comps(atl_ep_t& ep, struct fi_cq_tagged_entry* entries, ssize_t ret);
    atl_status_t prov_ep_handle_cq_err(atl_ofi_prov_ep_t* ep);
    atl_status_t open_providers(char* prov_env,
                                const atl_proc_coord_t& coord,
                                atl_attr_t* attr,
                                struct fi_info* base_hints,
                                int& open_nw_provs,
                                int fi_version,
                                std::shared_ptr<ipmi> pmi,
                                bool log_on_error);

    atl_ofi_ctx_t ctx;

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

    /* accumulates ep names from all comms */
    std::list<std::vector<char>> ep_names{};
};
