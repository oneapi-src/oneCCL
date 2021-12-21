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

#include "atl/atl_def.h"
#include "common/comm/atl_tag.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"

class ccl_executor;

class atl_base_comm {
protected:
    atl_base_comm() = default;

public:
    virtual ~atl_base_comm() = default;

    virtual atl_status_t main_addr_reserve(char* main_addr) = 0;

    virtual atl_status_t finalize() = 0;

    virtual atl_status_t update() = 0;

    virtual atl_status_t wait_notification() = 0;

    virtual atl_status_t set_resize_function(atl_resize_fn_t fn) = 0;

    virtual atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr) = 0;

    virtual atl_status_t mr_dereg(atl_mr_t* mr) = 0;

    virtual atl_status_t send(size_t ep_idx,
                              const void* buf,
                              size_t len,
                              int dst_proc_idx,
                              uint64_t tag,
                              atl_req_t* req) = 0;

    virtual atl_status_t recv(size_t ep_idx,
                              void* buf,
                              size_t len,
                              int src_proc_idx,
                              uint64_t tag,
                              atl_req_t* req) = 0;

    virtual atl_status_t probe(size_t ep_idx,
                               int src_proc_idx,
                               uint64_t tag,
                               int* found,
                               size_t* recv_len) = 0;

    virtual atl_status_t allgatherv(size_t ep_idx,
                                    const void* send_buf,
                                    size_t send_len,
                                    void* recv_buf,
                                    const int* recv_lens,
                                    const int* offsets,
                                    atl_req_t* req) = 0;

    virtual atl_status_t allreduce(size_t ep_idx,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t len,
                                   atl_datatype_t dtype,
                                   atl_reduction_t op,
                                   atl_req_t* req) = 0;

    virtual atl_status_t alltoall(size_t ep_idx,
                                  const void* send_buf,
                                  void* recv_buf,
                                  int len,
                                  atl_req_t* req) = 0;

    virtual atl_status_t alltoallv(size_t ep_idx,
                                   const void* send_buf,
                                   const int* send_lens,
                                   const int* send_offsets,
                                   void* recv_buf,
                                   const int* recv_lens,
                                   const int* recv_offsets,
                                   atl_req_t* req) = 0;

    virtual atl_status_t barrier(size_t ep_idx, atl_req_t* req) = 0;

    virtual atl_status_t bcast(size_t ep_idx, void* buf, size_t len, int root, atl_req_t* req) = 0;

    virtual atl_status_t reduce(size_t ep_idx,
                                const void* send_buf,
                                void* recv_buf,
                                size_t len,
                                int root,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t* req) = 0;

    virtual atl_status_t reduce_scatter(size_t ep_idx,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_len,
                                        atl_datatype_t dtype,
                                        atl_reduction_t op,
                                        atl_req_t* req) = 0;

    virtual atl_status_t read(size_t ep_idx,
                              void* buf,
                              size_t len,
                              atl_mr_t* mr,
                              uint64_t addr,
                              uintptr_t remote_key,
                              int dst_proc_idx,
                              atl_req_t* req) = 0;

    virtual atl_status_t write(size_t ep_idx,
                               const void* buf,
                               size_t len,
                               atl_mr_t* mr,
                               uint64_t addr,
                               uintptr_t remote_key,
                               int dst_proc_idx,
                               atl_req_t* req) = 0;

    virtual atl_status_t wait(size_t ep_idx, atl_req_t* req) = 0;

    virtual atl_status_t wait_all(size_t ep_idx, atl_req_t* req, size_t count) = 0;

    virtual atl_status_t cancel(size_t ep_idx, atl_req_t* req) = 0;

    virtual atl_status_t poll(size_t ep_idx) = 0;

    virtual atl_status_t check(size_t ep_idx, atl_req_t* req) = 0;

    virtual size_t get_threads_per_process() = 0;

    virtual size_t get_ranks_per_process() = 0;

    virtual int get_rank() = 0;

    virtual int get_size() = 0;

    virtual int get_r2r_color() = 0;

    virtual int get_host_color() = 0;

    virtual std::shared_ptr<atl_base_comm> comm_split(int color) = 0;

    virtual std::vector<int> get_rank2rank_map() = 0;

    /*
     * TODO: Temporary change.
     * Need to define correct to unique id
     */
    virtual size_t get_id() = 0;
    std::unique_ptr<ccl_atl_tag> tag;
    static atl_attr_t attr;

protected:
    void init_tag();
    void print_atl_attrs();
    void executor_update();

    friend class atl_comm_manager;
    static ccl_executor* executor;

    int rank;
    int size;

    size_t threads_per_process;
    size_t ranks_per_process;

    std::vector<int> rank2rank_map;
    atl_proc_coord_t coord;
    int parent_rank;
    int parent_size;

    std::shared_ptr<ipmi> pmi;
};

class atl_comm_manager {
public:
    static std::shared_ptr<atl_base_comm> create_comm();

    static std::shared_ptr<atl_base_comm> create_comm(std::shared_ptr<ikvs_wrapper> k);

    static std::shared_ptr<atl_base_comm> create_comm(int total_rank_count,
                                                      const std::vector<int>& ranks,
                                                      std::shared_ptr<ikvs_wrapper> k);
    static void set_internal_env(const atl_attr_t& attr);
    static void set_exec(ccl_executor* exec);
};
