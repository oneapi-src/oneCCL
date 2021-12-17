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
#ifdef CCL_ENABLE_MPI
#include <mpi.h>

#include "atl_mpi_global_data.hpp"

typedef enum { ATL_MPI_COMP_POSTED, ATL_MPI_COMP_COMPLETED } atl_mpi_comp_state_t;

typedef struct {
    MPI_Request native_req;
    atl_mpi_comp_state_t comp_state;
} atl_mpi_req_t;

typedef struct {
    MPI_Comm mpi_comm;

    /* dummy recv operation to ensure progress in atl_poll */
    atl_mpi_req_t dummy_req;
    MPI_Comm dummy_comm;
    size_t idx;
    atl_proc_coord_t* coord;
} atl_mpi_ep_t;

typedef struct atl_mpi_env_info {
    int found;
    char key[MPI_MAX_INFO_KEY];
    char value[MPI_MAX_INFO_VAL];

    atl_mpi_env_info() {
        found = 0;
        memset(key, 0, MPI_MAX_INFO_KEY);
        memset(value, 0, MPI_MAX_INFO_VAL);
    }
} atl_mpi_env_info_t;

typedef struct atl_mpi_comm_info : atl_mpi_env_info_t {
    MPI_Comm comm;

    atl_mpi_comm_info() {
        comm = MPI_COMM_WORLD;
    }
} atl_mpi_comm_info_t;

class atl_mpi {
public:
    atl_mpi() = default;
    ~atl_mpi();

    atl_status_t init(int* argc,
                      char*** argv,
                      atl_attr_t* attr,
                      const char* main_addr,
                      std::shared_ptr<ipmi> pmi);

    atl_status_t update(std::shared_ptr<ipmi> pmi);

    atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr);

    atl_status_t mr_dereg(atl_mr_t* mr);

    atl_status_t send(atl_mpi_ep_t& ep,
                      const void* buf,
                      size_t len,
                      int dst_proc_idx,
                      uint64_t tag,
                      atl_req_t* req);

    atl_status_t recv(atl_mpi_ep_t& ep,
                      void* buf,
                      size_t len,
                      int src_proc_idx,
                      uint64_t tag,
                      atl_req_t* req);

    atl_status_t probe(atl_mpi_ep_t& ep,
                       int src_proc_idx,
                       uint64_t tag,
                       int* found,
                       size_t* recv_len);

    atl_status_t allgatherv(atl_mpi_ep_t& ep,
                            const void* send_buf,
                            size_t send_len,
                            void* recv_buf,
                            const int* recv_lens,
                            const int* offsets,
                            atl_req_t* req);

    atl_status_t allreduce(atl_mpi_ep_t& ep,
                           const void* send_buf,
                           void* recv_buf,
                           size_t len,
                           atl_datatype_t dtype,
                           atl_reduction_t op,
                           atl_req_t* req);

    atl_status_t alltoall(atl_mpi_ep_t& ep,
                          const void* send_buf,
                          void* recv_buf,
                          int len,
                          atl_req_t* req);

    atl_status_t alltoallv(atl_mpi_ep_t& ep,
                           const void* send_buf,
                           const int* send_lens,
                           const int* send_offsets,
                           void* recv_buf,
                           const int* recv_lens,
                           const int* recv_offsets,
                           atl_req_t* req);

    atl_status_t barrier(atl_mpi_ep_t& ep, atl_req_t* req);

    atl_status_t bcast(atl_mpi_ep_t& ep, void* buf, size_t len, int root, atl_req_t* req);

    atl_status_t reduce(atl_mpi_ep_t& ep,
                        const void* send_buf,
                        void* recv_buf,
                        size_t len,
                        int root,
                        atl_datatype_t dtype,
                        atl_reduction_t op,
                        atl_req_t* req);

    atl_status_t reduce_scatter(atl_mpi_ep_t& ep,
                                const void* send_buf,
                                void* recv_buf,
                                size_t recv_len,
                                atl_datatype_t dtype,
                                atl_reduction_t op,
                                atl_req_t* req);

    atl_status_t read(atl_mpi_ep_t& ep,
                      void* buf,
                      size_t len,
                      atl_mr_t* mr,
                      uint64_t addr,
                      uintptr_t remote_key,
                      int dst_proc_idx,
                      atl_req_t* req);

    atl_status_t write(atl_mpi_ep_t& ep,
                       const void* buf,
                       size_t len,
                       atl_mr_t* mr,
                       uint64_t addr,
                       uintptr_t remote_key,
                       int dst_proc_idx,
                       atl_req_t* req);

    atl_status_t wait(atl_mpi_ep_t& ep, atl_req_t* req);

    atl_status_t wait_all(atl_mpi_ep_t& ep, atl_req_t* req, size_t count);

    atl_status_t cancel(atl_mpi_ep_t& ep, atl_req_t* req);

    atl_status_t poll(atl_mpi_ep_t& ep);

    atl_status_t check(atl_mpi_ep_t& ep, atl_req_t* req);

    void comms_free(std::vector<atl_mpi_ep_t>& eps);

    atl_status_t finalize();

    int get_rank() {
        return global_coord.global_idx;
    }
    int get_size() {
        return global_coord.global_count;
    }
    bool is_inited() {
        return inited;
    }

    static void set_env(const atl_attr_t& attr);
    void coord_update(MPI_Comm base_comm, atl_proc_coord_t& coord);
    atl_status_t ep_init(std::vector<atl_mpi_ep_t>& eps);
    atl_status_t comm_split(const std::vector<atl_mpi_ep_t>& base_eps,
                            std::vector<atl_mpi_ep_t>& eps,
                            size_t color);

    static atl_mpi_env_info_t get_env_info(const char* key);
    static atl_mpi_comm_info_t get_comm_info(MPI_Comm comm, const char* key);

private:
    MPI_Datatype atl2mpi_dtype(atl_datatype_t dtype);
    void init_req(atl_req_t* req);
    inline atl_status_t ep_progress(atl_mpi_ep_t& ep, atl_mpi_req_t* req);
    MPI_Op atl2mpi_op(atl_reduction_t rtype, MPI_Datatype dtype);
    void check_comm_nic_idx(MPI_Comm comm, size_t expected_idx);
    void check_comm_ep_idx(MPI_Comm comm, size_t expected_idx);
    void check_comm_info(MPI_Comm comm, const char* key, const char* expected_value);
    size_t get_ep_idx(size_t ep_idx);

#ifdef ENABLE_DEBUG
    void check_ep(atl_mpi_ep_t& ep);
#else
#define check_ep(ep)
#endif

    bool is_finalized{ false };
    bool inited{ false };
    static atl_mpi_global_data global_data;
    atl_progress_mode_t progress_mode;
    bool sync_coll;
    size_t ep_count;
    atl_proc_coord_t global_coord;
};
#endif // CCL_ENABLE_MPI
