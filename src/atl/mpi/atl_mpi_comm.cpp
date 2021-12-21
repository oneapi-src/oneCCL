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
#ifdef CCL_ENABLE_MPI

#include "atl/mpi/atl_mpi_comm.hpp"
#include "exec/exec.hpp"

std::atomic<size_t> atl_mpi_comm::comm_count{ 0 };
atl_mpi* atl_mpi_comm::transport{ nullptr };

atl_mpi_comm::~atl_mpi_comm() {
    static std::mutex memory_mutex;
    std::lock_guard<std::mutex> lock(memory_mutex);
    tag.reset();
    comm_count--;
    if (comm_count.load() == 0) {
        delete transport;
        transport = nullptr;
    }
}

atl_mpi_comm::atl_mpi_comm() {
    init_transport(true);
}

atl_mpi_comm::atl_mpi_comm(std::shared_ptr<ikvs_wrapper> k) : atl_mpi_comm() {
    (void)k;
}

atl_mpi_comm::atl_mpi_comm(int total_rank_count,
                           const std::vector<int>& ranks,
                           std::shared_ptr<ikvs_wrapper> k)
        : atl_mpi_comm() {
    (void)total_rank_count;
    (void)ranks;
    (void)k;
}

atl_mpi_comm::atl_mpi_comm(std::vector<atl_mpi_ep_t>& parent_eps,
                           int parent_rank,
                           int parent_size,
                           int color) {
    this->parent_rank = parent_rank;
    this->parent_size = parent_size;

    transport->comm_split(parent_eps, eps, color);
    transport->coord_update(eps[0].mpi_comm, coord);
    rank = coord.global_idx;
    size = coord.global_count;
    init_transport(false);
    rank2rank_map.resize(size);
    MPI_Allgather(&parent_rank, 1, MPI_INT, rank2rank_map.data(), 1, MPI_INT, eps[0].mpi_comm);
}

void atl_mpi_comm::eps_update() {
    for (auto& ep : eps) {
        ep.coord = &coord;
    }
}

std::shared_ptr<atl_base_comm> atl_mpi_comm::comm_split(int color) {
    std::shared_ptr<atl_mpi_comm> comm =
        std::shared_ptr<atl_mpi_comm>(new atl_mpi_comm(eps, parent_rank, parent_size, color));

    return static_cast<std::shared_ptr<atl_mpi_comm>>(comm);
}

void atl_mpi_comm::init_transport(bool is_new) {
    LOG_DEBUG("init ATL, requested ep_count ", attr.in.ep_count);
    if (is_new) {
        static std::mutex memory_mutex;
        {
            std::lock_guard<std::mutex> lock(memory_mutex);
            if (!transport) {
                transport = new atl_mpi();
            }
            if (!transport->is_inited()) {
                CCL_THROW_IF_NOT(
                    transport->init(nullptr, nullptr, &attr, nullptr, pmi) == ATL_STATUS_SUCCESS,
                    "failed to initialize ATL");

                int mpi_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                if (mpi_rank == 0) {
                    print_atl_attrs();
                }
            }
        }

        transport->ep_init(eps);
        transport->coord_update(MPI_COMM_WORLD, coord);
        parent_rank = rank = coord.global_idx;
        parent_size = size = coord.global_count;
        rank2rank_map.resize(size);

        for (int i = 0; i < size; i++) {
            rank2rank_map[i] = i;
        }
    }

    threads_per_process = 1;
    ranks_per_process = 1;

    eps_update();
    init_tag();

    comm_count++;

    executor_update();
}
std::vector<int> atl_mpi_comm::get_rank2rank_map() {
    return rank2rank_map;
}
#endif //CCL_ENABLE_MPI
