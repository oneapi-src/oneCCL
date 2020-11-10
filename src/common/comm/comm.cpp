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
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"
#include "exec/exec.hpp"
#include "common/comm/comm.hpp"
#include "common/global/global.hpp"
#include "sched/sched.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/kvs.hpp"

void ccl_comm::allocate_resources() {
    if (ccl::global_data::env().enable_unordered_coll) {
        unordered_coll_manager =
            std::unique_ptr<ccl_unordered_coll_manager>(new ccl_unordered_coll_manager(*this));
    }

    auto& env_object = ccl::global_data::env();

    allreduce_2d_builder = std::unique_ptr<ccl_allreduce_2d_builder>(new ccl_allreduce_2d_builder(
        (env_object.allreduce_2d_base_size != CCL_ENV_SIZET_NOT_SPECIFIED)
            ? env_object.allreduce_2d_base_size
            : ccl::global_data::get().executor->get_local_proc_count(),
        env_object.allreduce_2d_switch_dims,
        this));

    if (m_rank == 0)
        env_object.print();
}

ccl_comm::ccl_comm(int rank,
                   int size,
                   ccl_comm_id_storage::comm_id&& id,
                   std::shared_ptr<atl_wrapper> atl,
                   bool share_resources)
        : ccl_comm(rank, size, std::move(id), ccl_rank2rank_map{}, atl, share_resources) {}

ccl_comm::ccl_comm(int rank,
                   int size,
                   ccl_comm_id_storage::comm_id&& id,
                   ccl_rank2rank_map&& rank_map,
                   std::shared_ptr<atl_wrapper> atl,
                   bool share_resources)
        : atl(atl),
          m_id(std::move(id)),
          m_local2global_map(std::move(rank_map)),
          m_dtree(size, rank),
          thread_number(1),
          on_process_ranks_number(1) {
    reset(rank, size);

    if (!share_resources) {
        allocate_resources();
    }
}

//TODO non-implemented
//TODO rude simulation of multi-thread barrier
static std::atomic<size_t> thread_counter{};
static std::atomic<size_t> thread_ranks_counter{};
void ccl_comm::ccl_comm_reset_thread_barrier() {
    // recharge counters again
    thread_counter.store(0);
    thread_ranks_counter.store(0);
}

ccl_comm::ccl_comm(const std::vector<int>& local_ranks,
                   int comm_size,
                   std::shared_ptr<ccl::kvs_interface> kvs_instance,
                   ccl_comm_id_storage::comm_id&& id,
                   bool share_resources)
        : m_id(std::move(id)),
          m_local2global_map(),
          m_dtree(local_ranks.size(), comm_size) {
    std::shared_ptr<ikvs_wrapper> kvs_wrapper(new users_kvs(kvs_instance));

    atl = std::shared_ptr<atl_wrapper>(new atl_wrapper(comm_size, local_ranks, kvs_wrapper));

    thread_number = atl->get_threads_per_process();
    on_process_ranks_number = atl->get_ranks_per_process();

    reset(atl->get_rank(), atl->get_size());

    if (!share_resources) {
        allocate_resources();
    }
}

ccl_comm* ccl_comm::create_with_colors(const std::vector<int>& colors,
                                       ccl_comm_id_storage* comm_ids,
                                       const ccl_comm* parent_comm,
                                       bool share_resources) {
    ccl_rank2rank_map rank_map;
    int new_comm_size = 0;
    int new_comm_rank = 0;
    int color = colors[parent_comm->rank()];

    for (int i = 0; i < parent_comm->size(); ++i) {
        if (colors[i] == color) {
            LOG_DEBUG("map local rank ", new_comm_size, " to global ", i);
            rank_map.emplace_back(i);
            ++new_comm_size;
            if (i < parent_comm->rank()) {
                ++new_comm_rank;
            }
        }
    }

    if (new_comm_size == 0) {
        throw ccl::exception(std::string("no colors matched to ") + std::to_string(color) +
                             " seems to be exchange issue");
    }

    if (new_comm_size == parent_comm->size()) {
        // exact copy of the global communicator, use empty map
        rank_map.clear();
    }

    ccl_comm* comm = new ccl_comm(new_comm_rank,
                                  new_comm_size,
                                  comm_ids->acquire(),
                                  std::move(rank_map),
                                  parent_comm->atl,
                                  share_resources);

    LOG_DEBUG("new comm: color ",
              color,
              ", rank ",
              comm->rank(),
              ", size ",
              comm->size(),
              ", comm_id ",
              comm->id());

    return comm;
}

std::shared_ptr<ccl_comm> ccl_comm::clone_with_new_id(ccl_comm_id_storage::comm_id&& id) {
    ccl_rank2rank_map rank_map{ m_local2global_map };
    return std::make_shared<ccl_comm>(
        m_rank, m_size, std::move(id), std::move(rank_map), atl, true /*share_resources*/);
}

int ccl_comm::get_global_rank(int rank) const {
    if (m_local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return rank;
    }

    CCL_THROW_IF_NOT((int)m_local2global_map.size() > rank,
                     "no rank ",
                     rank,
                     " was found in comm ",
                     this,
                     ", id ",
                     m_id.value());
    int global_rank = m_local2global_map[rank];
    LOG_DEBUG(
        "comm , ", this, " id ", m_id.value(), ", map rank ", rank, " to global ", global_rank);
    return global_rank;
}
