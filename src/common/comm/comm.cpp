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
//#include "ccl.h"
#include "common/comm/comm.hpp"
#include "common/global/global.hpp"
#include "sched/sched.hpp"
#include "oneapi/ccl/ccl_types.hpp"
#include "oneapi/ccl/ccl_kvs.hpp"

ccl_comm::ccl_comm(size_t rank, size_t size, ccl_comm_id_storage::comm_id&& id)
        : ccl_comm(rank, size, std::move(id), ccl_rank2rank_map{}) {}

ccl_comm::ccl_comm(size_t rank,
                   size_t size,
                   ccl_comm_id_storage::comm_id&& id,
                   ccl_rank2rank_map&& rank_map)
        : m_id(std::move(id)),
          m_local2global_map(std::move(rank_map)),
          m_dtree(size, rank),
          thread_number(1),
          on_process_ranks_number(1) {
    reset(rank, size);
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

ccl_comm::ccl_comm(const std::vector<size_t>& local_thread_device_ranks,
                   size_t cluster_devices_count,
                   std::shared_ptr<ccl::kvs_interface> kvs_instance,
                   ccl_comm_id_storage::comm_id&& id)
        : m_id(std::move(id)),
          m_local2global_map(),
          m_dtree(local_thread_device_ranks.size(), cluster_devices_count) {
    //TODO use multithreaded  atl_init
    //...

    //TODO rude simulation of multi-thread barrier
    thread_counter.fetch_add(1); //calc entered threads
    thread_ranks_counter.fetch_add(
        local_thread_device_ranks.size()); //calc total thread device ranks

    std::this_thread::sleep_for(std::chrono::seconds(
        ccl::global_data::get().thread_barrier_wait_timeout_sec)); //simulate barrier

    thread_number = thread_counter.load(); // obtain total thread count
    on_process_ranks_number = thread_ranks_counter.load(); // obtain total thread ranks

    //WA for single device case
    if (on_process_ranks_number == 1) {
        reset(*local_thread_device_ranks.begin(), cluster_devices_count);
    }
}

static ccl_status_t ccl_comm_exchange_colors(std::vector<int>& colors) {
    throw ccl::ccl_error("ccl_comm_exchange_colors not implemented yet");

    // const size_t exchange_count = 1;
    // std::vector<size_t> recv_counts(colors.size(), exchange_count);
    // ccl_coll_attr_t coll_attr{};
    // coll_attr.to_cache = false;
    // ccl_request_t request;

    // ccl_status_t status;

    // CCL_CALL(ccl_allgatherv(colors.data(), exchange_count,
    //                         colors.data(), recv_counts.data(),
    //                         ccl_dtype_int, &coll_attr,
    //                         nullptr, /* comm */
    //                         nullptr, /* stream */
    //                         &request));

    // CCL_CALL(ccl_wait(request));

    // return status;
}

ccl_comm* ccl_comm::create_with_color(int color,
                                      ccl_comm_id_storage* comm_ids,
                                      const ccl_comm* global_comm) {
    if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        throw ccl::ccl_error(
            "MPI transport doesn't support creation of communicator with color yet");
    }

    ccl_status_t status = ccl_status_success;

    std::vector<int> colors(global_comm->size());
    colors[global_comm->rank()] = color;
    status = ccl_comm_exchange_colors(colors);
    if (status != ccl_status_success) {
        throw ccl::ccl_error("failed to exchange colors during comm creation");
    }

    return create_with_colors(colors, comm_ids, global_comm);
}

ccl_comm* ccl_comm::create_with_colors(const std::vector<int>& colors,
                                       ccl_comm_id_storage* comm_ids,
                                       const ccl_comm* global_comm) {
    ccl_rank2rank_map rank_map;
    size_t new_comm_size = 0;
    size_t new_comm_rank = 0;
    int color = colors[global_comm->rank()];

    for (size_t i = 0; i < global_comm->size(); ++i) {
        if (colors[i] == color) {
            LOG_DEBUG("map local rank ", new_comm_size, " to global ", i);
            rank_map.emplace_back(i);
            ++new_comm_size;
            if (i < global_comm->rank()) {
                ++new_comm_rank;
            }
        }
    }

    if (new_comm_size == 0) {
        throw ccl::ccl_error(std::string("no colors matched to ") + std::to_string(color) +
                             " seems to be exchange issue");
    }

    if (new_comm_size == global_comm->size()) {
        // exact copy of the global communicator, use empty map
        rank_map.clear();
    }

    ccl_comm* comm =
        new ccl_comm(new_comm_rank, new_comm_size, comm_ids->acquire(), std::move(rank_map));

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
    return std::make_shared<ccl_comm>(m_rank, m_size, std::move(id), std::move(rank_map));
}

size_t ccl_comm::get_global_rank(size_t rank) const {
    if (m_local2global_map.empty()) {
        // global comm and its copies do not have entries in the map
        return rank;
    }

    CCL_THROW_IF_NOT(m_local2global_map.size() > rank,
                     "no rank ",
                     rank,
                     " was found in comm ",
                     this,
                     ", id ",
                     m_id.value());
    size_t global_rank = m_local2global_map[rank];
    LOG_DEBUG(
        "comm , ", this, " id ", m_id.value(), ", map rank ", rank, " to global ", global_rank);
    return global_rank;
}
