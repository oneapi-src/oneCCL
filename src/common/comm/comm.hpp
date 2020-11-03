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

#include <atomic>
#include <unordered_map>

#include "atl/atl_wrapper.h"
#include "coll/algorithms/allreduce/allreduce_2d.hpp"
#include "common/comm/comm_id_storage.hpp"
#include "common/comm/atl_tag.hpp"
#include "common/log/log.hpp"
#include "common/utils/tree.hpp"
#include "common/utils/utils.hpp"
#include "unordered_coll/unordered_coll.hpp"

// index = local_rank, value = global_rank
using ccl_rank2rank_map = std::vector<size_t>;

namespace ccl {
class kvs_interface;
}

class alignas(CACHELINE_SIZE) ccl_comm {
public:
    //TODO
    static void ccl_comm_reset_thread_barrier();
    ccl_comm() = delete;
    ccl_comm(const ccl_comm& other) = delete;
    ccl_comm& operator=(const ccl_comm& other) = delete;

    ccl_comm(size_t rank,
             size_t size,
             ccl_comm_id_storage::comm_id&& id,
             std::shared_ptr<atl_wrapper> atl,
             bool share_resources = false);
    ccl_comm(size_t rank,
             size_t size,
             ccl_comm_id_storage::comm_id&& id,
             ccl_rank2rank_map&& ranks,
             std::shared_ptr<atl_wrapper> atl,
             bool share_resources = false);

    //TODO non-implemented
    //1) cluster_devices_count (devices 1000) -> (processes 10)
    //2) blocking until all thread -> calls ccl_comm
    //3) return 'thread_count'

    // ccl_comm( {0,1,2,3...}, 1000, kvs )
    // from 20 processes from ranks 0,1,2,3. Each rank contains 10 threads
    // communicator: size in {20} and ranks in {0..19}
    // communicator: return threads count in process {10}
    // communicator: return devices counts per thread in process
    ccl_comm(const std::vector<size_t>& local_thread_device_ranks,
             size_t cluster_devices_count,
             std::shared_ptr<ccl::kvs_interface> kvs_instance,
             ccl_comm_id_storage::comm_id&& id,
             bool share_resources = false);

    ~ccl_comm() = default;

    /* version with user-provided colors, allows to skip allgatherv */
    static ccl_comm* create_with_colors(const std::vector<int>& colors,
                                        ccl_comm_id_storage* comm_ids,
                                        const ccl_comm* parent_comm,
                                        bool share_resources = false);

    std::shared_ptr<ccl_comm> clone_with_new_id(ccl_comm_id_storage::comm_id&& id);

    size_t rank() const noexcept {
        return m_rank;
    }

    size_t size() const noexcept {
        return m_size;
    }

    size_t pof2() const noexcept {
        return m_pof2;
    }

    ccl_comm_id_t id() const noexcept {
        return m_id.value();
    }

    size_t thread_count() const noexcept {
        return thread_number;
    }

    size_t on_process_ranks_count() const noexcept {
        return on_process_ranks_number;
    }

    ccl_sched_id_t get_sched_id(bool use_internal_space) {
        ccl_sched_id_t& next_sched_id =
            (use_internal_space) ? m_next_sched_id_internal : m_next_sched_id_external;

        ccl_sched_id_t first_sched_id =
            (use_internal_space) ? static_cast<ccl_sched_id_t>(0) : ccl_comm::max_sched_count / 2;

        ccl_sched_id_t max_sched_id =
            (use_internal_space) ? ccl_comm::max_sched_count / 2 : ccl_comm::max_sched_count;

        ccl_sched_id_t id = next_sched_id;

        ++next_sched_id;

        if (next_sched_id == max_sched_id) {
            /* wrap the sched numbers around to the start */
            next_sched_id = first_sched_id;
        }

        LOG_DEBUG("sched_id ", id, ", comm_id ", m_id.value(), ", next sched_id ", next_sched_id);

        return id;
    }

    void reset(size_t rank, size_t size) {
        m_rank = rank;
        m_size = size;
        m_pof2 = ccl_pof2(m_size);

        m_next_sched_id_internal = ccl_comm::max_sched_count / 2;
        m_next_sched_id_external = 0;
    }

    /**
     * Returns the number of @c rank in the global communicator
     * @param rank a rank which is part of the current communicator
     * @return number of @c rank in the global communicator
     */
    size_t get_global_rank(size_t rank) const;

    const ccl_double_tree& dtree() const {
        return m_dtree;
    }

    /**
     * Maximum available number of active communicators
     */
    static constexpr ccl_sched_id_t max_comm_count = std::numeric_limits<ccl_comm_id_t>::max();
    /**
     * Maximum value of schedule id in scope of the current communicator
     */
    static constexpr ccl_sched_id_t max_sched_count = std::numeric_limits<ccl_sched_id_t>::max();

    std::shared_ptr<atl_wrapper> atl;
    std::unique_ptr<ccl_unordered_coll_manager> unordered_coll_manager;
    std::unique_ptr<ccl_allreduce_2d_builder> allreduce_2d_builder;

private:

    void allocate_resources();

    size_t m_rank;
    size_t m_size;
    size_t m_pof2;

    ccl_comm_id_storage::comm_id m_id;
    ccl_sched_id_t m_next_sched_id_internal;
    ccl_sched_id_t m_next_sched_id_external;
    ccl_rank2rank_map m_local2global_map{};
    ccl_double_tree m_dtree;

    size_t thread_number;
    size_t on_process_ranks_number;
};
