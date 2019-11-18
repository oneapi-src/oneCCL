/*
 Copyright 2016-2019 Intel Corporation
 
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

#include "ccl.h"
#include "common/comm/comm.hpp"
#include "sched/sched.hpp"

std::atomic_size_t ccl_comm::comm_count{};

ccl_comm::ccl_comm(size_t rank,
                   size_t size,
                   ccl_comm_id_storage::comm_id &&id) : ccl_comm(rank, size, std::move(id), rank_to_global_rank_map{})
{}

ccl_comm::ccl_comm(size_t rank,
                   size_t size,
                   ccl_comm_id_storage::comm_id &&id,
                   rank_to_global_rank_map&& ranks) :
    m_id(std::move(id)),
    m_ranks_map(std::move(ranks)),
    m_dtree(size, rank)
{
    reset(rank, size);
}

static ccl_status_t ccl_comm_exchange_colors(std::vector<int>& colors)
{
    const size_t exchange_count = 1;
    std::vector<size_t> recv_counts(colors.size(), exchange_count);
    ccl_coll_attr_t coll_attr{};
    coll_attr.to_cache = false;
    ccl_request_t request;

    ccl_status_t status;

    CCL_CALL(ccl_allgatherv(colors.data(), exchange_count,
                            colors.data(), recv_counts.data(),
                            ccl_dtype_int, &coll_attr,
                            nullptr, /* comm */
                            nullptr, /* stream */
                            &request));

    CCL_CALL(ccl_wait(request));

    return status;
}

ccl_comm* ccl_comm::create_with_color(int color,
                                      ccl_comm_id_storage* comm_ids,
                                      const ccl_comm* global_comm)
{
    ccl_status_t status = ccl_status_success;
    std::vector<int> all_colors(global_comm->size());
    all_colors[global_comm->rank()] = color;

    status = ccl_comm_exchange_colors(all_colors);
    if (status != ccl_status_success)
    {
        throw ccl::ccl_error("failed to exchange colors during comm creation");
    }

    size_t colors_match = 0;
    size_t new_rank = 0;

    rank_to_global_rank_map ranks_map;
    for (size_t i = 0; i < global_comm->size(); ++i)
    {
        if (all_colors[i] == color)
        {
            LOG_DEBUG("map local rank ", colors_match, " to global ", i);
            ranks_map[colors_match] = i;
            ++colors_match;
            if (i < global_comm->rank())
            {
                ++new_rank;
            }
        }
    }
    if (colors_match == 0)
    {
        throw ccl::ccl_error(
            std::string("no colors matched to ") + std::to_string(color) + " seems to be exchange issue");
    }

    if (colors_match == global_comm->size())
    {
        //Exact copy of the global communicator, use empty map
        ranks_map.clear();
    }

    ccl_comm* comm = new ccl_comm(new_rank, colors_match, comm_ids->acquire(),
                                  std::move(ranks_map));
    LOG_DEBUG("new comm: color ", color, ", rank ", comm->rank(), ", size ", comm->size(), ", comm_id ", comm->id());

    return comm;
}

std::shared_ptr<ccl_comm> ccl_comm::clone_with_new_id(ccl_comm_id_storage::comm_id &&id)
{
    rank_to_global_rank_map ranks_copy{m_ranks_map};
    return std::make_shared<ccl_comm>(m_rank, m_size, std::move(id), std::move(ranks_copy));
}

size_t ccl_comm::get_global_rank(size_t rank) const
{
    if (m_ranks_map.empty())
    {
        //global comm and its copies do not have entries in the map
        LOG_DEBUG("direct mapping of rank ", rank);
        return rank;
    }

    auto result = m_ranks_map.find(rank);
    if (result == m_ranks_map.end())
    {
        CCL_THROW("no rank ", rank, " was found in comm ", this, ", id ", m_id.value());
    }

    LOG_DEBUG("comm , ", this, " id ", m_id.value(), ", map rank ", rank, " to global ", result->second);
    return result->second;
}
