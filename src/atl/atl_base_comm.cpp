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
#include "atl/mpi/atl_mpi.hpp"
#include "atl/mpi/atl_mpi_comm.hpp"
#endif // CCL_ENABLE_MPI

#include "atl/atl_base_comm.hpp"
#include "atl/ofi/atl_ofi_comm.hpp"
#include "atl/ofi/atl_ofi.hpp"
#include "atl/util/pm/pm_rt.h"
#include "exec/exec.hpp"

atl_attr_t atl_base_comm::attr = {
    /* in */
    {
        0, /* enable_shm */
        0, /* enable_rma */
        0, /* enable_hmem */
        0, /* enable_sync_coll */
        0, /* enable_extra_ep */
        1, /* ep_count */
        ATL_MNIC_NONE, /* mnic_type */
        "", /* mnic_name */
        1, /* mnic_count */
        ATL_MNIC_OFFSET_NONE /* mnic_offset */
    },

    /* out */
    {
        0, /* enable_shm */
        0, /* enable_rma */
        0, /* enable_hmem */
        ATL_MNIC_NONE, /* mnic_type */
        0, /* mnic_count */
        0, /* tag_bits */
        0, /* max_tag */
        0, /* max_order_waw_size */
    }
};

ccl_executor* atl_base_comm::executor = nullptr;

void atl_base_comm::init_tag() {
    tag = std::unique_ptr<ccl_atl_tag>(new ccl_atl_tag(attr.out.tag_bits, attr.out.max_tag));
    if (rank == 0) {
        LOG_DEBUG("atl tag: ", tag->to_string());
    }
}

void atl_base_comm::print_atl_attrs() {
    std::stringstream ss;

    ss << "atl attrs:\n{\n"
       << "  in: { "
       << "shm: " << attr.in.enable_shm << ", hmem: " << attr.in.enable_hmem
       << ", sync_coll: " << attr.in.enable_sync_coll << ", extra_ep: " << attr.in.enable_extra_ep
       << ", ep_count: " << attr.in.ep_count << ", mnic_type: " << to_string(attr.in.mnic_type)
       << ", mnic_count: " << attr.in.mnic_count
       << ", mnic_offset: " << to_string(attr.in.mnic_offset) << " }\n"
       << "  out: { "
       << "shm: " << attr.out.enable_shm << ", hmem: " << attr.out.enable_hmem
       << ", mnic_type: " << to_string(attr.out.mnic_type)
       << ", mnic_count: " << attr.out.mnic_count << ", tag_bits: " << attr.out.tag_bits
       << ", max_tag: " << attr.out.max_tag << " }\n}";

    LOG_INFO(ss.str());
}

void atl_base_comm::executor_update() {
    if (!executor->are_workers_started()) {
        if (rank < coord.local_count)
            LOG_INFO(
                "start workers for local process [", coord.local_idx, ":", coord.local_count, "]");
        executor->start_workers(coord.local_idx, coord.local_count);
    }
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create_comm() {
    std::shared_ptr<atl_base_comm> atl_comm;

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm()); break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm()); break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("Unsupported yet"); break;
    }
    return atl_comm;
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create_comm(std::shared_ptr<ikvs_wrapper> k) {
    std::shared_ptr<atl_base_comm> atl_comm;

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm(k)); break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm(k)); break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("Unsupported yet"); break;
    }
    return atl_comm;
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create_comm(int total_rank_count,
                                                             const std::vector<int>& ranks,
                                                             std::shared_ptr<ikvs_wrapper> k) {
    std::shared_ptr<atl_base_comm> atl_comm;

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi:
            atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm(total_rank_count, ranks, k));
            break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi:
            atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm(total_rank_count, ranks, k));
            break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("Unsupported yet"); break;
    }
    return atl_comm;
}

void atl_comm_manager::set_internal_env(const atl_attr_t& attr) {
    auto transport_type = ccl::global_data::env().atl_transport;
    atl_base_comm::attr = attr;

    if (transport_type == ccl_atl_ofi)
        atl_ofi::atl_set_env(attr);
#ifdef CCL_ENABLE_MPI
    else if (transport_type == ccl_atl_mpi)
        atl_mpi::set_env(attr);
#endif // CCL_ENABLE_MPI
}

void atl_comm_manager::set_exec(ccl_executor* exec) {
    atl_base_comm::executor = exec;
}
