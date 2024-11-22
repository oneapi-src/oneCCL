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
#include "coll/algorithms/utils/sycl_selection.hpp"

bool can_use_sycl_kernels(const ccl_selector_param& param) {
// TODO: mitigate overhead added by can_use_sycl_kernels
#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(!param.comm->get_env()->get_enable_topo_algo(), "topo (sycl) algo is disabled");
#else // CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL
    auto supported_colls = { ccl_coll_allgatherv, ccl_coll_allreduce, ccl_coll_reduce_scatter };
    RETURN_FALSE_IF(!checkers::is_coll_supported(supported_colls, param.ctype),
                    "coll is not supported");

    // these fields are not expected to be set for sycl kernels
    CCL_THROW_IF_NOT(!param.is_vector_buf, "unexpected is_vector_buf value");
    CCL_THROW_IF_NOT(!param.is_sycl_buf, "unexpected is_sycl_buf value");
    CCL_THROW_IF_NOT(param.peer_rank == CCL_INVALID_PEER_RANK_IDX, "unexpected peer_rank value");
    CCL_THROW_IF_NOT(!param.is_scaleout, "unexpected is_scaleout value");

    size_t local_proc_count = ccl::global_data::get().get_local_proc_count();
    LOG_DEBUG("coll ",
              ccl_coll_type_to_str(param.ctype),
              ", local_proc_count ",
              local_proc_count,
              ", comm ",
              param.comm->to_string());

    if (param.comm->get_node_comm()->size() % 2 == 1) {
        // if a tile has a proc, the other tile on 2-tile systems should have it as well
        // case when subsequent cards use a single tile but other cards use two tiles
        // e. g. 2 2 2 2 1 1
        // is not possible due to how procs are allocated.
        // odd count of ranks on a node should be on a single plane
        RETURN_FALSE_IF(param.comm->get_even_comm()->size() != param.comm->get_node_comm()->size(),
                        "ranks on a node with odd count not on a single plane");
    }

    RETURN_FALSE_IF(!checkers::is_gpu_stream(param), "non-gpu stream is not supported");
    RETURN_FALSE_IF(!checkers::is_device_buf(param), "non-device buffers is not supported");
    RETURN_FALSE_IF(!checkers::is_l0_backend(param), "non-l0 backend is not supported");

    // Removed duplicate check for backend_mode::native
    RETURN_FALSE_IF(ccl::global_data::env().backend != backend_mode::native,
                    "stub backend is not supported");

    RETURN_FALSE_IF(ccl::global_data::env().worker_count != 1, "unsupported count of workers");
#ifdef CCL_ENABLE_SYCL
    // it hangs if we try to use sycl kernels without ze cache
    RETURN_FALSE_IF(ccl::global_data::env().enable_ze_cache == 0, "ze cache is not enabled");
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_p2p_access(),
                    "no p2p access between devices");
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_all_vertices_connected(),
                    "no connection between vertices");
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_ppn(),
                    "ppn is not the same among the nodes");
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_domains(),
                    "processes are not properly distributed among domains");

    const ccl::topo_manager& topo_manager = param.comm->get_topo_manager();
    bool is_single_node = topo_manager.is_single_node;
    bool is_oversubscription = topo_manager.has_oversubscription();
    bool is_dtype_supported =
        (param.dtype.idx() == ccl::datatype::float16 ||
         param.dtype.idx() == ccl::datatype::bfloat16 ||
         param.dtype.idx() == ccl::datatype::float32 || param.dtype.idx() == ccl::datatype::int32);

    // Common conditions for all collective operations
    RETURN_FALSE_IF(!ccl::global_data::env().enable_sycl_kernels, "SYCL kernels are not enabled");
    RETURN_FALSE_IF(!param.stream->get_native_stream().is_in_order(), "Stream is not in order");
    RETURN_FALSE_IF(!is_dtype_supported, "Data type is not supported");
    RETURN_FALSE_IF(is_oversubscription, "Oversubscription is not allowed");

    // Conditions specific to allreduce
    if (param.ctype == ccl_coll_allreduce) {
        RETURN_FALSE_IF(!ccl::global_data::env().allreduce_algo_raw.empty() &&
                            ccl::global_data::env().allreduce_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().allreduce_algo_raw,
                        " not supported");
        RETURN_FALSE_IF(param.reduction != ccl::reduction::sum,
                        "Allreduce only supports sum reduction");
    }

    // Conditions specific to allgatherv
    if (param.ctype == ccl_coll_allgatherv) {
        RETURN_FALSE_IF(!ccl::global_data::env().allgatherv_algo_raw.empty() &&
                            ccl::global_data::env().allgatherv_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().allgatherv_algo_raw,
                        " not supported");
    }

    // Conditions specific to reduce_scatter
    if (param.ctype == ccl_coll_reduce_scatter) {
        RETURN_FALSE_IF(!ccl::global_data::env().reduce_scatter_algo_raw.empty() &&
                            ccl::global_data::env().reduce_scatter_algo_raw != "topo",
                        "algo of coll: ",
                        ccl_coll_type_to_str(param.ctype),
                        " is specified explicitly as: ",
                        ccl::global_data::env().reduce_scatter_algo_raw,
                        " not supported");
        RETURN_FALSE_IF(param.reduction != ccl::reduction::sum,
                        "Reduce_scatter only supports sum reduction");
    }

    if (!ccl::global_data::env().disable_ze_port_check) {
        RETURN_FALSE_IF(!checkers::is_single_card(param) && topo_manager.has_failed_ports(),
                        "failed fabric ports");
    }

    if (!ccl::global_data::env().disable_ze_family_check) {
        RETURN_FALSE_IF(checkers::is_family1_card(param) && !checkers::is_single_card(param),
                        "multi-card ",
                        ccl_coll_type_to_str(param.ctype),
                        " is not supported for family1");
    }

    if (param.comm->get_node_comm()->size() != local_proc_count) {
        CCL_ASSERT(param.comm->get_node_comm()->size() < local_proc_count);
        RETURN_FALSE_IF(!ccl::global_data::env().sycl_sub_communicator,
                        "SYCL kernels are not enabled for sub-communicators");

        RETURN_FALSE_IF(ccl::global_data::env().sycl_esimd,
                        "SYCL ESIMD kernels are not enabled for sub-communicators");
    }

    if (checkers::is_unknown_device_family(param)) {
        LOG_WARN("Applying sycl-kernels, but device family is not recognized");
    }

#endif // CCL_ENABLE_SYCL

    LOG_DEBUG("selected algo: coll ", ccl_coll_type_to_str(param.ctype), ", algo ", "topo sycl");

    return true;
}
