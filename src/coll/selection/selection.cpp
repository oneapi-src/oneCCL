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
#include "coll/selection/selection.hpp"
#include "comm/comm.hpp"
#include "common/global/global.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

std::string to_string(const ccl_selector_param& param) {
    std::stringstream ss;

    ss << "{ "
       << "coll: " << ccl_coll_type_to_str(param.ctype) << ", count: " << param.count
       << ", dt: " << ccl::global_data::get().dtypes->name(param.dtype);

    if (param.comm) {
        ss << ", comm: { rank: " << param.comm->rank() << ", size: " << param.comm->size() << " }";
    }

    if (param.stream) {
        ss << ", stream: " << param.stream->to_string();
    }

    if (param.buf) {
        ss << ", buf: " << param.buf;
    }

    if (param.is_vector_buf) {
        ss << ", vector_buf";
    }

#ifdef CCL_ENABLE_SYCL
    if (param.is_sycl_buf) {
        ss << ", sycl_buf";
    }
#endif // CCL_ENABLE_SYCL

    if (param.hint_algo.has_value()) {
        ss << ", hint_algo: " << param.hint_algo.value;
    }

    ss << " }";

    return ss.str();
}

bool ccl_is_direct_algo(const ccl_selector_param& param) {
    bool res = false;

    auto& selector = ccl::global_data::get().algorithm_selector;

    if (param.ctype == ccl_coll_allgatherv) {
        res = (selector->get<ccl_coll_allgatherv>(param) == ccl_coll_allgatherv_direct);
    }
    else if (param.ctype == ccl_coll_allreduce) {
        res = (selector->get<ccl_coll_allreduce>(param) == ccl_coll_allreduce_direct);
    }
    else if (param.ctype == ccl_coll_alltoall) {
        res = (selector->get<ccl_coll_alltoall>(param) == ccl_coll_alltoall_direct);
    }
    else if (param.ctype == ccl_coll_alltoallv) {
        res = (selector->get<ccl_coll_alltoallv>(param) == ccl_coll_alltoallv_direct);
    }
    else if (param.ctype == ccl_coll_barrier) {
        res = (selector->get<ccl_coll_barrier>(param) == ccl_coll_barrier_direct);
    }
    else if (param.ctype == ccl_coll_bcast) {
        res = (selector->get<ccl_coll_bcast>(param) == ccl_coll_bcast_direct);
    }
    else if (param.ctype == ccl_coll_reduce) {
        res = (selector->get<ccl_coll_reduce>(param) == ccl_coll_reduce_direct);
    }
    else if (param.ctype == ccl_coll_reduce_scatter) {
        res = (selector->get<ccl_coll_reduce_scatter>(param) == ccl_coll_reduce_scatter_direct);
    }

    return res;
}

namespace checkers {

bool is_unknown_device_family(const ccl_selector_param& param) {
    if (param.stream) {
        return param.stream->get_device_family() == ccl::device_family::unknown;
    }
    return false;
}

bool is_family1_card(const ccl_selector_param& param) {
    if (param.stream) {
        return param.stream->get_device_family() == ccl::device_family::family1;
    }
    return false;
}

bool is_coll_supported(std::initializer_list<ccl_coll_type> colls, ccl_coll_type value) {
    return std::find(colls.begin(), colls.end(), value) != colls.end();
}

bool is_sycl_buf(const ccl_selector_param& param) {
#ifdef CCL_ENABLE_SYCL
    return param.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    return false;
}

bool is_device_buf(const ccl_selector_param& param) {
#ifdef CCL_ENABLE_SYCL
    if (param.buf && param.stream) {
        auto ctx = param.stream->get_native_stream().get_context();
        return sycl::get_pointer_type(param.buf, ctx) == sycl::usm::alloc::device;
    }
#endif // CCL_ENABLE_SYCL
    return true;
}

bool is_l0_backend(const ccl_selector_param& param) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (param.stream) {
        return param.stream->get_backend() == ccl::utils::get_level_zero_backend();
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    return false;
}

bool is_gpu_stream(const ccl_selector_param& param) {
    if (param.stream) {
        return param.stream->is_gpu();
    }
    return false;
}

bool is_single_node(const ccl_selector_param& param) {
    size_t local_proc_count = ccl::global_data::get().get_local_proc_count();
    return static_cast<size_t>(param.comm->size()) <= local_proc_count;
}

bool is_single_card(const ccl_selector_param& param) {
    return param.comm->get_topo_manager().is_single_card;
}

} // namespace checkers

#define RETURN_FALSE_IF(cond, ...) \
    do { \
        if (cond) { \
            LOG_DEBUG("selection checker: ", ##__VA_ARGS__); \
            return false; \
        } \
    } while (0)

static bool ccl_is_device_side_algo(ccl_coll_algo algo, const ccl_selector_param& param) {
    CCL_THROW_IF_NOT(algo.has_value(), "empty algo value");

    if (param.ctype == ccl_coll_allgatherv) {
        return algo.allgatherv == ccl_coll_allgatherv_topo;
    }
    else if (param.ctype == ccl_coll_allreduce) {
        return algo.allreduce == ccl_coll_allreduce_topo;
    }
    else if (param.ctype == ccl_coll_alltoall) {
        return algo.alltoall == ccl_coll_alltoall_topo;
    }
    else if (param.ctype == ccl_coll_alltoallv) {
        return algo.alltoallv == ccl_coll_alltoallv_topo;
    }
    else if (param.ctype == ccl_coll_bcast) {
        return algo.bcast == ccl_coll_bcast_topo;
    }
    else if (param.ctype == ccl_coll_reduce) {
        return algo.reduce == ccl_coll_reduce_topo;
    }
    else if (param.ctype == ccl_coll_reduce_scatter) {
        // limitation: topo algorithm is disabled for reduce_scatter with unidirectional algo
#ifdef CCL_ENABLE_SYCL
        return algo.reduce_scatter == ccl_coll_reduce_scatter_topo &&
               ccl::global_data::env().enable_ze_bidir_algo;
#else
        return algo.reduce_scatter == ccl_coll_reduce_scatter_topo;
#endif // CCL_ENABLE_SYCL
    }

    return false;
}

bool ccl_is_device_side_algo(const ccl_selector_param& param) {
#ifndef CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL

    auto supported_colls = { ccl_coll_allgatherv,    ccl_coll_allreduce, ccl_coll_alltoall,
                             ccl_coll_alltoallv,     ccl_coll_bcast,     ccl_coll_reduce,
                             ccl_coll_reduce_scatter };
    RETURN_FALSE_IF(!checkers::is_coll_supported(supported_colls, param.ctype),
                    "coll ",
                    ccl_coll_type_to_str(param.ctype),
                    " is not supported");

    ccl_coll_algo algo{};
    auto& selector = ccl::global_data::get().algorithm_selector;

    if (param.ctype == ccl_coll_allgatherv) {
        algo.allgatherv = selector->get<ccl_coll_allgatherv>(param);
    }
    else if (param.ctype == ccl_coll_allreduce) {
        algo.allreduce = selector->get<ccl_coll_allreduce>(param);
    }
    else if (param.ctype == ccl_coll_alltoall) {
        algo.alltoall = selector->get<ccl_coll_alltoall>(param);
    }
    else if (param.ctype == ccl_coll_alltoallv) {
        algo.alltoallv = selector->get<ccl_coll_alltoallv>(param);
    }
    else if (param.ctype == ccl_coll_bcast) {
        algo.bcast = selector->get<ccl_coll_bcast>(param);
    }
    else if (param.ctype == ccl_coll_reduce) {
        algo.reduce = selector->get<ccl_coll_reduce>(param);
    }
    else if (param.ctype == ccl_coll_reduce_scatter) {
        algo.reduce_scatter = selector->get<ccl_coll_reduce_scatter>(param);
    }

    return ccl_is_device_side_algo(algo, param);
}

bool ccl_can_use_topo_algo(const ccl_selector_param& param) {
#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(!param.comm->get_env()->get_enable_topo_algo(), "topo algo is disabled");
#else // CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL

    auto supported_colls = { ccl_coll_allgatherv,    ccl_coll_allreduce, ccl_coll_alltoall,
                             ccl_coll_alltoallv,     ccl_coll_bcast,     ccl_coll_reduce,
                             ccl_coll_reduce_scatter };
    RETURN_FALSE_IF(!checkers::is_coll_supported(supported_colls, param.ctype),
                    "coll is not supported");

    size_t local_proc_count = ccl::global_data::get().get_local_proc_count();
    int comm_size = param.comm->size();

    LOG_DEBUG("coll ",
              ccl_coll_type_to_str(param.ctype),
              ", local_proc_count ",
              local_proc_count,
              ", comm ",
              param.comm->to_string());

    RETURN_FALSE_IF(!checkers::is_gpu_stream(param), "non-gpu stream is not supported");
    RETURN_FALSE_IF(checkers::is_sycl_buf(param), "sycl buffer is not supported");
    RETURN_FALSE_IF(!checkers::is_device_buf(param), "non-device buffers is not supported");
    RETURN_FALSE_IF(!checkers::is_l0_backend(param), "non-l0 backend is not supported");

    RETURN_FALSE_IF(ccl::global_data::env().enable_fusion, "fusion is not supported");
    RETURN_FALSE_IF(ccl::global_data::env().enable_unordered_coll,
                    "unordered coll is not supported");
    RETURN_FALSE_IF(ccl::global_data::env().priority_mode != ccl_priority_none, "wrong priority");
    RETURN_FALSE_IF(ccl::global_data::env().worker_count != 1
#ifdef CCL_ENABLE_SYCL
                        && !ccl::global_data::env().ze_multi_workers
#endif
                    ,
                    "unsupported count of workers");

    // this check is for multi-level scale-out for device buffers case:
    // we can't use topo algorithm without sub-communicators
    RETURN_FALSE_IF(!param.comm->get_even_comm().get(), "sub-communicators are not available");

#ifdef CCL_ENABLE_SYCL
    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_p2p_access(),
                    "no p2p access between devices");

    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_ppn(),
                    "ppn is not the same among the nodes");

    RETURN_FALSE_IF(!param.comm->get_topo_manager().has_same_domains(),
                    "processes are not properly distributed among domains");

    if (!ccl::global_data::env().ze_disable_oversubscription_check) {
        RETURN_FALSE_IF(param.comm->get_topo_manager().has_oversubscription(),
                        "oversubscription case: one rank per device is only supported");
    }

    RETURN_FALSE_IF(!ccl::global_data::env().enable_ze_bidir_algo &&
                        (param.ctype == ccl_coll_alltoall || param.ctype == ccl_coll_alltoallv),
                    "alltoall(v) is supported with bidir only");

    if (!ccl::global_data::env().disable_ze_port_check) {
        RETURN_FALSE_IF(
            !checkers::is_single_card(param) && param.comm->get_topo_manager().has_failed_ports(),
            "failed fabric ports");
    }

    if (!ccl::global_data::env().disable_ze_family_check) {
        RETURN_FALSE_IF(checkers::is_family1_card(param) && !checkers::is_single_card(param),
                        "multi-card ",
                        ccl_coll_type_to_str(param.ctype),
                        " is not supported for family1");

        RETURN_FALSE_IF(
            checkers::is_family1_card(param) && ccl::global_data::env().enable_ze_bidir_algo,
            "bidir ",
            ccl_coll_type_to_str(param.ctype),
            " is not supported for family1");
    }

    RETURN_FALSE_IF(checkers::is_unknown_device_family(param),
                    "topo algo is not supported for unknown device family");
#ifndef CCL_BF16_GPU_TRUNCATE
    RETURN_FALSE_IF(checkers::is_unknown_device_family(param) &&
                        (param.dtype.idx() == ccl::datatype::bfloat16) &&
                        (param.ctype == ccl_coll_allreduce || param.ctype == ccl_coll_reduce ||
                         param.ctype == ccl_coll_reduce_scatter),
                    "bfloat16 reduction is not supported for unknown device family");
#endif // !CCL_BF16_GPU_TRUNCATE
#endif // CCL_ENABLE_SYCL

    RETURN_FALSE_IF((((param.ctype == ccl_coll_allreduce) || (param.ctype == ccl_coll_bcast) ||
                      (param.ctype == ccl_coll_reduce)) &&
                     ((comm_size < 2) || (local_proc_count == 1))),
                    "unsupported comm size for ",
                    ccl_coll_type_to_str(param.ctype));

    RETURN_FALSE_IF(param.ctype == ccl_coll_bcast && !checkers::is_single_node(param),
                    "multi-node for ",
                    ccl_coll_type_to_str(param.ctype),
                    " is not supported");

    RETURN_FALSE_IF(((param.ctype == ccl_coll_reduce) && (comm_size % local_proc_count != 0)),
                    "ppn must be equal");

    RETURN_FALSE_IF(param.ctype == ccl_coll_allgatherv && !checkers::is_single_card(param) &&
                        comm_size % local_proc_count != 0,
                    "ppn must be equal");

    RETURN_FALSE_IF((comm_size % 2 != 0), "odd comm_size is not supported");

    RETURN_FALSE_IF(!checkers::is_single_card(param) && !checkers::is_single_node(param) &&
                        (local_proc_count % 2 != 0),
                    "odd proc count per node is not supported");
    RETURN_FALSE_IF((param.ctype == ccl_coll_reduce) && (param.count < size_t(param.comm->size())),
                    "reduce with count < comm_size not supported");
    return true;
}

bool ccl_can_use_datatype(ccl_coll_algo algo, const ccl_selector_param& param) {
    if (param.dtype.idx() != ccl::datatype::float16) {
        return true;
    }

    bool can_use = true;

    // algorithms running on device side support fp16
    // so we don't need to require their support on the host
    bool device_side_algo = ccl_is_device_side_algo(algo, param);
    if (device_side_algo) {
        return true;
    }

    if (param.dtype == ccl::datatype::float16) {
        bool fp16_hw_support =
            ccl::global_data::env().fp16_impl_type != ccl_fp16_no_hardware_support;
        bool fp16_compiler_support =
            ccl::global_data::env().fp16_impl_type != ccl_fp16_no_compiler_support;

        can_use = fp16_hw_support && fp16_compiler_support;

        if (!can_use) {
            LOG_DEBUG("FP16 datatype is requested for ",
                      ccl_coll_type_to_str(param.ctype),
                      " running on CPU but not fully supported: hw: ",
                      fp16_hw_support,
                      " compiler: ",
                      fp16_compiler_support);
        }
    }

    return can_use;
}
