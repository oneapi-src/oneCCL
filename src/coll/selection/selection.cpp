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
#include "common/global/global.hpp"

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

static bool ccl_is_device_side_algo(ccl_coll_algo algo, const ccl_selector_param& param) {
    if (param.ctype == ccl_coll_allreduce) {
        return algo.allreduce == ccl_coll_allreduce_topo_ring;
    }
    else if (param.ctype == ccl_coll_reduce) {
        return algo.reduce == ccl_coll_reduce_topo_ring;
    }
    else if (param.ctype == ccl_coll_bcast) {
        return algo.bcast == ccl_coll_bcast_topo_ring;
    }

    return false;
}

bool ccl_can_use_datatype(ccl_coll_algo algo, const ccl_selector_param& param) {
    // A regular type, so we don't need to check for an additional support
    if (param.dtype.idx() != ccl::datatype::bfloat16 &&
        param.dtype.idx() != ccl::datatype::float16) {
        return true;
    }

    bool can_use = true;

    bool device_side_algo = ccl_is_device_side_algo(algo, param);

    // Algorithms running on GPU device support both fp16 and bf16, so we don't need to require their
    // support on the host.
    if (!device_side_algo) {
        if (param.dtype.idx() == ccl::datatype::bfloat16) {
            bool bf16_hw_support =
                ccl::global_data::env().bf16_impl_type != ccl_bf16_no_hardware_support;
            bool bf16_compiler_support =
                ccl::global_data::env().bf16_impl_type != ccl_bf16_no_compiler_support;

            can_use = bf16_compiler_support && bf16_hw_support;

            if (!can_use) {
                LOG_DEBUG("BF16 datatype is requested for ",
                          ccl_coll_type_to_str(param.ctype),
                          " running on CPU but not fully supported: hw: ",
                          bf16_hw_support,
                          " compiler: ",
                          bf16_compiler_support);
            }
        }
        else if (param.dtype.idx() == ccl::datatype::float16) {
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
    }

    return can_use;
}

bool ccl_is_topo_ring_algo(const ccl_selector_param& param) {
#ifndef CCL_ENABLE_SYCL
    return false;
#endif // CCL_ENABLE_SYCL

    if ((param.ctype != ccl_coll_allreduce) && (param.ctype != ccl_coll_bcast) &&
        (param.ctype != ccl_coll_reduce)) {
        return false;
    }

    bool res = false;

    auto& selector = ccl::global_data::get().algorithm_selector;

    if (param.ctype == ccl_coll_allreduce) {
        res = (selector->get<ccl_coll_allreduce>(param) == ccl_coll_allreduce_topo_ring);
    }
    else if (param.ctype == ccl_coll_bcast) {
        res = (selector->get<ccl_coll_bcast>(param) == ccl_coll_bcast_topo_ring);
    }
    else if (param.ctype == ccl_coll_reduce) {
        res = (selector->get<ccl_coll_reduce>(param) == ccl_coll_reduce_topo_ring);
    }

    return res;
}

bool ccl_can_use_topo_ring_algo(const ccl_selector_param& param) {
    if ((param.ctype != ccl_coll_allreduce) && (param.ctype != ccl_coll_bcast) &&
        (param.ctype != ccl_coll_reduce)) {
        return false;
    }

    bool is_sycl_buf = false;
    bool is_device_buf = true;
    bool is_l0_backend = false;

    size_t local_proc_count = ccl::global_data::get().executor->get_local_proc_count();

#ifdef CCL_ENABLE_SYCL
    is_sycl_buf = param.is_sycl_buf;
    if (param.buf && param.stream) {
        auto ctx = param.stream->get_native_stream().get_context();
        is_device_buf =
            (sycl::get_pointer_type(param.buf, ctx) == sycl::usm::alloc::device) ? true : false;
    }
#ifdef MULTI_GPU_SUPPORT
    if (param.stream && param.stream->get_backend() == sycl::backend::level_zero) {
        is_l0_backend = true;
    }
#endif // MULTI_GPU_SUPPORT
#endif // CCL_ENABLE_SYCL

    if ((param.comm->size() != 2 && param.comm->size() != 4) ||
        (param.comm->size() == 2 && param.comm->size() != static_cast<int>(local_proc_count)) ||
        (param.comm->size() == 4 && local_proc_count != 2 && local_proc_count != 4) ||
        (param.comm->size() != 2 && (ccl::global_data::env().atl_transport == ccl_atl_mpi)) ||
        !param.stream || (param.stream->get_type() != stream_type::gpu) || is_sycl_buf ||
        !is_device_buf || !is_l0_backend || ccl::global_data::env().enable_fusion ||
        ccl::global_data::env().enable_unordered_coll ||
        (ccl::global_data::env().priority_mode != ccl_priority_none) ||
        (ccl::global_data::env().worker_count != 1)) {
        return false;
    }

    return true;
}
