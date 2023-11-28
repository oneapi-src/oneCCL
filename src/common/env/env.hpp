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

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "oneapi/ccl/types.hpp"
#include "coll/coll.hpp"
#include "common/framework/framework.hpp"
#include "common/log/log.hpp"
#include "common/utils/utils.hpp"
#include "common/utils/yield.hpp"
#include "comp/bf16/bf16_utils.hpp"
#include "comp/fp16/fp16_utils.hpp"
#include "sched/cache/cache.hpp"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/global/ze/ze_fd_manager.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
#include "topology/topo_manager.hpp"

#include "common/env/vars.hpp"
#include "common/env/vars_experimental.hpp"

enum ccl_priority_mode { ccl_priority_none,
                         ccl_priority_direct,
                         ccl_priority_lifo };

enum ccl_atl_transport { ccl_atl_ofi,
                         ccl_atl_mpi };

enum ccl_atl_send_proxy {
    ccl_atl_send_proxy_none,
    ccl_atl_send_proxy_regular,
    ccl_atl_send_proxy_usm
};

enum ccl_staging_buffer { ccl_staging_regular,
                          ccl_staging_usm };

enum class backend_mode {
    native,
#ifdef CCL_ENABLE_STUB_BACKEND
    stub
#endif // CCL_ENABLE_STUB_BACKEND
};

enum class process_launcher_mode {
    hydra,
    torch,
#ifdef CCL_ENABLE_PMIX
    pmix,
#endif // CCL_ENABLE_PMIX
    none
};

namespace ccl {

class env_data {
public:
    env_data();
    ~env_data() = default;

    env_data(const env_data&) = delete;
    env_data(env_data&&) = delete;

    env_data& operator=(const env_data&) = delete;
    env_data& operator=(env_data&&) = delete;

    void parse();
    void print(int rank);
    void set_internal_env();

    bool was_printed;
    ccl_spinlock print_guard{};

    ccl_log_level log_level;
    int abort_on_throw;
    int queue_dump;
    int sched_dump;
    int sched_profile;
    ssize_t entry_max_update_time_sec;

    ccl_framework_type fw_type;

    size_t worker_count;
    int worker_offload;
    int worker_wait;
    std::vector<ssize_t> worker_affinity;
    std::vector<ssize_t> worker_mem_affinity;

    ccl_atl_transport atl_transport;
    int enable_shm;
    int enable_rma;
    int enable_hmem;
    ccl_atl_send_proxy atl_send_proxy;
    int enable_atl_cache;
    int enable_sync_coll;
    int enable_extra_ep;

    atl_mnic_t mnic_type;
    std::string mnic_name_raw;
    ssize_t mnic_count;
    atl_mnic_offset_t mnic_offset;

    /*
       parsing logic can be quite complex
       so hide it inside algorithm_selector module
       and store only raw strings in env_data
    */
    int enable_algo_fallback;
    // main algorithm selection
    std::string allgatherv_algo_raw;
    std::string allreduce_algo_raw;
    std::string alltoall_algo_raw;
    std::string alltoallv_algo_raw;
    std::string barrier_algo_raw;
    std::string bcast_algo_raw;
    std::string recv_algo_raw;
    std::string reduce_algo_raw;
    std::string reduce_scatter_algo_raw;
    std::string send_algo_raw;
    // scale-out selection part
    std::string allgatherv_scaleout_algo_raw;
    std::string allreduce_scaleout_algo_raw;
    std::string alltoall_scaleout_algo_raw;
    std::string alltoallv_scaleout_algo_raw;
    std::string barrier_scaleout_algo_raw;
    std::string bcast_scaleout_algo_raw;
    std::string recv_scaleout_algo_raw;
    std::string reduce_scaleout_algo_raw;
    std::string reduce_scatter_scaleout_algo_raw;
    std::string send_scaleout_algo_raw;
    int enable_unordered_coll;

    int enable_fusion;
    int fusion_bytes_threshold;
    int fusion_count_threshold;
    int fusion_check_urgent;
    float fusion_cycle_ms;

    ccl_priority_mode priority_mode;
    size_t spin_count;
    ccl_yield_type yield_type;
    size_t max_short_size;
    ssize_t bcast_part_count;
    ccl_cache_key_type cache_key_type;
    int enable_cache_flush;
    int enable_buffer_cache;
    int enable_strict_order;
    ccl_staging_buffer staging_buffer;
    int enable_op_sync;

    size_t chunk_count;
    size_t min_chunk_size;
    size_t rs_chunk_count;
    size_t rs_min_chunk_size;

#ifdef CCL_ENABLE_SYCL
    int allgatherv_topo_large_scale;
    int allgatherv_topo_read;
    int alltoallv_topo_read;
    int reduce_scatter_topo_read;
    int reduce_scatter_monolithic_kernel;
    int reduce_scatter_monolithic_pipeline_kernel;
    int reduce_scatter_fallback_algo;
    int allgatherv_monolithic_kernel;
    int allgatherv_monolithic_pipeline_kernel;
    int alltoallv_monolithic_kernel;
    int alltoallv_monolithic_read_kernel;

    size_t allgatherv_pipe_chunk_count;
    size_t allreduce_pipe_chunk_count;
    size_t reduce_scatter_pipe_chunk_count;
    size_t reduce_pipe_chunk_count;
#endif // CCL_ENABLE_SYCL

    int allreduce_nreduce_buffering;
    ssize_t allreduce_nreduce_segment_size;

    size_t allreduce_2d_chunk_count;
    size_t allreduce_2d_min_chunk_size;
    int allreduce_2d_switch_dims;

    ssize_t alltoall_scatter_max_ops;

    backend_mode backend;

    int local_rank;
    int local_size;
    process_launcher_mode process_launcher;

    int enable_topo_algo;
    topo_color_mode topo_color;
    int enable_p2p_access;

#ifdef CCL_ENABLE_MPI
    std::string mpi_lib_path;
#endif // CCL_ENABLE_MPI
    std::string ofi_lib_path;

#ifdef CCL_ENABLE_SYCL
    std::string kernel_path;
    int kernel_debug;
    ssize_t kernel_group_size;
    ssize_t kernel_group_count;
    ssize_t kernel_mem_align;
    int enable_kernel_sync;
    int kernel_1s_lead;
    int enable_kernel_1s_copy_ops;
    int enable_kernel_1s_ipc_wa;
    int enable_kernel_single_reduce_peers;
    int enable_close_fd_wa;

    int enable_sycl_output_event;
    int use_hmem;

    int sync_barrier;

    int enable_ze_barrier;
    int enable_ze_bidir_algo;
    int enable_ze_cache;
    int ze_device_cache_evict_smallest;
    long ze_device_cache_upper_limit;
    int ze_device_cache_num_blocks_in_chunk;
    ccl::ze::device_cache_policy_mode ze_device_cache_policy;
    int enable_ze_cache_cmdlists;
    int enable_ze_cache_cmdqueues;
    int enable_ze_cache_event_pools;
    int enable_ze_cache_open_ipc_handles;
    int ze_cache_open_ipc_handles_threshold;
    int enable_ze_cache_get_ipc_handles;
    int enable_ze_single_list;
    int disable_ze_family_check;
    int disable_ze_port_check;
    int ze_enable_oversubscription_fallback;
    int ze_enable_oversubscription_throw;
    int ze_serialize_mode;
    ccl::ze::copy_engine_mode ze_copy_engine;
    ccl::ze::h2d_copy_engine_mode ze_h2d_copy_engine;
    ssize_t ze_max_compute_queues;
    ssize_t ze_max_copy_queues;
    int ze_enable_ccs_fallback_for_copy;
    int enable_ze_list_dump;
    int ze_queue_index_offset;
    int ze_close_ipc_wa;
    std::string ze_lib_path;
    int ze_enable;
    int ze_fini_wa;
    int ze_multi_workers;
    int enable_ze_auto_tune_ports;
    ccl::ze::ipc_exchange_mode ze_ipc_exchange;
    int ze_drm_bdf_support;
    int ze_pt2pt_read;
    type2_tune_mode type2_mode;
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_PMIX
    std::string pmix_lib_path;
#endif // CCL_ENABLE_PMIX

#ifdef CCL_ENABLE_ITT
    int itt_level;
#endif // CCL_ENABLE_ITT

    ccl_bf16_impl_type bf16_impl_type;
    ccl_fp16_impl_type fp16_impl_type;

    template <class T>
    static int env_2_type(const char* env_name, T& val) {
        const char* env_val = getenv(env_name);
        if (env_val) {
            std::stringstream ss;
            ss << env_val;
            ss >> val;
            return 1;
        }
        return 0;
    }

    template <class T>
    static int env_2_enum(const char* env_name, const std::map<T, std::string>& values, T& val) {
        const char* env_val = getenv(env_name);
        if (env_val) {
            val = enum_by_str(env_name, values, env_val);
            return 1;
        }
        return 0;
    }

    template <class T>
    static T enum_by_str(const std::string& env_name,
                         const std::map<T, std::string>& e2s_map,
                         std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        for (const auto& pair : e2s_map) {
            if (!str.compare(pair.second)) {
                return pair.first;
            }
        }

        std::vector<std::string> values;
        std::transform(e2s_map.begin(),
                       e2s_map.end(),
                       std::back_inserter(values),
                       [](const typename std::map<T, std::string>::value_type& pair) {
                           return pair.second;
                       });

        std::string expected_values;
        for (size_t idx = 0; idx < values.size(); idx++) {
            expected_values += values[idx];
            if (idx != values.size() - 1)
                expected_values += ", ";
        }

        CCL_THROW(env_name, ": unexpected value: ", str, ", expected values: ", expected_values);
    }

    template <class T>
    static std::string str_by_enum(const std::map<T, std::string>& values, const T& val) {
        typename std::map<T, std::string>::const_iterator it;

        it = values.find(val);
        if (it != values.end()) {
            return it->second;
        }
        else {
            CCL_THROW("unexpected val ", static_cast<int>(val));
            return NULL;
        }
    }

    template <class T>
    static int env_2_topo(const char* env_name, const std::map<T, std::string>& values, T& val) {
        int status = 0;
        char* env_to_parse = getenv(env_name);
        if (env_to_parse) {
            std::string env_str(env_to_parse);
            if (env_str.find(std::string(topo_manager::card_domain_name)) != std::string::npos &&
                env_str.find(std::string(topo_manager::plane_domain_name)) != std::string::npos) {
                val = topo_color_mode::env;
                status = 1;
            }
            else {
                status = env_2_enum(env_name, values, val);
            }
        }

        return status;
    }

    static bool with_mpirun();

    static std::map<ccl_priority_mode, std::string> priority_mode_names;
    static std::map<ccl_atl_transport, std::string> atl_transport_names;
    static std::map<ccl_atl_send_proxy, std::string> atl_send_proxy_names;
    static std::map<ccl_staging_buffer, std::string> staging_buffer_names;
    static std::map<backend_mode, std::string> backend_names;
    static std::map<process_launcher_mode, std::string> process_launcher_names;

    int env_2_worker_affinity(int local_proc_idx, int local_proc_count);
    int env_2_worker_mem_affinity(int local_proc_count);
    void env_2_atl_transport();

private:
    int env_2_worker_affinity_auto(int local_proc_idx, size_t workers_per_process);

    int parse_affinity(const std::string& input,
                       std::vector<ssize_t>& output,
                       size_t expected_output_size);
    int parse_number(const std::string& number_str, size_t& result);
};

} // namespace ccl
