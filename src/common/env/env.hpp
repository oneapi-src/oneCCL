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

constexpr const char* CCL_ENV_STR_NOT_SPECIFIED = "<not specified>";
constexpr const ssize_t CCL_ENV_SIZET_NOT_SPECIFIED = -1;

constexpr const char* CCL_LOG_LEVEL = "CCL_LOG_LEVEL";
constexpr const char* CCL_SCHED_DUMP = "CCL_SCHED_DUMP";

constexpr const char* CCL_FRAMEWORK = "CCL_FRAMEWORK";

constexpr const char* CCL_WORKER_COUNT = "CCL_WORKER_COUNT";
constexpr const char* CCL_WORKER_OFFLOAD = "CCL_WORKER_OFFLOAD";
constexpr const char* CCL_WORKER_WAIT = "CCL_WORKER_WAIT";
constexpr const char* CCL_WORKER_AFFINITY = "CCL_WORKER_AFFINITY";

constexpr const char* I_MPI_AVAILABLE_CORES_ENV = "I_MPI_PIN_INFO";
constexpr const char* I_MPI_AVAILABLE_CORES_DELIMS = ",x";

constexpr const char* CCL_ATL_TRANSPORT = "CCL_ATL_TRANSPORT";
constexpr const char* CCL_ATL_SHM = "CCL_ATL_SHM";
constexpr const char* CCL_ATL_SYNC_COLL = "CCL_ATL_SYNC_COLL";
constexpr const char* CCL_ATL_EXTRA_EP = "CCL_ATL_EXTRA_EP";

constexpr const char* CCL_ALLGATHERV = "CCL_ALLGATHERV";
constexpr const char* CCL_ALLREDUCE = "CCL_ALLREDUCE";
constexpr const char* CCL_ALLTOALL = "CCL_ALLTOALL";
constexpr const char* CCL_ALLTOALLV = "CCL_ALLTOALLV";
constexpr const char* CCL_BARRIER = "CCL_BARRIER";
constexpr const char* CCL_BCAST = "CCL_BCAST";
constexpr const char* CCL_REDUCE = "CCL_REDUCE";
constexpr const char* CCL_REDUCE_SCATTER = "CCL_REDUCE_SCATTER";
constexpr const char* CCL_SPARSE_ALLREDUCE = "CCL_SPARSE_ALLREDUCE";
constexpr const char* CCL_UNORDERED_COLL = "CCL_UNORDERED_COLL";

constexpr const char* CCL_FUSION = "CCL_FUSION";
constexpr const char* CCL_FUSION_BYTES_THRESHOLD = "CCL_FUSION_BYTES_THRESHOLD";
constexpr const char* CCL_FUSION_COUNT_THRESHOLD = "CCL_FUSION_COUNT_THRESHOLD";
constexpr const char* CCL_FUSION_CHECK_URGENT = "CCL_FUSION_CHECK_URGENT";
constexpr const char* CCL_FUSION_CYCLE_MS = "CCL_FUSION_CYCLE_MS";

constexpr const char* CCL_RMA = "CCL_RMA";
constexpr const char* CCL_PRIORITY = "CCL_PRIORITY";
constexpr const char* CCL_SPIN_COUNT = "CCL_SPIN_COUNT";
constexpr const char* CCL_YIELD = "CCL_YIELD";
constexpr const char* CCL_MAX_SHORT_SIZE = "CCL_MAX_SHORT_SIZE";
constexpr const char* CCL_BCAST_PART_COUNT = "CCL_BCAST_PART_COUNT";
constexpr const char* CCL_CACHE_KEY = "CCL_CACHE_KEY";
constexpr const char* CCL_CACHE_FLUSH = "CCL_CACHE_FLUSH";
constexpr const char* CCL_STRICT_ORDER = "CCL_STRICT_ORDER";
constexpr const char* CCL_STAGING_BUFFER = "CCL_STAGING_BUFFER";

constexpr const char* CCL_CHUNK_COUNT = "CCL_CHUNK_COUNT";
constexpr const char* CCL_MIN_CHUNK_SIZE = "CCL_MIN_CHUNK_SIZE";
constexpr const char* CCL_RS_CHUNK_COUNT = "CCL_RS_CHUNK_COUNT";
constexpr const char* CCL_RS_MIN_CHUNK_SIZE = "CCL_RS_MIN_CHUNK_SIZE";
constexpr const char* CCL_AR2D_CHUNK_COUNT = "CCL_AR2D_CHUNK_COUNT";
constexpr const char* CCL_AR2D_MIN_CHUNK_SIZE = "CCL_AR2D_MIN_CHUNK_SIZE";

constexpr const char* CCL_ALLREDUCE_2D_BASE_SIZE = "CCL_ALLREDUCE_2D_BASE_SIZE";
constexpr const char* CCL_ALLREDUCE_2D_SWITCH_DIMS = "CCL_ALLREDUCE_2D_SWITCH_DIMS";

constexpr const char* CCL_ALLTOALL_SCATTER_MAX_OPS = "CCL_ALLTOALL_SCATTER_MAX_OPS";
constexpr const char* CCL_ALLTOALL_SCATTER_PLAIN = "CCL_ALLTOALL_SCATTER_PLAIN";

constexpr const char* CCL_COMM_KERNELS = "CCL_COMM_KERNELS";
constexpr const char* CCL_COMM_KERNELS_PATH = "CCL_COMM_KERNELS_PATH";
constexpr const char* CCL_GPU_THREAD_COUNT = "CCL_GPU_THREAD_COUNT";

constexpr const char* CCL_BF16 = "CCL_BF16";
constexpr const char* CCL_FP16 = "CCL_FP16";

enum ccl_priority_mode {
    ccl_priority_none,
    ccl_priority_direct,
    ccl_priority_lifo,

    ccl_priority_last_value
};

enum ccl_atl_transport {
    ccl_atl_ofi,
    ccl_atl_mpi,

    ccl_atl_last_value
};

enum ccl_staging_buffer {
    ccl_staging_regular,
    ccl_staging_usm,

    ccl_staging_last_value
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
    int sched_dump;

    ccl_framework_type fw_type;

    size_t worker_count;
    int worker_offload;
    int worker_wait;
    std::vector<size_t> worker_affinity;

    ccl_atl_transport atl_transport;
    int enable_shm;
    int sync_coll;
    int extra_ep;

    /*
       parsing logic can be quite complex
       so hide it inside algorithm_selector module
       and store only raw strings in env_data
    */
    std::string allgatherv_algo_raw;
    std::string allreduce_algo_raw;
    std::string alltoall_algo_raw;
    std::string alltoallv_algo_raw;
    std::string barrier_algo_raw;
    std::string bcast_algo_raw;
    std::string reduce_algo_raw;
    std::string reduce_scatter_algo_raw;
    std::string sparse_allreduce_algo_raw;
    int enable_unordered_coll;

    int enable_fusion;
    int fusion_bytes_threshold;
    int fusion_count_threshold;
    int fusion_check_urgent;
    float fusion_cycle_ms;

    int enable_rma;
    ccl_priority_mode priority_mode;
    size_t spin_count;
    ccl_yield_type yield_type;
    size_t max_short_size;
    ssize_t bcast_part_count;
    ccl_cache_key_type cache_key_type;
    int enable_cache_flush;
    int enable_strict_order;
    ccl_staging_buffer staging_buffer;

    size_t chunk_count;
    size_t min_chunk_size;
    size_t rs_chunk_count;
    size_t rs_min_chunk_size;
    size_t ar2d_chunk_count;
    size_t ar2d_min_chunk_size;

    ssize_t allreduce_2d_base_size;
    int allreduce_2d_switch_dims;

    ssize_t alltoall_scatter_max_ops;
    int alltoall_scatter_plain;

    int enable_comm_kernels;
    std::string comm_kernels_path;
    ssize_t gpu_thread_count;

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

    static bool with_mpirun();

    static std::map<ccl_priority_mode, std::string> priority_mode_names;
    static std::map<ccl_atl_transport, std::string> atl_transport_names;
    static std::map<ccl_staging_buffer, std::string> staging_buffer_names;

    int env_2_worker_affinity(size_t local_proc_idx, size_t local_proc_count);
    void env_2_atl_transport();

private:
    int env_2_worker_affinity_auto(size_t local_proc_idx, size_t workers_per_process);
};

} /* namespace ccl */
