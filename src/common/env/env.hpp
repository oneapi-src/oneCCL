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
constexpr const char* CCL_QUEUE_DUMP = "CCL_QUEUE_DUMP";
constexpr const char* CCL_SCHED_DUMP = "CCL_SCHED_DUMP";
constexpr const char* CCL_SCHED_PROFILE = "CCL_SCHED_PROFILE";

constexpr const char* CCL_FRAMEWORK = "CCL_FRAMEWORK";

constexpr const char* CCL_WORKER_COUNT = "CCL_WORKER_COUNT";
constexpr const char* CCL_WORKER_OFFLOAD = "CCL_WORKER_OFFLOAD";
constexpr const char* CCL_WORKER_WAIT = "CCL_WORKER_WAIT";
constexpr const char* CCL_WORKER_AFFINITY = "CCL_WORKER_AFFINITY";
constexpr const char* CCL_WORKER_MEM_AFFINITY = "CCL_WORKER_MEM_AFFINITY";

constexpr const char* I_MPI_AVAILABLE_CORES_ENV = "I_MPI_PIN_INFO";
constexpr const char* I_MPI_AVAILABLE_CORES_DELIMS = ",x";

constexpr const char* CCL_ATL_TRANSPORT = "CCL_ATL_TRANSPORT";
constexpr const char* CCL_ATL_SHM = "CCL_ATL_SHM";
constexpr const char* CCL_ATL_RMA = "CCL_ATL_RMA";
constexpr const char* CCL_ATL_HMEM = "CCL_ATL_HMEM";
constexpr const char* CCL_ATL_SEND_PROXY = "CCL_ATL_SEND_PROXY";
constexpr const char* CCL_ATL_SYNC_COLL = "CCL_ATL_SYNC_COLL";
constexpr const char* CCL_ATL_EXTRA_EP = "CCL_ATL_EXTRA_EP";
constexpr const char* CCL_ATL_CACHE = "CCL_ATL_CACHE";

constexpr const char* CCL_MNIC = "CCL_MNIC";
constexpr const char* CCL_MNIC_NAME = "CCL_MNIC_NAME";
constexpr const char* CCL_MNIC_COUNT = "CCL_MNIC_COUNT";
constexpr const char* CCL_MNIC_OFFSET = "CCL_MNIC_OFFSET";

constexpr const char* CCL_ALGO_FALLBACK = "CCL_ALGO_FALLBACK";
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

constexpr const char* CCL_PRIORITY = "CCL_PRIORITY";
constexpr const char* CCL_SPIN_COUNT = "CCL_SPIN_COUNT";
constexpr const char* CCL_YIELD = "CCL_YIELD";
constexpr const char* CCL_MAX_SHORT_SIZE = "CCL_MAX_SHORT_SIZE";
constexpr const char* CCL_BCAST_PART_COUNT = "CCL_BCAST_PART_COUNT";
constexpr const char* CCL_CACHE_KEY = "CCL_CACHE_KEY";
constexpr const char* CCL_CACHE_FLUSH = "CCL_CACHE_FLUSH";
constexpr const char* CCL_BUFFER_CACHE = "CCL_BUFFER_CACHE";
constexpr const char* CCL_STRICT_ORDER = "CCL_STRICT_ORDER";
constexpr const char* CCL_STAGING_BUFFER = "CCL_STAGING_BUFFER";
constexpr const char* CCL_OP_SYNC = "CCL_OP_SYNC";

constexpr const char* CCL_CHUNK_COUNT = "CCL_CHUNK_COUNT";
constexpr const char* CCL_MIN_CHUNK_SIZE = "CCL_MIN_CHUNK_SIZE";
constexpr const char* CCL_RS_CHUNK_COUNT = "CCL_RS_CHUNK_COUNT";
constexpr const char* CCL_RS_MIN_CHUNK_SIZE = "CCL_RS_MIN_CHUNK_SIZE";
constexpr const char* CCL_AR2D_CHUNK_COUNT = "CCL_AR2D_CHUNK_COUNT";
constexpr const char* CCL_AR2D_MIN_CHUNK_SIZE = "CCL_AR2D_MIN_CHUNK_SIZE";

constexpr const char* CCL_ALLREDUCE_2D_BASE_SIZE = "CCL_ALLREDUCE_2D_BASE_SIZE";
constexpr const char* CCL_ALLREDUCE_2D_SWITCH_DIMS = "CCL_ALLREDUCE_2D_SWITCH_DIMS";

constexpr const char* CCL_ALLREDUCE_NREDUCE_BUFFERING = "CCL_ALLREDUCE_NREDUCE_BUFFERING";
constexpr const char* CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE = "CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE";

constexpr const char* CCL_ALLTOALL_SCATTER_MAX_OPS = "CCL_ALLTOALL_SCATTER_MAX_OPS";
constexpr const char* CCL_ALLTOALL_SCATTER_PLAIN = "CCL_ALLTOALL_SCATTER_PLAIN";

constexpr const char* CCL_KERNEL_PATH = "CCL_KERNEL_PATH";
constexpr const char* CCL_KERNEL_DEBUG = "CCL_KERNEL_DEBUG";
constexpr const char* CCL_KERNEL_GROUP_SIZE = "CCL_KERNEL_GROUP_SIZE";
constexpr const char* CCL_KERNEL_GROUP_COUNT = "CCL_KERNEL_GROUP_COUNT";
constexpr const char* CCL_KERNEL_SYNC = "CCL_KERNEL_SYNC";
constexpr const char* CCL_KERNEL_1S_LEAD = "CCL_KERNEL_1S_LEAD";
constexpr const char* CCL_KERNEL_1S_USE_COPY_OPS = "CCL_KERNEL_1S_USE_COPY_OPS";
constexpr const char* CCL_KERNEL_1S_IPC_WA = "CCL_KERNEL_1S_IPC_WA";
constexpr const char* CCL_KERNEL_PROFILE = "CCL_KERNEL_PROFILE";
constexpr const char* CCL_KERNEL_CLOSE_FD_WA = "CCL_KERNEL_CLOSE_FD_WA";

constexpr const char* CCL_SYCL_OUTPUT_EVENT = "CCL_SYCL_OUTPUT_EVENT";

constexpr const char* CCL_ZE_BARRIER = "CCL_ZE_BARRIER";
constexpr const char* CCL_ZE_CACHE = "CCL_ZE_CACHE";
constexpr const char* CCL_ZE_SERIALIZE = "CCL_ZE_SERIALIZE";
constexpr const char* CCL_ZE_COPY_ENGINE = "CCL_ZE_COPY_ENGINE";
constexpr const char* CCL_ZE_QUEUE_INDEX = "CCL_ZE_QUEUE_INDEX";
constexpr const char* CCL_ZE_CLOSE_IPC_WA = "CCL_ZE_CLOSE_IPC_WA";
constexpr const char* CCL_ZE_SINGLE_LIST = "CCL_ZE_SINGLE_LIST";
constexpr const char* CCL_ZE_DISABLE_FAMILY_CHECK = "CCL_ZE_DISABLE_FAMILY_CHECK";

constexpr const char* CCL_BF16 = "CCL_BF16";
constexpr const char* CCL_FP16 = "CCL_FP16";

enum ccl_priority_mode { ccl_priority_none, ccl_priority_direct, ccl_priority_lifo };

enum ccl_atl_transport { ccl_atl_ofi, ccl_atl_mpi };

enum ccl_atl_send_proxy {
    ccl_atl_send_proxy_none,
    ccl_atl_send_proxy_regular,
    ccl_atl_send_proxy_usm
};

enum ccl_staging_buffer { ccl_staging_regular, ccl_staging_usm };

enum ccl_ze_copy_engine_mode {
    ccl_ze_copy_engine_none,
    ccl_ze_copy_engine_main,
    ccl_ze_copy_engine_link
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
    int queue_dump;
    int sched_dump;
    int sched_profile;

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
    size_t ar2d_chunk_count;
    size_t ar2d_min_chunk_size;

    ssize_t allreduce_2d_base_size;
    int allreduce_2d_switch_dims;
    int allreduce_nreduce_buffering;
    ssize_t allreduce_nreduce_segment_size;

    ssize_t alltoall_scatter_max_ops;
    int alltoall_scatter_plain;

#ifdef CCL_ENABLE_SYCL
    std::string kernel_path;
    int kernel_debug;
    ssize_t kernel_group_size;
    ssize_t kernel_group_count;
    int enable_kernel_sync;
    int kernel_1s_lead;
    int enable_kernel_1s_copy_ops;
    int enable_kernel_1s_ipc_wa;
    int enable_kernel_profile;
    int enable_close_fd_wa;

    int enable_sycl_output_event;

    int enable_ze_barrier;
    int enable_ze_cache;
    int enable_ze_single_list;
    int disable_ze_family_check;
    int ze_serialize_mode;
    ccl_ze_copy_engine_mode ze_copy_engine;
    int ze_queue_index;
    int ze_close_ipc_wa;
#endif // CCL_ENABLE_SYCL

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
    static std::map<ccl_atl_send_proxy, std::string> atl_send_proxy_names;
    static std::map<ccl_staging_buffer, std::string> staging_buffer_names;
    static std::map<ccl_ze_copy_engine_mode, std::string> ze_copy_engine_names;

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
