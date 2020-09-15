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
#include <iterator>
#include <sstream>
#include <unistd.h>

#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

namespace ccl {

std::map<ccl_priority_mode, std::string> env_data::priority_mode_names = {
    std::make_pair(ccl_priority_none, "none"),
    std::make_pair(ccl_priority_direct, "direct"),
    std::make_pair(ccl_priority_lifo, "lifo")
};

std::map<ccl_atl_transport, std::string> env_data::atl_transport_names = {
    std::make_pair(ccl_atl_ofi, "ofi"),
    std::make_pair(ccl_atl_mpi, "mpi")
};

env_data::env_data()
        : log_level(static_cast<int>(ccl_log_level::ERROR)),
          sched_dump(0),

          worker_count(1),
          worker_offload(1),

          atl_transport(ccl_atl_mpi),
          enable_shm(0),

          enable_unordered_coll(0),

          enable_fusion(0),
          fusion_bytes_threshold(16384),
          fusion_count_threshold(256),
          fusion_check_urgent(1),
          fusion_cycle_ms(0.2),

          enable_rma(0),
          priority_mode(ccl_priority_none),
          spin_count(100),
          yield_type(ccl_yield_pause),
          max_short_size(4096),
          bcast_part_count(CCL_ENV_SIZET_NOT_SPECIFIED),
          cache_key_type(ccl_cache_key_match_id),
          enable_cache_flush(1),

          chunk_count(1),
          min_chunk_size(65536),
          rs_chunk_count(1),
          rs_min_chunk_size(65536),
          ar2d_chunk_count(1),
          ar2d_min_chunk_size(65536),

          allreduce_2d_base_size(CCL_ENV_SIZET_NOT_SPECIFIED),
          allreduce_2d_switch_dims(0),

          alltoall_scatter_max_ops(CCL_ENV_SIZET_NOT_SPECIFIED),
          alltoall_scatter_plain(0),

          default_resizable(0) {}

void env_data::parse() {
    env_2_type(CCL_LOG_LEVEL, log_level);
    ccl_logger::set_log_level(static_cast<ccl_log_level>(log_level));
    env_2_type(CCL_SCHED_DUMP, sched_dump);

    env_2_type(CCL_WORKER_COUNT, worker_count);
    CCL_THROW_IF_NOT(worker_count >= 1, "incorrect ", CCL_WORKER_COUNT, " ", worker_count);
    env_2_type(CCL_WORKER_OFFLOAD, worker_offload);

    env_2_enum(CCL_ATL_TRANSPORT, atl_transport_names, atl_transport);
    env_2_type(CCL_ATL_SHM, enable_shm);

    env_2_type(CCL_ALLGATHERV, allgatherv_algo_raw);
    env_2_type(CCL_ALLREDUCE, allreduce_algo_raw);
    env_2_type(CCL_ALLTOALL, alltoall_algo_raw);
    env_2_type(CCL_ALLTOALLV, alltoallv_algo_raw);
    env_2_type(CCL_BARRIER, barrier_algo_raw);
    env_2_type(CCL_BCAST, bcast_algo_raw);
    env_2_type(CCL_REDUCE, reduce_algo_raw);
    env_2_type(CCL_SPARSE_ALLREDUCE, sparse_allreduce_algo_raw);
    env_2_type(CCL_UNORDERED_COLL, enable_unordered_coll);
    if (enable_unordered_coll && atl_transport != ccl_atl_ofi) {
        CCL_THROW("unordered collectives are supported for OFI transport only");
    }

    env_2_type(CCL_FUSION, enable_fusion);
    env_2_type(CCL_FUSION_BYTES_THRESHOLD, fusion_bytes_threshold);
    env_2_type(CCL_FUSION_COUNT_THRESHOLD, fusion_count_threshold);
    env_2_type(CCL_FUSION_CHECK_URGENT, fusion_check_urgent);
    env_2_type(CCL_FUSION_CYCLE_MS, fusion_cycle_ms);
    if (enable_fusion) {
        CCL_THROW_IF_NOT(fusion_bytes_threshold >= 1,
                         "incorrect ",
                         CCL_FUSION_BYTES_THRESHOLD,
                         " ",
                         fusion_bytes_threshold);
        CCL_THROW_IF_NOT(fusion_count_threshold >= 1,
                         "incorrect ",
                         CCL_FUSION_COUNT_THRESHOLD,
                         " ",
                         fusion_count_threshold);
    }

    env_2_type(CCL_RMA, enable_rma);
    env_2_enum(CCL_PRIORITY, priority_mode_names, priority_mode);
    env_2_type(CCL_SPIN_COUNT, spin_count);
    env_2_enum(CCL_YIELD, ccl_yield_type_names, yield_type);
    env_2_type(CCL_MAX_SHORT_SIZE, max_short_size);
    env_2_type(CCL_BCAST_PART_COUNT, (size_t&)bcast_part_count);
    env_2_enum(CCL_CACHE_KEY, ccl_sched_key::key_type_names, cache_key_type);
    env_2_type(CCL_CACHE_FLUSH, enable_cache_flush);

    env_2_type(CCL_CHUNK_COUNT, chunk_count);
    CCL_THROW_IF_NOT(chunk_count >= 1, "incorrect ", CCL_CHUNK_COUNT, " ", chunk_count);
    env_2_type(CCL_MIN_CHUNK_SIZE, min_chunk_size);
    CCL_THROW_IF_NOT(min_chunk_size >= 1, "incorrect ", CCL_MIN_CHUNK_SIZE, " ", min_chunk_size);

    env_2_type(CCL_RS_CHUNK_COUNT, rs_chunk_count);
    CCL_THROW_IF_NOT(rs_chunk_count >= 1, "incorrect ", CCL_RS_CHUNK_COUNT, " ", rs_chunk_count);
    env_2_type(CCL_RS_MIN_CHUNK_SIZE, rs_min_chunk_size);
    CCL_THROW_IF_NOT(
        rs_min_chunk_size >= 1, "incorrect ", CCL_RS_MIN_CHUNK_SIZE, " ", rs_min_chunk_size);

    env_2_type(CCL_AR2D_CHUNK_COUNT, ar2d_chunk_count);
    CCL_THROW_IF_NOT(
        ar2d_chunk_count >= 1, "incorrect ", CCL_AR2D_CHUNK_COUNT, " ", ar2d_chunk_count);
    env_2_type(CCL_AR2D_MIN_CHUNK_SIZE, ar2d_min_chunk_size);
    CCL_THROW_IF_NOT(
        ar2d_min_chunk_size >= 1, "incorrect ", CCL_AR2D_MIN_CHUNK_SIZE, " ", ar2d_min_chunk_size);

    env_2_type(CCL_ALLREDUCE_2D_BASE_SIZE, (size_t&)allreduce_2d_base_size);
    env_2_type(CCL_ALLREDUCE_2D_SWITCH_DIMS, allreduce_2d_switch_dims);

    env_2_type(CCL_ALLTOALL_SCATTER_MAX_OPS, (size_t&)alltoall_scatter_max_ops);
    env_2_type(CCL_ALLTOALL_SCATTER_PLAIN, alltoall_scatter_plain);

    env_2_type(CCL_DEFAULT_RESIZABLE, default_resizable);
    CCL_THROW_IF_NOT(
        default_resizable <= 2, "incorrect ", CCL_DEFAULT_RESIZABLE, " ", default_resizable);
}

void env_data::print() {
#ifdef ENABLE_DEBUG
    const char* build_mode = "debug";
#else
    const char* build_mode = "release";
#endif
    LOG_INFO("build mode : ", build_mode);

    ccl_version_t version;
    if (ccl_get_version(&version) != ccl_status_success) {
        throw std::runtime_error("cannot determine CCL version!");
    }

    LOG_INFO("version : ", version.full);

    char* ccl_root = getenv("CCL_ROOT");
    LOG_INFO("CCL_ROOT : ", (ccl_root) ? ccl_root : CCL_ENV_STR_NOT_SPECIFIED);

    char* impi_root = getenv("I_MPI_ROOT");
    LOG_INFO("I_MPI_ROOT : ", (impi_root) ? impi_root : CCL_ENV_STR_NOT_SPECIFIED);

    LOG_INFO(CCL_LOG_LEVEL, ": ", log_level);
    LOG_INFO(CCL_SCHED_DUMP, ": ", sched_dump);

    LOG_INFO(CCL_WORKER_COUNT, ": ", worker_count);
    LOG_INFO(CCL_WORKER_OFFLOAD, ": ", worker_offload);
    for (size_t w_idx = 0; w_idx < worker_affinity.size(); w_idx++) {
        LOG_INFO(CCL_WORKER_AFFINITY, ": worker: ", w_idx, ", processor: ", worker_affinity[w_idx]);
    }

    LOG_INFO(CCL_ATL_TRANSPORT, ": ", str_by_enum(atl_transport_names, atl_transport));
    LOG_INFO(CCL_ATL_SHM, ": ", enable_shm);

    LOG_INFO(CCL_ALLGATHERV,
             ": ",
             (allgatherv_algo_raw.length()) ? allgatherv_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_ALLREDUCE,
             ": ",
             (allreduce_algo_raw.length()) ? allreduce_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_ALLTOALL,
             ": ",
             (alltoall_algo_raw.length()) ? alltoall_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_ALLTOALLV,
             ": ",
             (alltoallv_algo_raw.length()) ? alltoallv_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_BARRIER,
             ": ",
             (barrier_algo_raw.length()) ? barrier_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(
        CCL_BCAST, ": ", (bcast_algo_raw.length()) ? bcast_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(
        CCL_REDUCE, ": ", (reduce_algo_raw.length()) ? reduce_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_SPARSE_ALLREDUCE,
             ": ",
             (sparse_allreduce_algo_raw.length()) ? sparse_allreduce_algo_raw
                                                  : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_UNORDERED_COLL, ": ", enable_unordered_coll);

    LOG_INFO(CCL_FUSION, ": ", enable_fusion);
    LOG_INFO(CCL_FUSION_BYTES_THRESHOLD, ": ", fusion_bytes_threshold);
    LOG_INFO(CCL_FUSION_COUNT_THRESHOLD, ": ", fusion_count_threshold);
    LOG_INFO(CCL_FUSION_CHECK_URGENT, ": ", fusion_check_urgent);
    LOG_INFO(CCL_FUSION_CYCLE_MS, ": ", fusion_cycle_ms);

    LOG_INFO(CCL_RMA, ": ", enable_rma);
    LOG_INFO(CCL_PRIORITY, ": ", str_by_enum(priority_mode_names, priority_mode));
    LOG_INFO(CCL_SPIN_COUNT, ": ", spin_count);
    LOG_INFO(CCL_YIELD, ": ", str_by_enum(ccl_yield_type_names, yield_type));
    LOG_INFO(CCL_MAX_SHORT_SIZE, ": ", max_short_size);
    LOG_INFO(CCL_BCAST_PART_COUNT,
             ": ",
             (bcast_part_count != CCL_ENV_SIZET_NOT_SPECIFIED) ? std::to_string(bcast_part_count)
                                                               : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_CACHE_KEY, ": ", str_by_enum(ccl_sched_key::key_type_names, cache_key_type));
    LOG_INFO(CCL_CACHE_FLUSH, ": ", enable_cache_flush);

    LOG_INFO(CCL_CHUNK_COUNT, ": ", chunk_count);
    LOG_INFO(CCL_MIN_CHUNK_SIZE, ": ", min_chunk_size);
    LOG_INFO(CCL_RS_CHUNK_COUNT, ": ", rs_chunk_count);
    LOG_INFO(CCL_RS_MIN_CHUNK_SIZE, ": ", rs_min_chunk_size);
    LOG_INFO(CCL_AR2D_CHUNK_COUNT, ": ", ar2d_chunk_count);
    LOG_INFO(CCL_AR2D_MIN_CHUNK_SIZE, ": ", ar2d_min_chunk_size);

    LOG_INFO(CCL_ALLREDUCE_2D_BASE_SIZE,
             ": ",
             (allreduce_2d_base_size != CCL_ENV_SIZET_NOT_SPECIFIED)
                 ? std::to_string(allreduce_2d_base_size)
                 : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_ALLREDUCE_2D_SWITCH_DIMS, ": ", allreduce_2d_switch_dims);

    LOG_INFO(CCL_ALLTOALL_SCATTER_MAX_OPS,
             ": ",
             (alltoall_scatter_max_ops != CCL_ENV_SIZET_NOT_SPECIFIED)
                 ? std::to_string(alltoall_scatter_max_ops)
                 : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_ALLTOALL_SCATTER_PLAIN, ": ", alltoall_scatter_plain);
}

int env_data::env_2_worker_affinity_auto(size_t local_proc_idx, size_t workers_per_process) {
    char* available_cores = std::getenv(I_MPI_AVAILABLE_CORES_ENV);
    CCL_THROW_IF_NOT(available_cores && strlen(available_cores) != 0,
                     "auto pinning requires ",
                     I_MPI_AVAILABLE_CORES_ENV,
                     " env variable to be set");
    std::vector<size_t> cores;
    ccl_str_to_array(available_cores + 1, cores, ',');

    LOG_DEBUG("available_cores ", available_cores);

    CCL_THROW_IF_NOT(worker_count <= cores.size(),
                     "count of workers ",
                     worker_count,
                     " exceeds the number of available cores ",
                     cores.size());

    size_t ccl_cores_start = cores.size() - worker_count;
    for (size_t idx = 0; idx < worker_count; ++idx) {
        worker_affinity[local_proc_idx * workers_per_process + idx] = cores[ccl_cores_start + idx];
    }
    return 1;
}

int env_data::env_2_worker_affinity(size_t local_proc_idx, size_t local_proc_count) {
    CCL_THROW_IF_NOT(local_proc_count > 0);

    int read_env = 0;
    size_t workers_per_process = worker_count;
    size_t w_idx, read_count = 0;
    char* affinity_copy = nullptr;
    char* affinity_to_parse = getenv(CCL_WORKER_AFFINITY);
    char* proc_id_str;
    char* tmp;
    size_t proccessor_count;

    size_t affinity_size = local_proc_count * workers_per_process;
    worker_affinity.assign(affinity_size, 0);

    if (affinity_to_parse && strcmp(affinity_to_parse, "auto") == 0) {
        return env_2_worker_affinity_auto(local_proc_idx, workers_per_process);
    }

    if (!affinity_to_parse || strlen(affinity_to_parse) == 0) {
        /* generate default affinity */
        proccessor_count = sysconf(_SC_NPROCESSORS_ONLN);
        for (w_idx = 0; w_idx < affinity_size; w_idx++) {
            if (w_idx < proccessor_count) {
                worker_affinity[w_idx] = proccessor_count - w_idx - 1;
            }
            else {
                worker_affinity[w_idx] = worker_affinity[w_idx % proccessor_count];
            }
        }

        read_env = 1;
        CCL_FREE(affinity_copy);
        return read_env;
    }

    /* create copy of original buffer because it will be modified in strsep */
    size_t affinity_len = strlen(affinity_to_parse);
    affinity_copy = static_cast<char*>(CCL_CALLOC(affinity_len + 1, "affinity_copy"));
    CCL_MEMCPY(affinity_copy, affinity_to_parse, affinity_len);
    tmp = affinity_copy;

    for (w_idx = 0; w_idx < affinity_size; w_idx++) {
        proc_id_str = strsep(&tmp, ",");
        if (proc_id_str != NULL) {
            if (atoi(proc_id_str) < 0) {
                LOG_ERROR(
                    "unexpected proc_id ", proc_id_str, ", affinity string ", affinity_to_parse);
                read_env = 0;
                CCL_FREE(affinity_copy);
                return read_env;
            }
            worker_affinity[w_idx] = std::strtoul(proc_id_str, nullptr, 10);
            read_count++;
        }
        else {
            LOG_ERROR("unexpected end of affinity string, expected ",
                      affinity_size,
                      " numbers, read ",
                      read_count,
                      ", affinity string ",
                      affinity_to_parse);
            read_env = 0;
            CCL_FREE(affinity_copy);
            return read_env;
        }
    }
    if (read_count < affinity_size) {
        LOG_ERROR(
            "unexpected number of processors (specify 1 logical processor per 1 progress thread), affinity string ",
            affinity_to_parse);
        read_env = 0;
        CCL_FREE(affinity_copy);
        return read_env;
    }
    read_env = 1;

    CCL_FREE(affinity_copy);
    return read_env;
}

} /* namespace ccl */
