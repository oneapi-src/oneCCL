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
#include <climits>
#include <dlfcn.h>
#include <iterator>
#include <memory>
#include <sstream>
#include <unistd.h>

#include "coll/selection/selection.hpp"
#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "exec/exec.hpp"
#include "oneapi/ccl/environment.hpp"
#include "common/utils/version.hpp"

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

std::map<ccl_staging_buffer, std::string> env_data::staging_buffer_names = {
    std::make_pair(ccl_staging_regular, "regular"),
    std::make_pair(ccl_staging_usm, "usm")
};

std::map<atl_mnic_t, std::string> env_data::mnic_type_names = {
    std::make_pair(ATL_MNIC_NONE, "none"),
    std::make_pair(ATL_MNIC_LOCAL, "local"),
    std::make_pair(ATL_MNIC_GLOBAL, "global")
};

env_data::env_data()
        : was_printed(false),

          log_level(ccl_log_level::warn),
          sched_dump(0),

          fw_type(ccl_framework_none),

          worker_count(1),
          worker_offload(1),
          worker_wait(1),

          atl_transport(ccl_atl_mpi),
          enable_shm(0),
          enable_rma(0),
          enable_device_buf(0),
          enable_sync_coll(0),
          enable_extra_ep(0),

          mnic_type(ATL_MNIC_NONE),
          mnic_count(4),

          enable_unordered_coll(0),

          enable_fusion(0),
          fusion_bytes_threshold(16384),
          fusion_count_threshold(256),
          fusion_check_urgent(1),
          fusion_cycle_ms(0.2),

          priority_mode(ccl_priority_none),
          spin_count(100),
          yield_type(ccl_yield_pause),
          max_short_size(0),
          bcast_part_count(CCL_ENV_SIZET_NOT_SPECIFIED),
          cache_key_type(ccl_cache_key_match_id),
          enable_cache_flush(0),
          enable_strict_order(0),
          staging_buffer(ccl_staging_usm),

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

          enable_comm_kernels(0),
          comm_kernels_path(),
          comm_kernels_debug(0),
          gpu_thread_count(CCL_ENV_SIZET_NOT_SPECIFIED),

          bf16_impl_type(ccl_bf16_no_compiler_support),
          fp16_impl_type(ccl_fp16_no_compiler_support) {}

void env_data::parse() {
    env_2_enum(CCL_LOG_LEVEL, ccl_logger::level_names, log_level);
    ccl_logger::set_log_level(log_level);
    env_2_type(CCL_SCHED_DUMP, sched_dump);

    if (fw_type == ccl_framework_none) {
        /* try to automatically detect framework */
        void* handle = dlopen(NULL, RTLD_GLOBAL | RTLD_NOW);
        if (handle) {
            horovod_init_function =
                (ccl_horovod_init_function)dlsym(handle, horovod_init_function_name);
            dlclose(handle);
        }

        if (horovod_init_function) {
            LOG_INFO("found horovod_init function");
            fw_type = ccl_framework_horovod;
        }
    }
    env_2_enum(CCL_FRAMEWORK, ccl_framework_type_names, fw_type);

    if (fw_type == ccl_framework_horovod) {
        worker_wait = 1;
        enable_sync_coll = 1;
        enable_extra_ep = 1;
        yield_type = ccl_yield_sched_yield;
    }

    env_2_type(CCL_WORKER_COUNT, worker_count);
    CCL_THROW_IF_NOT(worker_count >= 1, "incorrect ", CCL_WORKER_COUNT, " ", worker_count);
    env_2_type(CCL_WORKER_OFFLOAD, worker_offload);
    env_2_type(CCL_WORKER_WAIT, worker_wait);

    env_2_atl_transport();
    env_2_type(CCL_ATL_SHM, enable_shm);
    env_2_type(CCL_ATL_RMA, enable_rma);
    env_2_type(CCL_ATL_DEVICE_BUF, enable_device_buf);
    env_2_type(CCL_ATL_SYNC_COLL, enable_sync_coll);
    env_2_type(CCL_ATL_EXTRA_EP, enable_extra_ep);

    env_2_enum(CCL_MNIC, mnic_type_names, mnic_type);
    env_2_type(CCL_MNIC_COUNT, mnic_count);

    env_2_type(CCL_ALLGATHERV, allgatherv_algo_raw);
    env_2_type(CCL_ALLREDUCE, allreduce_algo_raw);
    env_2_type(CCL_ALLTOALL, alltoall_algo_raw);
    env_2_type(CCL_ALLTOALLV, alltoallv_algo_raw);
    env_2_type(CCL_BARRIER, barrier_algo_raw);
    env_2_type(CCL_BCAST, bcast_algo_raw);
    env_2_type(CCL_REDUCE, reduce_algo_raw);
    env_2_type(CCL_REDUCE_SCATTER, reduce_scatter_algo_raw);
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

    if (!worker_offload || enable_fusion)
        worker_wait = 0;

    if (worker_wait)
        spin_count = 1000;

    env_2_enum(CCL_PRIORITY, priority_mode_names, priority_mode);
    env_2_type(CCL_SPIN_COUNT, spin_count);
    env_2_enum(CCL_YIELD, ccl_yield_type_names, yield_type);
    env_2_type(CCL_MAX_SHORT_SIZE, max_short_size);
    env_2_type(CCL_BCAST_PART_COUNT, (size_t&)bcast_part_count);
    env_2_enum(CCL_CACHE_KEY, ccl_sched_key::key_type_names, cache_key_type);
    env_2_type(CCL_CACHE_FLUSH, enable_cache_flush);
    env_2_type(CCL_STRICT_ORDER, enable_strict_order);
    env_2_enum(CCL_STAGING_BUFFER, staging_buffer_names, staging_buffer);

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

    env_2_type(CCL_COMM_KERNELS, enable_comm_kernels);
    if (enable_comm_kernels) {
        env_2_type(CCL_COMM_KERNELS_PATH, comm_kernels_path);
        if (comm_kernels_path.empty()) {
            std::string ccl_root = getenv("CCL_ROOT");
            CCL_THROW_IF_NOT(!ccl_root.empty(), "incorrect comm kernels path, CCL_ROOT not found!");
            comm_kernels_path = ccl_root + "/lib/kernels/";
        }
        env_2_type(CCL_COMM_KERNELS_DEBUG, comm_kernels_debug);
    }
    env_2_type(CCL_GPU_THREAD_COUNT, gpu_thread_count);

    auto bf16_impl_types = ccl_bf16_get_impl_types();
    ccl_bf16_impl_type bf16_env_impl_type;
    if (env_2_enum(CCL_BF16, bf16_env_impl_names, bf16_env_impl_type)) {
        CCL_THROW_IF_NOT(bf16_impl_types.find(bf16_env_impl_type) != bf16_impl_types.end(),
                         "unsupported BF16 impl type: ",
                         bf16_env_impl_names[bf16_env_impl_type]);
        bf16_impl_type = bf16_env_impl_type;
    }
    else {
        bf16_impl_type = *bf16_impl_types.rbegin();
    }

    auto fp16_impl_types = ccl_fp16_get_impl_types();
    ccl_fp16_impl_type fp16_env_impl_type;
    if (env_2_enum(CCL_FP16, fp16_env_impl_names, fp16_env_impl_type)) {
        CCL_THROW_IF_NOT(fp16_impl_types.find(fp16_env_impl_type) != fp16_impl_types.end(),
                         "unsupported FP16 impl type: ",
                         fp16_env_impl_names[fp16_env_impl_type]);
        fp16_impl_type = fp16_env_impl_type;
    }
    else {
        fp16_impl_type = *fp16_impl_types.rbegin();
    }
}

void env_data::print(int rank) {
    std::lock_guard<ccl_spinlock> lock{ print_guard };

    if (was_printed)
        return;
    else
        was_printed = true;

    if (rank == 0) {
        auto version = utils::get_library_version();
        LOG_INFO("library version: ", version.full);
        LOG_INFO("specification version: ", ONECCL_SPEC_VERSION);
#ifdef CCL_ENABLE_SYCL
        LOG_INFO("compute backend: ", version.cl_backend_name);
#endif /* CCL_ENABLE_SYCL */

#ifdef ENABLE_DEBUG
        const char* build_mode = "debug";
#else /* ENABLE_DEBUG */
        const char* build_mode = "release";
#endif /* ENABLE_DEBUG */
        LOG_INFO("build mode: ", build_mode);
        LOG_INFO("C compiler: ", CCL_C_COMPILER);
        LOG_INFO("C++ compiler: ", CCL_CXX_COMPILER);
    }

    auto& global_data = ccl::global_data::get();
    auto local_proc_idx = global_data.executor->get_local_proc_idx();
    auto local_proc_count = global_data.executor->get_local_proc_count();

    if (rank < (int)local_proc_count) {
        for (size_t w_idx = 0; w_idx < worker_count; w_idx++) {
            LOG_INFO(CCL_WORKER_AFFINITY,
                     ": local process [",
                     local_proc_idx,
                     ":",
                     local_proc_count,
                     "]: worker: ",
                     w_idx,
                     ", core: ",
                     worker_affinity[local_proc_idx * worker_count + w_idx]);
        }
    }

    if (rank != 0)
        return;

    LOG_INFO(CCL_WORKER_COUNT, ": ", worker_count);
    LOG_INFO(CCL_WORKER_OFFLOAD, ": ", worker_offload);
    LOG_INFO(CCL_WORKER_WAIT, ": ", worker_wait);

    LOG_INFO(CCL_LOG_LEVEL, ": ", str_by_enum(ccl_logger::level_names, log_level));
    LOG_INFO(CCL_SCHED_DUMP, ": ", sched_dump);

    LOG_INFO(CCL_FRAMEWORK, ": ", str_by_enum(ccl_framework_type_names, fw_type));

    LOG_INFO(CCL_ATL_TRANSPORT, ": ", str_by_enum(atl_transport_names, atl_transport));
    LOG_INFO(CCL_ATL_SHM, ": ", enable_shm);
    LOG_INFO(CCL_ATL_RMA, ": ", enable_rma);
    LOG_INFO(CCL_ATL_DEVICE_BUF, ": ", enable_device_buf);
    LOG_DEBUG(CCL_ATL_SYNC_COLL, ": ", enable_sync_coll);
    LOG_DEBUG(CCL_ATL_EXTRA_EP, ": ", enable_extra_ep);

    LOG_INFO(CCL_MNIC, ": ", str_by_enum(mnic_type_names, mnic_type));
    LOG_INFO(CCL_MNIC_COUNT, ": ", mnic_count);

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
    LOG_INFO(
        CCL_REDUCE_SCATTER,
        ": ",
        (reduce_scatter_algo_raw.length()) ? reduce_scatter_algo_raw : CCL_ENV_STR_NOT_SPECIFIED);
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
    LOG_INFO(CCL_STRICT_ORDER, ": ", enable_strict_order);
    LOG_INFO(CCL_STAGING_BUFFER, ": ", str_by_enum(staging_buffer_names, staging_buffer));

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

#ifdef CCL_ENABLE_SYCL
    LOG_INFO(CCL_COMM_KERNELS, ": ", enable_comm_kernels);
    LOG_INFO(CCL_COMM_KERNELS_PATH,
             ": ",
             (!comm_kernels_path.empty()) ? comm_kernels_path : CCL_ENV_STR_NOT_SPECIFIED);
    LOG_INFO(CCL_COMM_KERNELS_DEBUG, ": ", comm_kernels_debug);
    LOG_INFO(CCL_GPU_THREAD_COUNT,
             ": ",
             (gpu_thread_count != CCL_ENV_SIZET_NOT_SPECIFIED) ? std::to_string(gpu_thread_count)
                                                               : CCL_ENV_STR_NOT_SPECIFIED);
#endif /* CCL_ENABLE_SYCL  */

    LOG_INFO(CCL_BF16, ": ", str_by_enum(bf16_impl_names, bf16_impl_type));
    LOG_INFO(CCL_FP16, ": ", str_by_enum(fp16_impl_names, fp16_impl_type));

    char* ccl_root = getenv("CCL_ROOT");
    LOG_INFO("CCL_ROOT: ", (ccl_root) ? ccl_root : CCL_ENV_STR_NOT_SPECIFIED);

    char* impi_root = getenv("I_MPI_ROOT");
    LOG_INFO("I_MPI_ROOT: ", (impi_root) ? impi_root : CCL_ENV_STR_NOT_SPECIFIED);

    char* fi_provider_path = getenv("FI_PROVIDER_PATH");
    LOG_INFO("FI_PROVIDER_PATH: ",
             (fi_provider_path) ? fi_provider_path : CCL_ENV_STR_NOT_SPECIFIED);

    char* fi_provider = getenv("FI_PROVIDER");
    LOG_INFO("FI_PROVIDER: ", (fi_provider) ? fi_provider : CCL_ENV_STR_NOT_SPECIFIED);

    global_data.algorithm_selector->print();
}

void env_data::set_internal_env() {
    auto attr = ccl_executor::generate_atl_attr(*this);
    atl_wrapper::set_internal_env(attr);
    if (log_level >= ccl_log_level::info) {
        setenv("I_MPI_DEBUG", "4", 0);
    }
}

int env_data::env_2_worker_affinity_auto(size_t local_proc_idx, size_t workers_per_process) {
    char* available_cores = std::getenv(I_MPI_AVAILABLE_CORES_ENV);
    CCL_THROW_IF_NOT(available_cores && strlen(available_cores) != 0,
                     "auto pinning requires ",
                     I_MPI_AVAILABLE_CORES_ENV,
                     " env variable to be set");

    LOG_DEBUG("available_cores ", available_cores);

    std::set<char> delims;
    for (char c : std::string(I_MPI_AVAILABLE_CORES_DELIMS)) {
        delims.insert(c);
    }
    std::vector<size_t> cores;
    ccl_str_to_array(available_cores, delims, cores);

    CCL_THROW_IF_NOT(workers_per_process <= cores.size(),
                     "failed to implicitly set workers affinity, "
                     "the number of workers (",
                     workers_per_process,
                     ") exceeds the number of available cores per process (",
                     cores.size(),
                     "), consider increasing the number of cores per process ",
                     "or explicitly setting of workers affinity using ",
                     CCL_WORKER_AFFINITY);

    if ((workers_per_process == cores.size()) && worker_offload) {
        LOG_WARN("the number of workers (",
                 workers_per_process,
                 ") matches the number of available cores per process,"
                 " this may lead to contention between workers and"
                 " application threads");
        if (!std::getenv(CCL_WORKER_OFFLOAD)) {
            worker_offload = 0;
            LOG_WARN("workers are disabled,"
                     " to forcibly enable them set ",
                     CCL_WORKER_OFFLOAD,
                     "=1");
        }
        else {
            LOG_WARN("consider increasing the number of cores per process",
                     " or disabling workers using ",
                     CCL_WORKER_OFFLOAD,
                     "=0");
        }
    }

    size_t worker_cores_start = cores.size() - workers_per_process;
    for (size_t idx = 0; idx < workers_per_process; ++idx) {
        worker_affinity[local_proc_idx * workers_per_process + idx] =
            cores[worker_cores_start + idx];
    }
    return 1;
}

int env_data::parse_core_id(const std::string& core_id_str, size_t& result) {
    char* end_ptr;
    const char* core_id_str_ptr = core_id_str.c_str();

    errno = 0;
    auto core_id = std::strtol(core_id_str_ptr, &end_ptr, 10);

    if ((errno == ERANGE && (core_id == LONG_MAX || core_id == LONG_MIN)) ||
        (errno != 0 && core_id == 0)) {
        LOG_ERROR("core id value is invalid in string: ", core_id_str);
        return 0;
    }
    if (end_ptr == core_id_str_ptr) {
        LOG_ERROR("no digits were found in string: ", core_id_str);
        return 0;
    }
    if (core_id < 0) {
        LOG_ERROR(
            "core id cannot be less than zero but got ", core_id, " in string: ", core_id_str);
        return 0;
    }
    result = core_id;
    return 1;
}

int env_data::env_2_worker_affinity(size_t local_proc_idx, size_t local_proc_count) {
    CCL_THROW_IF_NOT(local_proc_count > 0);

    size_t idx;
    std::unique_ptr<char> affinity_copy;
    char* affinity_to_parse = getenv(CCL_WORKER_AFFINITY);
    char* core_range_str;
    char* tmp;
    size_t system_core_count;

    size_t affinity_size = local_proc_count * worker_count;

    if (!affinity_to_parse || (strlen(affinity_to_parse) == 0) ||
        (strcmp(affinity_to_parse, "auto") == 0)) {
        worker_affinity.assign(affinity_size, 0);
        if (std::getenv(I_MPI_AVAILABLE_CORES_ENV)) {
            /* generate auto affinity based on IMPI process pinning */
            return env_2_worker_affinity_auto(local_proc_idx, worker_count);
        }
        else {
            /* generate auto affinity as last N cores */
            system_core_count = sysconf(_SC_NPROCESSORS_ONLN);
            for (idx = 0; idx < affinity_size; idx++) {
                if (idx < system_core_count) {
                    worker_affinity[idx] = system_core_count - idx - 1;
                }
                else {
                    worker_affinity[idx] = worker_affinity[idx % system_core_count];
                }
            }
            return 1;
        }
    }

    /* create copy of original buffer because it will be modified in strsep */
    size_t affinity_len = strlen(affinity_to_parse);
    affinity_copy =
        std::unique_ptr<char>(static_cast<char*>(CCL_CALLOC(affinity_len + 1, "affinity_copy")));
    CCL_MEMCPY(affinity_copy.get(), affinity_to_parse, affinity_len);
    tmp = affinity_copy.get();

    while (tmp) {
        core_range_str = strsep(&tmp, ",");
        if (!core_range_str) {
            break;
        }

        auto core_range = tokenize<std::vector<std::string>>(std::string(core_range_str), '-');

        if ((core_range.size() != 2) && (core_range.size() != 1)) {
            LOG_ERROR(
                "unexpected format in affinity: ",
                affinity_to_parse,
                ", specify core range using <first_core>-<last_core> or single core using <core>");
            return 0;
        }

        if (core_range.size() == 1) {
            /* to unify logic below */
            core_range.push_back(*core_range.begin());
        }

        CCL_ASSERT(core_range.size() == 2, "unexpected number of cores in range");

        size_t first_core, last_core;
        if (!parse_core_id(core_range[0], first_core) || !parse_core_id(core_range[1], last_core)) {
            return 0;
        }

        if (first_core > last_core) {
            LOG_ERROR("unexpected first and last cores in range: ",
                      core_range_str,
                      ", first core should be less or equal to last core");
            return 0;
        }

        for (idx = first_core; idx <= last_core; idx++) {
            worker_affinity.push_back(idx);
        }
    }

    if (worker_affinity.size() < affinity_size) {
        LOG_ERROR("unexpected number of cores in affinity: ",
                  affinity_to_parse,
                  ", specify 1 core per 1 worker thread");
        return 0;
    }
    return 1;
}

void env_data::env_2_atl_transport() {
    if (!getenv(CCL_ATL_TRANSPORT) && !with_mpirun()) {
        LOG_WARN("did not find MPI-launcher specific variables, switch to ATL/OFI, "
                 "to force enable ATL/MPI set CCL_ATL_TRANSPORT=mpi");

        atl_transport = ccl_atl_ofi;
    }
    else
        env_2_enum(CCL_ATL_TRANSPORT, atl_transport_names, atl_transport);
}

bool env_data::with_mpirun() {
    return (getenv("MPI_LOCALRANKID") || getenv("MPI_LOCALNRANKS") || getenv("PMI_RANK") ||
            getenv("PMI_SIZE"))
               ? true
               : false;
}

} /* namespace ccl */
