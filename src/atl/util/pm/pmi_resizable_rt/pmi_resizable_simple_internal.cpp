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
#include <unistd.h>

#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs_keeper.hpp"
#include "pmi_resizable_simple_internal.h"
#include "util/pm/codec/pm_rt_codec.h"

#define RESIZABLE_PMI_RT_KEY_FORMAT "%s-%d"
#define RANKS_PER_THREAD            "RANKS_PER_THREAD"
#define PROCESS_THREAD_NAME         "PROCESS_THREAD_NAME"

#define REQUESTED_RANK_TO_NAME "REQUESTED_RANK_TO_NAME"
#define GLOBAL_NAME_TO_RANK    "GLOBAL_NAME_TO_RANK"
#define GLOBAL_RANK_TO_NAME    "GLOBAL_RANK_TO_NAME"
#define LOCAL_KVS_ID           "LOCAL_KVS_ID"

#define INTERNAL_REGISTRATION                 "INTERNAL_REGISTRATION"
#define RANKCOUNT_RANK_PROCID_THREADID_FORMAT "%ld_%d_%d_%ld"

pmi_resizable_simple_internal::pmi_resizable_simple_internal(int size,
                                                             const std::vector<int>& ranks,
                                                             std::shared_ptr<internal_kvs> k,
                                                             const char* main_addr)
        : total_rank_count(size),
          ranks(ranks),
          k(k) {
    max_keylen = MAX_KVS_KEY_LENGTH;
    max_vallen = MAX_KVS_VAL_LENGTH;
    pmrt_init(main_addr);
}

int pmi_resizable_simple_internal::is_pm_resize_enabled() {
    return 0;
}

atl_status_t pmi_resizable_simple_internal::pmrt_init(const char* main_addr) {
    (void)main_addr;

    char* kvs_get_timeout_str = getenv("CCL_KVS_GET_TIMEOUT");
    if (kvs_get_timeout_str) {
        kvs_get_timeout = atoi(kvs_get_timeout_str);
    }

    local_id = 0;
    val_storage = (char*)calloc(1, max_vallen);
    if (!val_storage)
        return ATL_STATUS_FAILURE;
    local_id = get_local_kvs_id();
    barrier_full_reg();

    registration();

    if (ranks[0] == 0) {
        size_t tmp_local_id = get_local_kvs_id();
        tmp_local_id++;
        set_local_kvs_id(tmp_local_id);
    }
    if (thread_num == 0) {
        barrier_reg();
    }

    return ATL_STATUS_SUCCESS;
}

void pmi_resizable_simple_internal::registration() {
    std::string total_local_rank_count_str = std::to_string(total_rank_count);
    std::string result_kvs_name = std::string(INTERNAL_REGISTRATION) + std::to_string(local_id);
    memset(val_storage, 0, max_vallen);
    snprintf(val_storage,
             max_vallen,
             RANKCOUNT_RANK_PROCID_THREADID_FORMAT,
             ranks.size(),
             ranks[0],
             getpid(),
             gettid());
    k->kvs_set_size(
        result_kvs_name.c_str(), result_kvs_name.c_str(), total_local_rank_count_str.c_str());
    /*return string: %PROC_COUNT%_%RANK_NUM%_%PROCESS_RANK_COUNT%_%THREADS_COUNT%_%THREAD_NUM% */
    k->kvs_register(result_kvs_name.c_str(), result_kvs_name.c_str(), val_storage);

    char* proc_count_str = val_storage;
    char* rank_str = strstr(proc_count_str, "_");
    rank_str[0] = '\0';
    rank_str++;
    char* proc_rank_count_str = strstr(rank_str, "_");
    proc_rank_count_str[0] = '\0';
    proc_rank_count_str++;
    char* threads_count_str = strstr(proc_rank_count_str, "_");
    threads_count_str[0] = '\0';
    threads_count_str++;
    char* thread_num_str = strstr(threads_count_str, "_");
    thread_num_str[0] = '\0';
    thread_num_str++;

    proc_count = std::stoi(proc_count_str);
    rank = std::stoi(rank_str);
    proc_rank_count = std::stoi(proc_rank_count_str);
    threads_count = std::stoi(threads_count_str);
    thread_num = std::stoi(thread_num_str);
}

void pmi_resizable_simple_internal::barrier_full_reg() {
    std::string empty_line("");
    std::string total_local_rank_count_str =
        std::to_string(total_rank_count) + "_" + std::to_string(ranks.size());
    std::string result_kvs_name = std::string(KVS_BARRIER_FULL) + std::to_string(local_id);

    k->kvs_barrier_register(
        result_kvs_name.c_str(), result_kvs_name.c_str(), total_local_rank_count_str.c_str());
    pmrt_barrier_full();
}

void pmi_resizable_simple_internal::barrier_reg() {
    std::string empty_line("");
    std::string proc_count_str = std::to_string(proc_count);
    std::string result_kvs_name = std::string(KVS_BARRIER) + std::to_string(local_id);

    k->kvs_barrier_register(
        result_kvs_name.c_str(), result_kvs_name.c_str(), proc_count_str.c_str());
    pmrt_barrier_full();
}

atl_status_t pmi_resizable_simple_internal::pmrt_main_addr_reserve(char* main_addr) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple_internal::pmrt_set_resize_function(atl_resize_fn_t resize_fn) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple_internal::pmrt_update() {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple_internal::pmrt_wait_notification() {
    return ATL_STATUS_UNSUPPORTED;
}

void pmi_resizable_simple_internal::pmrt_finalize() {
    is_finalized = true;
    free(val_storage);

    if (getenv("CCL_PMI_FORCE_FINALIZE")) {
        printf("skip pmi_resizable_simple::pmrt_finalize\n");
        fflush(stdout);
        return;
    }

    char kvs_name[MAX_KVS_NAME_LENGTH];
    char kvs_key[MAX_KVS_KEY_LENGTH];
    char kvs_val[MAX_KVS_VAL_LENGTH];

    while (cut_head(kvs_name, kvs_key, kvs_val, ST_CLIENT)) {
        k->kvs_remove_name_key(kvs_name, kvs_key);
    }
}

void pmi_resizable_simple_internal::pmrt_barrier() {
    std::string empty_line("");
    std::string result_kvs_name = std::string(KVS_BARRIER) + std::to_string(local_id);

    k->kvs_barrier(result_kvs_name.c_str(), result_kvs_name.c_str(), empty_line.c_str());
}

void pmi_resizable_simple_internal::pmrt_barrier_full() {
    std::string empty_line("");
    std::string result_kvs_name = std::string(KVS_BARRIER_FULL) + std::to_string(local_id);

    k->kvs_barrier(result_kvs_name.c_str(), result_kvs_name.c_str(), (empty_line.c_str()));
}

atl_status_t pmi_resizable_simple_internal::pmrt_kvs_put(char* kvs_key,
                                                         int proc_idx,
                                                         const void* kvs_val,
                                                         size_t kvs_val_len) {
    int ret;
    char key_storage[max_keylen];
    if (kvs_val_len > max_vallen)
        return ATL_STATUS_FAILURE;

    ret = snprintf(key_storage, max_keylen - 1, RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = encode(kvs_val, kvs_val_len, val_storage, max_vallen);
    if (ret)
        return ATL_STATUS_FAILURE;

    kvs_set_value(KVS_NAME, key_storage, val_storage);

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple_internal::pmrt_kvs_get(char* kvs_key,
                                                         int proc_idx,
                                                         void* kvs_val,
                                                         size_t kvs_val_len) {
    int ret;
    char key_storage[max_keylen];

    ret = snprintf(key_storage, max_keylen - 1, RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    kvs_get_value(KVS_NAME, key_storage, val_storage);

    ret = decode(val_storage, kvs_val, kvs_val_len);
    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

int pmi_resizable_simple_internal::get_size() {
    return proc_count;
}

int pmi_resizable_simple_internal::get_rank() {
    return rank;
}

size_t pmi_resizable_simple_internal::get_local_thread_idx() {
    return thread_num;
}

size_t pmi_resizable_simple_internal::get_threads_per_process() {
    return threads_count;
}

size_t pmi_resizable_simple_internal::get_ranks_per_process() {
    return proc_rank_count;
}

int pmi_resizable_simple_internal::kvs_set_value(const char* kvs_name,
                                                 const char* key,
                                                 const char* value) {
    std::string result_kvs_name = std::string(kvs_name) + std::to_string(local_id);
    put_key(result_kvs_name.c_str(), key, value, ST_CLIENT);

    return k->kvs_set_value(result_kvs_name.c_str(), key, value);
}

int pmi_resizable_simple_internal::kvs_get_value(const char* kvs_name,
                                                 const char* key,
                                                 char* value) {
    std::string result_kvs_name = std::string(kvs_name) + std::to_string(local_id);

    time_t start_time = time(NULL);
    size_t kvs_get_time = 0;

    while (k->kvs_get_value_by_name_key(result_kvs_name.c_str(), key, value) == 0 &&
           kvs_get_time < kvs_get_timeout) {
        kvs_get_time = time(NULL) - start_time;
    }

    if (kvs_get_time >= kvs_get_timeout) {
        printf("KVS get error: timeout limit (%zu > %zu), prefix: %s, key %s\n",
               kvs_get_time,
               kvs_get_timeout,
               result_kvs_name.c_str(),
               key);
        exit(1);
    }

    return ATL_STATUS_SUCCESS;
}

size_t pmi_resizable_simple_internal::get_local_kvs_id() {
    char local_kvs_id[MAX_KVS_VAL_LENGTH];
    /*TODO: change it for collect local_per_rank id, not global*/
    if (k->kvs_get_value_by_name_key(LOCAL_KVS_ID, "ID", local_kvs_id) == 0)
        return 0;
    return atoi(local_kvs_id);
}

void pmi_resizable_simple_internal::set_local_kvs_id(size_t local_kvs_id) {
    /*TODO: change it for collect local_per_rank id, not global*/
    put_key(LOCAL_KVS_ID, "ID", std::to_string(local_kvs_id).c_str(), ST_CLIENT);
    k->kvs_set_value(LOCAL_KVS_ID, "ID", std::to_string(local_kvs_id).c_str());
}

pmi_resizable_simple_internal::~pmi_resizable_simple_internal() {
    if (!is_finalized)
        pmrt_finalize();
}
