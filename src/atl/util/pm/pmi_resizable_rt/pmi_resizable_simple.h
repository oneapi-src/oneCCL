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
#include <list>
#include <map>
#include <memory>
#include <vector>

#include "atl/atl_def.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "atl/util/pm/pm_rt.h"

#define PMIR_SUCCESS                0
#define PMIR_FAIL                   -1
#define PMIR_ERR_INIT               1
#define PMIR_ERR_NOMEM              2
#define PMIR_ERR_INVALID_ARG        3
#define PMIR_ERR_INVALID_KEY        4
#define PMIR_ERR_INVALID_KEY_LENGTH 5
#define PMIR_ERR_INVALID_VAL        6
#define PMIR_ERR_INVALID_VAL_LENGTH 7
#define PMIR_ERR_INVALID_LENGTH     8
#define PMIR_ERR_INVALID_NUM_ARGS   9
#define PMIR_ERR_INVALID_ARGS       10
#define PMIR_ERR_INVALID_NUM_PARSED 11
#define PMIR_ERR_INVALID_KEYVALP    12
#define PMIR_ERR_INVALID_SIZE       13

class pmi_resizable_simple final : public ipmi {
public:
    pmi_resizable_simple() = delete;
    pmi_resizable_simple(int total_rank_count,
                         const std::vector<int>& ranks,
                         std::shared_ptr<ikvs_wrapper> k,
                         const char* main_addr = nullptr);

    ~pmi_resizable_simple() override;

    int is_pm_resize_enabled() override;

    atl_status_t pmrt_main_addr_reserve(char* main_addr) override;

    atl_status_t pmrt_set_resize_function(atl_resize_fn_t resize_fn) override;

    atl_status_t pmrt_update() override;

    atl_status_t pmrt_wait_notification() override;

    void pmrt_barrier() override;

    atl_status_t pmrt_kvs_put(char* kvs_key,
                              int proc_idx,
                              const void* kvs_val,
                              size_t kvs_val_len) override;

    atl_status_t pmrt_kvs_get(char* kvs_key,
                              int proc_idx,
                              void* kvs_val,
                              size_t kvs_val_len) override;

    int get_size() override;

    int get_rank() override;

    size_t get_local_thread_idx() override;

    size_t get_local_kvs_id() override;

    void set_local_kvs_id(size_t local_kvs_id) override;

    size_t get_threads_per_process() override {
        return threads_per_proc[assigned_proc_idx].size();
    }

    size_t get_ranks_per_process() override {
        size_t res = 0;
        std::list<size_t>& thread_idxs = threads_per_proc[assigned_proc_idx];
        for (auto it = thread_idxs.begin(); it != thread_idxs.end(); it++) {
            res += ranks_per_thread_map[*it];
        }
        return res;
    }

    void pmrt_finalize() override;

private:
    bool is_finalized{ false };
    atl_status_t pmrt_init(const char* main_addr = nullptr);

    int kvs_set_value(const char* kvs_name, const char* key, const char* value);
    int kvs_get_value(const char* kvs_name, const char* key, char* value);
    int kvs_iget_value(const char* kvs_name, const char* key, char* value);

    size_t get_barrier_idx();
    size_t get_barrier_full_idx();

    void calculate_local_thread_idx();
    void register_first_rank_idx_and_rank_count();
    void assign_thread_idx_and_fill_ranks_per_thread_map();
    void register_my_proc_name();
    void get_my_proc_idx_and_proc_count();
    void make_requested_info();
    void remove_initial_data();
    void make_map_requested2global();
    void pmrt_barrier_full();

    int total_rank_count;
    int assigned_proc_idx;

    size_t assigned_thread_idx;
    size_t local_thread_idx;
    std::string my_proccess_name;
    std::vector<int> ranks;
    std::vector<size_t> ranks_per_thread_map;
    std::map<size_t, std::list<size_t>> threads_per_proc;
    std::shared_ptr<ikvs_wrapper> k;
    size_t max_keylen;
    size_t max_vallen;
    char* val_storage = nullptr;
    size_t barrier_num = 0;
    size_t barrier_num_full = 0;
    std::vector<int> requested2global;
    size_t local_id;
    size_t connection_timeout = 120; /* in seconds */
};
