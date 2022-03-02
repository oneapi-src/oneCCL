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
#include <memory>

#include "atl/atl_def.h"
#include "atl/util/pm/pm_rt.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/helper.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/pmi_listener.hpp"

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

typedef enum {
    KVS_RA_WAIT = 0,
    KVS_RA_RUN = 1,
    KVS_RA_FINALIZE = 2,
} kvs_resize_action_t;
typedef kvs_resize_action_t (*pmir_resize_fn_t)(int comm_size);

class helper;
class pmi_resizable final : public ipmi {
public:
    pmi_resizable() = delete;
    explicit pmi_resizable(std::shared_ptr<ikvs_wrapper> k, const char* main_addr = "")
            : main_addr(main_addr),
              h(std::make_shared<helper>(k)) {}

    ~pmi_resizable() override;

    int is_pm_resize_enabled() override;

    atl_status_t pmrt_main_addr_reserve(char* main_addr) override;

    atl_status_t pmrt_set_resize_function(atl_resize_fn_t resize_fn) override;

    atl_status_t pmrt_update() override;

    atl_status_t pmrt_wait_notification() override;

    atl_status_t pmrt_barrier() override;

    atl_status_t pmrt_kvs_put(char* kvs_key,
                              int proc_idx,
                              const void* kvs_val,
                              size_t kvs_val_len) override;

    atl_status_t pmrt_kvs_get(char* kvs_key,
                              int proc_idx,
                              void* kvs_val,
                              size_t kvs_val_len) override;

    kvs_status_t hard_finalize(int sig);

    int get_rank() override;

    int get_size() override;

    size_t get_local_thread_idx() override;

    atl_status_t get_local_kvs_id(size_t& res) override;

    atl_status_t set_local_kvs_id(size_t local_kvs_id) override;

    size_t get_threads_per_process() override {
        return 1;
    }

    size_t get_ranks_per_process() override {
        return 1;
    }
    atl_status_t pmrt_finalize() override;

    atl_status_t pmrt_init() override;

private:
    bool is_finalized{ false };
    /*Was in API ->*/
    kvs_status_t PMIR_Main_Addr_Reserve(char* main_addr);

    kvs_status_t PMIR_Init(const char* main_addr);

    kvs_status_t PMIR_Finalize(void);

    kvs_status_t PMIR_Get_size(int* size);

    kvs_status_t PMIR_Get_rank(int* rank);

    kvs_status_t PMIR_KVS_Get_my_name(char* kvs_name, size_t length);

    kvs_status_t PMIR_KVS_Get_name_length_max(size_t* length);

    kvs_status_t PMIR_Barrier(void);

    kvs_status_t PMIR_Update(void);

    kvs_status_t PMIR_KVS_Get_key_length_max(size_t* length);

    kvs_status_t PMIR_KVS_Get_value_length_max(size_t* length);

    kvs_status_t PMIR_KVS_Put(const char* kvs_name, const char* key, const char* value);

    kvs_status_t PMIR_KVS_Commit(const char* kvs_name);

    kvs_status_t PMIR_KVS_Get(const char* kvs_name, const char* key, char* value, size_t length);

    kvs_status_t PMIR_set_resize_function(pmir_resize_fn_t resize_fn);

    kvs_status_t PMIR_Wait_notification(void);
    /* <- Was in API*/
    kvs_resize_action_t default_checker(int comm_size);
    kvs_resize_action_t call_resize_fn(int comm_size);

    int rank = -1;
    int size = -1;
    std::string main_addr;

    pmir_resize_fn_t resize_function = nullptr;
    std::shared_ptr<helper> h;
    pmi_listener listener;
    bool initialized = false;
    size_t max_keylen{};
    size_t max_vallen{};
    char* key_storage = nullptr;
    char* val_storage = nullptr;
    char* kvsname = nullptr;
};
