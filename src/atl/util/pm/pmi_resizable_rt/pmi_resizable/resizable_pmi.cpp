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
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"

static int root_rank = 0;
static size_t is_new_root = 0;
static size_t ask_only_framework = 0;
static size_t finalized = 0;
static size_t extreme_finalize = 0;
static struct sigaction old_act;
char pmi_hostname[MAX_KVS_VAL_LENGTH];

// TODO: rework it for multi kvs
static pmi_resizable* pmi_object;

void call_hard_finalize(int sig) {
    if (pmi_object->hard_finalize(sig) != KVS_STATUS_SUCCESS) {
        LOG_ERROR("failed to hard finalize");
    }
}

kvs_resize_action_t pmi_resizable::default_checker(int comm_size) {
    char* comm_size_to_start_env;
    size_t comm_size_to_start;

    comm_size_to_start_env = getenv(CCL_WORLD_SIZE_ENV);

    if (comm_size_to_start_env != NULL) {
        if (safe_strtol(comm_size_to_start_env, comm_size_to_start) != KVS_STATUS_SUCCESS) {
            LOG_ERROR("failed to convert comm_size");
            return KVS_RA_FINALIZE;
        }
    }
    else {
        if (h->get_replica_size(comm_size_to_start) != KVS_STATUS_SUCCESS) {
            LOG_ERROR("failed to get comm_size");
            return KVS_RA_FINALIZE;
        }
    }
    if (comm_size >= static_cast<int>(comm_size_to_start))
        return KVS_RA_RUN;

    return KVS_RA_WAIT;
}

kvs_resize_action_t pmi_resizable::call_resize_fn(int comm_size) {
    if (resize_function != nullptr)
        return resize_function(comm_size);

    return default_checker(comm_size);
}

kvs_status_t pmi_resizable::PMIR_Update(void) {
    std::string up_idx_str(MAX_KVS_VAL_LENGTH, 0);
    int prev_new_ranks_count = 0;
    int prev_killed_ranks_count = 0;
    int prev_idx = -1;
    kvs_resize_action_t answer;
    std::list<int> dead_up_idx{};
    std::list<shift_rank_t> list{};

    // this code is a part of undocumented resizable_pmi functionality
    new_ranks_count = 0;
    killed_ranks_count = 0;
    if (finalized == 1) {
        LOG_ERROR("is finalized");
        return KVS_STATUS_FAILURE;
    }
    if (applied == 1) {
        size_t is_wait = 1;
        size_t is_first_collect = 0;

        KVS_CHECK_STATUS(h->get_value_by_name_key(KVS_UP, KVS_IDX, up_idx_str),
                         "failed to get KVS IDx");

        KVS_CHECK_STATUS(safe_strtol(up_idx_str.c_str(), up_idx), "failed to convert KVS IDx");
        if (up_idx == 0)
            is_first_collect = 1;

        do {
            size_t count_clean_checks = 0;
            size_t count_applied_changes = 0;
            do {
                /*Waiting new pods*/
                usleep(10000);
                KVS_CHECK_STATUS(h->get_value_by_name_key(KVS_UP, KVS_IDX, up_idx_str),
                                 "failed to get KVS IDx");

                KVS_CHECK_STATUS(safe_strtol(up_idx_str.c_str(), up_idx),
                                 "failed to convert KVS IDx");
                if (prev_idx == (int)up_idx) {
                    count_clean_checks = 0;

                    h->keep_first_n_up(prev_new_ranks_count, prev_killed_ranks_count);
                    count_applied_changes -= new_ranks_count - prev_new_ranks_count +
                                             killed_ranks_count - prev_killed_ranks_count;
                    new_ranks_count = prev_new_ranks_count;
                    killed_ranks_count = prev_killed_ranks_count;

                    //TODO: Add logic for getting new root

                    //                    while (int_list_is_contained(killed_ranks, root_rank) == 1)
                    {
                        int old_root = root_rank;

                        KVS_CHECK_STATUS(h->get_new_root(&root_rank), "failed to new root rank");

                        if (my_rank == root_rank && old_root != root_rank)
                            is_new_root = 1;
                        else
                            is_new_root = 0;
                    }
                }
                else {
                    prev_idx = up_idx;
                    prev_new_ranks_count = new_ranks_count;
                    prev_killed_ranks_count = killed_ranks_count;

                    KVS_CHECK_STATUS(h->get_update_ranks(), "failed to update ranks");
                    if (killed_ranks_count != prev_killed_ranks_count)
                        dead_up_idx.push_back(up_idx);
                }
                KVS_CHECK_STATUS(PMIR_Barrier(), "barrier failed");
                if (my_rank == root_rank && is_new_root == 0) {
                    up_idx++;
                    if (up_idx > 0 && up_idx > MAX_UP_IDX)
                        up_idx = 1;

                    SET_STR(const_cast<char*>(up_idx_str.data()),
                            INT_STR_SIZE,
                            SIZE_T_TEMPLATE,
                            up_idx);
                    KVS_CHECK_STATUS(h->set_value(KVS_UP, KVS_IDX, up_idx_str),
                                     "failed to set KVS IDx");
                    KVS_CHECK_STATUS(h->up_kvs_new_and_dead(), "failed to update KVS");
                }
                KVS_CHECK_STATUS(PMIR_Barrier(), "barrier failed");

                if (finalized == 1) {
                    killed_ranks.clear();
                    new_ranks.clear();
                    dead_up_idx.clear();
                    LOG_ERROR("is finalized")
                    return KVS_STATUS_FAILURE;
                }

                is_new_root = 0;
                count_applied_changes += new_ranks_count - prev_new_ranks_count +
                                         killed_ranks_count - prev_killed_ranks_count;

                if ((prev_new_ranks_count != new_ranks_count) ||
                    (prev_killed_ranks_count != killed_ranks_count))
                    count_clean_checks = 0;
                else
                    count_clean_checks++;
            } while (count_clean_checks != MAX_CLEAN_CHECKS);

            if (!is_first_collect || ask_only_framework == 1)
                answer = call_resize_fn(count_pods - killed_ranks_count + new_ranks_count);
            else {
                size_t replica_size;
                KVS_CHECK_STATUS(h->get_replica_size(replica_size), "failed to get replica size");
                if (replica_size != count_pods - killed_ranks_count + new_ranks_count)
                    answer = KVS_RA_WAIT;
                else
                    answer = KVS_RA_RUN;
            }

            switch (answer) {
                case KVS_RA_WAIT: {
                    break;
                }
                case KVS_RA_RUN: {
                    is_wait = 0;
                    break;
                }
                case KVS_RA_FINALIZE: {
                    KVS_CHECK_STATUS(PMIR_Finalize(), "failed to finalize");
                    break;
                }
                default: {
                    LOG_ERROR("Unknown resize action: %d\n", answer);
                    KVS_CHECK_STATUS(PMIR_Finalize(), "failed to finalize");
                    return KVS_STATUS_FAILURE;
                }
            }
            listener.set_applied_count(count_applied_changes);
        } while (is_wait == 1);
    }
    else {
        KVS_CHECK_STATUS(listener.send_notification(0, h), "failed to send notification");
        KVS_CHECK_STATUS(h->wait_accept(), "failed to wait accept");
    }

    h->get_shift(list);
    count_pods = count_pods - killed_ranks_count + new_ranks_count;

    KVS_CHECK_STATUS(h->update(list, dead_up_idx, root_rank), "failed to update root");

    root_rank = 0;

    KVS_CHECK_STATUS(PMIR_Barrier(), "barrier failed");
    KVS_CHECK_STATUS(h->up_pods_count(), "failed to update pods count");

    killed_ranks.clear();
    new_ranks.clear();
    dead_up_idx.clear();
    list.clear();
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::hard_finalize(int sig) {
    char rank_str[INT_STR_SIZE];

    SET_STR(rank_str, INT_STR_SIZE, RANK_TEMPLATE, my_rank);

    KVS_CHECK_STATUS(h->set_value(KVS_DEAD_POD, pmi_hostname, rank_str), "failed to set dead rank");

    KVS_CHECK_STATUS(listener.send_notification(sig, h), "failed to send notification");

    extreme_finalize = 1;
    KVS_CHECK_STATUS(PMIR_Finalize(), "failed to finalize");
    if (old_act.sa_handler != NULL)
        old_act.sa_handler(sig);

    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_Main_Addr_Reserve(char* addr) {
    return h->main_server_address_reserve(addr);
}

kvs_status_t pmi_resizable::PMIR_Init(const char* addr) {
    struct sigaction act;
    FILE* fp;
    finalized = 0;
    memset(pmi_hostname, 0, MAX_KVS_VAL_LENGTH);
    if ((fp = popen("hostname", READ_ONLY)) == NULL) {
        printf("Can't get hostname\n");
        exit(1);
    }
    CHECK_FGETS(fgets(pmi_hostname, MAX_KVS_VAL_LENGTH, fp), pmi_hostname, fp);
    while (pmi_hostname[strlen(pmi_hostname) - 1] == '\n' ||
           pmi_hostname[strlen(pmi_hostname) - 1] == ' ')
        pmi_hostname[strlen(pmi_hostname) - 1] = '\0';

    SET_STR(&(pmi_hostname[strlen(pmi_hostname)]),
            MAX_KVS_VAL_LENGTH - (int)strlen(pmi_hostname) - 1,
            "-%d",
            getpid());

    KVS_CHECK_STATUS(h->init(addr), "failed to init");

    KVS_CHECK_STATUS(h->reg_rank(), "failed to rank register");

    KVS_CHECK_STATUS(h->up_pods_count(), "failed to update pods count");

    // TODO: rework it for multi kvs
    pmi_object = this;
    memset(&act, 0, sizeof(act));
    act.sa_handler = &call_hard_finalize;
    act.sa_flags = 0;
    sigaction(SIGTERM, &act, &old_act);

    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_Finalize(void) {
    char kvs_name[MAX_KVS_NAME_LENGTH];
    char kvs_key[MAX_KVS_KEY_LENGTH];
    char kvs_val[MAX_KVS_VAL_LENGTH];
    char rank_str[INT_STR_SIZE];
    if (finalized) {
        return KVS_STATUS_SUCCESS;
    }

    if (my_rank == 0) {
        KVS_CHECK_STATUS(PMIR_Barrier(), "barrier failed");
    }

    finalized = 1;

    applied = 0;

    SET_STR(rank_str, INT_STR_SIZE, RANK_TEMPLATE, my_rank);

    KVS_CHECK_STATUS(h->remove_name_key(KVS_POD_NUM, rank_str), "failed to remove rank");

    while (cut_head(kvs_name, kvs_key, kvs_val, ST_CLIENT)) {
        KVS_CHECK_STATUS(h->remove_name_key(kvs_name, kvs_key), "failed to remove info");
    }

    if (my_rank == 0 && extreme_finalize != 1) {
        KVS_CHECK_STATUS(h->remove_name_key(KVS_UP, KVS_IDX), "failed to remove IDx");
    }
    KVS_CHECK_STATUS(h->remove_name_key(KVS_BARRIER, pmi_hostname),
                     "failed to remove barrier info");

    KVS_CHECK_STATUS(h->finalize(), "failed to finalize");

    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_Barrier(void) {
    size_t min_barrier_num;
    char barrier_num_str[INT_STR_SIZE];

    if (finalized)
        return KVS_STATUS_SUCCESS;

    SET_STR(barrier_num_str, INT_STR_SIZE, SIZE_T_TEMPLATE, barrier_num);

    KVS_CHECK_STATUS(h->set_value(KVS_BARRIER, pmi_hostname, barrier_num_str),
                     "failed to set barrier info");

    KVS_CHECK_STATUS(h->get_barrier_idx(min_barrier_num), "failed to get barrier IDx");
    while (min_barrier_num != barrier_num && finalized != 1) {
        KVS_CHECK_STATUS(h->get_barrier_idx(min_barrier_num), "failed to get barrier IDx");
    }

    barrier_num++;
    if (barrier_num > BARRIER_NUM_MAX)
        barrier_num = 0;

    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_Get_size(int* size_ptr) {
    *size_ptr = count_pods;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_Get_rank(int* rank_ptr) {
    *rank_ptr = my_rank;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Get_my_name(char* kvs_name, size_t length) {
    kvs_str_copy(kvs_name, KVS_NAME, length);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Get_name_length_max(size_t* length) {
    *length = MAX_KVS_NAME_LENGTH;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Get_key_length_max(size_t* length) {
    *length = MAX_KVS_KEY_LENGTH;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Get_value_length_max(size_t* length) {
    *length = MAX_KVS_VAL_LENGTH;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Commit(const char* kvs_name) {
    (void)kvs_name;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Put(const char* kvs_name, const char* key, const char* value) {
    put_key(kvs_name, key, value, ST_CLIENT);

    KVS_CHECK_STATUS(h->set_value(kvs_name, key, value), "failed to set value");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_KVS_Get(const char* kvs_name,
                                         const char* key,
                                         char* value,
                                         size_t length) {
    (void)length;
    std::string value_vec;
    do {
        KVS_CHECK_STATUS(h->get_value_by_name_key(kvs_name, key, value_vec), "failed to get value");
    } while (value_vec.empty());

    snprintf(value, value_vec.length(), "%s", value_vec.c_str());
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_set_resize_function(pmir_resize_fn_t resize_fn) {
    resize_function = resize_fn;
    return KVS_STATUS_SUCCESS;
}

kvs_status_t pmi_resizable::PMIR_Wait_notification(void) {
    return listener.run_listener(h);
}

int pmi_resizable::get_rank() {
    return rank;
}

int pmi_resizable::get_size() {
    return size;
}

size_t pmi_resizable::get_local_thread_idx() {
    return 0;
}
atl_status_t pmi_resizable::get_local_kvs_id(size_t& res) {
    res = 0;
    return ATL_STATUS_SUCCESS;
}
atl_status_t pmi_resizable::set_local_kvs_id(size_t local_kvs_id) {
    return ATL_STATUS_SUCCESS;
}
pmi_resizable::~pmi_resizable() {
    if (!is_finalized) {
        CCL_THROW_IF_NOT(pmrt_finalize(), "pmi finalize failed");
    }
}
