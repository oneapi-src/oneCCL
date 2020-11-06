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

static size_t root_rank = 0;
static size_t is_new_root = 0;
static size_t ask_only_framework = 0;
static size_t finalized = 0;
static size_t extreme_finalize = 0;
static struct sigaction old_act;
char my_hostname[MAX_KVS_VAL_LENGTH];

// TODO: rework it for multi kvs
static pmi_resizable* pmi_object;

void Call_Hard_finilize(int sig) {
    pmi_object->Hard_finilize(sig);
}

kvs_resize_action_t pmi_resizable::default_checker(size_t comm_size) {
    char* comm_size_to_start_env;
    size_t comm_size_to_start;

    comm_size_to_start_env = getenv(CCL_WORLD_SIZE_ENV);

    if (comm_size_to_start_env != NULL)
        comm_size_to_start = strtol(comm_size_to_start_env, NULL, 10);
    else
        comm_size_to_start = h->get_replica_size();
    if (comm_size >= comm_size_to_start)
        return KVS_RA_RUN;

    return KVS_RA_WAIT;
}

kvs_resize_action_t pmi_resizable::call_resize_fn(size_t comm_size) {
    if (resize_function != nullptr)
        return resize_function(comm_size);

    return default_checker(comm_size);
}

int pmi_resizable::PMIR_Update(void) {
    char up_idx_str[MAX_KVS_VAL_LENGTH];
    size_t prev_new_ranks_count = 0;
    size_t prev_killed_ranks_count = 0;
    int prev_idx = -1;
    kvs_resize_action_t answer;
    rank_list_t* dead_up_idx = NULL;
    shift_list_t* list = NULL;

    new_ranks_count = 0;
    killed_ranks_count = 0;
    if (finalized == 1) {
        return 1;
    }
    if (applied == 1) {
        size_t is_wait = 1;
        size_t is_first_collect = 0;

        h->get_value_by_name_key(KVS_UP, KVS_IDX, up_idx_str);

        up_idx = strtol(up_idx_str, NULL, 10);
        if (up_idx == 0)
            is_first_collect = 1;

        do {
            size_t count_clean_checks = 0;
            size_t count_applied_changes = 0;
            do {
                /*Waiting new pods*/
                usleep(10000);
                h->get_value_by_name_key(KVS_UP, KVS_IDX, up_idx_str);

                up_idx = strtol(up_idx_str, NULL, 10);
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
                        size_t old_root = root_rank;
                        h->get_new_root(&root_rank);

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

                    h->get_update_ranks();
                    if (killed_ranks_count != prev_killed_ranks_count)
                        rank_list_add(&dead_up_idx, up_idx);
                }
                PMIR_Barrier();
                if (my_rank == root_rank && is_new_root == 0) {
                    up_idx++;
                    if (up_idx > MAX_UP_IDX)
                        up_idx = 1;

                    SET_STR(up_idx_str, INT_STR_SIZE, SIZE_T_TEMPLATE, up_idx);
                    h->set_value(KVS_UP, KVS_IDX, up_idx_str);
                    h->up_kvs_new_and_dead();
                }
                PMIR_Barrier();

                if (finalized == 1) {
                    rank_list_clean(&killed_ranks);
                    rank_list_clean(&new_ranks);
                    rank_list_clean(&dead_up_idx);
                    return 1;
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
                if (h->get_replica_size() != count_pods - killed_ranks_count + new_ranks_count)
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
                    PMIR_Finalize();
                    return 1;
                }
                default: {
                    printf("Unknown resize action: %d\n", answer);
                    PMIR_Finalize();
                    return 1;
                }
            }
            listener.set_applied_count(count_applied_changes);
        } while (is_wait == 1);
    }
    else {
        listener.send_notification(0, h);
        h->wait_accept();
    }

    h->get_shift(&list);
    count_pods = count_pods - killed_ranks_count + new_ranks_count;
    h->update(&list, &dead_up_idx, root_rank);

    root_rank = 0;

    PMIR_Barrier();
    h->up_pods_count();

    rank_list_clean(&killed_ranks);
    rank_list_clean(&new_ranks);
    rank_list_clean(&dead_up_idx);
    shift_list_clean(&list);
    return 0;
}

void pmi_resizable::Hard_finilize(int sig) {
    char rank_str[INT_STR_SIZE];

    SET_STR(rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, my_rank);

    h->set_value(KVS_DEAD_POD, my_hostname, rank_str);

    listener.send_notification(sig, h);

    extreme_finalize = 1;
    PMIR_Finalize();
    if (old_act.sa_handler != NULL)
        old_act.sa_handler(sig);
}

int pmi_resizable::PMIR_Main_Addr_Reserv(char* main_addr) {
    h->main_server_address_reserve(main_addr);
    return 0;
}

int pmi_resizable::PMIR_Init(const char* main_addr) {
    struct sigaction act;
    FILE* fp;
    finalized = 0;
    memset(my_hostname, 0, MAX_KVS_VAL_LENGTH);
    if ((fp = popen("hostname", READ_ONLY)) == NULL) {
        printf("Can't get hostname\n");
        exit(1);
    }
    CHECK_FGETS(fgets(my_hostname, MAX_KVS_VAL_LENGTH, fp), my_hostname);
    pclose(fp);
    while (my_hostname[strlen(my_hostname) - 1] == '\n' ||
           my_hostname[strlen(my_hostname) - 1] == ' ')
        my_hostname[strlen(my_hostname) - 1] = '\0';

    SET_STR(&(my_hostname[strlen(my_hostname)]),
            MAX_KVS_VAL_LENGTH - (int)strlen(my_hostname) - 1,
            "-%d",
            getpid());

    if (h->init(main_addr)) {
        return 1;
    }

    h->reg_rank();

    h->up_pods_count();

    // TODO: rework it for multi kvs
    pmi_object = this;
    memset(&act, 0, sizeof(act));
    act.sa_handler = &Call_Hard_finilize;
    act.sa_flags = 0;
    sigaction(SIGTERM, &act, &old_act);

    return 0;
}

int pmi_resizable::PMIR_Finalize(void) {
    char kvs_name[MAX_KVS_NAME_LENGTH];
    char kvs_key[MAX_KVS_KEY_LENGTH];
    char kvs_val[MAX_KVS_VAL_LENGTH];
    char rank_str[INT_STR_SIZE];
    if (finalized)
        return 0;

    if (my_rank == 0)
        PMIR_Barrier();

    finalized = 1;

    applied = 0;

    SET_STR(rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, my_rank);

    h->remove_name_key(KVS_POD_NUM, rank_str);

    while (cut_head(kvs_name, kvs_key, kvs_val, ST_CLIENT)) {
        h->remove_name_key(kvs_name, kvs_key);
    }

    if (my_rank == 0 && extreme_finalize != 1) {
        h->remove_name_key(KVS_UP, KVS_IDX);
    }
    h->remove_name_key(KVS_BARRIER, my_hostname);

    h->finalize();

    return 0;
}

int pmi_resizable::PMIR_Barrier(void) {
    size_t min_barrier_num;
    char barrier_num_str[INT_STR_SIZE];

    if (finalized)
        return 0;

    SET_STR(barrier_num_str, INT_STR_SIZE, SIZE_T_TEMPLATE, barrier_num);

    h->set_value(KVS_BARRIER, my_hostname, barrier_num_str);

    min_barrier_num = h->get_barrier_idx();
    while (min_barrier_num != barrier_num && finalized != 1) {
        min_barrier_num = h->get_barrier_idx();
    }

    barrier_num++;
    if (barrier_num > BARRIER_NUM_MAX)
        barrier_num = 0;

    return 0;
}

int pmi_resizable::PMIR_Get_size(size_t* size) {
    *size = count_pods;
    return 0;
}

int pmi_resizable::PMIR_Get_rank(size_t* rank) {
    *rank = my_rank;
    return 0;
}

int pmi_resizable::PMIR_KVS_Get_my_name(char* kvs_name, size_t length) {
    STR_COPY(kvs_name, KVS_NAME, length);
    return 0;
}

int pmi_resizable::PMIR_KVS_Get_name_length_max(size_t* length) {
    *length = MAX_KVS_NAME_LENGTH;
    return 0;
}

int pmi_resizable::PMIR_KVS_Get_key_length_max(size_t* length) {
    *length = MAX_KVS_KEY_LENGTH;
    return 0;
}

int pmi_resizable::PMIR_KVS_Get_value_length_max(size_t* length) {
    *length = MAX_KVS_VAL_LENGTH;
    return 0;
}

int pmi_resizable::PMIR_KVS_Commit(const char* kvs_name) {
    (void)kvs_name;
    return 0;
}

int pmi_resizable::PMIR_KVS_Put(const char* kvs_name, const char* key, const char* value) {
    put_key(kvs_name, key, value, ST_CLIENT);

    h->set_value(kvs_name, key, value);
    return 0;
}

int pmi_resizable::PMIR_KVS_Get(const char* kvs_name, const char* key, char* value, size_t length) {
    (void)length;
    while (h->get_value_by_name_key(kvs_name, key, value) == 0) {
    }

    return 0;
}

int pmi_resizable::PMIR_set_resize_function(pmir_resize_fn_t resize_fn) {
    resize_function = resize_fn;
    return 0;
}

int pmi_resizable::PMIR_Wait_notification(void) {
    return listener.run_listener(h);
}

size_t pmi_resizable::get_rank() {
    return rank;
}

size_t pmi_resizable::get_size() {
    return size;
}

size_t pmi_resizable::get_thread() {
    return 0;
}
size_t pmi_resizable::get_local_kvs_id() {
    return 0;
}
void pmi_resizable::set_local_kvs_id(size_t local_kvs_id) {}
pmi_resizable::~pmi_resizable() {
    if (!is_finalized)
        pmrt_finalize();
}
