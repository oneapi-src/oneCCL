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
#include "helper.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"

size_t my_rank, count_pods;
size_t barrier_num = 0;
size_t up_idx;
size_t applied = 0;

rank_list_t* killed_ranks = NULL;
size_t killed_ranks_count = 0;

rank_list_t* new_ranks = NULL;
size_t new_ranks_count = 0;

size_t helper::replace_str(char* str, size_t old_rank, size_t new_rank) {
    char old_str[INT_STR_SIZE];
    char new_str[INT_STR_SIZE];
    char* point_to_replace;
    size_t old_str_size;
    size_t new_str_size;

    SET_STR(old_str, INT_STR_SIZE, SIZE_T_TEMPLATE, old_rank);

    SET_STR(new_str, INT_STR_SIZE, SIZE_T_TEMPLATE, new_rank);

    point_to_replace = strstr(str, old_str);
    if (point_to_replace == NULL)
        return 1;
    old_str_size = strlen(old_str);
    new_str_size = strlen(new_str);

    if (old_str_size != new_str_size) {
        size_t rest_len = strlen(point_to_replace);
        memmove(point_to_replace + new_str_size, point_to_replace + old_str_size, rest_len);
    }
    STR_COPY(point_to_replace, new_str, new_str_size);
    return 0;
}

void helper::update_ranks(size_t* old_count, rank_list_t** origin_list, const char* kvs_name) {
    char** rank_nums = NULL;
    size_t rank_count = get_keys_values_by_name(kvs_name, NULL, &rank_nums);
    size_t i;
    size_t cur_count = 0;

    if (rank_count == 0) {
        // *old_count = 0;
        return;
    }

    for (i = 0; i < rank_count; i++) {
        if (rank_list_contains(*origin_list, strtol(rank_nums[i], NULL, 10)))
            continue;

        rank_list_add(origin_list, strtol(rank_nums[i], NULL, 10));
        cur_count++;
    }

    for (i = 0; i < rank_count; i++) {
        free(rank_nums[i]);
    }
    free(rank_nums);

    *old_count += cur_count;
}

void helper::keep_first_n_up(size_t prev_new_ranks_count, size_t prev_killed_ranks_count) {
    rank_list_keep_first_n(&killed_ranks, prev_killed_ranks_count);
    rank_list_keep_first_n(&new_ranks, prev_new_ranks_count);
}

void helper::get_update_ranks(void) {
    update_ranks(&killed_ranks_count, &killed_ranks, KVS_APPROVED_DEAD_POD);
    update_ranks(&new_ranks_count, &new_ranks, KVS_APPROVED_NEW_POD);
}

void helper::get_shift(shift_list_t** list) {
    size_t shift_pods_count = 0;
    size_t max_rank_survivor_pod = count_pods;
    rank_list_t* cur_new = new_ranks;
    rank_list_t* cur_killed = killed_ranks;

    if (killed_ranks != NULL)
        rank_list_sort(killed_ranks);
    if (new_ranks != NULL)
        rank_list_sort(new_ranks);

    while (cur_killed != NULL) {
        if (cur_new != NULL) {
            shift_list_add(list, cur_killed->rank, cur_killed->rank, CH_T_UPDATE);
            cur_new = cur_new->next;
        }
        else {
            while (rank_list_contains(cur_killed, max_rank_survivor_pod - shift_pods_count - 1) ==
                   1)
                max_rank_survivor_pod--;

            if (cur_killed->rank < max_rank_survivor_pod - shift_pods_count) {
                shift_list_add(list,
                               max_rank_survivor_pod - shift_pods_count - 1,
                               cur_killed->rank,
                               CH_T_SHIFT);
                shift_pods_count++;
            }
            else {
                while (cur_killed != NULL) {
                    shift_list_add(list, cur_killed->rank, cur_killed->rank, CH_T_DEAD);
                    cur_killed = cur_killed->next;
                }
                break;
            }
        }
        cur_killed = cur_killed->next;
    }
    while (cur_new != NULL) {
        shift_list_add(list, cur_new->rank, cur_new->rank, CH_T_NEW);
        cur_new = cur_new->next;
    }
}

void helper::up_pods_count(void) {
    count_pods = get_count_names(KVS_POD_NUM);
}

void helper::wait_accept(void) {
    char my_rank_str[MAX_KVS_VAL_LENGTH];

    my_rank = 0;

    while (1) {
        if (get_value_by_name_key(KVS_ACCEPT, my_hostname, my_rank_str) != 0) {
            my_rank = strtol(my_rank_str, NULL, 10);
            break;
        }
    }
}

void helper::clean_dead_pods_info(rank_list_t* dead_up_idx) {
    size_t i;
    size_t count_death;
    char** kvs_keys = NULL;

    while (dead_up_idx != NULL) {
        count_death = get_keys_values_by_name(KVS_APPROVED_DEAD_POD, &kvs_keys, NULL);

        for (i = 0; i < count_death; i++) {
            remove_name_key(KVS_APPROVED_DEAD_POD, kvs_keys[i]);
            dead_up_idx = dead_up_idx->next;
            if (dead_up_idx == NULL) {
                for (; i < count_death; i++) {
                    free(kvs_keys[i]);
                }
                break;
            }
            free(kvs_keys[i]);
        }
    }
    if (kvs_keys != NULL)
        free(kvs_keys);
}

void helper::accept_new_ranks(shift_list_t* cur_list) {
    char new_rank_str[INT_STR_SIZE];
    char old_rank_str[INT_STR_SIZE];
    char** kvs_values = NULL;
    char** kvs_keys = NULL;
    size_t count_values;
    size_t i = 0;

    while (cur_list != NULL) {
        if (cur_list->shift.type == CH_T_UPDATE || cur_list->shift.type == CH_T_NEW) {
            SET_STR(old_rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, cur_list->shift.old_rank);
            SET_STR(new_rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, cur_list->shift.new_rank);

            count_values = get_keys_values_by_name(KVS_APPROVED_NEW_POD, &kvs_keys, &kvs_values);

            for (i = 0; i < count_values; i++) {
                if (!strcmp(kvs_values[i], old_rank_str)) {
                    set_value(KVS_ACCEPT, kvs_keys[i], new_rank_str);
                    break;
                }
            }
            for (i = 0; i < count_values; i++) {
                free(kvs_keys[i]);
                free(kvs_values[i]);
            }
        }
        cur_list = cur_list->next;
    }

    while ((count_values = get_keys_values_by_name(KVS_ACCEPT, NULL, &kvs_values)) != 0) {
        for (i = 0; i < count_values; i++) {
            free(kvs_values[i]);
        }
    }

    if (kvs_keys != NULL)
        free(kvs_keys);
    if (kvs_values != NULL)
        free(kvs_values);
}

void helper::update_kvs_info(size_t new_rank) {
    char kvs_name[MAX_KVS_NAME_LENGTH];
    char kvs_key[MAX_KVS_KEY_LENGTH];
    char kvs_val[MAX_KVS_VAL_LENGTH];
    size_t kvs_list_size = get_kvs_list_size(ST_CLIENT);
    size_t k;

    for (k = 0; k < kvs_list_size; k++) {
        cut_head(kvs_name, kvs_key, kvs_val, ST_CLIENT);

        remove_name_key(kvs_name, kvs_key);

        replace_str(kvs_key, my_rank, new_rank);

        set_value(kvs_name, kvs_key, kvs_val);

        put_key(kvs_name, kvs_key, kvs_val, ST_CLIENT);
    }
}

void helper::move_to_new_rank(size_t new_rank) {
    char rank_str[INT_STR_SIZE];

    update_kvs_info(new_rank);
    my_rank = new_rank;

    SET_STR(rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, my_rank);

    //    request_set_val(KVS_POD_REQUEST, my_hostname, rank_str);

    set_value(KVS_POD_NUM, rank_str, my_hostname);
}

void helper::update_my_info(shift_list_t* list) {
    char rank_str[INT_STR_SIZE];

    while (list != NULL) {
        if (list->shift.old_rank == my_rank) {
            size_t old_rank = my_rank;
            move_to_new_rank(list->shift.new_rank);

            SET_STR(rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, old_rank);

            remove_name_key(KVS_POD_NUM, rank_str);

            break;
        }
        list = list->next;
    }
}

size_t helper::get_barrier_idx(void) {
    char** kvs_values = NULL;
    size_t count_kvs_values = 0;
    size_t tmp_barrier_num;
    size_t min_barrier_num;
    size_t i = 0;

    count_kvs_values = get_keys_values_by_name(KVS_BARRIER, NULL, &kvs_values);
    if (count_kvs_values == 0)
        return 0;

    min_barrier_num = strtol(kvs_values[0], NULL, 10);
    for (i = 1; i < count_kvs_values; i++) {
        tmp_barrier_num = strtol(kvs_values[i], NULL, 10);
        if (min_barrier_num > tmp_barrier_num)
            min_barrier_num = tmp_barrier_num;
    }
    for (i = 0; i < count_kvs_values; i++) {
        free(kvs_values[i]);
    }
    free(kvs_values);
    return min_barrier_num;
}

void helper::post_my_info(void) {
    char barrier_num_str[INT_STR_SIZE];
    char my_rank_str[INT_STR_SIZE];

    applied = 1;

    SET_STR(my_rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, my_rank);

    set_value(KVS_POD_NUM, my_rank_str, my_hostname);

    barrier_num = get_barrier_idx();

    SET_STR(barrier_num_str, INT_STR_SIZE, SIZE_T_TEMPLATE, barrier_num);

    set_value(KVS_BARRIER, my_hostname, barrier_num_str);

    remove_name_key(KVS_ACCEPT, my_hostname);

    remove_name_key(KVS_APPROVED_NEW_POD, my_hostname);

    barrier_num++;
    if (barrier_num > BARRIER_NUM_MAX)
        barrier_num = 0;
}

size_t helper::update(shift_list_t** list, rank_list_t** dead_up_idx, size_t root_rank) {
    if (applied == 1) {
        if ((*list) != NULL) {
            if (my_rank == root_rank) {
                if ((*dead_up_idx) != NULL)
                    clean_dead_pods_info(*dead_up_idx);

                accept_new_ranks(*list);
            }
            update_my_info(*list);
        }
    }
    else
        post_my_info();

    return 0;
}

size_t helper::get_val_count(const char* name, const char* val) {
    size_t res = 0;
    char** kvs_values = NULL;
    size_t count_values;
    size_t i;

    count_values = get_keys_values_by_name(name, NULL, &kvs_values);

    if (count_values == 0)
        return res;

    for (i = 0; i < count_values; i++) {
        if (!strcmp(val, kvs_values[i])) {
            res++;
        }
        free(kvs_values[i]);
    }
    free(kvs_values);

    return res;
}

size_t helper::get_occupied_ranks_count(char* rank) {
    char occupied_rank_val_str[MAX_KVS_VAL_LENGTH];
    size_t is_occupied_rank;
    size_t count_new_pod = 0;
    size_t count_seen_new_pod = 0;

    is_occupied_rank =
        (get_value_by_name_key(KVS_POD_NUM, rank, occupied_rank_val_str) == 0) ? 0 : 1;

    count_new_pod = get_val_count(KVS_NEW_POD, rank);

    count_seen_new_pod = get_val_count(KVS_APPROVED_NEW_POD, rank);

    return is_occupied_rank + count_new_pod + count_seen_new_pod;
}

size_t helper::get_count_requested_ranks(char* rank) {
    size_t count_pods_with_my_rank = 0;

    count_pods_with_my_rank = get_val_count(KVS_POD_REQUEST, rank);

    return count_pods_with_my_rank;
}

void helper::occupied_rank(char* rank) {
    char idx_val[MAX_KVS_VAL_LENGTH];
    size_t is_inited;

    is_inited = get_value_by_name_key(KVS_UP, KVS_IDX, idx_val);

    if ((is_inited == 0) && (my_rank == 0)) {
        set_value(KVS_UP, KVS_IDX, INITIAL_UPDATE_IDX);

        count_pods = 1;

        update(NULL, NULL, 0);
    }
    else {
        set_value(KVS_NEW_POD, my_hostname, rank);
    }
}

void helper::reg_rank(void) {
    char rank_str[INT_STR_SIZE];
    size_t wait_shift = 0;
    char** kvs_values = NULL;
    char** kvs_keys = NULL;
    size_t count_values = 0;
    size_t my_num_in_pod_request_line = 0;
    size_t i;

    my_rank = 0;
    set_value(KVS_POD_REQUEST, my_hostname, INITIAL_RANK_NUM);

    SET_STR(rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, my_rank);

    while (1) {
        wait_shift = 0;
        my_num_in_pod_request_line = 0;

        count_values = get_keys_values_by_name(KVS_POD_REQUEST, &kvs_keys, &kvs_values);

        for (i = 0; i < count_values; i++) {
            if (!strcmp(kvs_values[i], rank_str)) {
                my_num_in_pod_request_line++;
                if (!strcmp(kvs_keys[i], my_hostname))
                    break;
            }
        }
        for (i = 0; i < count_values; i++) {
            free(kvs_keys[i]);
            free(kvs_values[i]);
        }

        if (my_num_in_pod_request_line == 1) {
            if (get_occupied_ranks_count(rank_str) != 0) {
                wait_shift = 0;
            }
            else {
                wait_shift = 1;
                if (get_count_requested_ranks(rank_str) == 1) {
                    occupied_rank(rank_str);
                    break;
                }
            }
        }

        if (!wait_shift) {
            my_rank++;
            SET_STR(rank_str, INT_STR_SIZE, SIZE_T_TEMPLATE, my_rank);
            set_value(KVS_POD_REQUEST, my_hostname, rank_str);
        }
    }

    remove_name_key(KVS_POD_REQUEST, my_hostname);

    if (kvs_keys != NULL)
        free(kvs_keys);
    if (kvs_values != NULL)
        free(kvs_values);
}

size_t helper::get_replica_size(void) {
    return k->kvs_get_replica_size();
}

void helper::up_kvs(const char* new_kvs_name, const char* old_kvs_name) {
    char** kvs_values = NULL;
    char** kvs_keys = NULL;
    size_t i = 0;
    size_t count_values;

    count_values = get_keys_values_by_name(old_kvs_name, &kvs_keys, &kvs_values);
    for (i = 0; i < count_values; i++) {
        remove_name_key(old_kvs_name, kvs_keys[i]);

        set_value(new_kvs_name, kvs_keys[i], kvs_values[i]);

        free(kvs_keys[i]);
        free(kvs_values[i]);
    }
    if (kvs_keys != NULL)
        free(kvs_keys);
    if (kvs_values != NULL)
        free(kvs_values);
}

void helper::up_kvs_new_and_dead(void) {
    up_kvs(KVS_APPROVED_NEW_POD, KVS_NEW_POD);
    up_kvs(KVS_APPROVED_DEAD_POD, KVS_DEAD_POD);
}

void helper::get_new_root(size_t* old_root) {
    size_t i;
    char** rank_nums = NULL;
    size_t rank_count = get_keys_values_by_name(KVS_DEAD_POD, NULL, &rank_nums);

    for (i = 0; i < rank_count; i++) {
        if (*old_root == (size_t)strtol(rank_nums[i], NULL, 10))
            (*old_root)++;
        free(rank_nums[i]);
    }
    if (rank_nums != NULL)
        free(rank_nums);
}

size_t helper::get_keys_values_by_name(const char* kvs_name, char*** kvs_keys, char*** kvs_values) {
    return k->kvs_get_keys_values_by_name(kvs_name, kvs_keys, kvs_values);
}
size_t helper::set_value(const char* kvs_name, const char* kvs_key, const char* kvs_val) {
    return k->kvs_set_value(kvs_name, kvs_key, kvs_val);
}
size_t helper::remove_name_key(const char* kvs_name, const char* kvs_key) {
    return k->kvs_remove_name_key(kvs_name, kvs_key);
}
size_t helper::get_value_by_name_key(const char* kvs_name, const char* kvs_key, char* kvs_val) {
    return k->kvs_get_value_by_name_key(kvs_name, kvs_key, kvs_val);
}
size_t helper::init(const char* main_addr) {
    return k->kvs_init(main_addr);
}
size_t helper::main_server_address_reserve(char* main_addr) {
    return k->kvs_main_server_address_reserve(main_addr);
}
size_t helper::get_count_names(const char* kvs_name) {
    return k->kvs_get_count_names(kvs_name);
}
size_t helper::finalize(void) {
    return k->kvs_finalize();
}
