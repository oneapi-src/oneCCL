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
#ifndef HELPER_H_INCLUDED
#define HELPER_H_INCLUDED

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <ctype.h>
#include <iostream>
#include <memory>
#include <utility>
#include "def.h"
#include "rank_list.h"
#include "shift_list.h"
#include "kvs_keeper.h"
#include "kvs/ikvs_wrapper.h"

extern size_t my_rank, count_pods;
extern size_t barrier_num;
extern size_t up_idx;
extern size_t applied;

extern rank_list_t* killed_ranks;
extern size_t killed_ranks_count;

extern rank_list_t* new_ranks;
extern size_t new_ranks_count;

class helper {
public:
    helper() = delete;
    explicit helper(std::shared_ptr<ikvs_wrapper> k) : k(std::move(k)){};
    ~helper() = default;

    void get_update_ranks(void);

    size_t get_replica_size(void);

    void wait_accept(void);

    size_t update(shift_list_t** list, rank_list_t** dead_up_idx, size_t root_rank);

    void up_pods_count(void);

    void get_shift(shift_list_t** list);

    void reg_rank(void);

    size_t get_barrier_idx(void);

    void up_kvs_new_and_dead(void);

    void keep_first_n_up(size_t prev_new_ranks_count, size_t prev_killed_ranks_count);

    void get_new_root(size_t* old_root);

    /*Work with KVS, new*/
    size_t set_value(const char* kvs_name, const char* kvs_key, const char* kvs_val);

    size_t remove_name_key(const char* kvs_name, const char* kvs_key);

    size_t get_value_by_name_key(const char* kvs_name, const char* kvs_key, char* kvs_val);

    size_t init(const char* main_addr);

    size_t main_server_address_reserve(char* main_addr);

    size_t get_count_names(const char* kvs_name);

    size_t finalize(void);

    size_t get_keys_values_by_name(const char* kvs_name, char*** kvs_keys, char*** kvs_values);

    /*Work with KVS, new*/

private:
    size_t replace_str(char* str, size_t old_rank, size_t new_rank);

    void update_ranks(size_t* old_count, rank_list_t** origin_list, const char* kvs_name);

    void clean_dead_pods_info(rank_list_t* dead_up_idx);

    void accept_new_ranks(shift_list_t* cur_list);

    void update_kvs_info(size_t new_rank);

    void move_to_new_rank(size_t new_rank);

    void update_my_info(shift_list_t* list);

    void post_my_info(void);

    size_t get_val_count(const char* name, const char* val);

    size_t get_occupied_ranks_count(char* rank);

    size_t get_count_requested_ranks(char* rank);

    void occupied_rank(char* rank);

    void up_kvs(const char* new_kvs_name, const char* old_kvs_name);
    std::shared_ptr<ikvs_wrapper> k;
};
#endif
