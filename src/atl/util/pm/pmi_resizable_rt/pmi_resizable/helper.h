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

#include "def.h"
#include "rank_list.h"
#include "shift_list.h"
#include "kvs_keeper.h"
#include "listener.h"

extern size_t my_rank, count_pods;
extern size_t barrier_num;
extern size_t up_idx;
extern size_t initialized;

extern rank_list_t* killed_ranks;
extern size_t killed_ranks_count;

extern rank_list_t* new_ranks;
extern size_t new_ranks_count;

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

#endif
