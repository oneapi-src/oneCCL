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
#ifndef SHIFT_LIST_H_INCLUDED
#define SHIFT_LIST_H_INCLUDED

typedef enum change_type {
    CH_T_SHIFT = 0,
    CH_T_DEAD = 1,
    CH_T_NEW = 2,
    CH_T_UPDATE = 3,
} change_type_t;

typedef struct shift_rank {
    size_t old;
    size_t new;
    change_type_t type;
} shift_rank_t;

typedef struct shift_list {
    shift_rank_t shift;
    struct shift_list* next;
} shift_list_t;

void shift_list_clean(shift_list_t** list);

void shift_list_add(shift_list_t** list, size_t old, size_t new, change_type_t type);

#endif
