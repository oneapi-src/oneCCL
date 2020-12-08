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
#ifndef INT_LIST_H_INCLUDED
#define INT_LIST_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif
typedef struct rank_list {
    int rank;
    struct rank_list* next;
} rank_list_t;

size_t rank_list_contains(rank_list_t* list, int rank);

void rank_list_clean(rank_list_t** list);

void rank_list_sort(rank_list_t* list);

void rank_list_keep_first_n(rank_list_t** origin_list, size_t n);

void rank_list_add(rank_list_t** origin_list, int rank);

#ifdef __cplusplus
}
#endif
#endif
