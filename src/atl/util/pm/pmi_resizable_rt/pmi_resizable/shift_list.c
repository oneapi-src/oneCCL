/*
 Copyright 2016-2019 Intel Corporation
 
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

#include <stdio.h>
#include <stdlib.h>

#include "shift_list.h"

void shift_list_clean(shift_list_t** list)
{
    shift_list_t* cur_list = (*list);
    shift_list_t* node_to_remove;
    while (cur_list != NULL)
    {
        node_to_remove = cur_list;
        cur_list = cur_list->next;
        free(node_to_remove);
    }
    (*list) = NULL;
}

void shift_list_add(shift_list_t** list, size_t old, size_t new, change_type_t type)
{
    shift_list_t* cur_list;
    if ((*list) == NULL)
    {
        (*list) = (shift_list_t*)malloc(sizeof(shift_list_t));
        cur_list = (*list);
    }
    else
    {
        cur_list = (*list);
        while (cur_list->next != NULL)
            cur_list = cur_list->next;
        cur_list->next = (shift_list_t*)malloc(sizeof(shift_list_t));
        cur_list = cur_list->next;
    }
    cur_list->shift.old = old;
    cur_list->shift.new = new;
    cur_list->shift.type = type;
    cur_list->next = NULL;
}
